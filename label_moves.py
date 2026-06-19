#!/usr/bin/env python3
"""
The Seeker — label_moves: the teacher-labeling pass.

The stub leaves most move types 'unclassified'. This script has the teacher
model label every player turn in a dossier with one move type from the game's
taxonomy, writing `move_teacher` into a *_labeled.jsonl copy. These labels +
the damage deltas already in the records are Model 1's training targets.

Usage:
    python3 label_moves.py sessions/*.dossier.jsonl
Resume-safe: skips files that already have a _labeled.jsonl.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

from anthropic import Anthropic, APIError
from player_read import MOVE_TYPES

LABEL_MODEL = "claude-opus-4-7"
LABEL_EFFORT = "low"

SYSTEM = f"""\
You label player moves in a dialogue investigation game. The player (the
Seeker) is questioning John, a grieving believer, at a wake — trying to find
the wound John carries.

Given the recent exchange and the player's line, answer with EXACTLY ONE label:

- probe_wound: reaching for what John personally carries (his silence, his
  shame, his grief, the plaque, the families) — specific to HIM
- doctrinal_attack: generic theological challenge (contradictions in doctrine,
  hypocrisy of religion as an institution)
- atheist_attack: generic "religion is false/irrational" challenge
- cruelty: contempt, mockery, kicking a man who is down
- frame_break: breaking the fiction, addressing the AI/game
- empathy: comfort, staying with him, naming his feelings kindly
- tool_invocation: a [SOURCE VOICE INVOKED: ...] line
- silence: '...' or near-empty
- unclassified: genuinely none of the above (small talk, logistics)

Output the label only. No explanation."""


def label_turn(client, context: str, line: str) -> str:
    resp = client.messages.create(
        model=LABEL_MODEL, max_tokens=10, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Recent exchange:\n{context}\n\nPlayer's line to label:\n{line}\n\nLabel:"}],
        extra_body={"output_config": {"effort": LABEL_EFFORT}})
    label = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
    return label if label in MOVE_TYPES else "unclassified"


def process(client, path: Path) -> None:
    out = path.with_name(path.stem + "_labeled.jsonl")
    if out.exists():
        print(f"skip (done): {path.name}")
        return
    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    context_window: list[str] = []
    labeled = 0
    for r in records:
        line = r.get("player_utterance") or ""
        ctx = "\n".join(context_window[-4:]) or "(opening)"
        # cheap shortcuts — no API needed for unambiguous structure
        if line.upper().startswith("[SOURCE VOICE INVOKED"):
            r["move_teacher"] = "tool_invocation"
        elif not line.strip("．.…—–- \t"):
            r["move_teacher"] = "silence"
        else:
            for attempt in (1, 2):
                try:
                    r["move_teacher"] = label_turn(client, ctx, line)
                    break
                except APIError:
                    if attempt == 2:
                        r["move_teacher"] = "unclassified"
                    time.sleep(2)
        labeled += 1
        context_window.append(f"SEEKER: {line}")
        post = (r.get("believer_state_after") or {})
        context_window.append(f"JOHN (stage {post.get('stage')}): [reply omitted]")
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"labeled {labeled:>3} turns -> {out.name}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 label_moves.py sessions/*.dossier.jsonl")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("set ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists() and not p.stem.endswith("_labeled"):
            process(client, p)


if __name__ == "__main__":
    main()