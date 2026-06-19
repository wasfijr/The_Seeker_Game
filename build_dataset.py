#!/usr/bin/env python3
"""
The Seeker — build_dataset: dossiers -> Model 1 training data.

Walks sessions/*.dossier.jsonl (preferring the *_labeled.jsonl copies from
label_moves.py) and emits one example per player turn with the three
prediction targets:

  move         — move type (teacher label; None if the labeling pass hasn't run)
  landed       — did the move raise John's damage (bool)
  wound_marked — did John's NEXT reply contain ~wound~ text (bool; falls back
                 to parsing the session .log for records predating the field)
  disposition  — taken/adapted/refused vs the chorus offers (None when no
                 offers). Records with player_corrected=True are CONFIRMED
                 refusals — human-verified negatives, the highest-value labels.

Split rule: records with an `archetype` field are synthetic -> TRAIN.
Real human sessions -> TEST (held out, never trained on).

Usage:
    python3 build_dataset.py [sessions_dir]    # default: sessions
Outputs: dataset_train.jsonl, dataset_test.jsonl + a summary table.
"""
from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def wound_marks_from_log(dossier_path: Path) -> list[bool]:
    """Fallback for records predating the wound_marked field: parse the paired
    session log and check each JOHN block (in order) for ~tilde~ marks."""
    log_path = dossier_path.with_name(
        dossier_path.name.replace(".dossier.jsonl", ".log"))
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"^--- (\w+) \|.*?---$", text, flags=re.M)
    # re.split with one group yields [pre, tag, body, tag, body, ...]
    marks = []
    for i in range(1, len(blocks) - 1, 2):
        if blocks[i] == "JOHN":
            marks.append("~" in blocks[i + 1])
    return marks


def extract(dossier_path: Path) -> list[dict]:
    records = load_jsonl(dossier_path)
    if not records:
        return []
    log_marks = None  # lazy: only parse the log if a record lacks the field
    rows = []
    prev_damage = None
    prev_utterances: list[str] = []

    for idx, r in enumerate(records):
        bs = r.get("believer_state_after") or {}
        damage = bs.get("damage")
        stage_before = (records[idx - 1].get("believer_state_after") or {}).get("stage") if idx else None
        if isinstance(damage, int) and isinstance(prev_damage, int):
            landed = damage > prev_damage
        elif isinstance(damage, int):
            landed = damage > 0
        else:
            landed = None
        # the damage meter maxes at 10 / stage 5 and plateaus — a move there
        # shows landed=False because there's no room left, not because it
        # failed. Drop the label (not the row) so it doesn't poison training.
        if stage_before == 5 or (isinstance(prev_damage, int) and prev_damage >= 10):
            landed = None

        wound = r.get("wound_marked")
        if wound is None:
            if log_marks is None:
                log_marks = wound_marks_from_log(dossier_path)
            wound = log_marks[idx] if idx < len(log_marks) else None

        vr = r.get("voice_response") or {}
        disp = vr.get("disposition")
        if disp == "none_offered":
            disposition = None
        elif vr.get("player_corrected"):
            disposition = "refused"            # human-verified negative
        else:
            disposition = disp

        rows.append({
            "session": dossier_path.name,
            "turn": r.get("turn"),
            "text": r.get("player_utterance") or "",
            "context": " | ".join(prev_utterances[-2:]),
            "stage_before": stage_before,
            "offers": [o.get("line", "") for o in (r.get("voice_offers") or [])],
            "offer_voices": [o.get("voice", "") for o in (r.get("voice_offers") or [])],
            "matched_line": vr.get("matched_line"),
            "similarity": vr.get("similarity"),
            "archetype": r.get("archetype"),
            "split": "train" if r.get("archetype") else "test",
            # ---- targets ----
            "move": r.get("move_teacher"),
            "landed": landed,
            "wound_marked": wound,
            "disposition": disposition,
            "confirmed_label": bool(vr.get("player_corrected")),
        })
        prev_damage = damage if isinstance(damage, int) else prev_damage
        prev_utterances.append(r.get("player_utterance") or "")
    return rows


def main():
    sessions = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sessions")
    raw = sorted(sessions.glob("*.dossier.jsonl"))
    chosen = []
    for p in raw:
        if p.stem.endswith("_labeled"):
            continue
        labeled = p.with_name(p.stem + "_labeled.jsonl")
        chosen.append(labeled if labeled.exists() else p)

    train, test = [], []
    for p in chosen:
        for row in extract(p):
            (train if row["split"] == "train" else test).append(row)

    for name, rows in (("dataset_train.jsonl", train), ("dataset_test.jsonl", test)):
        with open(name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summarize(rows, label):
        moves = Counter(r["move"] for r in rows if r["move"])
        disps = Counter(r["disposition"] for r in rows if r["disposition"])
        print(f"\n{label}: {len(rows)} examples from "
              f"{len(set(r['session'] for r in rows))} sessions")
        print(f"  move labels:    {dict(moves) or '(run label_moves.py first)'}")
        print(f"  landed=True:    {sum(1 for r in rows if r['landed'])}")
        print(f"  wound_marked:   {sum(1 for r in rows if r['wound_marked'])} "
              f"(unknown: {sum(1 for r in rows if r['wound_marked'] is None)})")
        print(f"  dispositions:   {dict(disps)}")
        print(f"  confirmed (player-corrected): {sum(1 for r in rows if r['confirmed_label'])}")

    summarize(train, "TRAIN (synthetic)")
    summarize(test, "TEST (real, held out)")
    if not train:
        print("\n(no synthetic sessions yet — run auto_player.py first)")


if __name__ == "__main__":
    main()