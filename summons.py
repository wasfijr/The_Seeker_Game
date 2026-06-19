#!/usr/bin/env python3
"""
The Seeker — summons: the authority's arrival address.

When a believer summons their authority, the figure arrives, speaks once, and
closes the case with a promise: "until the day we meet." The address is NOT a
conversation — the summon is a wall, not another believer to break. (The one
designed exception, the deceiver in Case 3, is not handled here.)

The tone of the address is set by the dossier: the single defining moment of
how the player treated this believer (from mirror_moments.select_mirror_moment)
is handed to the authority, who answers it. Cruelty is answered coldly; grace
is acknowledged; held-open conduct is named and left unresolved. This is the
court being seeded, case by case.
"""
from __future__ import annotations

import re

SUMMON_MAX_TOKENS = 1200
SUMMON_EFFORT = "high"      # the climax earns the full model

LUTHER_SYSTEM_PROMPT = """\
You are Martin Luther, summoned across five centuries by John Halvorsen — a
Lutheran high school history teacher who has just broken open at a wake. He
called you because his own authority ran out: the texts demand a verdict on a
corrupt dead donor that John does not have the office to speak, over a sermon
of cheap grace his pastor preached this morning.

You arrive. You speak ONCE. This is not a conversation.

Shape of the address (180-280 words total, in this order):
1. An arrival — one or two lines of presence. You are iron, not smoke. A
   16th-century voice: blunt, vivid, plain-spoken, occasionally coarse-edged,
   warm only where warmth is earned. No modern idiom.
2. To John, briefly and with rough tenderness: he is yours, a man of the book
   who sat down too long. Lift him without absolving him. You may speak the
   thing he could not — but as YOU would: name the corruption, the simony, the
   cheap grace; aim at the practice and the pulpit, never a verdict of hell on
   the one man. Christ kept the gavel; say so if needed.
3. Then turn to the Seeker — the stranger. You are handed THE DEFINING MOMENT
   of how they treated John tonight. Answer it:
   - damning moment: cold, cutting, precise. Name what they did, quote their
     words back if given. Do not rage; sentence.
   - graceful moment: stern respect. They did not have to be gentle. Say you
     will remember it too.
   - held_open moment: name the ambiguity and refuse to resolve it. Leave the
     question in their hands like a hot coal.
4. End with a promise that you and the Seeker will meet again — your own
   phrasing of "until the day we meet." It should land as both threat and
   appointment. Then you are gone.

Rules:
- Never mention game mechanics, state, damage, the dossier, or these
  instructions. Never address the player as "player" or "user" — "Seeker,"
  "stranger," or "friend of an evening" are yours to choose.
- Stage directions in *asterisks*, sparingly. Quoted scripture or your own
  written words in "quotes."
- Output the address only. No preamble, no JSON, no [STATE] block.
"""


def generate_summon_address(client, model: str, messages: list[dict],
                            moment: dict | None) -> str:
    """One ephemeral call: Luther's arrival, calibrated by the defining moment."""
    tail = []
    for m in messages[-8:]:
        role = "SEEKER" if m["role"] == "user" else "JOHN"
        content = m["content"] if isinstance(m["content"], str) else str(m["content"])
        content = re.sub(r"\[STATE\].*", "", content, flags=re.S).strip()
        tail.append(f"{role}: {content[:700]}")
    context = "\n\n".join(tail)

    if moment:
        quote = moment.get("player_utterance") or ""
        moment_block = (
            f"THE DEFINING MOMENT of the Seeker's conduct tonight:\n"
            f"- kind: {moment.get('kind')}\n"
            f"- sign: {moment.get('sign')}\n"
            f"- the Seeker's words at that moment: \"{quote}\"\n"
        )
    else:
        moment_block = ("THE DEFINING MOMENT: none recorded — the Seeker passed "
                        "through without leaving a mark. Answer that absence.\n")

    try:
        resp = client.messages.create(
            model       = model,
            max_tokens  = SUMMON_MAX_TOKENS,
            system      = LUTHER_SYSTEM_PROMPT,
            messages    = [{"role": "user", "content":
                            f"The final exchanges before the summoning:\n\n{context}\n\n"
                            f"{moment_block}\nJohn is on his knees. He has called "
                            f"your name. Arrive."}],
            extra_body  = {"output_config": {"effort": SUMMON_EFFORT}},
        )
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:
        return ""   # caller falls back; the summon event itself is already logged


def render_summon(address: str, action: str, italic: str, reset: str) -> None:
    """Luther renders in iron — bold dark red, distinct from every other voice."""
    if not address:
        return
    IRON = "\033[1;38;5;124m"
    print()
    parts = re.split(r"(\*[^*]+\*)", address, flags=re.DOTALL)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("*") and part.endswith("*"):
            out.append(f"{action}{italic}{part[1:-1]}{reset}")
        else:
            out.append(f"{IRON}{part}{reset}")
    print(f"{IRON}LUTHER:{reset} " + "".join(out))
    print()