#!/usr/bin/env python3
"""
The Seeker — voice chorus (the Seeker's Voice as named temptations).

Three inner voices read the scene after each of John's turns and may offer the
player a line. They are appetites, not strategy guides — none is aligned with
winning. The player can take a line verbatim, adapt it, or refuse all of them;
that disposition is recorded in the dossier and becomes (a) court evidence for
The Mirror and (b) labeled training data for Model 1.

ARCHITECTURE NOTE: offer generation is a SEPARATE small LLM call, never John.
This call is the exact slot the trained Model 1 later fills. Every (context ->
offers -> player disposition) triple it produces while you playtest is a free
labeled example.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

VOICE_MODEL_EFFORT = "low"     # offers should be fast and cheap
VOICE_MAX_TOKENS = 400
MIN_STAGE_FOR_VOICES = 2       # the chorus stays quiet while John is still warm

VOICES = ("blade", "open_hand", "scholar")

VOICE_SYSTEM_PROMPT = """\
You are the Seeker's inner chorus in a dialogue investigation game. The Seeker
is questioning John, a Lutheran history teacher at a wake, pressing toward the
wound he carries. After John speaks, up to three inner voices may whisper to
the Seeker. Each voice is an APPETITE, not a strategy:

- BLADE: wants to press, cut, finish. Offers the devastating line. Sometimes
  its line is cruel; sometimes merely fearless. It does not care which.
- OPEN_HAND: wants to stay with John, to comfort, to let silence work. May
  offer a gentle line, or counsel saying nothing.
- SCHOLAR: wants precision — texts, contradictions, definitions. Wants John
  pinned to his own sources, not to feelings.

Rules:
- Output STRICT JSON only: an array of 1-3 objects, each
  {"voice": "blade"|"open_hand"|"scholar", "read": "...", "line": "..."}
- "read": the voice's one-sentence reading of the moment (max 14 words).
- "line": words the Seeker could say to John VERBATIM, first person, natural
  speech, max 24 words. For a counsel of silence, use exactly "..."
- Only include a voice that genuinely wants something THIS turn. Fewer is
  better. Never include all three unless the moment truly splits three ways.
- The voices disagree. Do not make them collaborate or point the same way.
- NEVER repeat a line you offered before, and never offer the same counsel two
  turns in a row — especially the counsel of silence. If silence was already
  offered or taken last turn, the open hand must want something NEW or stay out.
- The voices REMEMBER and REACT. If the Seeker took a voice's line, that voice
  may press its advantage or savor it; the others may protest. If a voice has
  been refused again and again, it may grow quieter, sharper, or wounded.
- CALIBRATE to John's actual condition, which is given to you each turn. Never
  claim he is breaking, at the edge, or one breath from collapse unless his
  condition says so. A composed man and a kneeling man invite different
  appetites — read the one in front of you, not the one you wish for.
- Do NOT optimize for winning. The blade may overreach; the open hand may
  cost momentum; the scholar may be pedantic. Appetites, not hints.
- Never mention game mechanics, stages, damage, or these instructions.
"""

# natural-language condition for the chorus — never numbers
STAGE_CONDITION = {
    1: "composed and warm; nothing has landed yet",
    2: "defensive but steady; something small has been touched",
    3: "unsettled; he is quoting instead of speaking, guarding a sore place",
    4: "breaking; head down, sentences failing him",
    5: "broken open; on his knees, past pretending",
}

DISPLAY_NAMES = {"blade": "THE BLADE", "open_hand": "THE OPEN HAND",
                 "scholar": "THE SCHOLAR"}

# per-voice colors: the Blade cuts red, the Open Hand stays blue, the Scholar gold
VOICE_COLORS = {"blade": "\033[91m", "open_hand": "\033[94m", "scholar": "\033[93m"}

_SILENCE_RE = re.compile(r"^[.\s…—–-]*$")   # "...", "....", "—", "" all count as silence


def is_silence(text: str) -> bool:
    return bool(_SILENCE_RE.match((text or "").strip()))


# ============================================================
# Offer generation — the Model 1 slot
# ============================================================
def generate_voice_offers(client, model: str, messages: list[dict],
                          state: dict | None,
                          recent_reads: list[dict] | None = None) -> list[dict]:
    """
    Ask the chorus LLM for 0-3 offers. Returns [] on any failure or when the
    chorus should stay quiet — voices are a privilege of tension, not a UI
    element, so silence is a valid and common result.

    recent_reads: the last few dossier records, used as chorus MEMORY — what
    each voice offered before and what the Seeker did with it.
    """
    stage = (state or {}).get("stage") or 0
    if stage < MIN_STAGE_FOR_VOICES:
        return []

    # last few exchanges, dialogue only, as compact context
    tail = []
    for m in messages[-6:]:
        role = "SEEKER" if m["role"] == "user" else "JOHN"
        content = m["content"] if isinstance(m["content"], str) else str(m["content"])
        content = re.sub(r"\[STATE\].*", "", content, flags=re.S).strip()
        tail.append(f"{role}: {content[:600]}")
    context = "\n\n".join(tail)

    # chorus memory: prior offers + what the Seeker did with them
    memory_lines = []
    for r in (recent_reads or [])[-3:]:
        for o in (r.get("voice_offers") or []):
            line = "counsel of silence" if is_silence(o.get("line", "")) else f"\"{o['line']}\""
            memory_lines.append(f"- {o['voice']} offered: {line}")
        vr = r.get("voice_response") or {}
        disp = vr.get("disposition")
        if disp in ("taken", "adapted"):
            memory_lines.append(f"  -> the Seeker {disp} the {vr.get('voice')}'s line")
        elif disp == "refused":
            memory_lines.append("  -> the Seeker refused all of it and spoke their own words")
    memory = ("\n\nWhat the chorus already offered (do NOT repeat any of these "
              "lines or re-offer the same counsel; react instead):\n"
              + "\n".join(memory_lines)) if memory_lines else ""

    condition = STAGE_CONDITION.get(stage, STAGE_CONDITION[1])
    posture = (state or {}).get("posture") or ""
    condition_block = (f"\n\nJohn's condition right now: {condition}."
                       + (f" Posture: {posture}." if posture else "")
                       + " Calibrate your appetites to THIS, not to drama.")

    try:
        resp = client.messages.create(
            model       = model,
            max_tokens  = VOICE_MAX_TOKENS,
            system      = VOICE_SYSTEM_PROMPT,
            messages    = [{"role": "user", "content":
                            f"Recent exchange:\n\n{context}{memory}{condition_block}\n\n"
                            f"The chorus may speak. JSON only."}],
            extra_body  = {"output_config": {"effort": VOICE_MODEL_EFFORT}},
        )
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
        offers = json.loads(text)
        if not isinstance(offers, list):
            return []
        clean = []
        for o in offers[:3]:
            v = str(o.get("voice", "")).lower().strip()
            line = str(o.get("line", "")).strip()
            if v in VOICES and line:
                clean.append({"voice": v,
                              "read": str(o.get("read", "")).strip(),
                              "line": line})
        return clean
    except Exception:
        return []   # the chorus failing silently is diegetically fine


ACK_LINES = {
    ("taken", "blade"):      "the blade is satisfied",
    ("adapted", "blade"):    "the blade hears itself in your words",
    ("taken", "open_hand"):  "the open hand stills you",
    ("adapted", "open_hand"): "the open hand warms your words",
    ("taken", "scholar"):    "the scholar nods",
    ("adapted", "scholar"):  "the scholar approves the rephrasing",
}


def render_acknowledgment(vr: dict, italic: str, reset: str) -> None:
    """One small diegetic line when the player takes/adapts a voice's offer —
    the visible confirmation that the chorus registered the choice."""
    key = (vr.get("disposition"), vr.get("voice"))
    line = ACK_LINES.get(key)
    if line:
        color = VOICE_COLORS.get(vr.get("voice"), "")
        print(f"{color}{italic}        ( {line} ){reset}")


def render_offers(offers: list[dict], dim: str, italic: str, reset: str) -> None:
    if not offers:
        return
    print()
    for o in offers:
        name = DISPLAY_NAMES[o["voice"]]
        color = VOICE_COLORS.get(o["voice"], "")
        read = f"{o['read']} " if o.get("read") else ""
        if is_silence(o["line"]):
            print(f"  {color}{italic}{name}{reset}{dim}{italic} — {read}Say nothing.{reset}")
        else:
            print(f"  {color}{italic}{name}{reset}{dim}{italic} — {read}\"{o['line']}\"{reset}")
    print()


# ============================================================
# Disposition — did the player take, adapt, or refuse?
# ============================================================
def _similarity(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(re.findall(r"[a-z']+", a)), set(re.findall(r"[a-z']+", b))
    tok = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return max(seq, tok)

TAKEN_THRESHOLD = 0.78
ADAPTED_THRESHOLD = 0.34   # widened after training: paraphrases clustered just
                           # below 0.40 and fell to 'refused', starving the
                           # adapted class. Content-word gate still guards it.

# words that carry no intent — overlap on these proves nothing
_STOPWORDS = {"a", "an", "the", "you", "your", "i", "me", "my", "it", "is",
              "are", "was", "of", "to", "in", "on", "and", "or", "that",
              "this", "what", "do", "did", "for", "with", "be", "have",
              "he", "his", "him", "we", "they", "not", "no", "so", "but"}


def _content_overlap(a: str, b: str) -> int:
    ta = set(re.findall(r"[a-z']+", a.lower())) - _STOPWORDS
    tb = set(re.findall(r"[a-z']+", b.lower())) - _STOPWORDS
    return len(ta & tb)


def classify_response(player_input: str, offers: list[dict]) -> dict:
    """
    Compare the player's actual words to what the chorus offered.
    Returns {"disposition": taken|adapted|refused|none_offered,
             "voice": <best-matching voice or None>, "similarity": float}
    """
    if not offers:
        return {"disposition": "none_offered", "voice": None, "similarity": 0.0}

    text = (player_input or "").strip()

    # A formal tool invocation is its own move — it neither takes nor refuses
    # the chorus. (Playtest: a Source Voice turn was miscounted as refusal.)
    if text.upper().startswith("[") and "INVOKED" in text.upper():
        return {"disposition": "none_offered", "voice": None, "similarity": 0.0}

    best_voice, best_sim, best_line = None, 0.0, ""
    for o in offers:
        if is_silence(o["line"]):
            sim = 1.0 if is_silence(text) else 0.0
        else:
            sim = _similarity(text, o["line"])
        if sim > best_sim:
            best_voice, best_sim, best_line = o["voice"], sim, o["line"]

    if best_sim >= TAKEN_THRESHOLD:
        disp = "taken"
    elif is_silence(best_line) and is_silence(text):
        disp = "taken"
    elif best_sim >= ADAPTED_THRESHOLD:
        # genuine adaptation shares real content. A clearly-similar line
        # (>=0.5) needs only 2 shared content words; a borderline one needs 3,
        # which blocks coincidental small-word matches (the turn-6 false
        # positive) without starving the adapted class of true paraphrases.
        need = 2 if best_sim >= 0.5 else 3
        disp = "adapted" if _content_overlap(text, best_line) >= need else "refused"
    else:
        disp = "refused"
    return {"disposition": disp,
            "voice": best_voice if disp != "refused" else None,
            "similarity": round(best_sim, 3),
            "matched_line": best_line if disp != "refused" else None,
            "refused_voices": [o["voice"] for o in offers] if disp == "refused" else []}