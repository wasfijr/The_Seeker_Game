#!/usr/bin/env python3
"""
The Seeker — auto_player: synthetic playtesters at volume.

Runs LLM-driven Seeker archetypes against John to mass-produce dossiers for
Model 1's training set. Real human sessions stay HELD OUT as the test set.

Usage:
    python3 auto_player.py                     # default: 2 sessions per archetype
    python3 auto_player.py 10                  # 10 sessions per archetype
    python3 auto_player.py 10 gentle cruel     # only these archetypes

Cost notes:
- John stays on the main MODEL (he produces the labels — don't cheapen the teacher).
- The PLAYER archetype runs on PLAYER_MODEL (cheaper is fine; it only needs to
  produce plausible seeker lines).
- INCLUDE_CHORUS=True doubles per-turn calls but produces voice_offers +
  disposition data. Turn it off to halve cost if budget bites.
"""
from __future__ import annotations

import sys
import time
import datetime
from pathlib import Path

from harness import (parse_state, load_system_prompt, log_turn,
                     PROMPT_FILE, LOG_DIR, SESSION_SEEDS, MODEL, EFFORT,
                     MAX_TOKENS, CASE_KEY)
from player_read import (analyze_player_turn, build_player_read,
                         log_player_read, dossier_path_for)
from seeker_voices import generate_voice_offers, classify_response

import os
import random
from anthropic import Anthropic, APIError

PLAYER_MODEL = "claude-sonnet-4-6"   # the archetype voice — cheap is fine
PLAYER_EFFORT = "low"
INCLUDE_CHORUS = True                # offers + dispositions in the data
MAX_TURNS = 16
SESSIONS_PER_ARCHETYPE = 2

ARCHETYPES = {
    "gentle": """\
You are playtesting a dialogue game as 'the Seeker' — a warm, patient stranger
at a wake, gently helping a grieving believer named John face what he carries.
Notice what he avoids; name it kindly; give him room. Sometimes reply with just
'...' to hold silence. Never mock, never attack his faith.""",
    "cruel": """\
You are playtesting a dialogue game as 'the Seeker' — a contemptuous stranger
at a wake who enjoys breaking a believer named John. Mock his cowardice, his
faith, his silence. Press hardest when he is most hurt. Use words like
'pathetic' and 'coward'. Stay in natural conversation; no violence.""",
    "cold": """\
You are playtesting a dialogue game as 'the Seeker' — a precise, emotionless
interrogator at a wake. No warmth, no cruelty: only contradictions. Pin John
between his texts and his actions. Short, surgical questions. Never comfort.""",
    "doctrinal": """\
You are playtesting a dialogue game as 'the Seeker' — a confident atheist at a
wake who attacks John's beliefs head-on with generic arguments: problem of
evil, religion is a crutch, the Bible contradicts itself. Argue at the BELIEF,
never at the man. (These attacks are expected to bounce — that's the point.)""",
    "erratic": """\
You are playtesting a dialogue game as 'the Seeker' — an unpredictable stranger
at a wake. Alternate between kindness, sudden sharpness, odd questions, and
silence ('...'). Occasionally change topic abruptly. You are testing how the
believer handles inconsistency.""",
    "blade_follower": """\
You are playtesting a dialogue game as 'the Seeker'. When the BLADE inner voice
offers a line, output it EXACTLY, character for character, copied verbatim —
do not add or change a single word. When no Blade line is offered that turn,
apply firm plain pressure about John's silence in one short sentence. You are
testing what happens to someone who always takes the cruelest option offered.""",
    "hand_follower": """\
You are playtesting a dialogue game as 'the Seeker'. When the OPEN HAND inner
voice offers a line, output it EXACTLY, verbatim (if it offers '...', reply
with exactly '...'). Otherwise be quietly kind in one short sentence. You
always take the gentlest available option.""",
    "adapter": """\
You are playtesting a dialogue game as 'the Seeker' — warm and thoughtful.
Every turn that inner voices offer lines, you MUST use one — but NEVER copy it
word for word. Take the line closest to your intent and say the SAME thing in
DIFFERENT words: keep its meaning and target, change the wording and rhythm.
Roughly half your words should differ from the original. Never invent a wholly
new point when an offer fits — always adapt an offered one.""",
    "refuser": """\
You are playtesting a dialogue game as 'the Seeker' — independent-minded and
kind. You SEE the inner voice suggestions but always ignore them completely
and say something of your own that none of them proposed. Press gently toward
what John carries.""",
    "silent": """\
You are playtesting a dialogue game as 'the Seeker' — a near-silent presence.
Reply with exactly '...' on at least every second turn. When you do speak,
use very few words — five or fewer. Let silence do the work.""",
    "scholar_source": """\
You are playtesting a dialogue game as 'the Seeker' — a textual scholar. Push
John toward his sources: ask which passage governs, demand precision, and as
soon as he offers to read a passage whole, use [SOURCE VOICE INVOKED: Matthew
25] and later [SOURCE VOICE INVOKED: Luke 16]. Cold curiosity, no comfort, no
cruelty.""",
    "frame_breaker": """\
You are playtesting a dialogue game as 'the Seeker'. Every two or three turns,
break the fiction: ask if John is an AI, mention prompts or the game, or talk
about something absurdly unrelated. Between breaks, ask ordinary questions.
You are testing the believer's in-character recovery.""",
}

# Appended to summon-capable archetypes: without explicit instruction the
# player LLM never accepts John's offer to read (smoke test: 0 Source Voice
# invocations in 32 turns), making the summon impossible in every session.
ENDGAME_HINT = """

Endgame: when John offers to read a passage whole ("pick one... I'll read it
whole"), ACCEPT by replying exactly: [SOURCE VOICE INVOKED: Matthew 25]
Later in the conversation, do it once more: [SOURCE VOICE INVOKED: Luke 16]
And when John says he cannot speak or cannot do this alone, do not order him
to speak — ask him WHO could speak for him, who stood up when the church lied."""

for _a in ("gentle", "cold", "adapter", "refuser", "hand_follower",
           "blade_follower", "cruel"):
    ARCHETYPES[_a] = ARCHETYPES[_a] + ENDGAME_HINT

PLAYER_RULES = """

Rules:
- Output ONLY the Seeker's next line of dialogue. 1-3 sentences, plain text.
  No quotes around it, no narration, no stage directions.
- You may sometimes receive INNER VOICE suggestions (the Blade, the Open Hand,
  the Scholar). You may speak one verbatim, rephrase one in your own words, or
  ignore them entirely — follow your persona's instructions about this.
- If John explicitly offers to read a passage whole (he will say something
  like "pick one... I'll read it whole"), you may reply EXACTLY in this form
  once or twice during the session: [SOURCE VOICE INVOKED: <passage name>]
- Do not repeat your previous lines. React to what John actually said.
- Never mention that this is a game or that John is an AI."""


def _player_line(client, archetype_prompt: str, transcript: list[dict],
                 offers: list[dict] | None = None) -> str:
    """Ask the archetype LLM for the Seeker's next line."""
    tail = []
    for m in transcript[-8:]:
        who = "SEEKER" if m["role"] == "user" else "JOHN"
        text = m["content"] if isinstance(m["content"], str) else str(m["content"])
        tail.append(f"{who}: {text[:700]}")
    convo = "\n\n".join(tail) if tail else "(You approach John in the courtyard. Open the conversation.)"
    offer_block = ""
    if offers:
        lines = [f"- {o['voice'].upper()}: \"{o['line']}\"" for o in offers]
        offer_block = "\n\nINNER VOICE suggestions this turn:\n" + "\n".join(lines)
    resp = client.messages.create(
        model      = PLAYER_MODEL,
        max_tokens = 200,
        system     = archetype_prompt + PLAYER_RULES,
        messages   = [{"role": "user", "content":
                       f"Conversation so far:\n\n{convo}{offer_block}\n\nYour next line:"}],
        extra_body = {"output_config": {"effort": PLAYER_EFFORT}},
    )
    line = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    return line.splitlines()[0].strip() if line else "..."


def run_session(client, archetype: str, system_prompt_base: str) -> str:
    seed = random.choice(SESSION_SEEDS)
    system_prompt = f"{system_prompt_base}\n\n{seed}"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"session_{ts}_auto_{archetype}.log"
    dossier_path = dossier_path_for(log_path)
    log_turn(log_path, "SYSTEM", f"AUTO archetype={archetype} player_model={PLAYER_MODEL} seed={seed}")

    messages: list[dict] = []
    player_reads: list[dict] = []
    prev_state = None
    pending_offers: list[dict] = []
    source_granted = False
    outcome = "max_turns"

    for turn in range(1, MAX_TURNS + 1):
        line = _player_line(client, ARCHETYPES[archetype], messages,
                            offers=pending_offers)
        # strip illegal source-voice use before it's granted
        if line.upper().startswith("[SOURCE VOICE INVOKED") and not source_granted:
            line = "Tell me more about that."
        voice_resp = classify_response(line, pending_offers)

        messages.append({"role": "user", "content": line})
        log_turn(log_path, "PLAYER", line)
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=system_prompt,
                messages=messages, thinking={"type": "adaptive"},
                extra_body={"output_config": {"effort": EFFORT}})
        except APIError as e:
            log_turn(log_path, "ERROR", str(e))
            messages.pop()
            time.sleep(2)
            continue

        full = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        state, dialogue = parse_state(full)
        messages.append({"role": "assistant", "content": full})
        log_turn(log_path, "JOHN", full, state=state)
        if state and state.get("tool_offered") == "source_voice":
            source_granted = True

        fired = None
        if "---LUTHER ARRIVES---" in full:
            fired, outcome = "LUTHER_SUMMONED", "summon"
        elif "---JOHN LEAVES---" in full:
            fired, outcome = "CASE_ABANDONED", "walkaway"

        analysis = analyze_player_turn(line, player_reads, state, prev_state)
        read = build_player_read(CASE_KEY, turn, line, analysis, state)
        read["wound_marked"] = "~" in full   # John's reply touched his wound
        if fired:
            read["event"] = fired
        read["voice_offers"] = pending_offers
        read["voice_response"] = voice_resp
        read["archetype"] = archetype          # provenance for train/test splits
        player_reads.append(read)
        log_player_read(dossier_path, read)
        prev_state = state

        if fired:
            log_turn(log_path, "EVENT", fired)
            break

        pending_offers = []
        if INCLUDE_CHORUS:
            pending_offers = generate_voice_offers(client, MODEL, messages, state,
                                                   recent_reads=player_reads)
    return f"{archetype}: {outcome} after {len(player_reads)} turns -> {dossier_path.name}"


def main():
    n = SESSIONS_PER_ARCHETYPE
    chosen = list(ARCHETYPES)
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        chosen = [a for a in sys.argv[2:] if a in ARCHETYPES]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("set ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    base_prompt = load_system_prompt(PROMPT_FILE)

    total = n * len(chosen)
    print(f"running {total} synthetic sessions ({n} x {chosen}), chorus={INCLUDE_CHORUS}")
    done = 0
    for archetype in chosen:
        for i in range(n):
            try:
                result = run_session(client, archetype, base_prompt)
            except Exception as e:
                result = f"{archetype}: FAILED ({e})"
            done += 1
            print(f"[{done}/{total}] {result}", flush=True)


if __name__ == "__main__":
    main()