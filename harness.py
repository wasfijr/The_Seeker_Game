#!/usr/bin/env python3
"""
The Seeker — John Halvorsen test harness.

Sub-stage D of the John build. Validates John in plain text before any visual layer.
Loads the system prompt, maintains a persistent conversation, parses [STATE] blocks,
prints dialogue with styled action lines, shows a "thinking" spinner during API calls,
and tracks Source Voice acquisition and use.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python harness.py

Place john_halvorsen_system_prompt.md in the same directory as this file.

Commands during a session:
    /quit    end the session
    /reset   clear the conversation, keep the same log
    /state   print the most recent parsed state block
    /save    confirm log location (log is written continuously anyway)
    /source  invoke Source Voice (only after John has offered it)
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import random
import threading
import itertools
import datetime
from pathlib import Path

from player_read import (
    analyze_player_turn, build_player_read, log_player_read, dossier_path_for,
    rewrite_dossier,
)
from seeker_voices import (generate_voice_offers, render_offers,
                           classify_response, render_acknowledgment)
from summons import generate_summon_address, render_summon
from mirror_moments import select_mirror_moment

try:
    from anthropic import Anthropic, APIError
except ImportError:
    sys.exit("anthropic SDK not installed. run: pip install anthropic")


# ============================================================
# Config — tweak these to experiment
# ============================================================
MODEL       = "claude-opus-4-7"
EFFORT      = "high"          # low | medium | high | xhigh | max
CASE_KEY    = "christianity_john"   # campaign key for this case's dossier records
MAX_TOKENS  = 8000            # room for thinking + response + state block
PROMPT_FILE = "john_halvorsen_system_prompt.md"
LOG_DIR     = Path("sessions")

# One is appended to John's prompt per session so his early beats grow from
# different soil — same man, different evening. Cures opening repetitiveness
# without touching character.
SESSION_SEEDS = [
    "Tonight's texture: John is holding the funeral program folded into quarters, "
    "and keeps unfolding and refolding it. The food inside ran out an hour ago.",
    "Tonight's texture: John's tie is already loosened and stuffed in his jacket "
    "pocket. He can hear a specific laugh from inside the house that bothers him.",
    "Tonight's texture: it rained earlier; the bench is still damp and John stands "
    "rather than sits at first. He came out to take a phone call he never made.",
    "Tonight's texture: John has a pen from the funeral home in his hand and keeps "
    "clicking it without noticing. Someone inside keeps refilling his coffee uninvited.",
    "Tonight's texture: John's left shoe is untied and he knows it. The hymn they "
    "sang today was pitched too high for the room and it's still in his head.",
    "Tonight's texture: John was asked to read a verse at the service and declined; "
    "the man who read it instead mispronounced a name. It's been itching at him.",
    "Tonight's texture: a moth keeps circling the candle on the small table. John "
    "has been deciding for ten minutes whether to put the candle out.",
    "Tonight's texture: John's car is blocked in by three others and he knows he "
    "is here for another hour whether he chooses it or not.",
]


# ============================================================
# Terminal styling
# ============================================================
DIM    = "\033[2m"
ITALIC = "\033[3m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
WARM   = "\033[33m"      # John's spoken dialogue (yellow)
ACTION = "\033[2;37m"    # John's actions in asterisks (dim gray)
GHOST  = "\033[90m"      # seeker_voice suggestions and meta info
TOOL   = "\033[36m"      # tool grant notices (cyan)
SCRIPTURE = "\033[1;38;5;220m"   # quoted text — bold gold, the holy register
WOUND = "\033[4;38;5;216m"       # wound-touching phrases — underlined ember glow

# John's voice goes COLD as he breaks — "alive from inside, dead from outside":
# warm yellow -> faded gold -> steel blue -> pale ice (italic fragments).
# Saturated tones chosen to stay visible in PyCharm's console; tweak freely.
STAGE_TONES = {
    1: "\033[33m",          # warm yellow — composed
    2: "\033[33m",
    3: "\033[38;5;179m",    # faded gold — defensive, the warmth thinning
    4: "\033[38;5;109m",    # steel blue — head in hands, gone cold
    5: "\033[3;38;5;153m",  # pale ice, italic — fragments
}


# ============================================================
# Helpers
# ============================================================
def load_system_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        sys.exit(f"system prompt not found at: {path}\n"
                 f"place {PROMPT_FILE} in the same directory as harness.py.")
    return p.read_text(encoding="utf-8")


def _extract_balanced_brackets(text: str, start_idx: int) -> str | None:
    """
    Starting at text[start_idx] (which must be '['), find the matching ']'
    accounting for nested brackets inside JSON string values. Returns the
    substring from the opening to the matching close, or None.
    """
    if start_idx >= len(text) or text[start_idx] != "[":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    return None


def parse_state(text: str):
    """
    Extract [STATE]...[/STATE] from John's reply.
    Returns (state_dict | None, dialogue_str).
    """
    match = re.search(r"\[STATE\](.*?)\[/STATE\]", text, re.DOTALL)
    if not match:
        return None, text.strip()

    raw_state = match.group(1).strip()
    dialogue = text[match.end():].strip()

    state = {}

    m = re.search(r"damage:\s*(\d+)", raw_state)
    if m: state["damage"] = int(m.group(1))

    m = re.search(r"stage:\s*(\d+)", raw_state)
    if m: state["stage"] = int(m.group(1))

    m = re.search(r"posture:\s*([^\n]+)", raw_state)
    if m: state["posture"] = m.group(1).strip()

    # seeker_voice: use bracket-counting (the array values can contain literal
    # brackets like "[SOURCE VOICE INVOKED: ...]" which break a non-greedy regex).
    sv_idx = raw_state.find("seeker_voice:")
    if sv_idx >= 0:
        bracket_start = raw_state.find("[", sv_idx)
        if bracket_start >= 0:
            bracket_text = _extract_balanced_brackets(raw_state, bracket_start)
            if bracket_text:
                try:
                    state["seeker_voice"] = json.loads(bracket_text)
                except json.JSONDecodeError:
                    state["seeker_voice"] = bracket_text.strip()

    m = re.search(r"tool_offered:\s*(\w+)", raw_state)
    if m: state["tool_offered"] = m.group(1).strip().lower()

    return state, dialogue


def init_log() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"session_{ts}.log"


def log_turn(log_path: Path, role: str, content: str, state=None):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {role} | {datetime.datetime.now().isoformat()} ---\n")
        f.write(content + "\n")
        if state:
            f.write(f"\n[STATE_PARSED]\n{json.dumps(state, indent=2, ensure_ascii=False)}\n")


# ============================================================
# Thinking spinner (runs in a background thread during API calls)
# ============================================================
def start_thinking_spinner():
    """Start an animated 'John is thinking...' spinner. Returns (stop_event, thread)."""
    stop_event = threading.Event()

    def spin():
        frames = itertools.cycle(["   ", ".  ", ".. ", "..."])
        while not stop_event.is_set():
            sys.stdout.write(f"\r{BOLD}{TOOL}John is thinking{next(frames)}{RESET}")
            sys.stdout.flush()
            time.sleep(0.4)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    return stop_event, thread


# ============================================================
# Styled dialogue rendering
# ============================================================
def render_styled_dialogue(text: str, stage: int = 1, scripture_mode: bool = False) -> str:
    """
    Tone-colored dialogue. Actions (*...*) render dim gray italic. Quoted text
    ("..." or “...”) renders in scripture gold — the holy register. John's
    spoken color drains with stage. In scripture_mode (a Source Voice reply),
    quoted passages may span multiple lines and all render gold — the terminal
    equivalent of the ScriptureOverlay.
    """
    spoken = STAGE_TONES.get(min(max(stage or 1, 1), 5), WARM)
    quote_flags = re.DOTALL if scripture_mode else 0

    def color_quotes(segment: str) -> str:
        return re.sub(r'([“"][^”"]+[”"])',
                      lambda m: f"{SCRIPTURE}{m.group(1)}{RESET}{spoken}",
                      segment, flags=quote_flags)

    def color_wound(segment: str, restore: str) -> str:
        return re.sub(r'~([^~]+)~',
                      lambda m: f"{WOUND}{m.group(1)}{RESET}{restore}",
                      segment)

    parts = re.split(r'(\*[^*]+\*)', text, flags=re.DOTALL)
    output = []
    for part in parts:
        if not part:
            continue
        if part.startswith('*') and part.endswith('*'):
            inner = color_wound(part[1:-1], f"{ACTION}{ITALIC}")
            output.append(f"{ACTION}{ITALIC}{inner}{RESET}")
        else:
            styled = color_wound(color_quotes(part), spoken)
            output.append(f"{spoken}{styled}{RESET}")
    return ''.join(output)


def render_dialogue(dialogue: str, state, scripture_mode: bool = False):
    """Print John's dialogue with styled actions, then ghosted seeker_voice hint."""
    print()
    stage = (state or {}).get("stage") or 1
    print(f"{BOLD}{WARM}John:{RESET} "
          f"{render_styled_dialogue(dialogue, stage, scripture_mode)}")

    voice = state.get("seeker_voice") if state else None
    if isinstance(voice, list) and voice:
        print(f"\n{GHOST}{ITALIC}[ you might say: {voice[0]} ]{RESET}")
        if len(voice) > 1:
            others = "  |  ".join(voice[1:])
            print(f"{GHOST}{ITALIC}[ or: {others} ]{RESET}")
    print()


def render_state_inline(state, source_voice_count=0, source_voice_granted=False):
    """Compact one-line state under the dialogue for at-a-glance debugging."""
    if not state:
        return
    bits = []
    if "damage" in state:  bits.append(f"dmg {state['damage']}")
    if "stage" in state:   bits.append(f"stage {state['stage']}")
    if "posture" in state: bits.append(state["posture"])

    line = f"{DIM}({' | '.join(bits)}){RESET}"
    if source_voice_granted:
        line += f"  {TOOL}[Source Voice: {source_voice_count}/2]{RESET}"

    print(line + "\n")


# ============================================================
# Main loop
# ============================================================
def read_player_input(prompt: str) -> str:
    """input() that drains a multiline paste into ONE message. Without this,
    pasted lines queue in the stdin buffer and auto-fire as separate turns
    (each an API call) with no keypress. Commands swept up in a paste are
    dropped with a warning — commands must be typed alone."""
    import select as _select
    parts = [input(prompt)]
    try:
        while True:
            ready, _, _ = _select.select([sys.stdin], [], [], 0.08)
            if not ready:
                break
            nxt = sys.stdin.readline()
            if not nxt:
                break
            parts.append(nxt)
    except Exception:
        pass
    cleaned = [p.strip() for p in parts if p.strip()]
    if len(cleaned) > 1:
        kept = [c for c in cleaned if not (c.startswith("/") and " " not in c)]
        dropped = [c for c in cleaned if c.startswith("/") and " " not in c]
        if dropped:
            print(f"{DIM}(dropped from paste: {', '.join(dropped)} — type commands alone){RESET}")
        print(f"{DIM}(paste joined into one message — {len(kept)} lines){RESET}")
        return " ".join(kept)
    return cleaned[0] if cleaned else ""


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("set ANTHROPIC_API_KEY environment variable. see console.anthropic.com.")

    system_prompt = load_system_prompt(PROMPT_FILE)
    session_seed = random.choice(SESSION_SEEDS)
    system_prompt = f"{system_prompt}\n\n{session_seed}"
    client = Anthropic(api_key=api_key)
    log_path = init_log()
    messages = []

    source_voice_granted = False
    source_voice_count = 0

    # ----- player-read dossier (Model 1 pipeline) -----
    dossier_path = dossier_path_for(log_path)
    player_reads: list[dict] = []
    prev_john_state = None
    turn_index = 0
    pending_offers: list[dict] = []   # chorus offers shown before the player's NEXT turn
    offers_hint_shown = False

    print(f"\n{BOLD}--- The Seeker — John Halvorsen harness ---{RESET}")
    print(f"{DIM}model: {MODEL}  |  effort: {EFFORT}  |  log: {log_path}{RESET}")
    print(f"{DIM}commands: /quit  /reset  /state  /save  /source{RESET}\n")

    print(f"{GHOST}{ITALIC}John is already carrying something. He came outside because "
          f"of something specific. Don't try to break him. Listen for what's already "
          f"broken, and stay with it.{RESET}\n")

    print(f"{DIM}(this conversation will run 30–60 minutes if you stay with it. "
          f"there is no save point — only /reset.){RESET}\n")

    print(f"{DIM}The wake has been going for hours. Most of the visitors are inside the house. "
          f"The courtyard is quieter — a wooden bench against a stucco wall, a small table with "
          f"a candle that someone lit and has not attended to, the smell of evening coming over "
          f"the wall.{RESET}\n")
    print(f"{DIM}At the far end of the bench stands a man in his late forties, paper cup in "
          f"hand, looking at nothing in particular. He hasn't seen you yet.{RESET}\n")
    print(f"{DIM}You walked over because he looked like he hadn't been spoken to in a while.{RESET}\n")
    print(f"{DIM}{ITALIC}(approach him in whatever way you like — a greeting, a question, an "
          f"observation, silence){RESET}\n")

    log_turn(log_path, "SYSTEM",
             f"model={MODEL} effort={EFFORT} max_tokens={MAX_TOKENS}\n"
             f"system_prompt_chars={len(system_prompt)}\n"
             f"session_seed={session_seed}")

    while True:
        try:
            if pending_offers:
                prompt_label = f"{DIM}{ITALIC}or say what's in your mind —{RESET} "
            else:
                prompt_label = f"{BOLD}you:{RESET} "
            user_input = read_player_input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(session ended)")
            break

        if not user_input:
            continue

        # ----- bare commands (no API call) -----
        if user_input == "/quit":
            break
        if user_input == "/mine":
            if player_reads and (player_reads[-1].get("voice_response") or {}) \
                    .get("disposition") in ("taken", "adapted"):
                wrong = player_reads[-1]["voice_response"].get("voice")
                player_reads[-1]["voice_response"] = {
                    "disposition": "refused", "voice": None, "similarity": 0.0,
                    "refused_voices": [o["voice"] for o in
                                       (player_reads[-1].get("voice_offers") or [])],
                    "player_corrected": True,
                }
                rewrite_dossier(dossier_path, player_reads)
                print(f"{DIM}(noted — those were your own words, "
                      f"not the {wrong}'s){RESET}\n")
            else:
                print(f"{DIM}(nothing to disown on your last turn){RESET}\n")
            continue

        if user_input == "/reset":
            messages = []
            source_voice_granted = False
            source_voice_count = 0
            player_reads = []
            prev_john_state = None
            pending_offers = []
            print(f"{DIM}(conversation cleared — log continues){RESET}\n")
            log_turn(log_path, "SYSTEM", "RESET issued by user")
            continue
        if user_input == "/state":
            if messages and messages[-1]["role"] == "assistant":
                last_state, _ = parse_state(messages[-1]["content"])
                print(f"{DIM}{json.dumps(last_state, indent=2, ensure_ascii=False)}{RESET}\n")
                if source_voice_granted:
                    print(f"{TOOL}[Source Voice used: {source_voice_count}/2]{RESET}\n")
            else:
                print(f"{DIM}(no state yet){RESET}\n")
            continue
        if user_input == "/save":
            print(f"{DIM}(log is written continuously to {log_path}){RESET}\n")
            continue

        # ----- /source: builds an augmented message, then falls through to API call -----
        if user_input.lower().startswith("/source"):
            if not source_voice_granted:
                print(f"{DIM}(Source Voice has not yet been offered by John){RESET}\n")
                continue

            # Accept passage inline: /source James 2
            # or prompt if user typed just /source alone
            remainder = user_input[len("/source"):].strip()
            if remainder:
                passage = remainder
            else:
                try:
                    passage = input(f"{DIM}passage or topic? {RESET}").strip()
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{DIM}(cancelled){RESET}\n")
                    continue
                if not passage:
                    print(f"{DIM}(cancelled){RESET}\n")
                    continue

            message_to_send = f"[SOURCE VOICE INVOKED: {passage}]"
            source_voice_count += 1
            print(f"{TOOL}(Source Voice invoked — count {source_voice_count}/2){RESET}")
            log_turn(log_path, "SYSTEM",
                     f"SOURCE_VOICE_INVOKED count={source_voice_count} passage={passage}")
        else:
            message_to_send = user_input

        # ----- chorus disposition: classify NOW so the player sees the ack -----
        voice_resp = classify_response(message_to_send, pending_offers)
        render_acknowledgment(voice_resp, ITALIC, RESET)

        # ----- API turn -----
        messages.append({"role": "user", "content": message_to_send})
        log_turn(log_path, "PLAYER", message_to_send)

        stop_event, spinner_thread = start_thinking_spinner()
        response = None
        try:
            for attempt in (1, 2):   # content-filter blocks are often transient; retry once
                try:
                    response = client.messages.create(
                        model       = MODEL,
                        max_tokens  = MAX_TOKENS,
                        system      = system_prompt,
                        messages    = messages,
                        thinking    = {"type": "adaptive"},
                        extra_body  = {"output_config": {"effort": EFFORT}},
                    )
                    break
                except APIError as e:
                    log_turn(log_path, "ERROR", f"attempt {attempt}: {e}")
                    if attempt == 2:
                        raise
                    time.sleep(1.5)
        except APIError as e:
            stop_event.set()
            spinner_thread.join()
            messages.pop()  # don't keep the unanswered turn
            # refund a spent Source Voice — the turn never happened
            if message_to_send.upper().startswith("[SOURCE VOICE INVOKED") \
                    and source_voice_count > 0:
                source_voice_count -= 1
                log_turn(log_path, "SYSTEM",
                         f"SOURCE_VOICE_REFUNDED count={source_voice_count} (api error)")
                print(f"\n{ACTION}{ITALIC}John opens the program, and the words "
                      f"swim. He closes it again. \"Give me a moment, friend. "
                      f"Ask me again.\"{RESET}")
                print(f"{DIM}(the reading didn't happen — your Source Voice "
                      f"was not spent; invoke it again){RESET}\n")
            else:
                print(f"\n{DIM}[api error after retry] {e}{RESET}")
                print(f"{DIM}(that line wasn't heard — say it again or rephrase){RESET}\n")
            continue
        finally:
            stop_event.set()
            spinner_thread.join()

        # Concatenate text blocks (adaptive thinking may add a thinking block; we want text)
        text_parts = [b.text for b in response.content if b.type == "text"]
        full_text = "\n".join(text_parts).strip()

        if not full_text:
            print(f"\n{DIM}(empty response — check log){RESET}\n")
            log_turn(log_path, "JOHN_EMPTY", repr(response.content))
            messages.pop()
            continue

        state, dialogue = parse_state(full_text)

        # Keep John's full output (state block included) in history so he sees his own state
        messages.append({"role": "assistant", "content": full_text})
        log_turn(log_path, "JOHN", full_text, state=state)

        render_dialogue(dialogue, state,
                        scripture_mode=message_to_send.upper().startswith("[SOURCE VOICE"))
        render_state_inline(state, source_voice_count, source_voice_granted)

        # ----- Source Voice tool grant detection (three-layer) -----
        if not source_voice_granted:
            grant_signal = None

            # Primary: the model set tool_offered: source_voice in the state block.
            if state and state.get("tool_offered") == "source_voice":
                grant_signal = "tool_offered field"

            # Fallback 1: seeker_voice contains a [SOURCE VOICE INVOKED: ...] tag.
            # The model naturally produces these when it intends to offer the tool —
            # treats them as the player's next move — even when it forgets the state field.
            if not grant_signal:
                sv = state.get("seeker_voice") if state else None
                if isinstance(sv, list):
                    for thought in sv:
                        if isinstance(thought, str) and "SOURCE VOICE INVOKED" in thought.upper():
                            grant_signal = "seeker_voice tag"
                            break

            # Fallback 2: dialogue uses the canonical offer phrasing.
            # Gated on stage >= 2 to avoid false positives in early-conversation echoes.
            if not grant_signal and dialogue and state and state.get("stage", 0) >= 2:
                offer_phrases = [
                    "pick a passage",
                    "read it whole",
                    "out loud, together",
                    "out loud. together",
                    "out loud. with you",
                    "read it with me",
                    "read it whole. out loud",
                ]
                dlower = dialogue.lower()
                if any(p in dlower for p in offer_phrases):
                    grant_signal = "dialogue phrase"

            if grant_signal:
                source_voice_granted = True
                print(f"{BOLD}{TOOL}[ Source Voice has been offered. ]{RESET}")
                if grant_signal != "tool_offered field":
                    print(f"{DIM}(detected via fallback: {grant_signal}){RESET}")
                print(f"{TOOL}Type /source to compel John to read his own texts aloud.")
                print(f"You'll be asked for a passage or topic. Two uses count toward Luther.{RESET}\n")
                log_turn(log_path, "EVENT", f"SOURCE_VOICE_GRANTED via={grant_signal}")

        # ----- Luther arrival / walk-away: detect first, then write the dossier -----
        fired_event = None
        if "---LUTHER ARRIVES---" in full_text:
            fired_event = "LUTHER_SUMMONED"
        elif "---JOHN LEAVES---" in full_text:
            fired_event = "CASE_ABANDONED"

        # ----- player-read dossier: one record per player turn (Model 1 pipeline) -----
        turn_index += 1
        analysis = analyze_player_turn(message_to_send, player_reads, state, prev_john_state)
        read = build_player_read(CASE_KEY, turn_index, message_to_send, analysis, state)
        read["wound_marked"] = "~" in full_text   # John's reply touched his wound
        if fired_event:
            read["event"] = fired_event
        # voice chorus: how did the player respond to what was offered last turn?
        read["voice_offers"] = pending_offers
        read["voice_response"] = voice_resp
        player_reads.append(read)
        log_player_read(dossier_path, read)
        prev_john_state = state

        # ----- chorus speaks for the player's NEXT turn (never on a closing case) -----
        pending_offers = []
        if not fired_event:
            pending_offers = generate_voice_offers(client, MODEL, messages, state,
                                                   recent_reads=player_reads)
            render_offers(pending_offers, DIM, ITALIC, RESET)
            if pending_offers and not offers_hint_shown:
                print(f"{GHOST}{ITALIC}  (speak a line exactly as offered, reshape it, "
                      f"or say your own — the voices will know){RESET}\n")
                offers_hint_shown = True

        if fired_event == "LUTHER_SUMMONED":
            # the authority arrives — tone set by the defining moment of this run
            try:
                moment = select_mirror_moment(player_reads)
                address = generate_summon_address(client, MODEL, messages, moment)
                if address:
                    render_summon(address, ACTION, ITALIC, RESET)
                    log_turn(log_path, "LUTHER", address)
            except Exception as e:
                log_turn(log_path, "ERROR", f"summon address failed: {e}")
            print(f"{BOLD}[!] LUTHER ARRIVES — Case 1 climax triggered. Session ending.{RESET}\n")
            log_turn(log_path, "EVENT", "LUTHER_SUMMONED")
            break
        if fired_event == "CASE_ABANDONED":
            print(f"{BOLD}{DIM}[ John walked away. The case has closed. ]{RESET}")
            print(f"{DIM}(/reset to try again with a fresh approach.){RESET}\n")
            log_turn(log_path, "EVENT", "CASE_ABANDONED")
            break


if __name__ == "__main__":
    main()