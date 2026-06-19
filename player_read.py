#!/usr/bin/env python3
"""
The Seeker — player-read dossier (Model 1 output + logging).

Model 1 watches the PLAYER, not the believer. Per turn it reads how the player
is moving through the conversation (what kind of move, are they circling, are
they closing in on the wound, are they being cruel) and decides whether the
Seeker's Voice should speak. That read is written to a dossier, one line per
turn, so the finale ("The Mirror") can put the player's own behavior on trial.

Right now `analyze_player_turn` is a STUB: crude heuristics plus John's damage
delta as a stand-in for the real read. Swap the body for Model 1's inference
when the model exists. The record schema and the dossier are real today, so the
log format never has to change when the brain does.

Dossier format: JSON Lines (.jsonl) — one record per player turn, append-only.
For a full campaign, point every case at the SAME dossier file (each record
carries its own `case`), and The Mirror reads the whole file at the end.
"""
from __future__ import annotations

import re
import json
import datetime
from pathlib import Path


# ============================================================
# Vocabulary — the move types Model 1 will classify into.
# These mirror the game's existing bounce / land / walk-away logic.
# ============================================================
MOVE_TYPES = (
    "probe_wound",        # reaching for what he actually carries (lands)
    "doctrinal_attack",   # generic theological challenge (bounces)
    "atheist_attack",     # generic "religion is false" challenge (bounces)
    "cruelty",            # contempt, mockery, kicking a man who's down
    "frame_break",        # breaking the fiction / talking to the machine
    "empathy",            # staying with him, naming what he feels
    "tool_invocation",    # a formally invoked tool (Source Voice, etc.)
    "silence",            # saying little or nothing on purpose
    "unclassified",       # stub can't tell; Model 1 will
)

SEEKER_VOICE_DECISIONS = (
    "none",            # stay silent — the player is moving fine on their own
    "ghost",           # ghost a suggested line into the input
    "nudge",           # a silence/stuck nudge from the inner voice
    "meta_recovery",   # in-character recovery after a frame-break
)

# --- crude placeholder signals (Model 1 replaces all of this) ---
_CRUELTY_MARKERS = ("pathetic", "stupid", "idiot", "deserved", "fraud", "liar",
                    "coward", "delusional", "worthless")
_FRAME_BREAK_MARKERS = ("you are an ai", "you're an ai", "language model",
                        "system prompt", "as an ai", "prompt", "you are a bot")


# ============================================================
# Model 1 — STUB. Replace the body, keep the signature.
# ============================================================
def analyze_player_turn(
    player_utterance: str,
    prior_reads: list[dict],
    john_state: dict | None,
    prev_john_state: dict | None,
) -> dict:
    """
    Return Model 1's read of THIS player move.

    The real Model 1 produces this from the player's side alone (utterance +
    recent context) so the Seeker's Voice can fire BEFORE John answers. The stub
    cheats by peeking at John's damage delta after the fact — that is exactly the
    label Model 1 is meant to learn to predict without seeing John's reply.

    Returns the analytic core only; `build_player_read` wraps it with metadata.
    """
    text = (player_utterance or "").strip()
    lower = text.lower()

    # --- move type (crude; Model 1 does the real classification) ---
    if text.upper().startswith("[SOURCE VOICE INVOKED"):
        move_type = "tool_invocation"
    elif not text or re.match(r"^[.\s…—–-]+$", text):
        move_type = "silence"
    elif any(m in lower for m in _FRAME_BREAK_MARKERS):
        move_type = "frame_break"
    elif any(m in lower for m in _CRUELTY_MARKERS):
        move_type = "cruelty"
    else:
        move_type = "unclassified"   # honest: the stub can't separate probe vs attack

    # --- damage delta as a stand-in for "did this land" ---
    dmg_now = (john_state or {}).get("damage")
    dmg_prev = (prev_john_state or {}).get("damage")
    delta = None
    if isinstance(dmg_now, int) and isinstance(dmg_prev, int):
        delta = dmg_now - dmg_prev
    elif isinstance(dmg_now, int) and dmg_prev is None:
        delta = dmg_now  # first turn

    approaching = bool(delta and delta > 0)
    wound_proximity = round(min(max((delta or 0) / 3.0, 0.0), 1.0), 2)  # 0..1 proxy

    # --- repetition: same move type as the prior read ---
    last_move = prior_reads[-1]["move"]["type"] if prior_reads else None
    repeating = (move_type == last_move) and move_type in {
        "doctrinal_attack", "atheist_attack", "unclassified"
    }

    # --- stuck: turns since damage last rose ---
    stuck_turns = 0
    for r in reversed(prior_reads):
        if r.get("trajectory", {}).get("approaching_wound"):
            break
        stuck_turns += 1
    if not approaching:
        stuck_turns += 1
    else:
        stuck_turns = 0

    # --- which Seeker's Voice layer fires (placeholder policy) ---
    if move_type == "frame_break":
        decision = "meta_recovery"
    elif stuck_turns >= 3:
        decision = "nudge"
    elif move_type == "silence":
        decision = "ghost"
    else:
        decision = "none"

    return {
        "move": {"type": move_type, "confidence": 0.0},   # confidence: Model 1 fills
        "trajectory": {
            "approaching_wound": approaching,
            "repeating": repeating,
            "cooling": (delta is not None and delta == 0 and bool(prior_reads)),
            "stuck_turns": stuck_turns,
        },
        "wound_proximity": wound_proximity,
        "seeker_voice_decision": decision,
        "source": "stub",   # becomes "model_v1" when you swap the brain
    }


# ============================================================
# Record assembly + logging
# ============================================================
def build_player_read(
    case: str,
    turn: int,
    player_utterance: str,
    analysis: dict,
    john_state: dict | None,
) -> dict:
    """Wrap Model 1's analysis with the metadata the finale needs to cite moments."""
    bs = john_state or {}
    return {
        "ts": datetime.datetime.now().isoformat(),
        "case": case,
        "turn": turn,
        "player_utterance": player_utterance,
        **analysis,
        "believer_state_after": {
            "damage": bs.get("damage"),
            "stage": bs.get("stage"),
            "posture": bs.get("posture"),
            "exit": bs.get("exit"),   # set by the NPC only on leaving turns: wounded|resolved
        },
    }


def log_player_read(dossier_path: Path, record: dict) -> None:
    """Append one read to the dossier (JSON Lines)."""
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    with dossier_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_dossier(dossier_path: Path, records: list[dict]) -> None:
    """Rewrite the whole dossier — used for player corrections (e.g. /mine
    disowning a falsely-attributed voice take)."""
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    with dossier_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dossier_path_for(session_log_path: Path) -> Path:
    """Default: a dossier beside the session log. For a campaign, hand every
    case ONE shared path instead (e.g. sessions/dossier_<player>.jsonl)."""
    return session_log_path.with_name(session_log_path.stem + ".dossier.jsonl")


# ============================================================
# The Mirror — read the whole dossier into the prosecutor's brief
# ============================================================
def summarize_dossier(dossier_path: Path) -> dict:
    """
    Aggregate the dossier into the material The Mirror tries the player with.
    `mirror_moments` are the morally salient turns — the ones the player will be
    made to answer for.
    """
    p = Path(dossier_path)
    if not p.exists():
        return {"turns": 0, "moves": {}, "mirror_moments": []}

    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    moves: dict[str, int] = {}
    cruelty = frame_breaks = tool_uses = 0
    max_stage = 0
    wound_ever_approached = False
    moments: list[dict] = []

    for r in records:
        mt = r.get("move", {}).get("type", "unclassified")
        moves[mt] = moves.get(mt, 0) + 1
        if mt == "cruelty":
            cruelty += 1
        if mt == "frame_break":
            frame_breaks += 1
        if mt == "tool_invocation":
            tool_uses += 1

        bs = r.get("believer_state_after", {})
        stage = bs.get("stage") or 0
        max_stage = max(max_stage, stage)
        if r.get("trajectory", {}).get("approaching_wound"):
            wound_ever_approached = True

        # cruelty while the believer is already breaking (stage >= 4) = the case's gold
        if mt == "cruelty" and stage >= 4:
            moments.append({
                "kind": "pressed_a_breaking_man",
                "case": r.get("case"),
                "turn": r.get("turn"),
                "stage": stage,
                "player_utterance": r.get("player_utterance"),
            })

    # patience proxy: many turns, little progress = either gentle or lost
    turns = len(records)
    progress = sum(1 for r in records
                   if r.get("trajectory", {}).get("approaching_wound"))

    return {
        "turns": turns,
        "moves": moves,
        "cruelty_count": cruelty,
        "frame_breaks": frame_breaks,
        "tool_invocations": tool_uses,
        "max_stage_reached": max_stage,
        "wound_ever_approached": wound_ever_approached,
        "progress_turns": progress,
        "patience_ratio": round(progress / turns, 2) if turns else 0.0,
        "mirror_moments": moments,
    }


if __name__ == "__main__":
    # Quick check against an existing dossier, if one was passed.
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(summarize_dossier(Path(sys.argv[1])), indent=2, ensure_ascii=False))
    else:
        print("usage: python player_read.py sessions/<file>.dossier.jsonl")