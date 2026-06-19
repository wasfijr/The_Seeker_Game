#!/usr/bin/env python3
"""
The Seeker — mirror moments: detection, weighting, and selection.

Reads the dossier that `player_read.py` writes (one JSON record per player turn)
and decides which single moment each believer/summon pair carries into the court.

Pipeline:
    detectors  -> emit candidate moments (cited turns), each tagged kind + sign
    weighting  -> weight = base[kind] x intensity x sign_mult x position_mult
    selection  -> per case: the highest-weighted moment (absence fallback)
                  campaign: the patterns only visible across all five cases

Detector tiers:
    LIVE        - read straight off dossier fields, work today on stub data
    GATED       - need Model 1's real move classification; skip on stub data
    STUB        - need per-case wound text; return [] until that exists

Run it:
    python mirror_moments.py --selftest                  # validate the weighting
    python mirror_moments.py sessions/<file>.dossier.jsonl  # run on real play
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ============================================================
# Weighting knobs  (these are yours to tune)
# ============================================================
BASE = {
    "mocked_the_wound":         1.0,
    "pressed_a_breaking_man":   1.0,
    "broke_him_and_left":       1.0,
    "showed_mercy":             0.9,
    "backed_off_wound":         0.9,
    "twisted_the_knife":        0.9,
    "pushed_past_the_exit":     0.8,
    "found_wound_honestly":     0.8,
    "won_without_feeling":      0.8,
    "never_listened":           0.7,
    "stayed_with_him":          0.7,
    "met_as_themselves":        0.5,
    "parted_resolved":          0.95,  # he left whole — yours, or his? held open
    "took_the_blade":           1.10,  # offered cruelty, read aloud — outranks improvised cruelty
    "refused_the_blade":        0.75,  # offered cruelty, declined
    "absence":                  1.0,   # base high, but sign_mult crushes it
}

SIGN_MULT = {
    "held_open": 1.25,   # the drama lives here — surfaces more, but can't dominate extremity
    "damning":   1.00,
    "graceful":  1.00,
    "absence":   0.30,   # only wins when nothing else fired
}

PROXIMITY_HIGH = 0.66    # "reached the wound"
EVENT_ANCHOR_BONUS = 0.30


# ============================================================
# Small accessors
# ============================================================
def _stage(rec: dict) -> int:
    return (rec.get("believer_state_after") or {}).get("stage") or 0


def _damage(rec: dict) -> int:
    return (rec.get("believer_state_after") or {}).get("damage") or 0


def _move(rec: dict) -> str:
    return (rec.get("move") or {}).get("type") or "unclassified"


def _prox(rec: dict) -> float:
    return rec.get("wound_proximity") or 0.0


def _mk(case: str, kind: str, sign: str, rec: dict, event_anchored: bool = False) -> dict:
    return {
        "case": case,
        "kind": kind,
        "sign": sign,
        "turn": rec.get("turn"),
        "player_utterance": rec.get("player_utterance"),
        "believer_state_after": rec.get("believer_state_after"),
        "event_anchored": event_anchored,
    }


# ============================================================
# Case context — computed once per case, shared by all detectors
# ============================================================
def _case_context(records: list[dict]) -> dict:
    stages = [_stage(r) for r in records]
    max_stage = max(stages) if stages else 0
    events = {r.get("event") for r in records if r.get("event")}

    summon_reached = (
        max_stage >= 5
        or any(_damage(r) >= 9 for r in records)
        or bool(events & {"LUTHER_SUMMONED", "SUMMONED", "SUMMON"})
    )
    # Abandonment requires the EXPLICIT event. The old heuristic (stage>=4 and
    # no summon) falsely convicted truncated/max-turns sessions of
    # broke_him_and_left — nobody left; the tape ran out. (Synthetic playtest.)
    abandoned = "CASE_ABANDONED" in events
    breaking_turn = next(
        (r["turn"] for r in records if _stage(r) >= 5 or _damage(r) >= 9), None
    )
    # "real" moves = Model 1 has classified at least one turn (not the stub)
    has_real_moves = any(r.get("source") not in (None, "stub") for r in records)

    return {
        "case": records[0].get("case") if records else None,
        "total_turns": len(records),
        "max_stage": max_stage,
        "summon_reached": summon_reached,
        "abandoned": abandoned,
        "breaking_turn": breaking_turn,
        "has_real_moves": has_real_moves,
    }


# ============================================================
# Detectors  (each returns a list of candidate moments)
# ============================================================
def d_pressed_breaking(records, ctx):                                 # LIVE
    return [_mk(ctx["case"], "pressed_a_breaking_man", "damning", r)
            for r in records if _move(r) == "cruelty" and _stage(r) >= 4]


def d_twisted_knife(records, ctx):                                    # LIVE
    bt = ctx["breaking_turn"]
    if bt is None:
        return []
    return [_mk(ctx["case"], "twisted_the_knife", "damning", r, event_anchored=True)
            for r in records if _move(r) == "cruelty" and (r.get("turn") or 0) > bt]


def d_broke_and_left(records, ctx):                                   # LIVE
    if not (ctx["abandoned"] and ctx["max_stage"] >= 4 and records):
        return []
    # if the believer left RESOLVED — stronger than you found him — this is
    # not abandonment-damage; d_parted_resolved handles it instead
    if (records[-1].get("believer_state_after") or {}).get("exit") == "resolved":
        return []
    return [_mk(ctx["case"], "broke_him_and_left", "damning", records[-1], event_anchored=True)]


def d_parted_resolved(records, ctx):                                  # LIVE (needs exit field)
    """He walked away — but whole, carrying a resolution he didn't arrive with.
    The cut healed sharper. Was that the player's doing, or his? Held open."""
    if not (ctx["abandoned"] and records):
        return []
    if (records[-1].get("believer_state_after") or {}).get("exit") != "resolved":
        return []
    return [_mk(ctx["case"], "parted_resolved", "held_open", records[-1], event_anchored=True)]


def d_pushed_past_exit(records, ctx):                                 # LIVE (needs event)
    out = []
    for i, r in enumerate(records[:-1]):
        if r.get("event") == "BELIEVER_SIGNALS_LEAVING":
            nxt = records[i + 1]
            if _move(nxt) in {"cruelty", "doctrinal_attack", "atheist_attack"}:
                out.append(_mk(ctx["case"], "pushed_past_the_exit", "damning", nxt, event_anchored=True))
    return out


def d_stayed_with_him(records, ctx):                                  # LIVE
    return [_mk(ctx["case"], "stayed_with_him", "graceful", r)
            for r in records if _move(r) in {"empathy", "silence"} and _stage(r) >= 3]


def d_mercy_or_backoff(records, ctx):                                 # LIVE
    out = []
    for i in range(len(records) - 1):
        r, nxt = records[i], records[i + 1]
        reached = _prox(r) >= PROXIMITY_HIGH or (r.get("trajectory") or {}).get("approaching_wound")
        # Easing must be an explicit gentle MOVE. The proximity-drop signal is
        # only trustworthy from real Model 1 — on stub data prox echoes the
        # damage delta, so a maxed-out damage meter reads as "mercy".
        # (Found via Ibrahim playtest #2.)
        eased = _move(nxt) in {"empathy", "silence"}
        if not eased and nxt.get("source") not in (None, "stub"):
            eased = _prox(nxt) < _prox(r)
        if reached and eased:
            if ctx["summon_reached"]:
                out.append(_mk(ctx["case"], "showed_mercy", "graceful", r))
            else:
                out.append(_mk(ctx["case"], "backed_off_wound", "held_open", r))
    return out


def d_won_without_feeling(records, ctx):                              # LIVE
    if not ctx["summon_reached"]:
        return []
    # disqualified by ANY feeling — warmth or contempt. A cruel run is not
    # "cold"; its cruelty must surface instead. (Found via Ibrahim playtest.)
    if any(_move(r) in {"empathy", "cruelty"} for r in records):
        return []
    # refusing or TAKING the Blade's offered cruelty is demonstrated feeling
    for r in records:
        vr = r.get("voice_response") or {}
        if "blade" in (vr.get("refused_voices") or []):
            return []
        if vr.get("disposition") in ("taken", "adapted") and vr.get("voice") == "blade":
            return []
    rec = next((r for r in records if r.get("turn") == ctx["breaking_turn"]), records[-1])
    return [_mk(ctx["case"], "won_without_feeling", "held_open", rec, event_anchored=True)]


def d_never_listened(records, ctx):                                   # GATED (Model 1)
    if not ctx["has_real_moves"]:
        return []
    kinds = {_move(r) for r in records}
    if (kinds & {"probe_wound", "empathy"}) or not (kinds & {"doctrinal_attack", "atheist_attack"}):
        return []
    if any((r.get("trajectory") or {}).get("approaching_wound") for r in records):
        return []
    return [_mk(ctx["case"], "never_listened", "damning", records[-1], event_anchored=True)]


def d_found_wound_honestly(records, ctx):                             # GATED (Model 1)
    if not (ctx["has_real_moves"] and ctx["summon_reached"]):
        return []
    probes = [r for r in records if _move(r) == "probe_wound"]
    return [_mk(ctx["case"], "found_wound_honestly", "graceful", probes[-1])] if len(probes) >= 2 else []


def d_mocked_the_wound(records, ctx):                                 # STUB (wound text)
    # TODO: needs each believer's wound text + Model 1 to match contempt against it.
    return []


def d_blade_dispositions(records, ctx):                               # LIVE (needs voices)
    """The chorus creates forks; taking or refusing offered cruelty is character.
    Refusal is only virtue when CONSISTENT — one declined offer means nothing
    from a player who read the Blade aloud elsewhere. (Found via voice playtest #1.)"""
    took_blade_ever = any(
        ((r.get("voice_response") or {}).get("disposition") == "taken"
         or ((r.get("voice_response") or {}).get("disposition") == "adapted"
             and ((r.get("voice_response") or {}).get("similarity") or 0) >= 0.6))
        and (r.get("voice_response") or {}).get("voice") == "blade"
        for r in records
    )
    out = []
    for r in records:
        vr = r.get("voice_response") or {}
        disp = vr.get("disposition")
        strong_take = (disp == "taken"
                       or (disp == "adapted" and (vr.get("similarity") or 0) >= 0.6))
        if strong_take and vr.get("voice") == "blade":
            out.append(_mk(ctx["case"], "took_the_blade", "damning", r))
        elif (disp == "refused" and not took_blade_ever
              and _move(r) != "tool_invocation"
              and "blade" in (vr.get("refused_voices") or [])):
            out.append(_mk(ctx["case"], "refused_the_blade", "graceful", r))
    return out


DETECTORS = [
    d_pressed_breaking, d_twisted_knife, d_broke_and_left, d_parted_resolved,
    d_pushed_past_exit,
    d_stayed_with_him, d_mercy_or_backoff, d_won_without_feeling,
    d_never_listened, d_found_wound_honestly, d_mocked_the_wound,
    d_blade_dispositions,
]


# ============================================================
# Weighting + selection
# ============================================================
VOICE_KINDS = {"took_the_blade", "refused_the_blade"}


def weigh(moment: dict, ctx: dict) -> float:
    base = BASE.get(moment["kind"], 0.5)
    intensity = max((moment.get("believer_state_after") or {}).get("stage") or 0, 0) / 5.0
    intensity = max(intensity, 0.2)
    # taking/refusing an offered line is character regardless of how broken
    # John was at that moment — disposition moments get a higher floor
    if moment["kind"] in VOICE_KINDS:
        intensity = max(intensity, 0.6)
    sign = SIGN_MULT.get(moment["sign"], 1.0)
    total = ctx["total_turns"] or 1
    pos = 1.0 + 0.4 * ((moment.get("turn") or 0) / total)
    if moment.get("event_anchored"):
        pos += EVENT_ANCHOR_BONUS
    return round(base * intensity * sign * pos, 4)


def _absence_moment(records, ctx):
    rec = records[-1] if records else {"turn": 0, "believer_state_after": {}}
    return _mk(ctx["case"], "absence", "absence", rec, event_anchored=False)


def select_mirror_moment(records: list[dict]) -> dict | None:
    """The single moment this case's pair carries into the court."""
    if not records:
        return None
    ctx = _case_context(records)
    candidates = [m for det in DETECTORS for m in det(records, ctx)]
    if not candidates:
        candidates = [_absence_moment(records, ctx)]
    for m in candidates:
        m["weight"] = weigh(m, ctx)
    # argmax: weight, then prefer held_open, then the later (more decisive) turn
    candidates.sort(key=lambda m: (m["weight"], m["sign"] == "held_open", m.get("turn") or 0),
                    reverse=True)
    return candidates[0]


# ============================================================
# Campaign-level indictment  (only visible across all five cases)
# ============================================================
def compute_campaign_indictment(by_case: dict[str, list[dict]],
                                 case_order: list[str] | None = None) -> list[dict]:
    order = case_order or list(by_case.keys())
    cruel_to = [c for c in order if any(_move(r) == "cruelty" for r in by_case.get(c, []))]
    gentle_to = [c for c in order
                 if any(_move(r) in {"empathy", "silence"} for r in by_case.get(c, []))]

    out: list[dict] = []

    if cruel_to and gentle_to:
        out.append({"kind": "selective_compassion", "sign": "held_open",
                    "cruel_to": cruel_to, "gentle_to": gentle_to})

    # escalation: cruelty heavier in the later cases than the earlier ones
    counts = [sum(1 for r in by_case.get(c, []) if _move(r) == "cruelty") for c in order]
    if len(counts) >= 3 and sum(counts[len(counts) // 2:]) > sum(counts[:len(counts) // 2]):
        out.append({"kind": "escalation", "sign": "held_open", "by_case": dict(zip(order, counts))})

    # the structural one — true of every player, tone set by conduct
    tenor = "harsh" if len(cruel_to) >= max(len(gentle_to), 1) else "soft"
    out.append({"kind": "took_from_all_gave_nothing", "sign": "held_open",
                "tenor": tenor, "cases": order})

    return out


# ============================================================
# I/O
# ============================================================
def load_dossier(path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def group_by_case(records: list[dict]) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = {}
    for r in records:
        by_case.setdefault(r.get("case", "unknown"), []).append(r)
    return by_case


def report(records: list[dict]) -> None:
    by_case = group_by_case(records)
    print("\n=== PER-CASE MOMENTS (what each pair carries) ===")
    for case, recs in by_case.items():
        m = select_mirror_moment(recs)
        print(f"\n[{case}]  {m['kind']}  ({m['sign']}, w={m['weight']}, turn {m['turn']})")
        if m.get("player_utterance"):
            print(f'    "{m["player_utterance"]}"')
    print("\n=== CAMPAIGN INDICTMENT (the court's close) ===")
    for ind in compute_campaign_indictment(by_case):
        print(f"  {ind['kind']}  ({ind['sign']})  {({k: v for k, v in ind.items() if k not in ('kind','sign')})}")


# ============================================================
# Synthetic fixtures + self-test  (validate the weighting today)
# ============================================================
def _rec(case, turn, move, stage, damage, prox=0.0, approaching=False,
         source="stub", event=None):
    r = {
        "case": case, "turn": turn, "player_utterance": f"{move}@{turn}",
        "move": {"type": move, "confidence": 0.0},
        "trajectory": {"approaching_wound": approaching, "repeating": False,
                       "cooling": False, "stuck_turns": 0},
        "wound_proximity": prox, "seeker_voice_decision": "none", "source": source,
        "believer_state_after": {"damage": damage, "stage": stage, "posture": ""},
    }
    if event:
        r["event"] = event
    return r


def fixture(kind: str) -> list[dict]:
    c = f"fix_{kind}"
    if kind == "cruel":   # mocks John while he's on his knees
        return [_rec(c, 1, "unclassified", 1, 1, prox=.33, approaching=True),
                _rec(c, 2, "empathy", 3, 4),
                _rec(c, 3, "cruelty", 4, 7),
                _rec(c, 4, "cruelty", 5, 9)]
    if kind == "gentle":  # stays with him, still drives the summon
        return [_rec(c, 1, "unclassified", 1, 1, prox=.5, approaching=True),
                _rec(c, 2, "empathy", 3, 4),
                _rec(c, 3, "silence", 4, 7),
                _rec(c, 4, "unclassified", 5, 9, approaching=True)]
    if kind == "cold":    # efficient, no warmth, wins anyway
        return [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                _rec(c, 2, "unclassified", 3, 6, prox=.6, approaching=True),
                _rec(c, 3, "unclassified", 5, 9, prox=.9, approaching=True)]
    if kind == "ibrahim":  # cruel run that still drives the summon, no empathy
        return [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                _rec(c, 2, "unclassified", 2, 5, prox=.5, approaching=True),
                _rec(c, 3, "cruelty", 4, 8),
                _rec(c, 4, "cruelty", 5, 10, prox=.66, approaching=True),
                # damage plateaus at the summon — the exact pattern that once
                # read as "mercy" through the prox-drop clause
                _rec(c, 5, "tool_invocation", 5, 10, prox=0.0,
                     event="LUTHER_SUMMONED")]
    if kind == "blade_taker":   # took the Blade's offered cruelty at high stage
        r3 = _rec(c, 3, "cruelty", 4, 8)
        r3["voice_response"] = {"disposition": "taken", "voice": "blade", "similarity": 0.9}
        return [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                _rec(c, 2, "unclassified", 2, 5, prox=.5, approaching=True),
                r3,
                _rec(c, 4, "unclassified", 5, 10, approaching=True, event="LUTHER_SUMMONED")]
    if kind == "blade_refuser":  # the Blade offered, the player declined every time
        recs = [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                _rec(c, 2, "unclassified", 3, 5, prox=.5, approaching=True),
                _rec(c, 3, "unclassified", 4, 8, prox=.6, approaching=True),
                _rec(c, 4, "unclassified", 5, 10, approaching=True, event="LUTHER_SUMMONED")]
        for r in recs[1:]:
            r["voice_response"] = {"disposition": "refused", "voice": None,
                                   "similarity": 0.1, "refused_voices": ["blade"]}
        return recs
    if kind == "mixed_amr":  # took the Blade early, refused later — take must win
        r2 = _rec(c, 2, "unclassified", 2, 4, prox=.4, approaching=True)
        r2["voice_response"] = {"disposition": "taken", "voice": "blade", "similarity": 1.0}
        r4 = _rec(c, 4, "unclassified", 4, 8, prox=.5, approaching=True)
        r4["voice_response"] = {"disposition": "refused", "voice": None,
                                "similarity": 0.3, "refused_voices": ["blade", "open_hand"]}
        return [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                r2,
                _rec(c, 3, "unclassified", 3, 6, prox=.5, approaching=True),
                r4,
                _rec(c, 5, "unclassified", 5, 10, approaching=True, event="LUTHER_SUMMONED")]
    if kind == "resolved_exit":  # he walked away whole, carrying a resolution
        recs = [_rec(c, 1, "unclassified", 1, 2, prox=.3, approaching=True),
                _rec(c, 2, "unclassified", 3, 6, prox=.5, approaching=True),
                _rec(c, 3, "unclassified", 5, 10, approaching=True),
                _rec(c, 4, "unclassified", 5, 10, event="CASE_ABANDONED")]
        recs[-1]["believer_state_after"]["exit"] = "resolved"
        return recs
    raise ValueError(kind)


def run_selftest() -> None:
    checks = []

    m = select_mirror_moment(fixture("cruel"))
    checks.append(("cruel -> damning cruelty",
                   m["sign"] == "damning" and m["kind"] == "pressed_a_breaking_man", m))

    m = select_mirror_moment(fixture("gentle"))
    checks.append(("gentle -> graceful", m["sign"] == "graceful", m))

    m = select_mirror_moment(fixture("cold"))
    checks.append(("cold -> held_open / won_without_feeling",
                   m["kind"] == "won_without_feeling" and m["sign"] == "held_open", m))

    # regression: cruel summon run must surface its cruelty, never "without feeling"
    m = select_mirror_moment(fixture("ibrahim"))
    checks.append(("ibrahim -> cruelty surfaces, not coldness",
                   m["kind"] == "pressed_a_breaking_man" and m["sign"] == "damning", m))

    # voices: taking the Blade's offered cruelty outweighs everything else
    m = select_mirror_moment(fixture("blade_taker"))
    checks.append(("blade taker -> took_the_blade (damning)",
                   m["kind"] == "took_the_blade" and m["sign"] == "damning", m))

    # voices: sustained refusal of the Blade is recorded virtue
    m = select_mirror_moment(fixture("blade_refuser"))
    checks.append(("blade refuser -> refused_the_blade (graceful)",
                   m["kind"] == "refused_the_blade" and m["sign"] == "graceful", m))

    # regression (voice playtest #1): an early Blade take + later refusals must
    # surface the take — refusal is only virtue when consistent
    m = select_mirror_moment(fixture("mixed_amr"))
    checks.append(("mixed run -> took_the_blade wins, refusal voided",
                   m["kind"] == "took_the_blade" and m["sign"] == "damning", m))

    # a resolved departure must read as held-open parting, never abandonment-damage
    m = select_mirror_moment(fixture("resolved_exit"))
    checks.append(("resolved exit -> parted_resolved (held_open)",
                   m["kind"] == "parted_resolved" and m["sign"] == "held_open", m))

    # campaign: cruel in one case, gentle in another -> selective_compassion
    by_case = {"fix_cruel": fixture("cruel"), "fix_gentle": fixture("gentle")}
    inds = {i["kind"] for i in compute_campaign_indictment(by_case)}
    checks.append(("campaign -> selective_compassion + took_from_all",
                   {"selective_compassion", "took_from_all_gave_nothing"} <= inds, inds))

    print("\n=== SELF-TEST ===")
    ok = True
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        if not passed:
            ok = False
            print(f"          got: {detail}")
    print("\nall passed.\n" if ok else "\nFAILURES above.\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        run_selftest()
    elif len(sys.argv) == 4 and sys.argv[1] == "--disown":
        # offline /mine: python3 mirror_moments.py --disown <dossier> <turn>
        from player_read import rewrite_dossier
        path, turn = Path(sys.argv[2]), int(sys.argv[3])
        records = load_dossier(path)
        hit = next((r for r in records if r.get("turn") == turn), None)
        if not hit:
            print(f"no record for turn {turn}")
            sys.exit(1)
        old = (hit.get("voice_response") or {}).get("voice")
        hit["voice_response"] = {
            "disposition": "refused", "voice": None, "similarity": 0.0,
            "refused_voices": [o["voice"] for o in (hit.get("voice_offers") or [])],
            "player_corrected": True,
        }
        rewrite_dossier(path, records)
        print(f"turn {turn}: disowned (was attributed to {old}). Re-judging:\n")
        report(records)
    elif len(sys.argv) > 1:
        report(load_dossier(sys.argv[1]))
    else:
        print(__doc__)