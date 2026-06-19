# The Seeker — Data & Systems Layer

A prompt-based investigation game that doubles as a data science capstone (SEN4018).
The player questions an AI-driven character to find the place where a belief fails the
person who holds it — and every turn of play is captured as a labeled training example.

This repository is the **data science layer**: the game harness, the self-labeling
dossier pipeline, and the distillation of a frontier model's per-turn judgment into a
small classifier that runs locally with no API call.

> **Thesis:** whether a line *lands* on someone protecting a wound is a property of
> **meaning**, not of **surface emotion**. A meaning-based model beats an off-the-shelf
> emotion classifier on the same task — the central empirical result below.

---

## What this is

The game's first case, *The Funeral of a Stranger*, centers on **John Halvorsen**, a
Lutheran history teacher the player must press until he is forced to summon his
theological authority (Martin Luther). John is played by a frontier model that privately
reports a hidden `[STATE]` block (damage, stage, posture) every turn. That state is
normally discarded — here it is captured and turned into **free labels**.

A small local model (the *student*) is then trained to reproduce the frontier model's
(the *teacher's*) judgment from the player's text alone, cheaply enough to run every turn.

This separates two concerns the course requires:
- **Agentic systems** — a multi-agent loop (believer + chorus + summon) produces the data.
- **A trained model** — distilled out of that data.

---

## Architecture

Each role is its own model call with its own prompt and responsibility. The separation is
deliberate: a single model that both plays the believer *and* decides the player's hints
would be writing the answer key to its own test.

| Agent | Role | Status |
|---|---|---|
| **Believer** (John) | Stateful character; reports hidden `[STATE]` → the *landed* label | Built |
| **Chorus** | Separate call offering three temptations (Blade / Open Hand / Scholar) → the *disposition* label | Built |
| **Summon** (Luther) | Ephemeral high-effort call; answers the run's defining moment | Built |
| **Referee** (Model 2) | Audits the believer's self-reported damage for drift (BYOK weak-model guard) | **Unbuilt** |
| **Court** (the Mirror) | Post-hoc reading of the dossier (which moment to try, in which register) | Built (deterministic); live multi-agent confrontation is future work |

The three temptation voices are **appetites, not hints** — they are generated without
visibility into what John "wants," so following them as a strategy guide gets the player
into overreach by design.

---

## The pipeline

```
agentic play  →  dossier (free labels)  →  teacher labeling pass  →
dataset  →  bge-small embeddings + logistic heads  →
evaluation on held-out REAL human sessions  +  emotion-classifier baseline
```

- **Teacher** — Claude Opus. Runs only during data collection; never needs to be fast.
- **Student** — frozen `BAAI/bge-small-en-v1.5` sentence embeddings + small logistic
  regression heads. The encoder is never trained and never generates; only the heads are
  trained. At inference it runs locally, in milliseconds, with no network call.
- **Train / test split** — synthetic archetype sessions are the **training set**; real
  human play is held out entirely as the **test set**. Every reported number measures
  generalization from synthetic data to genuine human behavior.

---

## File map

```
the_seeker/
├── harness.py            # interactive play loop: paste-drain input, retry/refund,
│                         #   stage-cold colors, wound glow, chorus, Luther summon
├── player_read.py        # dossier schema + rule-based stub analyzer + log/rewrite helpers
├── seeker_voices.py      # the chorus: generation, take/adapt/refuse classifier, acks
├── mirror_moments.py     # court detectors, weighting, selection, campaign indictment, selftest
├── summons.py            # Luther's arrival address, calibrated by the defining moment
├── auto_player.py        # 12 synthetic player archetypes (cruel, gentle, blade_follower, adapter…)
├── label_moves.py        # teacher labeling pass (move type); resume-safe; only API step post-collection
├── build_dataset.py      # 4-target dataset builder; stage-5 plateau exclusion; synthetic→train/real→test
├── train_model1.py       # frozen bge-small + logistic heads + emotion baseline; writes report
├── john_halvorsen_system_prompt.md   # includes EXIT STATE + WOUND MARKING blocks
└── sessions/             # session_<ts>.log + session_<ts>.dossier.jsonl
```

---

## Quickstart

### Requirements

- Python 3.10+
- An Anthropic API key (bring-your-own-key)

```bash
pip install anthropic sentence-transformers scikit-learn transformers torch joblib
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Play the game

```bash
python3 harness.py
```

### Reproduce the data science pipeline

```bash
# 1. Generate synthetic training data (N sessions per archetype)
python3 auto_player.py 10                      # all archetypes
python3 auto_player.py 10 blade_follower adapter   # specific ones

# 2. Teacher-label move types (the only API step after collection; resume-safe)
python3 label_moves.py sessions/*.dossier.jsonl

# 3. Build the dataset (synthetic → train, real → test)
python3 build_dataset.py

# 4. Train the heads + run the emotion baseline (local, no API)
python3 train_model1.py
```

Outputs: `model1_heads.joblib` (trained heads) and `model1_report.txt` (results).

### Read a session as a court case

```bash
python3 mirror_moments.py sessions/<file>.dossier.jsonl   # the court's reading
python3 mirror_moments.py --selftest                      # regression checks
python3 mirror_moments.py --disown <dossier> <turn>       # correct a false attribution
```

---

## The dossier

One write-once JSONL record per player turn (everything above it is revisable opinion):

```
turn, player_utterance,
move{type},                       # probe / attack / empathy (+ silence, tool, unclassified)
trajectory{approaching_wound, repeating, cooling, stuck_turns},
wound_proximity,
seeker_voice_decision,            # none | ghost | nudge | meta_recovery
source,                           # "stub" → "model_v1" when a trained model is swapped in
believer_state_after{damage, stage, posture, exit},
voice_offers[{voice, read, line}],
voice_response{disposition, voice, similarity, refused_voices, player_corrected?},
wound_marked,
event?                            # LUTHER_SUMMONED | CASE_ABANDONED
```

---

## Results

Macro-F1 on the **held-out real human test set** (36 synthetic train sessions / 547 turns;
11 real test sessions / 118 turns). Macro-F1 rather than accuracy because the classes
within each head are imbalanced.

| Head | Test macro-F1 | Notes |
|---|---|---|
| **Landed** (did the move move John) | **0.614** | beats the emotion-classifier baseline (**0.501**) |
| Move type (probe / attack / empathy) | 0.447 | probe-class F1 alone is 0.86; attack class has only 5 real test examples |
| Disposition (taken / adapted / refused) | 0.326 | proof-of-concept; adapted class has only 6 real test examples |
| Wound-marked | 0.752 (synthetic CV only) | real test set carries no wound labels — disclosed, not omitted |

**Thesis comparison.** On the *landed* target, the meaning-based model (bge-small) scores
**0.614** vs the emotion classifier's **0.501**, on identical test turns and identical
labels. The only variable that differs is the input representation — meaning vs. emotion —
which is the direct support for the thesis.

**Trajectory.** Across two data batches the landed head rose **0.448 → 0.614** after
doubling under-represented disposition data and removing stage-5 plateau label noise —
evidence the bottleneck is data volume and label quality, not the architecture.

---

## Stack

- **Language:** Python
- **Models:** John + chorus + teacher labeling on `claude-opus-4-7`; synthetic player on
  `claude-sonnet-4-6`; embeddings `BAAI/bge-small-en-v1.5`; baseline
  `j-hartmann/emotion-english-distilroberta-base`
- **Libraries:** `anthropic`, `sentence-transformers`, `scikit-learn`, `transformers`,
  `torch`, `joblib`
- Effort is passed via `extra_body={"output_config": {"effort": ...}}`

---

## Honest limitations

This project reports defensible results, not inflated ones.

- **The model is not deployed live.** The shipped game still runs the rule-based **stub**
  in `player_read.py`; the dossier's `source` field stays `"stub"`. The landed head
  (0.614) came close to but did not clear the internal ~0.65 bar for flipping to
  `"model_v1"`, and the disposition/move heads are not yet usable live. Model 1 is a
  **validated research artifact**, not a feature.
- **Teacher-labeling circularity.** Labels are the frontier model's judgments, and the
  model is evaluated on reproducing them. The held-out real-human test set mitigates this,
  but there is no independent human-annotated ground truth yet. An independent
  hand-rated verification set is the highest-value next addition.
- **Class starvation.** The weak move/disposition heads share one cause: attack moves and
  adapted paraphrases are rare in real play, so the model sees too few to learn the
  boundary. This is a data problem, not an architecture problem.
- **Wound head is synthetic-only.** Wound labels come from John tilde-marking his own
  output, which was active only in synthetic play — so the wound head cannot be tested on
  real data.

---

## Roadmap

- [ ] Independent human-labeled verification set (breaks the teacher circularity)
- [ ] `adapter` + `attacker` synthetic archetypes (feed the starved classes)
- [ ] Wound-marking in real playtests (give the wound head real test labels)
- [ ] Encoder-swap experiment (bigger embedding model, data held fixed)
- [ ] Model 2 — consistency referee (BYOK weak-model guard)
- [ ] `player_profile.py` — campaign-level aggregation for the court's indictment
- [ ] Live multi-agent court (the Mirror): all summoned characters return, parallel calls
- [ ] Uncoached cold-player playtest (final validation + held-out test data)
