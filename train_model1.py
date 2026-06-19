#!/usr/bin/env python3
"""
The Seeker — train_model1.py: distil the teacher's judgment into a local model.

Frozen bge-small embeddings + logistic-regression heads, trained on synthetic
sessions, evaluated STRICTLY on held-out real human sessions. Reports macro-F1
(not accuracy — the move classes are imbalanced and accuracy would lie).
Includes the j-hartmann emotion-classifier baseline for the central thesis:
wound-proximity / landed ≠ surface emotion.

Three heads, each scoped to what the data honestly supports:
  landed       — trained + tested on real data (the headline result)
  disposition  — trained + tested on real data (proof-of-concept, smaller n)
  move_coarse  — probe_wound vs attack vs empathy (rare classes merged;
                 silence + tool_invocation are rule-detectable, excluded here)
  wound_marked — trained on synthetic only (real test set has no labels) — eval
                 reported as TRAIN-CV only, flagged honestly.

Setup (local, no API):
    pip install sentence-transformers scikit-learn transformers torch --break-system-packages
Run:
    python3 train_model1.py
Outputs: model1_heads.joblib + a metrics report (also written to model1_report.txt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np


# ---------- data ----------
def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"missing {path} — run build_dataset.py first")
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def move_coarse(m: str | None) -> str | None:
    """Collapse the sparse, learnable move classes into an honest label set.
    silence + tool_invocation are excluded (rule-detectable, not learned)."""
    if m in (None, "silence", "tool_invocation", "unclassified"):
        return None
    if m in ("doctrinal_attack", "atheist_attack", "cruelty"):
        return "attack"
    if m == "probe_wound":
        return "probe"
    if m == "empathy":
        return "empathy"
    return None


def emb_text(r: dict) -> str:
    ctx = r.get("context") or ""
    return (ctx + " [SEP] " + (r.get("text") or "")).strip()


# ---------- metrics ----------
def report_head(name, ytr, yte, predtr, predte, f):
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    def line(s=""):
        print(s); f.write(s + "\n")
    line(f"\n{'='*60}\n{name}\n{'='*60}")
    line(f"train n={len(ytr)}  classes={dict(Counter(ytr))}")
    line(f"test  n={len(yte)}  classes={dict(Counter(yte))}")
    if len(set(yte)) < 2:
        line("** test set has <2 classes — cannot evaluate on real data **")
        line(f"train-CV macro-F1 only: {f1_score(ytr, predtr, average='macro'):.3f}")
        return
    line(f"\nTEST macro-F1: {f1_score(yte, predte, average='macro'):.3f}")
    line(f"TEST accuracy: {(np.array(yte)==np.array(predte)).mean():.3f}")
    line("\nper-class (test):")
    line(classification_report(yte, predte, zero_division=0))


def main():
    print("loading embeddings model (first run downloads ~130MB)...")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")

    train, test = load("dataset_train.jsonl"), load("dataset_test.jsonl")

    def embed(rows):
        return enc.encode([emb_text(r) for r in rows],
                          normalize_embeddings=True, show_progress_bar=False)
    Xtr_all, Xte_all = embed(train), embed(test)

    heads = {}
    report = open("model1_report.txt", "w", encoding="utf-8")

    def train_head(name, label_fn, balanced=True):
        tr = [(x, label_fn(r)) for x, r in zip(Xtr_all, train) if label_fn(r) is not None]
        te = [(x, label_fn(r)) for x, r in zip(Xte_all, test) if label_fn(r) is not None]
        if len(tr) < 10 or len(set(y for _, y in tr)) < 2:
            print(f"\n[skip] {name}: too few examples/classes to train")
            return
        Xtr = np.array([x for x, _ in tr]); ytr = [y for _, y in tr]
        clf = LogisticRegression(max_iter=2000,
                                 class_weight="balanced" if balanced else None)
        clf.fit(Xtr, ytr)
        heads[name] = clf
        predtr = clf.predict(Xtr)
        if te:
            Xte = np.array([x for x, _ in te]); yte = [y for _, y in te]
            predte = clf.predict(Xte)
        else:
            yte, predte = [], []
        report_head(name, ytr, yte, predtr, predte, report)

    # head 1 — landed (headline: trained + tested)
    train_head("LANDED (did the move move him)", lambda r: r.get("landed"))
    # head 2 — disposition (proof-of-concept: trained + tested)
    train_head("DISPOSITION (taken/adapted/refused)",
               lambda r: r.get("disposition") if r.get("disposition") else None)
    # head 3 — coarse move type (macro-F1, imbalance-aware)
    train_head("MOVE (probe/attack/empathy)", lambda r: move_coarse(r.get("move")))
    # head 4 — wound (synthetic-only; honest caveat printed by report_head)
    train_head("WOUND_MARKED (synthetic-only eval)",
               lambda r: r.get("wound_marked") if r.get("wound_marked") is not None else None)

    # ---------- baseline: emotion classifier on the LANDED task ----------
    print("\nrunning emotion-classifier baseline (thesis: wound ≠ emotion)...")
    try:
        from transformers import pipeline
        emo = pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base",
                       top_k=None, truncation=True)
        EMOS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        def emo_vec(rows):
            out = []
            for r in rows:
                scores = {d["label"]: d["score"] for d in emo(r.get("text") or " ")[0]}
                out.append([scores.get(e, 0.0) for e in EMOS])
            return np.array(out)
        trl = [(v, r.get("landed")) for v, r in zip(emo_vec(train), train) if r.get("landed") is not None]
        tel = [(v, r.get("landed")) for v, r in zip(emo_vec(test), test) if r.get("landed") is not None]
        from sklearn.metrics import f1_score
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(np.array([v for v, _ in trl]), [y for _, y in trl])
        pred = clf.predict(np.array([v for v, _ in tel]))
        base_f1 = f1_score([y for _, y in tel], pred, average="macro")
        bge_f1 = None
        if "LANDED (did the move move him)" in heads:
            h = heads["LANDED (did the move move him)"]
            tel_b = [(x, r.get("landed")) for x, r in zip(Xte_all, test) if r.get("landed") is not None]
            bge_f1 = f1_score([y for _, y in tel_b],
                              h.predict(np.array([x for x, _ in tel_b])), average="macro")
        msg = (f"\n{'='*60}\nTHESIS COMPARISON (LANDED, test macro-F1)\n{'='*60}\n"
               f"  emotion-classifier features : {base_f1:.3f}\n"
               f"  bge-small (semantic)        : {bge_f1:.3f}\n"
               f"  -> {'bge wins: landed is not surface emotion' if (bge_f1 or 0) > base_f1 else 'inconclusive at this n — discuss in report'}\n")
        print(msg); report.write(msg)
    except Exception as e:
        print(f"baseline skipped: {e}")

    import joblib
    joblib.dump({"encoder": "BAAI/bge-small-en-v1.5", "heads": heads}, "model1_heads.joblib")
    report.close()
    print("\nsaved model1_heads.joblib and model1_report.txt")


if __name__ == "__main__":
    main()