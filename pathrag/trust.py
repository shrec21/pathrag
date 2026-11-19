"""Step 6: when should the system be believed?

Each answered question gets trust signals computed without the gold answer:

  sim      mean cosine between the query image and its k retrieved
           neighbours. Low means the knowledge base has no close cases.
  support  open questions only: fraction of the k neighbours whose answer
           slots contain the model's answer. Low means the answer is not
           backed by any reference, so the model likely made it up.
  support+sim  open questions: support first, sim only to break its ties.
           Support takes six values (0 to 5 of 5 neighbours), so many answers
           tie; sim orders them without overruling the stronger signal.

Ranking questions by a signal and answering only the top fraction gives a
risk-coverage curve. A useful signal makes accuracy climb as coverage drops;
random order keeps it flat; the oracle (correct answers first) is the best
any signal could do. AURC (area under the risk-coverage curve, lower is
better) sums that up in one number.
"""
from __future__ import annotations

import numpy as np

from pathrag import scoring
from pathrag.data import Description
from pathrag.metrics import answer_slots, norm, present

COVERAGES = (0.2, 0.4, 0.6, 0.8, 1.0)


def is_closed(gold: str) -> bool:
    return norm(gold) in ("yes", "no")


def correct(rec: dict) -> bool:
    if is_closed(rec["gold"]):
        return scoring.yes_no(rec["pred"]) == norm(rec["gold"])
    return scoring.contains(rec["pred"], rec["gold"])


def similarity(query: np.ndarray, neighbours: np.ndarray) -> float:
    """Mean cosine of unit vectors; 0 with no neighbours."""
    return float((neighbours @ query).mean()) if len(neighbours) else 0.0


def support(pred: str, neighbours: list[str], descs: dict[str, Description]) -> float:
    """Fraction of neighbours with the prediction in an answer slot."""
    if not neighbours or not norm(pred):
        return 0.0
    return sum(present(pred, answer_slots(descs[n])) for n in neighbours) / len(neighbours)


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks in [0, 1], ties sharing a rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average over ties so no hidden tie-break helps
        m = x == v
        r[m] = r[m].mean()
    return r / max(len(x) - 1, 1)


def risk_coverage(score: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Accuracy of the top-n by score, for n = 1..N, with ties averaged:
    every question in a tied block counts at the block's mean accuracy."""
    score, ok = np.asarray(score, float), np.asarray(ok, float)
    order = np.argsort(-score, kind="stable")
    s, c = score[order], ok[order].copy()
    start = 0
    while start < len(s):  # smooth correctness within each tie block
        end = start
        while end + 1 < len(s) and s[end + 1] == s[start]:
            end += 1
        c[start:end + 1] = c[start:end + 1].mean()
        start = end + 1
    return np.cumsum(c) / np.arange(1, len(c) + 1)


def aurc(acc_curve: np.ndarray) -> float:
    return float(np.mean(1.0 - acc_curve))


def at_coverage(acc_curve: np.ndarray, cov: float) -> float:
    n = max(1, int(round(cov * len(acc_curve))))
    return float(acc_curve[n - 1])


def evaluate(signals: dict[str, np.ndarray], ok: np.ndarray) -> dict:
    """Accuracy at fixed coverages and AURC for each signal, plus the random
    and oracle bounds."""
    ok = np.asarray(ok, float)
    rows = {}
    curves = {name: risk_coverage(s, ok) for name, s in signals.items()}
    curves["random"] = np.full(len(ok), ok.mean())
    curves["oracle"] = risk_coverage(ok, ok)
    for name, curve in curves.items():
        rows[name] = {"aurc": aurc(curve), **{f"acc@{int(c * 100)}": at_coverage(curve, c)
                                               for c in COVERAGES}}
    return {"n": len(ok), "base_acc": float(ok.mean()) if len(ok) else float("nan"),
            "signals": rows, "curves": {k: v.tolist() for k, v in curves.items()}}


def support_then_sim(sup: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Sorts by support, then sim within equal support. Support steps are
    1/k apart and the sim term stays below that, so it only breaks ties."""
    k_step = np.diff(np.unique(sup)).min() if len(np.unique(sup)) > 1 else 1.0
    return sup + 0.9 * k_step * ranks(sim)
