"""Scores model answers against PathVQA gold answers.

Yes/no questions (by gold answer): accuracy on the first yes/no word of the
prediction. Open-ended questions: exact match after normalising, token F1,
and "contains" (the gold phrase appears as whole words in the prediction,
which forgives "hyaline arteriolosclerosis of the kidney" for
"hyaline arteriolosclerosis").
"""
from __future__ import annotations

import re
from collections import Counter

from pathrag.metrics import norm

_YN = re.compile(r"\b(yes|no)\b")
_ARTICLES = {"a", "an", "the"}


def _tokens(s: str) -> list[str]:
    return [t for t in norm(s).split() if t not in _ARTICLES]


def yes_no(pred: str) -> str | None:
    m = _YN.search(norm(pred))
    return m.group(1) if m else None


def exact(pred: str, gold: str) -> bool:
    return _tokens(pred) == _tokens(gold)


def contains(pred: str, gold: str) -> bool:
    g = " ".join(_tokens(gold))
    return bool(g) and re.search(rf"(^| ){re.escape(g)}( |$)", " ".join(_tokens(pred))) is not None


def f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(records: list[dict]) -> dict:
    closed = [r for r in records if norm(r["gold"]) in ("yes", "no")]
    open_ = [r for r in records if norm(r["gold"]) not in ("yes", "no")]

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_closed": len(closed),
        "n_open": len(open_),
        "closed_acc": mean(yes_no(r["pred"]) == norm(r["gold"]) for r in closed),
        "open_exact": mean(exact(r["pred"], r["gold"]) for r in open_),
        "open_contains": mean(contains(r["pred"], r["gold"]) for r in open_),
        "open_f1": mean(f1(r["pred"], r["gold"]) for r in open_),
        "tokens_in": sum(r.get("usage", {}).get("in", 0) for r in records),
        "tokens_out": sum(r.get("usage", {}).get("out", 0) for r in records),
    }
