"""Answer presence: can retrieval surface the gold answer at all?

Checked before any generator runs, so it isolates retrieval from
generation. Only the answer half of each retrieved fact counts: a
neighbour's "is this the liver: no" must not get credit for "liver".
Yes/no questions are excluded, since nearly every description holds a
"yes" and a "no".
"""
from __future__ import annotations

import re

from pathrag.data import QA, Description

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def answer_slots(desc: Description) -> list[str]:
    """The answer part of every fact ('question: answer.')."""
    return [norm(f.rsplit(":", 1)[1]) for f in desc.facts if ":" in f]


def present(gold: str, slots: list[str]) -> bool:
    """Gold matches a slot exactly or as a whole-word phrase inside it."""
    g = norm(gold)
    if not g:
        return False
    pat = re.compile(rf"(^| ){re.escape(g)}( |$)")
    return any(s == g or pat.search(s) for s in slots)


def presence_at_k(qa: list[QA], retrieved: dict[str, list[str]],
                  descs: dict[str, Description], ks: list[int]) -> dict[int, float]:
    """Fraction of open-ended eval questions whose gold answer appears in the
    answer slots of the top-k retrieved descriptions, for each k."""
    open_qa = [r for r in qa if not r.closed]
    if not open_qa:
        return {k: float("nan") for k in ks}
    slots = {iid: answer_slots(d) for iid, d in descs.items()}
    out = {}
    for k in ks:
        hit = 0
        for r in open_qa:
            pool = [s for iid in retrieved.get(r.image_id, [])[:k] for s in slots.get(iid, [])]
            hit += present(r.answer, pool)
        out[k] = hit / len(open_qa)
    return out
