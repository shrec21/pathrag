"""Step 7: a living knowledge base.

A hospital's reference library never sits still. New cases get annotated,
mislabeled ones get retracted, and old ones get re-ingested after review.
This simulates that on the patch index (the largest one) and asks: does
retrieval stay good while the index changes underneath it, and what does
each index pay for that?

Each round, in order:
  retract   delete every patch of some live images
  reingest  delete then re-add every patch of some live images (same label)
  annotate  add every patch of some new images from the unseen pool
Then, right away, with no flush (background repair may still be running),
the round's queries run: patches of a fixed sample of eval images.

Per index and round it records recall@10 against exact search over the
same live set, answer presence@5 via patch voting, per-patch op latency,
query latency, and, for Backstitch, the repair backlog.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pathrag.data import QA, Description
from pathrag.index import VectorIndex, make_index
from pathrag.metrics import presence_at_k


@dataclass
class Round:
    retract: list[str] = field(default_factory=list)
    reingest: list[str] = field(default_factory=list)
    annotate: list[str] = field(default_factory=list)


def schedule(train: list[str], rounds: int, init: float, annotate: float, retract: float,
             reingest: float, seed: int = 0) -> tuple[list[str], list[Round]]:
    """Initial live images and the per-round events. Fractions are of all
    train images, per round. Retracted images never come back."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(train))
    n = len(order)
    live = order[:int(init * n)]
    pool = order[int(init * n):]
    cur = list(live)
    plan = []
    for _ in range(rounds):
        r = Round()
        n_ret = min(int(retract * n), len(cur))
        idx = set(rng.choice(len(cur), n_ret, replace=False).tolist()) if n_ret else set()
        r.retract = [cur[i] for i in sorted(idx)]
        cur = [c for i, c in enumerate(cur) if i not in idx]
        n_re = min(int(reingest * n), len(cur))
        r.reingest = [cur[i] for i in sorted(rng.choice(len(cur), n_re, replace=False).tolist())] if n_re else []
        n_add = min(int(annotate * n), len(pool))
        r.annotate, pool = pool[:n_add], pool[n_add:]
        cur += r.annotate
        plan.append(r)
    return live, plan


def capacity_needed(live: list[str], plan: list[Round], patches_per: dict[str, list[int]]) -> int:
    """Slots a non-reusing index needs: every vector ever inserted."""
    total = sum(len(patches_per[i]) for i in live)
    for r in plan:
        total += sum(len(patches_per[i]) for i in r.reingest + r.annotate)
    return total


def vote(hits: np.ndarray, owner: list[str], slices: list[tuple[int, int]], k: int) -> list[list[str]]:
    """Patch hits -> top-k images per query image, each patch voting 1/rank."""
    out = []
    for a, b in slices:
        score = defaultdict(float)
        for row in hits[a:b]:
            for rank, h in enumerate(row):
                if h >= 0:
                    score[owner[h]] += 1.0 / (rank + 1)
        out.append(sorted(score, key=lambda i: -score[i])[:k])
    return out


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    vals = []
    for f, t in zip(found, truth):
        t = set(t[t >= 0].tolist())
        if t:
            vals.append(len(set(f[f >= 0].tolist()) & t) / len(t))
    return float(np.mean(vals)) if vals else float("nan")


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def simulate(kinds: list[str], vectors: np.ndarray, owner: list[str], patches_per: dict[str, list[int]],
             live: list[str], plan: list[Round], query_images: list[str], qa: list[QA],
             descs: dict[str, Description], k: int = 10, presence_k: int = 5,
             index_kw: dict | None = None, log=print) -> dict:
    """Runs the same schedule on every index kind. 'flat' is always run, as
    the exact reference the others are scored against."""
    index_kw = index_kw or {}
    kinds = ["flat"] + [x for x in kinds if x != "flat"]
    cap = capacity_needed(live, plan, patches_per)
    label_cap = len(vectors)

    qrows, slices = [], []
    for iid in query_images:
        rows = patches_per[iid]
        slices.append((len(qrows), len(qrows) + len(rows)))
        qrows += rows
    queries = vectors[qrows]
    qa_q = [r for r in qa if r.image_id in set(query_images)]

    results, reference = {}, {}
    for kind in kinds:
        kw = dict(index_kw.get(kind.split(":")[0], {}))
        idx: VectorIndex = make_index(kind, vectors.shape[1], capacity=max(cap, label_cap) + 1, **kw)
        init_rows = np.array([p for i in live for p in patches_per[i]], dtype=np.int64)
        t0 = time.perf_counter()
        idx.build(vectors[init_rows], init_rows)
        build_s = time.perf_counter() - t0
        n_live = len(live)
        rows_out, all_dels, all_adds = [], [], []
        for rn, r in enumerate([Round()] + plan):  # round 0: before any change
            if hasattr(idx, "next_round"):
                idx.next_round()
            dels, adds = [], []
            for iid in r.retract + r.reingest:
                for p in patches_per[iid]:
                    t = time.perf_counter()
                    idx.delete(p)
                    dels.append((time.perf_counter() - t) * 1e6)
            for iid in r.reingest + r.annotate:
                for p in patches_per[iid]:
                    t = time.perf_counter()
                    idx.add(p, vectors[p])
                    adds.append((time.perf_counter() - t) * 1e6)
            n_live += len(r.annotate) - len(r.retract)
            all_dels += dels
            all_adds += adds
            st = idx.stats()
            t = time.perf_counter()
            hits = idx.search(queries, k)
            q_us = (time.perf_counter() - t) * 1e6 / max(len(queries), 1)
            if kind == "flat":
                reference[rn] = hits
            ranked = vote(hits, owner, slices, presence_k)
            pres = presence_at_k(qa_q, dict(zip(query_images, ranked)), descs, [presence_k])[presence_k]
            rows_out.append({
                "round": rn, "live_images": n_live, "recall": recall(hits, reference[rn]),
                "presence": pres, "query_us": q_us,
                "delete_us_p50": pct(dels, 50), "delete_us_p99": pct(dels, 99),
                "insert_us_p50": pct(adds, 50), "insert_us_p99": pct(adds, 99),
                "deletes": len(dels), "inserts": len(adds), "backlog": st.get("backlog", 0),
                "unreachable_live": st.get("unreachable_live", 0),
            })
            log(f"  {kind:<22} round {rn:2d}: live {n_live:5d}  recall@{k} {rows_out[-1]['recall']:.4f}  "
                f"presence@{presence_k} {pres:.3f}  backlog {rows_out[-1]['backlog']}")
        t = time.perf_counter()
        idx.flush()
        results[kind] = {"build_s": build_s, "final_flush_s": time.perf_counter() - t,
                         "final_stats": idx.stats(), "rounds": rows_out,
                         "delete_us_p50": pct(all_dels, 50), "delete_us_p99": pct(all_dels, 99),
                         "insert_us_p50": pct(all_adds, 50), "insert_us_p99": pct(all_adds, 99)}
        del idx
    return {"capacity": cap, "query_images": len(query_images), "query_patches": len(queries),
            "open_questions": sum(not r.closed for r in qa_q), "results": results}
