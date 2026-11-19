"""Step 7: the knowledge base keeps changing while queries run.

    python scripts/churn.py --plot results/churn.png

Defaults: 60% of train images live at the start, then 10 rounds, each
retracting 4% of train images, re-ingesting 2% and annotating 4% new ones.
After 10 rounds about 40% of the starting library has been retracted or
replaced. No API calls. Backstitch kinds are skipped if it isn't built.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import churn, data, vecio  # noqa: E402

KINDS = "hnsw,backstitch:lazy_only,backstitch:inline,backstitch:background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--kinds", default=KINDS, help="flat always runs as the exact reference")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--init", type=float, default=0.6)
    ap.add_argument("--annotate", type=float, default=0.04)
    ap.add_argument("--retract", type=float, default=0.04)
    ap.add_argument("--reingest", type=float, default=0.02)
    ap.add_argument("--eval-images", type=int, default=300, help="test images queried every round")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2, help="Backstitch background repair threads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    kinds = args.kinds.split(",")
    if not importlib.util.find_spec("pathrag._backstitch"):
        dropped = [x for x in kinds if x.startswith("backstitch")]
        kinds = [x for x in kinds if not x.startswith("backstitch")]
        if dropped:
            print(f"backstitch backend not built, skipping {', '.join(dropped)}")

    vectors, ids = vecio.load(args.data / "emb/patches")
    owner = [r[0] for r in ids]
    split = {r[0]: r[1] for r in ids}
    patches_per: dict[str, list[int]] = {}
    for row, r in enumerate(ids):
        patches_per.setdefault(r[0], []).append(row)
    train = sorted(i for i, s in split.items() if s == "train")
    test = sorted(i for i, s in split.items() if s == "test")
    rng = np.random.default_rng(args.seed)
    query_images = sorted(rng.choice(test, min(args.eval_images, len(test)), replace=False).tolist())

    live, plan = churn.schedule(train, args.rounds, args.init, args.annotate, args.retract,
                                args.reingest, seed=args.seed)
    qa = [r for r in data.read_qa(args.data / "qa.jsonl") if r.split == "test"]
    descs = data.read_descriptions(args.data / "descriptions.jsonl")
    print(f"{len(train)} train images, {len(live)} live at start; per round retract "
          f"{len(plan[0].retract)}, reingest {len(plan[0].reingest)}, annotate {len(plan[0].annotate)} "
          f"(x{len(patches_per[train[0]])} patches each); {len(query_images)} query images")

    res = churn.simulate(kinds, vectors, owner, patches_per, live, plan, query_images, qa, descs, k=args.k,
                         index_kw={"backstitch": {"workers": args.workers}})
    res.update(vars(args) | {"data": str(args.data), "plot": str(args.plot), "out": str(args.out)})

    print(f"\n{'index':<22}{'build s':>8}{'min recall':>11}{'end recall':>11}{'end pres@5':>11}"
          f"{'del p50 us':>11}{'del p99 us':>11}{'ins p50 us':>11}{'query us':>9}{'flush s':>8}")
    for kind, r in res["results"].items():
        rs = r["rounds"]
        print(f"{kind:<22}{r['build_s']:>8.1f}{min(x['recall'] for x in rs):>11.4f}{rs[-1]['recall']:>11.4f}"
              f"{rs[-1]['presence']:>11.3f}{r['delete_us_p50']:>11.1f}{r['delete_us_p99']:>11.1f}"
              f"{r['insert_us_p50']:>11.1f}{np.mean([x['query_us'] for x in rs]):>9.1f}{r['final_flush_s']:>8.2f}")
    print("latencies are per patch vector and include Python call overhead (a few us)")

    if args.plot:
        plot(res, args.plot, args.k)
        print(f"plot saved to {args.plot}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"churn_{int(time.time())}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"saved {path}")


def plot(res: dict, path: Path, k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for kind, r in res["results"].items():
        rs = r["rounds"]
        x = [z["round"] for z in rs]
        ax[0, 0].plot(x, [z["recall"] for z in rs], marker="o", ms=3, label=kind)
        ax[0, 1].plot(x, [z["presence"] for z in rs], marker="o", ms=3, label=kind)
        ax[1, 0].plot(x, [z["query_us"] for z in rs], marker="o", ms=3, label=kind)
    names = [kn for kn in res["results"] if kn != "flat"]
    p50 = [res["results"][kn]["delete_us_p50"] for kn in names]
    p99 = [res["results"][kn]["delete_us_p99"] for kn in names]
    xs = np.arange(len(names))
    ax[1, 1].bar(xs - 0.2, p50, 0.4, label="p50")
    ax[1, 1].bar(xs + 0.2, p99, 0.4, label="p99")
    ax[1, 1].set_xticks(xs, [n.replace("backstitch:", "bs:") for n in names], rotation=15)
    ax[1, 1].set_yscale("log")
    for a, (t, yl) in zip(ax.flat, [(f"recall@{k} vs exact, live set", "recall"),
                                    ("answer presence@5 (patch voting)", "presence"),
                                    ("query latency per patch", "us"),
                                    ("delete latency per patch (all rounds)", "us, log scale")]):
        a.set(title=t, ylabel=yl)
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    for a in ax[0, 0], ax[0, 1], ax[1, 0]:
        a.set_xlabel("round")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
