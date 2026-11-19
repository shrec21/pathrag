"""Step 6: trust scores and abstention over answers step 5 already produced.

    python scripts/trust.py                       # reads the cached rag answers
    python scripts/trust.py --plot runs/trust.png

No API calls. Retrieval is rerun with the same mode, k and index as step 5
(deterministic), so each answer is scored against the neighbours it saw.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import data, trust  # noqa: E402
from pathrag.retrieve import KnowledgeBase  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--gen", type=Path, default=None,
                    help="rag answers jsonl (default: the newest runs/gen/*/rag-*.jsonl)")
    ap.add_argument("--mode", default="patch")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index", default="flat")
    ap.add_argument("--no-near-dups", action="store_true", help="drop eval images near a train image")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    gen = args.gen or max(Path("runs/gen").glob("*/rag-*.jsonl"), key=lambda p: p.stat().st_mtime)
    recs = [json.loads(l) for l in open(gen) if l.strip()]
    recs = [r for r in recs if r.get("pred", "").strip()]
    if args.no_near_dups:
        dups = set(json.loads((args.data / "emb/near_dups.json").read_text())["images"])
        recs = [r for r in recs if r["image_id"] not in dups]
    print(f"{len(recs)} answers from {gen}")

    descs = data.read_descriptions(args.data / "descriptions.jsonl")
    kb = KnowledgeBase(args.data / "emb", index_kind=args.index, modes=[args.mode])
    images = sorted({r["image_id"] for r in recs})
    neigh = dict(zip(images, kb.retrieve(args.mode, images, args.k)))

    sim = np.array([trust.similarity(kb.image_vec[r["image_id"]],
                                     np.stack([kb.image_vec[n] for n in neigh[r["image_id"]]]))
                    for r in recs])
    ok = np.array([trust.correct(r) for r in recs])
    closed = np.array([trust.is_closed(r["gold"]) for r in recs])

    results = {"gen": str(gen), "mode": args.mode, "k": args.k, "no_near_dups": args.no_near_dups}
    o = ~closed
    sup = np.array([trust.support(recs[i]["pred"], neigh[recs[i]["image_id"]], descs) for i in np.flatnonzero(o)])
    results["open"] = trust.evaluate({
        "sim": sim[o], "support": sup,
        "support+sim": trust.support_then_sim(sup, sim[o])}, ok[o])
    results["closed"] = trust.evaluate({"sim": sim[closed]}, ok[closed])

    for kind in ("open", "closed"):
        r = results[kind]
        print(f"\n{kind} questions: {r['n']}, accuracy when answering all {r['base_acc']:.3f}")
        print(f"  {'signal':<12}{'AURC':>7}" + "".join(f"{f'acc@{int(c*100)}%':>10}" for c in trust.COVERAGES))
        for name, row in r["signals"].items():
            print(f"  {name:<12}{row['aurc']:>7.3f}" +
                  "".join(f"{row[f'acc@{int(c*100)}']:>10.3f}" for c in trust.COVERAGES))
    print("\nacc@60% = accuracy when the system answers its 60% most trusted questions and abstains on the rest."
          "\nAURC: lower is better; random is the floor, oracle the ceiling.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, kind in zip(axes, ("open", "closed")):
            curves = results[kind]["curves"]
            for name, curve in curves.items():
                cov = np.arange(1, len(curve) + 1) / len(curve)
                style = {"random": dict(color="0.6", ls="--"), "oracle": dict(color="0.3", ls=":")}.get(name, {})
                ax.plot(cov, curve, label=name, **style)
            ax.set(title=f"{kind} questions (n={results[kind]['n']})", xlabel="coverage (fraction answered)",
                   ylabel="accuracy on answered", ylim=(0, 1.02))
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"plot saved to {args.plot}")

    for kind in ("open", "closed"):
        results[kind].pop("curves")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"trust_{gen.stem}{'_nodup' if args.no_near_dups else ''}_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
