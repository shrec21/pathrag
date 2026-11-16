"""Step 4: answer presence at k, the first result, no LLM involved.

    python scripts/answer_presence.py --data data/pathvqa --index flat
    python scripts/answer_presence.py --data data/pathvqa --index backstitch:background

For each open-ended eval question, retrieve k train images by each mode and
check whether the gold answer appears in their descriptions. Reported for
all eval images and again without near-duplicates of train images.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import data, metrics  # noqa: E402
from pathrag.retrieve import MODES, KnowledgeBase  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--index", default="flat")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--ks", default="1,3,5,10,20")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    modes = args.modes.split(",")
    descs = data.read_descriptions(args.data / "descriptions.jsonl")
    qa = [r for r in data.read_qa(args.data / "qa.jsonl") if r.split == args.split]
    dups_file = args.data / "emb/near_dups.json"
    dups = set(json.loads(dups_file.read_text())["images"]) if dups_file.exists() else set()
    eval_images = sorted({r.image_id for r in qa})

    t0 = time.time()
    kb = KnowledgeBase(args.data / "emb", index_kind=args.index,
                       modes=[m for m in modes if m != "random"])
    print(f"built {args.index} indexes in {time.time() - t0:.1f}s; "
          f"{len(kb.train)} train images, {len(eval_images)} {args.split} images, "
          f"{sum(not r.closed for r in qa)} open-ended questions")

    results = {}
    for mode in modes:
        got = kb.retrieve(mode, eval_images, max(ks))
        retrieved = dict(zip(eval_images, got))
        results[mode] = {
            "all": metrics.presence_at_k(qa, retrieved, descs, ks),
            "no_near_dups": metrics.presence_at_k([r for r in qa if r.image_id not in dups],
                                                  retrieved, descs, ks),
        }

    head = "mode      subset        " + "".join(f"  @{k:<5}" for k in ks)
    print(head + "\n" + "-" * len(head))
    for mode, res in results.items():
        for subset, row in res.items():
            cells = ("    -   " if row[k] != row[k] else f"  {row[k]:.3f} " for k in ks)  # nan: no rows
            print(f"{mode:<9} {subset:<13} " + "".join(cells))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"presence_{args.split}_{args.index.replace(':', '-')}_{int(time.time())}.json"
    path.write_text(json.dumps({"index": args.index, "split": args.split, "ks": ks,
                                "near_dups_excluded": len(dups), "results": results}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
