"""Step 5: answer eval questions with an API vision model, with and without RAG.

Check the prompt and the request count first (free, no API calls):
    python scripts/generate.py --limit 200 --dry-run

Small paid pilot, then the full run (both resume from the cache):
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY with --provider openai
    python scripts/generate.py --limit 200
    python scripts/generate.py

The same questions go through every condition, so results compare row by
row. Answers are cached under runs/gen/, so rerunning only pays for what is
missing; raising --limit later reuses everything already answered.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import data, generate, scoring  # noqa: E402
from pathrag.retrieve import KnowledgeBase  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None, help="default: claude-haiku-5-5 or gpt-5-mini")
    ap.add_argument("--conditions", default=",".join(generate.CONDITIONS))
    ap.add_argument("--mode", default="patch", choices=["image", "patch", "text"],
                    help="retrieval mode for rag and refs_only")
    ap.add_argument("--index", default="flat")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="random sample of N questions, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=generate.DEFAULT_MAX_TOKENS,
                    help="output budget per answer, reasoning included")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/gen"))
    args = ap.parse_args()

    model = args.model or {"anthropic": "claude-haiku-5-5", "openai": "gpt-5-mini"}[args.provider]
    conditions = args.conditions.split(",")
    for c in conditions:
        if c not in generate.CONDITIONS:
            ap.error(f"unknown condition {c}")

    descs = data.read_descriptions(args.data / "descriptions.jsonl")
    rows = [(n, r) for n, r in enumerate(data.read_qa(args.data / "qa.jsonl")) if r.split == args.split]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    images = sorted({r.image_id for _, r in rows})
    dups_file = args.data / "emb/near_dups.json"
    dups = set(json.loads(dups_file.read_text())["images"]) if dups_file.exists() else set()

    kb = KnowledgeBase(args.data / "emb", index_kind=args.index, modes=[args.mode])
    neigh = dict(zip(images, kb.retrieve(args.mode, images, args.k)))
    rand = dict(zip(images, kb.retrieve("random", images, args.k, seed=args.seed)))
    png = {i: generate.load_png(args.data / "images" / f"{i}.png") for i in images}

    jobs = {}
    for c in conditions:
        src = rand if c == "rag_random" else neigh
        jobs[c] = [generate.Job(n, r, generate.build_prompt(r, c, src[r.image_id], descs, png[r.image_id]))
                   for n, r in rows]

    n_req = sum(len(j) for j in jobs.values())
    n_closed = sum(r.answer.lower() in ("yes", "no") for _, r in rows)
    print(f"{len(rows)} {args.split} questions ({n_closed} yes/no) on {len(images)} images; "
          f"{len(conditions)} conditions = {n_req} requests to {model}")
    if args.dry_run:
        sample = jobs.get("rag", next(iter(jobs.values())))[0]
        chars = sum(len(j.prompt.text) for js in jobs.values() for j in js)
        imgs = sum(j.prompt.image_png is not None for js in jobs.values() for j in js)
        print(f"about {chars // 4:,} text tokens plus {imgs:,} images (~512px each) in total\n")
        print("---- sample prompt ----\n" + generate.SYSTEM + "\n\n" + sample.prompt.text)
        print(f"---- gold: {sample.qa.answer}")
        return

    client = generate.make_client(args.provider, model, args.max_tokens)
    tag = f"{args.split}-{args.mode}-k{args.k}-{args.index.replace(':', '-')}"
    summary = {"max_tokens": args.max_tokens, "model": model, "provider": args.provider, "split": args.split, "mode": args.mode,
               "k": args.k, "index": args.index, "limit": args.limit, "seed": args.seed, "results": {}}
    for c in conditions:
        cache = generate.Cache(args.out / model / f"{c}-{tag}.jsonl")
        t0 = time.time()

        def progress(done, total, c=c):
            if done == total or done % 100 == 0:
                print(f"  {c}: {done}/{total}", flush=True)

        recs, failed = generate.run(jobs[c], client, cache, workers=args.workers, progress=progress)
        if failed:
            print(f"  {c}: {failed} requests failed; rerun to retry them (if they came back empty, "
                  f"raise --max-tokens)")
        summary["results"][c] = {
            "all": scoring.score(recs),
            "no_near_dups": scoring.score([r for r in recs if r["image_id"] not in dups]),
            "missing": len(jobs[c]) - len(recs),
        }
        print(f"  {c}: done in {time.time() - t0:.0f}s")

    print(f"\n{'condition':<12}{'subset':<14}{'yes/no acc':>11}{'open exact':>11}"
          f"{'open contains':>14}{'open F1':>9}")
    for c, res in summary["results"].items():
        for subset in ("all", "no_near_dups"):
            s = res[subset]
            print(f"{c:<12}{subset:<14}{s['closed_acc']:>11.3f}{s['open_exact']:>11.3f}"
                  f"{s['open_contains']:>14.3f}{s['open_f1']:>9.3f}")
    tin = sum(r["all"]["tokens_in"] for r in summary["results"].values())
    tout = sum(r["all"]["tokens_out"] for r in summary["results"].values())
    n_ans = sum(r["all"]["n_closed"] + r["all"]["n_open"] for r in summary["results"].values())
    print(f"\ntokens used (cached answers included): {tin:,} in, {tout:,} out, "
          f"{tout / max(n_ans, 1):.0f} out per answer")

    path = args.out / f"summary_{model}_{tag}_n{len(rows)}_{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
