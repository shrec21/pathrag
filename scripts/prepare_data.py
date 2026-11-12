"""Step 1: download PathVQA, group QA by image, write descriptions.

    python scripts/prepare_data.py --out data/pathvqa

Writes:
  images/<image_id>.png    each distinct image once
  qa.jsonl                 every QA row, image ids attached, cross-split rows dropped
  descriptions.jsonl       one pseudo-description per image
  split_report.json        counts and the images PathVQA puts in more than one split
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--dataset", default="flaviagiammarino/path-vqa")
    args = ap.parse_args()

    from datasets import load_dataset  # heavy import, only needed here

    ds = load_dataset(args.dataset)
    records = data.load_rows(data.iter_hf_rows(ds), image_dir=args.out / "images")
    report = data.split_report(records)
    kept = data.drop_cross_split(records)
    descs = data.build_descriptions(records)

    data.write_jsonl(args.out / "qa.jsonl", kept)
    data.write_jsonl(args.out / "descriptions.jsonl", descs.values())
    report["dropped_cross_split_qa"] = len(records) - len(kept)
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2))

    for split, c in report["counts"].items():
        print(f"{split:>10}: {c['images']:5d} images, {c['qa']:6d} QA ({c['closed']} yes/no)")
    print(f"images in more than one split: {len(report['shared_images'])}, "
          f"QA rows dropped: {report['dropped_cross_split_qa']}")


if __name__ == "__main__":
    main()
