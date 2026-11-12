"""Step 2: PLIP embeddings for everything later steps retrieve over.

    python scripts/embed.py --data data/pathvqa            # real PLIP (GPU, MPS or CPU)
    python scripts/embed.py --data data/pathvqa --encoder hash   # offline smoke test only

Writes under <data>/emb/, each as .fbin plus .ids.tsv:
  images         one vector per image          ids: image_id, split
  patches        grid x grid tiles per image   ids: image_id, split, patch
  descriptions   mean of per-fact text vectors ids: image_id, split
  questions      eval questions (val + test)   ids: qa_row, image_id, split
and near_dups.json: eval images whose closest train image is above the
threshold, so results can be reported with and without them.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathrag import data, embed, vecio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pathvqa"))
    ap.add_argument("--encoder", choices=["plip", "hash"], default="plip")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=4, help="patches per side; 0 skips patches")
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    args = ap.parse_args()

    enc = (embed.PLIPEncoder(device=args.device, batch=args.batch) if args.encoder == "plip"
           else embed.HashEncoder())
    out = args.data / "emb"
    descs = data.read_descriptions(args.data / "descriptions.jsonl")
    qa = data.read_qa(args.data / "qa.jsonl")
    order = sorted(descs, key=lambda i: (descs[i].split, i))
    split = {i: descs[i].split for i in order}

    def load(iid):
        return Image.open(args.data / "images" / f"{iid}.png").convert("RGB")

    img_vecs, patch_vecs, patch_ids = [], [], []
    for start in range(0, len(order), 256):  # chunks keep memory flat
        chunk = order[start:start + 256]
        imgs = [load(i) for i in chunk]
        img_vecs.append(enc.images(imgs))
        if args.grid:
            tiles, owners = [], []
            for iid, im in zip(chunk, imgs):
                ps = embed.patches(im, args.grid)
                tiles += ps
                owners += [(iid, split[iid], p) for p in range(len(ps))]
            patch_vecs.append(enc.images(tiles))
            patch_ids += owners
        print(f"images {min(start + 256, len(order))}/{len(order)}", flush=True)

    images = np.concatenate(img_vecs)
    vecio.save(out / "images", images, [(i, split[i]) for i in order])
    if args.grid:
        vecio.save(out / "patches", np.concatenate(patch_vecs), patch_ids)

    vecio.save(out / "descriptions", embed.embed_descriptions(enc, [descs[i].facts for i in order]),
               [(i, split[i]) for i in order])

    eval_rows = [(n, r) for n, r in enumerate(qa) if r.split != "train"]
    vecio.save(out / "questions", enc.texts([r.question for _, r in eval_rows]),
               [(n, r.image_id, r.split) for n, r in eval_rows])

    is_train = np.array([split[i] == "train" for i in order])
    eval_ids = [i for i in order if split[i] != "train"]
    mask, best = embed.near_duplicates(images[~is_train], images[is_train], args.dup_threshold)
    dups = {iid: float(s) for iid, s, m in zip(eval_ids, best, mask) if m}
    (out / "near_dups.json").write_text(json.dumps(
        {"threshold": args.dup_threshold, "encoder": args.encoder, "images": dups}, indent=2))
    print(f"{len(order)} images, {len(patch_ids)} patches, {len(eval_rows)} eval questions, "
          f"{len(dups)} eval images near-duplicate a train image")


if __name__ == "__main__":
    main()
