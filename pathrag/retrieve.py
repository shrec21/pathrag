"""Retrieval over the training knowledge base, by image, patch or text.

Every mode answers the same question: given an eval image, which train
images (and so which descriptions) come back? Only train images are ever
indexed, so eval answers can't leak in.

  image   eval image vector  -> train image vectors
  patch   eval image patches -> train patch vectors, votes summed per image
  text    eval image vector  -> train description vectors (PLIP is cross-modal)
  random  uniform train images, the floor every other mode must beat
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from pathrag import vecio
from pathrag.index import VectorIndex, make_index

MODES = ("image", "patch", "text", "random")


class KnowledgeBase:
    def __init__(self, emb_dir: Path, index_kind: str = "flat", headroom: int = 0,
                 modes=("image", "patch", "text"), **index_kw):
        self.emb_dir = Path(emb_dir)
        imgs, ids = vecio.load(self.emb_dir / "images")
        self.image_vec = {r[0]: v for r, v in zip(ids, imgs)}
        self.split = {r[0]: r[1] for r in ids}
        self.train = [r[0] for r in ids if r[1] == "train"]
        self.indexes: dict[str, VectorIndex] = {}
        self.owner: dict[str, list[str]] = {}
        self.patch_rows: dict[str, list[int]] = defaultdict(list)
        self.patch_vecs = None

        if "image" in modes:
            self._index("image", imgs, ids, index_kind, headroom, index_kw)
        if "text" in modes:
            dv, dids = vecio.load(self.emb_dir / "descriptions")
            self._index("text", dv, dids, index_kind, headroom, index_kw)
        if "patch" in modes and (self.emb_dir / "patches.fbin").exists():
            pv, pids = vecio.load(self.emb_dir / "patches")
            self.patch_vecs = pv
            for row, r in enumerate(pids):
                self.patch_rows[r[0]].append(row)
            self._index("patch", pv, pids, index_kind, headroom, index_kw)

    def _index(self, mode, vecs, ids, kind, headroom, kw):
        rows = np.array([i for i, r in enumerate(ids) if r[1] == "train"], dtype=np.int64)
        idx = make_index(kind, vecs.shape[1], capacity=len(vecs) + headroom, **kw)
        idx.build(vecs[rows], rows)
        self.indexes[mode] = idx
        self.owner[mode] = [r[0] for r in ids]  # row -> image id

    def retrieve(self, mode: str, image_ids: list[str], k: int, seed: int = 0) -> list[list[str]]:
        """Top-k distinct train image ids per query image."""
        if mode == "random":
            rng = np.random.default_rng(seed)
            return [list(rng.choice(self.train, size=min(k, len(self.train)), replace=False))
                    for _ in image_ids]
        if mode == "patch":
            return self._retrieve_patch(image_ids, k)
        q = np.stack([self.image_vec[i] for i in image_ids])
        hits = self.indexes[mode].search(q, k)
        own = self.owner[mode]
        return [[own[h] for h in row if h >= 0] for row in hits]

    def _retrieve_patch(self, image_ids, k, per_patch: int = 10):
        """Each query patch votes for the images its nearest train patches
        belong to, weighted 1/rank; images are ranked by total vote."""
        own = self.owner["patch"]
        out = []
        for iid in image_ids:
            rows = self.patch_rows.get(iid, [])
            if not rows:
                out.append([])
                continue
            hits = self.indexes["patch"].search(self.patch_vecs[rows], per_patch)
            score: dict[str, float] = defaultdict(float)
            for row in hits:
                for rank, h in enumerate(row):
                    if h >= 0:
                        score[own[h]] += 1.0 / (rank + 1)
            out.append(sorted(score, key=lambda i: -score[i])[:k])
        return out
