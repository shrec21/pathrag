"""One interface over three vector indexes, so retrieval code never cares
which one it runs on.

  flat        exact search (FAISS IndexFlatL2): ground truth for recall
  hnsw        FAISS HNSW, the standard baseline; deletes are filtered at
              query time and never repaired, which is how FAISS HNSW works
  backstitch  the team's graph index; deletes are tombstoned and repaired
              inline or in background workers, depending on `version`

Labels are the caller's integer ids (row numbers of an .fbin). Deleting a
label and adding it again is a replace. All three take unit vectors, where
L2 order equals cosine order.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

F32 = np.float32


def _rows(x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=F32)
    return x[None, :] if x.ndim == 1 else x


class VectorIndex(ABC):
    name = "base"

    @abstractmethod
    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None: ...

    @abstractmethod
    def add(self, label: int, vector: np.ndarray) -> None: ...

    @abstractmethod
    def delete(self, label: int) -> bool: ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """(m, k) int64 labels, nearest first, -1 where fewer than k exist."""

    def flush(self) -> None:
        """Waits for any background work. Only Backstitch has some."""

    def stats(self) -> dict:
        return {"index": self.name}


class _FaissTombstoned(VectorIndex):
    """FAISS index over internal sequential ids, with a label map and a
    deleted set applied as a search-time filter."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = self._make()
        self.label_of: list[int] = []
        self.slot_of: dict[int, int] = {}
        self.dead: set[int] = set()

    def _make(self):
        raise NotImplementedError

    def _params(self):
        import faiss
        p = self._search_params()
        if self.dead:
            self._dead_arr = np.fromiter(self.dead, dtype=np.int64)
            self._sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(self._dead_arr))
            p.sel = self._sel
        return p

    def _search_params(self):
        import faiss
        return faiss.SearchParameters()

    def build(self, vectors, labels):
        if self.label_of:
            raise RuntimeError("build() was already called")
        self._append(_rows(vectors), np.asarray(labels))

    def _append(self, x, labels):
        for lab in labels.tolist():
            if lab in self.slot_of:
                raise ValueError(f"label {lab} is already live")
        start = len(self.label_of)
        self.index.add(x)
        for i, lab in enumerate(labels.tolist()):
            self.slot_of[lab] = start + i
            self.label_of.append(lab)

    def add(self, label, vector):
        self._append(_rows(vector), np.array([label]))

    def delete(self, label):
        slot = self.slot_of.pop(int(label), None)
        if slot is None:
            return False
        self.dead.add(slot)
        return True

    def search(self, queries, k):
        q = _rows(queries)
        _, ids = self.index.search(q, k, params=self._params())
        lut = np.asarray(self.label_of + [-1], dtype=np.int64)
        return lut[np.where(ids < 0, len(self.label_of), ids)]

    def stats(self):
        return {"index": self.name, "live": len(self.slot_of), "tombstones": len(self.dead)}


class FlatIndex(_FaissTombstoned):
    name = "flat"

    def _make(self):
        import faiss
        return faiss.IndexFlatL2(self.dim)


class HNSWIndex(_FaissTombstoned):
    name = "hnsw"

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 100, ef_search: int = 100):
        self.M, self.efc, self.efs = M, ef_construction, ef_search
        super().__init__(dim)

    def _make(self):
        import faiss
        idx = faiss.IndexHNSWFlat(self.dim, self.M)
        idx.hnsw.efConstruction = self.efc
        return idx

    def _search_params(self):
        import faiss
        return faiss.SearchParametersHNSW(efSearch=self.efs)


class BackstitchIndex(VectorIndex):
    name = "backstitch"

    def __init__(self, dim: int, capacity: int, *, version: str = "background", workers: int = 2,
                 R: int = 64, L_build: int = 100, L_search: int = 100, alpha: float = 1.2,
                 trigger: str = "every_delete", policy: str = "routed", threads: int = 0):
        try:
            from pathrag import _backstitch
        except ImportError as e:
            raise ImportError("backstitch backend not built: set BACKSTITCH_DIR to your Backstitch "
                              "checkout and run `python setup.py build_ext --inplace`") from e

        self.L_search = L_search
        self.threads = threads or os.cpu_count() or 1  # build is the slow part
        self.version = version
        self.index = _backstitch.Index(dim=dim, capacity=capacity, label_capacity=capacity, R=R,
                                       L_build=L_build, alpha=alpha, version=version,
                                       trigger=trigger, policy=policy, workers=workers)

    def build(self, vectors, labels):
        self.index.build(_rows(vectors), np.asarray(labels, dtype=np.uint32), threads=self.threads)

    def add(self, label, vector):
        self.index.insert(int(label), np.ascontiguousarray(vector, dtype=F32).ravel())

    def delete(self, label):
        return self.index.remove(int(label))

    def search(self, queries, k):
        return self.index.search(_rows(queries), k, L=max(self.L_search, k))

    def flush(self):
        self.index.flush()

    def next_round(self):
        self.index.next_round()

    def stats(self):
        return {"index": self.name, **self.index.stats()}


def make_index(kind: str, dim: int, capacity: int = 0, **kw) -> VectorIndex:
    """kind: flat | hnsw | backstitch | backstitch:<version> (lazy_only,
    inline, background). capacity bounds Backstitch, which never reuses the
    slot of a deleted vector: size it for every insert of the run."""
    if kind == "flat":
        return FlatIndex(dim)
    if kind == "hnsw":
        return HNSWIndex(dim, **kw)
    if kind.startswith("backstitch"):
        if capacity <= 0:
            raise ValueError("backstitch needs a capacity")
        _, _, version = kind.partition(":")
        return BackstitchIndex(dim, capacity, version=version or "background", **kw)
    raise ValueError(f"unknown index kind: {kind}")
