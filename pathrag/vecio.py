"""Vector files in the big-ann .fbin layout Backstitch reads:
uint32 rows, uint32 dim, then rows*dim float32, little-endian.
Each .fbin has a sidecar .ids.tsv naming what every row is."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x, dtype="<f4")
    if x.ndim != 2:
        raise ValueError("expected a 2-d array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.array(x.shape, dtype="<u4").tofile(f)
        x.tofile(f)


def read_fbin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype="<u4", count=2)
        x = np.fromfile(f, dtype="<f4", count=int(n) * int(d))
    if x.size != int(n) * int(d):
        raise ValueError(f"{path}: truncated file")
    return x.reshape(int(n), int(d))


def write_ids(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


def read_ids(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [line.rstrip("\n").split("\t") for line in f]


def save(prefix: Path, x: np.ndarray, ids: list[tuple]) -> None:
    if len(ids) != len(x):
        raise ValueError("one id row per vector")
    write_fbin(prefix.with_suffix(".fbin"), x)
    write_ids(prefix.with_suffix(".ids.tsv"), ids)


def load(prefix: Path) -> tuple[np.ndarray, list[list[str]]]:
    x = read_fbin(prefix.with_suffix(".fbin"))
    ids = read_ids(prefix.with_suffix(".ids.tsv"))
    if len(ids) != len(x):
        raise ValueError(f"{prefix}: {len(x)} vectors but {len(ids)} ids")
    return x, ids
