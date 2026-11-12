import numpy as np
import pytest

from pathrag.embed import normalize
from pathrag.index import make_index

from conftest import kinds

DIM, N = 48, 1500
KINDS = kinds("flat", "hnsw", "backstitch:lazy_only", "backstitch:inline", "backstitch:background")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return normalize(rng.standard_normal((N, DIM))), normalize(rng.standard_normal((40, DIM)))


def build(kind, base):
    kw = {"R": 32, "L_build": 64} if kind.startswith("backstitch") else {}
    idx = make_index(kind, DIM, capacity=N + 200, **kw)
    idx.build(base, np.arange(N))
    return idx


def overlap(a, b):
    return np.mean([len(set(x[x >= 0]) & set(y[y >= 0])) / len(y[y >= 0]) for x, y in zip(a, b)])


@pytest.mark.parametrize("kind", KINDS)
def test_recall_against_exact(data, kind):
    base, q = data
    exact = build("flat", base).search(q, 10)
    assert overlap(build(kind, base).search(q, 10), exact) > 0.9


@pytest.mark.parametrize("kind", KINDS)
def test_delete_and_replace(data, kind):
    base, q = data
    idx = build(kind, base)
    exact = build("flat", base)
    gone = np.random.default_rng(4).choice(N, 300, replace=False)
    for lab in gone:
        assert idx.delete(int(lab)) and exact.delete(int(lab))
    assert not idx.delete(int(gone[0]))
    idx.flush()
    got = idx.search(q, 10)
    assert not set(got.ravel().tolist()) & set(gone.tolist())
    assert overlap(got, exact.search(q, 10)) > 0.85

    target = q[0]  # replace label 5 with a vector sitting on query 0
    idx.delete(5)
    idx.add(5, target)
    idx.flush()
    assert idx.search(target, 1)[0, 0] == 5
    with pytest.raises((ValueError, RuntimeError)):
        idx.add(5, target)  # already live


def test_short_results_pad_with_minus_one():
    idx = make_index("flat", 4)
    idx.build(normalize(np.eye(4)[:2]), np.array([10, 11]))
    assert idx.search(np.eye(4)[0], 5)[0].tolist() == [10, 11, -1, -1, -1]


def test_backstitch_needs_capacity():
    with pytest.raises(ValueError):
        make_index("backstitch", 8)
