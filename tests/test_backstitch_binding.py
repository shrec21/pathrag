import numpy as np
import pytest

bs = pytest.importorskip("pathrag._backstitch", reason="backstitch backend not built")

DIM = 64


def unit_rows(rng, n):
    x = rng.standard_normal((n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def exact_topk(base, labels, live, q, k):
    keep = np.flatnonzero(live)
    d = ((base[keep][None, :, :] - q[:, None, :]) ** 2).sum(-1)
    return labels[keep][np.argsort(d, axis=1)[:, :k]]


def recall(found, truth):
    return np.mean([len(set(f[f >= 0]) & set(t)) / len(t) for f, t in zip(found, truth)])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    base = unit_rows(rng, 2000)
    queries = unit_rows(rng, 50)
    return base, np.arange(2000, dtype=np.uint32), queries


def make(version, n, extra=500, workers=2):
    return bs.Index(dim=DIM, capacity=n + extra, label_capacity=n + extra, R=32, L_build=64,
                    version=version, workers=workers)


def test_static_recall(data):
    base, labels, q = data
    idx = make("background", len(base))
    idx.build(base, labels, threads=2)
    found = idx.search(q, 10, L=100)
    assert found.shape == (50, 10)
    assert recall(found, exact_topk(base, labels, np.ones(len(base), bool), q, 10)) > 0.95


@pytest.mark.parametrize("version", ["lazy_only", "inline", "background"])
def test_deletes_hidden_and_repaired(data, version):
    base, labels, q = data
    idx = make(version, len(base))
    idx.build(base, labels, threads=2)
    rng = np.random.default_rng(1)
    gone = rng.choice(len(base), 400, replace=False)
    for lab in gone:
        assert idx.remove(int(lab))
    assert not idx.remove(int(gone[0]))  # already deleted
    idx.flush()
    st = idx.stats()
    assert st["deletes"] == 400
    assert st["backlog"] == (400 if version == "lazy_only" else 0)

    live = np.ones(len(base), bool)
    live[gone] = False
    found = idx.search(q, 10, L=100)
    assert not set(found.ravel()) & set(gone.tolist())
    assert recall(found, exact_topk(base, labels, live, q, 10)) > 0.9


def test_replace_reuses_label(data):
    base, labels, _ = data
    idx = make("background", len(base))
    idx.build(base, labels, threads=2)
    new = base[7] * -1.0
    idx.remove(7)
    idx.insert(7, new)
    idx.flush()
    assert idx.is_live(7)
    assert idx.search(new[None, :], 1)[0, 0] == 7


def test_guards_raise_instead_of_abort(data):
    base, labels, _ = data
    idx = bs.Index(dim=DIM, capacity=len(base) + 1, label_capacity=len(base) + 1, R=32, L_build=64)
    with pytest.raises(RuntimeError):
        idx.search(base[:1], 1)  # not built
    idx.build(base, labels, threads=2)
    with pytest.raises(ValueError):
        idx.insert(3, base[3])  # already live
    with pytest.raises(ValueError):
        idx.insert(len(base) + 5, base[3])  # label overflow
    idx.insert(len(base), base[0])  # fills the last slot
    with pytest.raises(RuntimeError):
        idx.remove(1)
        idx.insert(1, base[1])  # deleted slots are not reused, so the index is full
    with pytest.raises(ValueError):
        idx.search(np.zeros((1, DIM + 1), np.float32), 1)


def test_duplicate_build_labels_rejected(data):
    base, _, _ = data
    idx = make("background", len(base))
    with pytest.raises(ValueError):
        idx.build(base[:10], np.zeros(10, np.uint32))
