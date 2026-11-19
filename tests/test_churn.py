import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pathrag import churn, data, synthetic

from conftest import HAS_BACKSTITCH

ROOT = Path(__file__).resolve().parents[1]


def test_schedule_never_reuses_retracted_images():
    train = [f"i{n}" for n in range(100)]
    live, plan = churn.schedule(train, rounds=5, init=0.5, annotate=0.05, retract=0.05, reingest=0.03, seed=1)
    assert len(live) == 50
    cur, gone = set(live), set()
    for r in plan:
        assert set(r.retract) <= cur and not set(r.annotate) & (cur | gone)
        cur -= set(r.retract)
        gone |= set(r.retract)
        assert set(r.reingest) <= cur
        cur |= set(r.annotate)
        assert len(r.retract) == 5 and len(r.reingest) == 3 and len(r.annotate) == 5
    seen = live + [i for r in plan for i in r.annotate]
    assert len(seen) == len(set(seen))


def test_capacity_counts_every_insert():
    per = {"a": [0, 1], "b": [2, 3], "c": [4, 5]}
    plan = [churn.Round(retract=["a"], reingest=["b"], annotate=["c"])]
    assert churn.capacity_needed(["a", "b"], plan, per) == 4 + 2 + 2


def test_vote_and_recall():
    owner = ["a", "a", "b", "c"]
    hits = np.array([[0, 2, -1], [2, 3, 1]])
    assert churn.vote(hits, owner, [(0, 2)], k=2) == [["b", "a"]]  # b: 1/2 + 1, a: 1 + 1/3
    assert churn.recall(np.array([[0, 2, -1]]), np.array([[0, 3, -1]])) == 0.5


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("churn")
    recs = data.load_rows(synthetic.rows(200, seed=7), image_dir=d / "images")
    data.write_jsonl(d / "qa.jsonl", data.drop_cross_split(recs))
    data.write_jsonl(d / "descriptions.jsonl", data.build_descriptions(recs).values())
    subprocess.run([sys.executable, str(ROOT / "scripts/embed.py"), "--data", str(d),
                    "--encoder", "hash", "--grid", "2"], check=True, capture_output=True)
    return d


def test_churn_script(synth, tmp_path):
    kinds = "hnsw,backstitch:inline,backstitch:background" if HAS_BACKSTITCH else "hnsw"
    out = subprocess.run([sys.executable, str(ROOT / "scripts/churn.py"), "--data", str(synth),
                          "--kinds", kinds, "--rounds", "3", "--eval-images", "15",
                          "--plot", str(tmp_path / "c.png"), "--out", str(tmp_path)],
                         check=True, capture_output=True, text=True).stdout
    assert (tmp_path / "c.png").exists()
    res = json.loads(next(tmp_path.glob("churn_*.json")).read_text())["results"]
    assert set(res) == {"flat", *kinds.split(",")}
    for kind, r in res.items():
        rs = r["rounds"]
        assert [x["round"] for x in rs] == [0, 1, 2, 3]
        assert rs[0]["deletes"] == 0 and rs[1]["deletes"] > 0 and rs[1]["inserts"] > 0
        assert all(x["recall"] > (0.999 if kind == "flat" else 0.8) for x in rs), (kind, out)
        assert rs[-1]["live_images"] == rs[0]["live_images"]  # retract and annotate balance
    if HAS_BACKSTITCH:
        assert res["backstitch:inline"]["final_stats"]["backlog"] == 0
        assert res["backstitch:background"]["final_stats"]["backlog"] == 0  # flushed at the end
