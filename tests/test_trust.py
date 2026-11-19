import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pathrag import data, synthetic, trust

ROOT = Path(__file__).resolve().parents[1]


def test_correct_by_question_type():
    assert trust.correct({"gold": "yes", "pred": "Yes."})
    assert not trust.correct({"gold": "no", "pred": "yes"})
    assert trust.correct({"gold": "granuloma", "pred": "caseating granuloma"})
    assert not trust.correct({"gold": "granuloma", "pred": "fibrosis"})


def test_support_counts_neighbours_backing_the_answer():
    descs = {"a": data.Description("a", "train", ["What is present: granuloma."]),
             "b": data.Description("b", "train", ["Is this granuloma: no.", "What organ: lung."]),
             "c": data.Description("c", "train", ["What is present: caseating granuloma."])}
    assert trust.support("granuloma", ["a", "b", "c"], descs) == pytest.approx(2 / 3)
    assert trust.support("", ["a"], descs) == 0.0


def test_risk_coverage_orders_and_averages_ties():
    ok = np.array([1, 0, 1, 0])
    curve = trust.risk_coverage(np.array([0.9, 0.1, 0.8, 0.2]), ok)
    assert curve.tolist() == [1.0, 1.0, pytest.approx(2 / 3), 0.5]
    flat = trust.risk_coverage(np.zeros(4), ok)  # all tied: no free ordering
    assert np.allclose(flat, 0.5)


def test_ranks_share_ties():
    assert trust.ranks(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [1.0, pytest.approx(1 / 6),
                                                                    pytest.approx(1 / 6), pytest.approx(2 / 3)]


def test_evaluate_bounds():
    rng = np.random.default_rng(0)
    ok = rng.random(500) < 0.4
    good = ok + rng.normal(0, 0.5, 500)
    r = trust.evaluate({"good": good}, ok)["signals"]
    assert r["oracle"]["aurc"] < r["good"]["aurc"] < r["random"]["aurc"]
    assert r["good"]["acc@20"] > r["good"]["acc@100"] == pytest.approx(0.4, abs=0.05)


def test_trust_script_end_to_end(tmp_path):
    d = tmp_path / "data"
    recs = data.load_rows(synthetic.rows(120, seed=5), image_dir=d / "images")
    data.write_jsonl(d / "qa.jsonl", data.drop_cross_split(recs))
    data.write_jsonl(d / "descriptions.jsonl", data.build_descriptions(recs).values())
    subprocess.run([sys.executable, str(ROOT / "scripts/embed.py"), "--data", str(d),
                    "--encoder", "hash", "--grid", "2"], check=True, capture_output=True)
    test = [r for r in data.read_qa(d / "qa.jsonl") if r.split == "test"]
    gen = tmp_path / "gen" / "fake" / "rag-test-patch-k5-flat.jsonl"
    gen.parent.mkdir(parents=True)
    with open(gen, "w") as f:  # a fake model that is right on every other question
        for n, r in enumerate(test):
            pred = r.answer if n % 2 == 0 else "unknown"
            f.write(json.dumps({"key": str(n), "image_id": r.image_id, "question": r.question,
                                "gold": r.answer, "pred": pred, "usage": {}}) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/trust.py"), "--data", str(d), "--gen", str(gen),
                          "--plot", str(tmp_path / "t.png"), "--out", str(tmp_path / "runs")],
                         check=True, capture_output=True, text=True).stdout
    assert "open questions" in out and "closed questions" in out and (tmp_path / "t.png").exists()
    res = json.loads(next((tmp_path / "runs").glob("trust_*.json")).read_text())
    sig = res["open"]["signals"]
    assert sig["support"]["aurc"] < sig["random"]["aurc"]  # "unknown" is never backed by a neighbour


def test_support_then_sim_only_breaks_ties():
    sup = np.array([0.2, 0.2, 0.0, 0.4])
    sim = np.array([0.1, 0.9, 1.0, 0.0])
    order = np.argsort(-trust.support_then_sim(sup, sim)).tolist()
    assert order == [3, 1, 0, 2]  # support decides, sim orders the two 0.2s
