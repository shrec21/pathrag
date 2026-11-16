import subprocess
import sys
from pathlib import Path

import pytest

from pathrag import data, metrics, synthetic
from pathrag.retrieve import KnowledgeBase

from conftest import kinds

ROOT = Path(__file__).resolve().parents[1]


def test_answer_slots_ignore_question_words():
    d = data.Description("a", "train", ["Is this the liver: no.", "What is present: hyaline necrosis."])
    slots = metrics.answer_slots(d)
    assert slots == ["no", "hyaline necrosis"]
    assert not metrics.present("liver", slots)
    assert metrics.present("necrosis", slots)
    assert metrics.present("Hyaline necrosis!", slots)
    assert not metrics.present("necro", slots)


def test_presence_counts_open_questions_only():
    descs = {"t1": data.Description("t1", "train", ["What is present: fibrosis."]),
             "t2": data.Description("t2", "train", ["What is present: granuloma."])}
    qa = [data.QA("e1", "test", "What is present?", "granuloma"),
          data.QA("e1", "test", "Is this lung?", "yes")]
    out = metrics.presence_at_k(qa, {"e1": ["t1", "t2"]}, descs, [1, 2])
    assert out == {1: 0.0, 2: 1.0}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    recs = data.load_rows(synthetic.rows(150, seed=1), image_dir=d / "images")
    data.write_jsonl(d / "qa.jsonl", data.drop_cross_split(recs))
    data.write_jsonl(d / "descriptions.jsonl", data.build_descriptions(recs).values())
    subprocess.run([sys.executable, str(ROOT / "scripts/embed.py"), "--data", str(d),
                    "--encoder", "hash", "--grid", "2"], check=True, capture_output=True)
    return d


@pytest.mark.parametrize("kind", kinds("flat", "hnsw", "backstitch:background"))
def test_retrieval_returns_only_train_images(synth, kind):
    kw = {"R": 16, "L_build": 32} if kind.startswith("backstitch") else {}
    kb = KnowledgeBase(synth / "emb", index_kind=kind, **kw)
    evals = [i for i, s in kb.split.items() if s != "train"][:10]
    for mode in ("image", "patch", "text", "random"):
        for row in kb.retrieve(mode, evals, 5):
            assert 0 < len(row) <= 5 and all(kb.split[i] == "train" for i in row)
            assert len(set(row)) == len(row)


def test_presence_script_beats_random_on_synthetic(synth, tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "scripts/answer_presence.py"),
                          "--data", str(synth), "--modes", "image,random", "--ks", "1,5",
                          "--out", str(tmp_path)], check=True, capture_output=True, text=True).stdout
    import json
    res = json.loads(next(tmp_path.glob("presence_*.json")).read_text())["results"]
    assert res["image"]["all"]["1"] > res["random"]["all"]["1"] + 0.3, out
