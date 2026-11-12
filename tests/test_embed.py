import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pathrag import data, embed, synthetic, vecio

ROOT = Path(__file__).resolve().parents[1]


def test_fbin_round_trip_matches_backstitch_layout(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    vecio.write_fbin(tmp_path / "a.fbin", x)
    raw = (tmp_path / "a.fbin").read_bytes()
    assert np.frombuffer(raw[:8], "<u4").tolist() == [3, 4]
    assert len(raw) == 8 + 12 * 4
    assert np.array_equal(vecio.read_fbin(tmp_path / "a.fbin"), x)


def test_save_requires_one_id_per_row(tmp_path):
    vecio.save(tmp_path / "v", np.zeros((2, 3)), [("a", "train"), ("b", "test")])
    x, ids = vecio.load(tmp_path / "v")
    assert x.shape == (2, 3) and ids == [["a", "train"], ["b", "test"]]


def test_patches_tile_row_major():
    img = Image.new("RGB", (10, 9))
    img.putpixel((9, 8), (255, 0, 0))
    ps = embed.patches(img, 2)
    assert len(ps) == 4 and all(p.size == (5, 4) for p in ps)
    assert embed.patches(Image.new("RGB", (1, 1)), 2) == []


def test_description_vectors_are_unit_and_similar_texts_match():
    enc = embed.HashEncoder()
    v = embed.embed_descriptions(enc, [["what is present: necrosis."],
                                       ["what is present: necrosis.", "what organ is this: liver."],
                                       ["is this the heart: yes."]])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_near_duplicates():
    rng = np.random.default_rng(0)
    base = embed.normalize(rng.standard_normal((20, 16)))
    q = np.vstack([base[3] + 1e-3, embed.normalize(rng.standard_normal((1, 16)))[0]])
    mask, best = embed.near_duplicates(embed.normalize(q), base, 0.95)
    assert mask.tolist() == [True, False]


def test_embed_script_end_to_end(tmp_path):
    recs = data.load_rows(synthetic.rows(40), image_dir=tmp_path / "images")
    data.write_jsonl(tmp_path / "qa.jsonl", data.drop_cross_split(recs))
    data.write_jsonl(tmp_path / "descriptions.jsonl", data.build_descriptions(recs).values())
    subprocess.run([sys.executable, str(ROOT / "scripts/embed.py"), "--data", str(tmp_path),
                    "--encoder", "hash", "--grid", "2"], check=True, cwd=ROOT,
                   capture_output=True)
    imgs, ids = vecio.load(tmp_path / "emb/images")
    n = len(ids)
    assert imgs.shape == (n, 64)
    assert vecio.load(tmp_path / "emb/patches")[0].shape == (4 * n, 64)
    assert vecio.load(tmp_path / "emb/descriptions")[0].shape == (n, 64)
    qs, qids = vecio.load(tmp_path / "emb/questions")
    assert all(s != "train" for _, _, s in qids)
    assert json.loads((tmp_path / "emb/near_dups.json").read_text())["encoder"] == "hash"
