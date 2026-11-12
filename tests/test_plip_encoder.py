"""Runs PLIPEncoder against a tiny random CLIP saved locally, so the
transformers calls are checked without downloading PLIP's weights."""
import json

import numpy as np
import pytest
from PIL import Image

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

from pathrag import embed  # noqa: E402


def byte_chars():
    """The 256 printable stand-ins GPT-2 style BPE uses for raw bytes."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory):
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor,
                              CLIPTokenizer)
    d = tmp_path_factory.mktemp("tiny_clip")
    chars = byte_chars()
    vocab = {c: i for i, c in enumerate(chars + [c + "</w>" for c in chars])}
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt"))
    proc = CLIPProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 32},
                                                            crop_size={"height": 32, "width": 32}),
                         tokenizer=tok)
    cfg = CLIPConfig(
        text_config=dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, image_size=32, patch_size=8),
        projection_dim=24)
    CLIPModel(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return str(d)


def test_plip_encoder_shapes_and_norms(tiny_clip):
    enc = embed.PLIPEncoder(name=tiny_clip, device="cpu", batch=3)
    assert enc.dim == 24
    imgs = [Image.new("RGB", (40, 30), c) for c in ["red", "blue", "green", "white", "black"]]
    iv = enc.images(imgs)
    tv = enc.texts(["what is present: necrosis.", "is this the kidney: yes.", "x " * 200])
    assert iv.shape == (5, 24) and tv.shape == (3, 24)
    assert np.allclose(np.linalg.norm(iv, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(tv, axis=1), 1, atol=1e-5)
    assert enc.images([]).shape == (0, 24)
