"""PLIP embeddings: whole images, image patches, descriptions, questions.

Every vector is L2-normalized. For unit vectors, squared L2 distance is
2 - 2*cosine, so Backstitch's L2-only search ranks exactly like cosine.
"""
from __future__ import annotations

import zlib
from typing import Protocol, Sequence

import numpy as np

PLIP = "vinid/plip"


class Encoder(Protocol):
    dim: int

    def images(self, imgs: Sequence) -> np.ndarray: ...

    def texts(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class PLIPEncoder:
    """PLIP is CLIP ViT-B/32 fine-tuned on pathology image-text pairs."""

    def __init__(self, name: str = PLIP, device: str | None = None, batch: int = 64):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.torch, self.device, self.batch = torch, device, batch
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.dim = self.model.config.projection_dim

    def _run(self, items, fn) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(items), self.batch):
                out.append(fn(items[i:i + self.batch]).float().cpu().numpy())
        return normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def images(self, imgs):
        def f(chunk):
            px = self.proc(images=[im.convert("RGB") for im in chunk], return_tensors="pt")
            # Pooled output through the projection, spelled out: get_image_features
            # returns a tensor in transformers 4.x but an output object in 5.x.
            pooled = self.model.vision_model(pixel_values=px["pixel_values"].to(self.device)).pooler_output
            return self.model.visual_projection(pooled)
        return self._run(list(imgs), f)

    def texts(self, texts):
        def f(chunk):
            tok = self.proc(text=list(chunk), return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
            tok = {k: v.to(self.device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
            return self.model.text_projection(self.model.text_model(**tok).pooler_output)
        return self._run(list(texts), f)


class HashEncoder:
    """Offline stand-in for smoke tests, never for results. Images map
    through a fixed random projection of their 8x8 thumbnail, texts through
    hashed bags of words, so similar inputs still land near each other."""

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((8 * 8 * 3, dim)).astype(np.float32)

    def images(self, imgs):
        if not len(imgs):
            return np.zeros((0, self.dim), np.float32)
        px = np.stack([np.asarray(im.convert("RGB").resize((8, 8)), np.float32).ravel() / 255.0
                       for im in imgs])
        return normalize((px - 0.5) @ self.proj)

    def texts(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(":", " ").replace(".", " ").split():
                h = zlib.crc32(w.encode())
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return normalize(out)


def patches(img, grid: int) -> list:
    """grid x grid tiles of an image, row-major. Edge pixels that don't
    divide evenly are dropped."""
    w, h = img.size
    tw, th = w // grid, h // grid
    if tw == 0 or th == 0:
        return []
    return [img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            for r in range(grid) for c in range(grid)]


def embed_descriptions(enc: Encoder, facts_per_doc: Sequence[Sequence[str]]) -> np.ndarray:
    """Mean of per-fact text embeddings, renormalized.

    CLIP's text encoder stops at 77 tokens, and many descriptions are longer,
    so each fact is embedded on its own instead of truncating the whole text.
    """
    flat = [f for facts in facts_per_doc for f in facts]
    owner = np.repeat(np.arange(len(facts_per_doc)), [len(f) for f in facts_per_doc])
    e = enc.texts(flat)
    out = np.zeros((len(facts_per_doc), e.shape[1] if len(e) else enc.dim), np.float32)
    np.add.at(out, owner, e)
    return normalize(out)


def near_duplicates(query: np.ndarray, base: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, its best cosine match in base. Returns (mask of
    rows at or above threshold, best similarity). Inputs must be unit rows."""
    import faiss

    idx = faiss.IndexFlatIP(base.shape[1])
    idx.add(np.ascontiguousarray(base, dtype=np.float32))
    sims, _ = idx.search(np.ascontiguousarray(query, dtype=np.float32), 1)
    best = sims[:, 0]
    return best >= threshold, best
