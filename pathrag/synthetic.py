"""A tiny fake PathVQA for offline smoke tests. Not data, never results.

Each image belongs to one (organ, finding) pair. Its pixels are a noisy
tint of that pair's colour, so near neighbours tend to share a pair, and
its QA pairs mirror PathVQA's mix of open and yes/no questions.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ORGANS = ["kidney", "liver", "lung", "heart", "brain", "colon"]
FINDINGS = ["amyloidosis", "necrosis", "fibrosis", "granuloma", "carcinoma"]


def rows(n_images: int = 120, seed: int = 0, size: int = 32):
    """(split, row) pairs shaped like iter_hf_rows output."""
    rng = np.random.default_rng(seed)
    pairs = [(o, f) for o in ORGANS for f in FINDINGS]
    base = {p: rng.uniform(0.15, 0.85, 3) for p in pairs}
    out = []
    for i in range(n_images):
        organ, finding = pairs[rng.integers(len(pairs))]
        px = np.clip(base[(organ, finding)] + rng.normal(0, 0.05, (size, size, 3)), 0, 1)
        img = Image.fromarray((px * 255).astype(np.uint8))
        split = "train" if i % 5 < 3 else ("validation" if i % 5 == 3 else "test")
        other = ORGANS[(ORGANS.index(organ) + 1) % len(ORGANS)]
        qa = [
            ("What is present?", finding),
            ("What organ is this?", organ),
            (f"Is this the {organ}?", "yes"),
            (f"Is this the {other}?", "no"),
        ]
        for q, a in qa:
            out.append((split, {"image": img, "question": q, "answer": a}))
    return out
