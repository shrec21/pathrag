"""PathVQA loading, image grouping and per-image pseudo-descriptions.

PathVQA ships question/answer rows with an image each, but no image id and
no caption. Rows are grouped by a hash of the image pixels, and each
image's QA pairs are folded into one description: that text is the
"existing description" the RAG side retrieves.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

YES_NO = {"yes", "no"}
SPLITS = ("train", "validation", "test")


@dataclass
class QA:
    image_id: str
    split: str
    question: str
    answer: str

    @property
    def closed(self) -> bool:
        return self.answer.lower() in YES_NO


@dataclass
class Description:
    image_id: str
    split: str
    facts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.facts)


def image_id(img) -> str:
    """Content hash of a PIL image, so repeated images share one id."""
    h = hashlib.sha1()
    h.update(f"{img.mode}{img.size}".encode())
    h.update(img.tobytes())
    return h.hexdigest()[:16]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def fact(question: str, answer: str) -> str:
    """One QA pair as a short fact line.

    No attempt is made to rewrite questions into prose: grammar rules break
    on textbook phrasing, and an LLM reads "q: a" pairs fine.
    """
    q = _clean(question).rstrip("?").rstrip()
    return f"{q}: {_clean(answer).lower()}."


def iter_hf_rows(dataset) -> Iterator[tuple[str, dict]]:
    """(split, row) pairs from a datasets.DatasetDict with image/question/answer."""
    for split in SPLITS:
        if split in dataset:
            for row in dataset[split]:
                yield split, row


def load_rows(rows: Iterable[tuple[str, dict]], image_dir: Path | None = None) -> list[QA]:
    """Turns raw rows into QA records, saving each distinct image once."""
    out, saved = [], set()
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
    for split, row in rows:
        img = row["image"]
        iid = image_id(img)
        if image_dir is not None and iid not in saved:
            path = image_dir / f"{iid}.png"
            if not path.exists():
                img.convert("RGB").save(path)
            saved.add(iid)
        out.append(QA(iid, split, _clean(row["question"]), _clean(str(row["answer"]))))
    return out


def split_report(records: list[QA]) -> dict:
    """Per-split counts, plus images that show up in more than one split."""
    splits_of = defaultdict(set)
    for r in records:
        splits_of[r.image_id].add(r.split)
    shared = {i: sorted(s) for i, s in splits_of.items() if len(s) > 1}
    counts = {}
    for split in SPLITS:
        rs = [r for r in records if r.split == split]
        counts[split] = {
            "qa": len(rs),
            "images": len({r.image_id for r in rs}),
            "closed": sum(r.closed for r in rs),
        }
    return {"counts": counts, "shared_images": shared}


def assign_splits(records: list[QA]) -> dict[str, str]:
    """One split per image. An image seen in train stays train, so eval
    images never also sit in the knowledge base; otherwise validation wins
    over test."""
    rank = {s: i for i, s in enumerate(SPLITS)}
    best: dict[str, str] = {}
    for r in records:
        cur = best.get(r.image_id)
        if cur is None or rank[r.split] < rank[cur]:
            best[r.image_id] = r.split
    return best


def build_descriptions(records: list[QA], include_no: bool = True) -> dict[str, Description]:
    """Folds each image's QA pairs into a description.

    Open-ended facts come first (they carry the specific findings), then
    yes, then no. Exact duplicate facts are dropped.
    """
    split_of = assign_splits(records)
    by_image: dict[str, list[QA]] = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)
    out = {}
    for iid, rs in by_image.items():
        open_ = [r for r in rs if not r.closed]
        yes = [r for r in rs if r.answer.lower() == "yes"]
        no = [r for r in rs if r.answer.lower() == "no"] if include_no else []
        facts, seen = [], set()
        for r in open_ + yes + no:
            f = fact(r.question, r.answer)
            if f.lower() not in seen:
                seen.add(f.lower())
                facts.append(f)
        out[iid] = Description(iid, split_of[iid], facts)
    return out


def drop_cross_split(records: list[QA]) -> list[QA]:
    """Keeps each image's QA only in the split assign_splits gave it."""
    split_of = assign_splits(records)
    return [r for r in records if r.split == split_of[r.image_id]]


def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            d = asdict(it) if hasattr(it, "__dataclass_fields__") else it
            if isinstance(it, Description):
                d["text"] = it.text
            f.write(json.dumps(d) + "\n")


def read_qa(path: Path) -> list[QA]:
    with open(path) as f:
        return [QA(**json.loads(line)) for line in f]


def read_descriptions(path: Path) -> dict[str, Description]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["image_id"]] = Description(d["image_id"], d["split"], d["facts"])
    return out
