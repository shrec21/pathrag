# pathrag

Can existing descriptions of pathology images describe new ones? This repo answers that with retrieval over PathVQA, using PLIP embeddings and the Backstitch graph index for CS 594 Vector Databases at UIC.

Backstitch is a separate research project and is not included here. pathrag touches it at exactly one seam, `bindings/pybackstitch.cpp`, which wraps its `Index` (build, search, insert) and `Engine` (remove, repair). Without a Backstitch checkout, pathrag runs on its FAISS backends and the Backstitch tests skip.

## Setup

```bash
pip install numpy faiss-cpu pillow pybind11 pytest torch transformers datasets

# optional: the Backstitch backend, from a local checkout of it
BACKSTITCH_DIR=/path/to/Backstitch python setup.py build_ext --inplace   # about 30s

pytest   # 38 tests, offline; the Backstitch ones skip without the backend
```

`setup.py` compiles the Backstitch core sources directly instead of going through its CMake build.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1 Data | `python scripts/prepare_data.py` | `data/pathvqa/`: images, `qa.jsonl`, `descriptions.jsonl`, `split_report.json` |
| 2 Embed | `python scripts/embed.py` | `data/pathvqa/emb/`: images, patches, descriptions, questions (`.fbin` + `.ids.tsv`), `near_dups.json` |
| 3 Index | `pathrag/index.py` | `make_index("flat" \| "hnsw" \| "backstitch:<lazy_only\|inline\|background>", ...)` |
| 4 Answer presence | `python scripts/answer_presence.py --index flat` | table and `runs/presence_*.json` |

Step 2 downloads PLIP (`vinid/plip`) and runs on CUDA, Apple MPS or CPU. With the default 4x4 grid it embeds about 5K images and 80K patches. `--encoder hash` exists only for offline smoke tests and its numbers mean nothing.

## Design decisions

- **Image ids are pixel hashes.** PathVQA has no image ids, so QA rows are grouped by a hash of the pixels. An image in more than one split stays in train and its eval QA rows are dropped, so no eval image ever sits in the knowledge base.
- **Descriptions are fact lines, not prose.** Each QA pair becomes `question: answer.`, open-ended first. Rewriting questions into sentences breaks on textbook phrasing.
- **Description vectors average per-fact embeddings.** CLIP's text encoder stops at 77 tokens and many descriptions run longer.
- **Every vector is unit length.** Squared L2 is then 2 - 2 cos, so Backstitch's L2-only search ranks exactly like cosine.
- **Answer presence checks answer slots only.** A neighbor's `is this the liver: no.` gets no credit for the gold answer "liver". Yes/no questions are left out of this metric.
- **FAISS deletes are filters.** FAISS HNSW can't remove nodes, so the `hnsw` and `flat` backends drop deleted labels at query time with no repair. That's the honest baseline for Backstitch.
- **Backstitch capacity counts every insert.** Deleted slots are not reused, so size `capacity` for the whole run, replaces included.

## Measured so far

On random 512-d unit vectors (the hard case; real PLIP vectors cluster), 2 cores, 10K points: Backstitch built in 112s and reached 0.818 recall@10 at L=100, against 3s and 0.738 for FAISS HNSW at efSearch=100. Expect the 80K patch index to build in a few minutes on an 8+ core laptop.

## Next

Step 5 (RAG generation) waits on the generator choice: an API model or a local open VLM. Steps 6 (trust and abstention) and 7 (living knowledge base under the three Backstitch versions) build on step 4's retrieval.
