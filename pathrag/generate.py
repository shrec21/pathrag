"""Step 5: answer PathVQA questions with an API vision model, with and
without retrieved descriptions.

Conditions, all on the same questions so they compare row by row:
  image_only   the image and the question, no references
  rag          the image plus the descriptions of its k retrieved neighbours
  refs_only    the references and the question, no image
  rag_random   the image plus k random train descriptions (does retrieval
               matter, or does any pathology text help?)

Every answer is cached on disk under a hash of the exact request, so a run
that dies (rate limit, lid closed, crash) resumes where it stopped and no
request is ever paid for twice.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from pathrag.data import QA, Description

CONDITIONS = ("image_only", "rag", "refs_only", "rag_random")
DEFAULT_MAX_TOKENS = 1024  # room for models that reason before answering
YES_NO_START = ("is ", "are ", "does ", "do ", "did ", "was ", "were ", "can ", "has ", "have ",
                "will ", "could ", "should ", "would ")

SYSTEM = (
    "You answer questions about pathology images from a textbook-style dataset. "
    "Reply with the answer only: a short phrase of a few words, no explanation, no full sentence. "
    "For a yes/no question reply with exactly yes or no."
)


def is_yes_no(question: str) -> bool:
    """From the question text alone, never the gold answer."""
    return question.strip().lower().startswith(YES_NO_START)


@dataclass
class Prompt:
    text: str
    image_png: bytes | None  # None when the condition has no image


def format_refs(neighbours: list[str], descs: dict[str, Description], max_facts: int) -> str:
    blocks = []
    for n, iid in enumerate(neighbours, 1):
        facts = descs[iid].facts[:max_facts]
        blocks.append(f"Reference case {n}:\n" + "\n".join(f"- {f}" for f in facts))
    return "\n\n".join(blocks)


def build_prompt(qa: QA, condition: str, neighbours: list[str], descs: dict[str, Description],
                 image_png: bytes | None, max_facts: int = 12) -> Prompt:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    parts = []
    if condition != "image_only":
        intro = ("Below are notes on reference cases that look similar to this image. They may or may "
                 "not apply to it; use them only where they fit." if condition != "refs_only" else
                 "You cannot see the image. Below are notes on reference cases that look similar to it. "
                 "Answer from them as best you can.")
        parts += [intro, format_refs(neighbours, descs, max_facts)]
    parts.append(f"Question: {qa.question}")
    parts.append("Answer yes or no." if is_yes_no(qa.question) else "Answer in a few words.")
    return Prompt("\n\n".join(parts), None if condition == "refs_only" else image_png)


def load_png(path: Path, max_side: int = 512) -> bytes:
    """The image as PNG, downscaled so its long side is at most max_side,
    which keeps image tokens (and cost) down."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- clients

class Client(Protocol):
    model: str

    def answer(self, prompt: Prompt) -> tuple[str, dict]: ...


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = DEFAULT_MAX_TOKENS, client=None):
        import anthropic

        self.model, self.max_tokens = model, max_tokens
        self.client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def answer(self, prompt):
        content = []
        if prompt.image_png is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(prompt.image_png).decode()}})
        content.append({"type": "text", "text": prompt.text})
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
                                        messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens, "stop": r.stop_reason}


class OpenAIClient:
    def __init__(self, model: str, max_tokens: int = DEFAULT_MAX_TOKENS, client=None):
        import openai

        self.model, self.max_tokens = model, max_tokens
        self.client = client or openai.OpenAI()  # reads OPENAI_API_KEY

    def answer(self, prompt):
        content = [{"type": "text", "text": prompt.text}]
        if prompt.image_png is not None:
            url = "data:image/png;base64," + base64.b64encode(prompt.image_png).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        r = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        c = r.choices[0]
        return c.message.content or "", {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens,
                                         "stop": c.finish_reason}


def make_client(provider: str, model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Client:
    if provider == "anthropic":
        return AnthropicClient(model, max_tokens)
    if provider == "openai":
        return OpenAIClient(model, max_tokens)
    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------- running

def request_key(model: str, prompt: Prompt, max_tokens: int = 0) -> str:
    """max_tokens is part of the key: answers cut off under an old limit
    must not be reused under a new one."""
    h = hashlib.sha256(f"{model}|{max_tokens}".encode())
    h.update(prompt.text.encode())
    h.update(prompt.image_png or b"")
    return h.hexdigest()[:24]


class Cache:
    """Append-only JSONL of finished answers, keyed by request hash."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("pred", "").strip():  # never serve a blank answer
                            self.done[rec["key"]] = rec
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, rec: dict) -> None:
        with self.lock:
            self.done[rec["key"]] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")


class EmptyAnswer(RuntimeError):
    """The model returned no text, usually because reasoning used up the
    whole token budget. Retrying the same request would only repeat it."""


def with_retries(fn: Callable, tries: int = 6, base: float = 2.0, sleep=time.sleep):
    for attempt in range(tries):
        try:
            return fn()
        except EmptyAnswer:
            raise
        except Exception:  # SDK errors differ by provider; rate limits and 5xx retry alike
            if attempt == tries - 1:
                raise
            sleep(base ** attempt + random.random())


@dataclass
class Job:
    row: int  # line number in qa.jsonl
    qa: QA
    prompt: Prompt


def run(jobs: list[Job], client: Client, cache: Cache, workers: int = 8,
        progress: Callable[[int, int], None] | None = None, sleep=time.sleep) -> tuple[list[dict], int]:
    """Answers every job not already cached. Returns (records in job order,
    number that failed after retries). Failed jobs are left out of the cache
    so a rerun tries them again."""
    keyed = [(request_key(client.model, j.prompt, getattr(client, "max_tokens", 0)), j) for j in jobs]
    todo = [(k, j) for k, j in keyed if k not in cache.done]
    failed, finished = 0, 0
    empty = []  # stop reasons of empty answers, so the caller can say why

    def one(key, job):
        def ask():
            text, usage = client.answer(job.prompt)
            if not text.strip():
                empty.append(usage.get("stop"))
                raise EmptyAnswer(f"no text (stop: {usage.get('stop')}, out tokens: {usage.get('out')})")
            return text, usage

        text, usage = with_retries(ask, sleep=sleep)
        cache.put({"key": key, "row": job.row, "image_id": job.qa.image_id, "question": job.qa.question,
                   "gold": job.qa.answer, "pred": text.strip(), "usage": usage})

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(one, k, j) for k, j in todo]
        for fut in as_completed(futs):
            finished += 1
            if fut.exception() is not None:
                failed += 1
                if failed == 1:
                    print(f"  first failure: {fut.exception()!r}", flush=True)
            if progress:
                progress(finished, len(todo))
    return [cache.done[k] for k, _ in keyed if k in cache.done], failed
