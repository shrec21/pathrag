import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pathrag import data, generate, scoring, synthetic

ROOT = Path(__file__).resolve().parents[1]
DESCS = {
    "t1": data.Description("t1", "train", ["What is present: fibrosis.", "Is this the lung: yes."]),
    "t2": data.Description("t2", "train", [f"fact {i}: x." for i in range(20)]),
}
OPEN = data.QA("e1", "test", "What is present?", "fibrosis")
CLOSED = data.QA("e1", "test", "Is this the lung?", "yes")


def test_yes_no_detection_uses_question_only():
    assert generate.is_yes_no("Is this the kidney?")
    assert generate.is_yes_no("  Does this show necrosis?")
    assert not generate.is_yes_no("What is present?")
    assert not generate.is_yes_no("Island of cells?")


def test_prompts_per_condition():
    img = b"png"
    p = generate.build_prompt(OPEN, "image_only", ["t1"], DESCS, img)
    assert p.image_png == img and "Reference" not in p.text and p.text.endswith("Answer in a few words.")
    p = generate.build_prompt(CLOSED, "rag", ["t1", "t2"], DESCS, img)
    assert p.image_png == img and "Reference case 2:" in p.text and p.text.endswith("Answer yes or no.")
    assert "fact 11" in p.text and "fact 12" not in p.text  # capped at 12 facts per reference
    p = generate.build_prompt(OPEN, "refs_only", ["t1"], DESCS, img)
    assert p.image_png is None and "cannot see the image" in p.text
    with pytest.raises(ValueError):
        generate.build_prompt(OPEN, "nope", [], DESCS, img)


def test_load_png_downscales(tmp_path):
    from PIL import Image
    Image.new("RGB", (1200, 600), "red").save(tmp_path / "a.png")
    out = Image.open(__import__("io").BytesIO(generate.load_png(tmp_path / "a.png")))
    assert out.size == (512, 256)


def test_scoring():
    assert scoring.yes_no("Yes.") == "yes" and scoring.yes_no("No, it is not") == "no"
    assert scoring.yes_no("unclear") is None
    assert scoring.exact("The Fibrosis.", "fibrosis")
    assert scoring.contains("hyaline arteriolosclerosis of the kidney", "hyaline arteriolosclerosis")
    assert not scoring.contains("fibrosis-like", "fibros")
    assert scoring.f1("fatty liver", "liver") == pytest.approx(2 / 3)
    s = scoring.score([
        {"gold": "yes", "pred": "Yes"}, {"gold": "no", "pred": "yes"},
        {"gold": "fibrosis", "pred": "fibrosis"}, {"gold": "granuloma", "pred": "a caseating granuloma"},
    ])
    assert s["n_closed"] == 2 and s["closed_acc"] == 0.5
    assert s["open_exact"] == 0.5 and s["open_contains"] == 1.0


class FakeClient:
    model = "fake-1"

    def __init__(self, fail_first=0, always_fail=()):
        self.calls, self.fail_first, self.always_fail = 0, fail_first, set(always_fail)
        self.lock = threading.Lock()

    def answer(self, prompt):
        with self.lock:
            self.calls += 1
            if self.calls <= self.fail_first:
                raise RuntimeError("429 rate limited")
        if any(q in prompt.text for q in self.always_fail):
            raise RuntimeError("500")
        return " fibrosis \n", {"in": 10, "out": 2}


def jobs():
    return [generate.Job(i, q, generate.build_prompt(q, "rag", ["t1"], DESCS, b"png"))
            for i, q in enumerate([OPEN, CLOSED])]


def test_run_caches_and_resumes(tmp_path):
    cache = generate.Cache(tmp_path / "c.jsonl")
    c = FakeClient(fail_first=1)
    recs, failed = generate.run(jobs(), c, cache, workers=2, sleep=lambda s: None)
    assert failed == 0 and [r["pred"] for r in recs] == ["fibrosis", "fibrosis"]
    assert c.calls == 3  # one retry
    again = FakeClient()
    recs2, _ = generate.run(jobs(), again, generate.Cache(tmp_path / "c.jsonl"), sleep=lambda s: None)
    assert again.calls == 0 and recs2 == recs  # all served from disk


def test_failures_stay_out_of_cache(tmp_path):
    cache = generate.Cache(tmp_path / "c.jsonl")
    recs, failed = generate.run(jobs(), FakeClient(always_fail=["Is this the lung?"]), cache, sleep=lambda s: None)
    assert failed == 1 and len(recs) == 1
    recs, failed = generate.run(jobs(), FakeClient(), generate.Cache(tmp_path / "c.jsonl"),
                                sleep=lambda s: None)
    assert failed == 0 and len(recs) == 2


def mock_http(handler, lib="httpx"):
    """A client whose requests go to handler. Newer anthropic SDKs use httpx2."""
    http = pytest.importorskip(lib)
    return http.Client(transport=http.MockTransport(handler)), http


def anthropic_http_lib():
    import importlib.util
    return "httpx2" if importlib.util.find_spec("httpx2") else "httpx"


def test_anthropic_request_shape():
    anthropic = pytest.importorskip("anthropic")
    seen = {}

    lib = anthropic_http_lib()

    def handler(req):
        seen.update(json.loads(req.content))
        return http.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "fibrosis"}],
            "usage": {"input_tokens": 120, "output_tokens": 3}})

    client, http = mock_http(handler, lib)
    sdk = anthropic.Anthropic(api_key="test", http_client=client, max_retries=0)
    text, usage = generate.AnthropicClient("claude-x", client=sdk).answer(
        generate.build_prompt(OPEN, "rag", ["t1"], DESCS, b"\x89PNG"))
    assert text == "fibrosis" and usage == {"in": 120, "out": 3, "stop": "end_turn"}
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
    assert content[1]["type"] == "text" and seen["system"] == generate.SYSTEM


def test_openai_request_shape():
    openai = pytest.importorskip("openai")
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return http.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-x",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "yes"}}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 1, "total_tokens": 91}})

    client, http = mock_http(handler)
    sdk = openai.OpenAI(api_key="test", http_client=client, max_retries=0)
    text, usage = generate.OpenAIClient("gpt-x", client=sdk).answer(
        generate.build_prompt(CLOSED, "refs_only", ["t1"], DESCS, b"\x89PNG"))
    assert text == "yes" and usage == {"in": 90, "out": 1, "stop": "stop"}
    user = seen["messages"][1]["content"]
    assert len(user) == 1 and user[0]["type"] == "text"  # refs_only sends no image
    assert seen["max_completion_tokens"] == generate.DEFAULT_MAX_TOKENS


def test_dry_run_script(tmp_path):
    recs = data.load_rows(synthetic.rows(60, seed=3), image_dir=tmp_path / "images")
    data.write_jsonl(tmp_path / "qa.jsonl", data.drop_cross_split(recs))
    data.write_jsonl(tmp_path / "descriptions.jsonl", data.build_descriptions(recs).values())
    subprocess.run([sys.executable, str(ROOT / "scripts/embed.py"), "--data", str(tmp_path),
                    "--encoder", "hash", "--grid", "2"], check=True, capture_output=True)
    out = subprocess.run([sys.executable, str(ROOT / "scripts/generate.py"), "--data", str(tmp_path),
                          "--limit", "10", "--dry-run"], check=True, capture_output=True, text=True).stdout
    assert "10 test questions" in out and "4 conditions = 40 requests" in out
    assert "Reference case 1:" in out and "---- gold:" in out


class BlankClient:
    """Mimics a model whose reasoning eats the whole budget."""
    model = "fake-1"

    def __init__(self, max_tokens):
        self.max_tokens, self.calls = max_tokens, 0

    def answer(self, prompt):
        self.calls += 1
        if self.max_tokens < 100:
            return "", {"in": 10, "out": self.max_tokens, "stop": "max_tokens"}
        return "yes", {"in": 10, "out": 60, "stop": "end_turn"}


def test_blank_answers_fail_once_and_are_redone_under_a_bigger_budget(tmp_path):
    path = tmp_path / "c.jsonl"
    small = BlankClient(32)
    recs, failed = generate.run(jobs(), small, generate.Cache(path), sleep=lambda s: None)
    assert recs == [] and failed == 2 and small.calls == 2  # no pointless retries
    big = BlankClient(1024)
    recs, failed = generate.run(jobs(), big, generate.Cache(path), sleep=lambda s: None)
    assert failed == 0 and [r["pred"] for r in recs] == ["yes", "yes"] and big.calls == 2


def test_cache_skips_blank_records_from_old_runs(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"key": "k1", "pred": ""}) + "\n" + json.dumps({"key": "k2", "pred": "no"}) + "\n")
    assert set(generate.Cache(path).done) == {"k2"}
