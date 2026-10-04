from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from apt_rag.evaluation.llm_benchmark import (
    citation_check, estimated_cost, freeze_inputs, load_config, run_provider, summarize_runs, validate_run,
)
from apt_rag.generation.providers import (
    DeepSeekProvider, GenerationResult, OllamaProvider, ProviderError, fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


def frozen_sample():
    chunk = {
        "chunk_id": "chunk1", "report_id": "report1", "document_title": "测试报告",
        "vendor": "CISA", "page": 2, "section": None, "source_url": "https://example.com/report",
        "text": "Example evidence", "text_sha256": "test",
    }
    benchmark = {"questions": [{"id": "q001", "question": "问题", "category": "ioc",
                               "expected_answer": "不能泄露的标准答案",
                               "evidence": [{"chunk_id": "chunk1", "report_id": "report1", "page": 2}]}]}
    prompt = {"system": "仅依据证据", "user_template": "{question}\n{context}"}
    data = {"controls": {"question_count": 1}, "inputs": freeze_inputs(benchmark, [chunk], prompt)}
    data["inputs_fingerprint"] = fingerprint(data)
    return data


def test_frozen_inputs_exclude_ground_truth_answer_and_resolve_sources():
    data = frozen_sample()
    assert "不能泄露的标准答案" not in json.dumps(data, ensure_ascii=False)
    item = data["inputs"][0]
    assert item["messages_sha256"] == fingerprint(item["messages"])
    assert item["sources"][0]["page"] == 2
    assert "[S1]" in item["messages"][1]["content"]


def test_citation_format_is_not_semantic_quality():
    value = citation_check("完全错误的事实 [S1]。未知 [S9]", [{"label": "S1"}])
    assert value["label_validity"] == 0.5
    assert value["unknown_labels"] == ["S9"]
    assert citation_check("没有引用", [])["label_validity"] is None
    assert citation_check("Tor [S0183]，证据 [S1, S2]", [{"label": "S1"}, {"label": "S2"}])["occurrences"] == ["S1", "S2"]


def test_resume_does_not_call_provider_and_rejects_changed_inputs(tmp_path):
    class FakeProvider:
        calls = []

        def identity(self):
            return {"model": "fake"}

        def generate(self, messages):
            self.calls.append(messages)
            return GenerationResult("答案 [S1]", "fake", "stop", 1, 100, 10)

    config = {"results_directory": "results", "max_output_tokens": 10}
    spec = {"id": "local", "kind": "ollama", "num_ctx": 1000, "model": "fake"}
    frozen = frozen_sample()
    fake = FakeProvider()
    first = run_provider(tmp_path, config, frozen, spec, provider=fake)
    assert fake.calls == [frozen["inputs"][0]["messages"]]
    assert first["answers"][0]["estimated_api_cost_usd"] is None
    assert summarize_runs([first])["results"][0]["semantic_quality"] is None
    assert run_provider(tmp_path, config, frozen, spec, provider=fake) == first
    assert len(fake.calls) == 1
    changed = deepcopy(frozen)
    changed["inputs_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="指纹"):
        run_provider(tmp_path, config, changed, spec, provider=fake)
    bad = deepcopy(first)
    bad["answers"][0]["answer"] = "修改"
    with pytest.raises(ValueError, match="答案指纹"):
        validate_run(bad, frozen)


def test_cost_uses_cached_and_uncached_tokens():
    spec = {"kind": "deepseek", "pricing": {
        "cache_hit_per_million": 1, "cache_miss_per_million": 2, "output_per_million": 3,
    }}
    result = GenerationResult("a", "m", "stop", 1, 100, 10, cache_hit_tokens=20)
    assert estimated_cost(result, spec) == pytest.approx(0.00021)
    with pytest.raises(ValueError, match="缓存"):
        estimated_cost(GenerationResult("a", "m", "stop", 1, 100, 10), spec)


def test_deepseek_payload_and_secret_safe_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-never-print")
    config = load_config(ROOT / "configs/llm_benchmark.json")
    spec = config["providers"][1]
    messages = frozen_sample()["inputs"][0]["messages"]

    def handler(request):
        payload = json.loads(request.content)
        assert payload["messages"] == messages
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["max_tokens"] == config["max_output_tokens"]
        return httpx.Response(200, json={"model": "deepseek-flash", "system_fingerprint": "fp",
            "choices": [{"finish_reason": "stop", "message": {"content": "答案"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 20}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert DeepSeekProvider(spec, config, client=client).generate(messages).answer == "答案"
    with httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(401, text="test-secret-never-print"))) as client:
        with pytest.raises(ProviderError) as error:
            DeepSeekProvider(spec, config, client=client).generate(messages)
        assert "test-secret-never-print" not in str(error.value)
    bad = {**spec, "base_url": "https://example.com"}
    with pytest.raises(ValueError, match="官方"):
        DeepSeekProvider(bad, config)


def test_ollama_payload_digest_and_loopback(monkeypatch):
    config = load_config(ROOT / "configs/llm_benchmark.json")
    spec = config["providers"][0]
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    messages = frozen_sample()["inputs"][0]["messages"]

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": spec["model"], "digest": "changed"}]})
        payload = json.loads(request.content)
        assert payload["think"] is False and payload["stream"] is False
        assert payload["messages"] == messages
        assert payload["options"]["presence_penalty"] == 0
        return httpx.Response(200, json={"model": spec["model"], "done": True, "done_reason": "stop",
            "message": {"content": "答案"}, "prompt_eval_count": 100, "eval_count": 10})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaProvider(spec, config, client=client)
        assert provider.generate(messages).answer == "答案"
        with pytest.raises(ValueError, match="digest"):
            provider.identity()
    monkeypatch.setenv("OLLAMA_HOST", "http://example.com:11434")
    with pytest.raises(ValueError, match="回环"):
        OllamaProvider(spec, config)


def test_assisted_review_rejects_missing_and_unbound_annotations():
    from apt_rag.evaluation.llm_review import summarize_review

    run = {"provider_id": "local", "answers": [{"question_id": "q001", "answer_sha256": "hash",
                                               "citation_check": {"occurrences": ["S1"]}}]}
    rubric = {"required_facts": {"q001": ["a", "b"]}}
    review = {"reviewer": "测试辅助复核", "independent_human_review": False,
              "rubric_fingerprint": fingerprint(rubric), "reviews": [{
        "provider_id": "local", "question_id": "q001", "answer_sha256": "hash",
        "fact_coverage": [True, False], "citation_supported": [True],
        "unsupported_claims": [], "notes": "覆盖第一项，漏答第二项，引用有依据。",
    }]}
    score = summarize_review([run], rubric, review)["local"]
    assert score["required_fact_coverage"] == 0.5
    assert score["semantic_citation_support"] == 1
    bad = deepcopy(review)
    bad["reviews"][0]["answer_sha256"] = "different"
    with pytest.raises(ValueError, match="答案指纹"):
        summarize_review([run], rubric, bad)
    bad = deepcopy(review)
    bad["reviews"] = []
    with pytest.raises(ValueError, match="漏题"):
        summarize_review([run], rubric, bad)


def test_failed_request_preserves_checkpoint_and_resume_only_missing_question(tmp_path):
    frozen = frozen_sample()
    second = deepcopy(frozen["inputs"][0])
    second["question_id"] = "q002"
    second["messages"][1]["content"] += " 第二题"
    second["messages_sha256"] = fingerprint(second["messages"])
    frozen["inputs"].append(second)
    frozen["controls"]["question_count"] = 2
    config = {"results_directory": "results", "max_output_tokens": 10}
    spec = {"id": "local", "kind": "ollama", "num_ctx": 1000, "model": "fake"}

    class InterruptedProvider:
        def __init__(self, fail_on_second):
            self.fail_on_second = fail_on_second
            self.calls = []

        def identity(self):
            return {"model": "fake"}

        def generate(self, messages):
            self.calls.append(messages)
            if self.fail_on_second and len(self.calls) == 2:
                raise ProviderError("测试网络中断")
            return GenerationResult("答案 [S1]", "fake", "stop", 1, 100, 10)

    with pytest.raises(ProviderError):
        run_provider(tmp_path, config, frozen, spec, provider=InterruptedProvider(True))
    saved = json.loads((tmp_path / "results/local.json").read_text(encoding="utf-8"))
    assert len(saved["answers"]) == 1 and saved["completed"] is False
    resumed = InterruptedProvider(False)
    assert run_provider(tmp_path, config, frozen, spec, provider=resumed)["completed"] is True
    assert resumed.calls == [second["messages"]]


def test_committed_results_are_offline_reproducible():
    from apt_rag.evaluation.llm_review import summarize_review

    config = load_config(ROOT / "configs/llm_benchmark.json")
    directory = ROOT / config["results_directory"]
    frozen = json.loads((directory / "inputs_manifest.json").read_text(encoding="utf-8"))
    runs = [json.loads((directory / f"{spec['id']}.json").read_text(encoding="utf-8"))
            for spec in config["providers"]]
    for run in runs:
        validate_run(run, frozen)
    rubric = json.loads((ROOT / "data/benchmark/llm_rubric_v1.json").read_text(encoding="utf-8"))
    review = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    scores = summarize_review(runs, rubric, review)
    summary = summarize_runs(runs)
    for row in summary["results"]:
        row["semantic_quality"] = scores[row["provider_id"]]
    assert summary == json.loads((directory / "summary.json").read_text(encoding="utf-8"))
