import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from apt_rag.evaluation.ablation_benchmark import (
    citation_audit, protocol, review_messages, schedule, score_review, select_references, validate_review,
)
from apt_rag.generation.providers import DeepSeekProvider, GenerationResult, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def sample():
    row = {"answer": "归属于 GRU [S1]。细节不确定。", "sources": [{"label": "S1", "chunk_id": "c1"}],
           "citation_audit": {"labels": ["S1"]}, "strategy": "must_not_leak"}
    value = {"required_facts": [{"covered": True, "reason": "已覆盖"}],
             "claims": [{"text": "归属于 GRU", "verdict": "supported", "reference_chunk_ids": ["c1"], "reason": "明确提及"}],
             "citations": [{"label": "S1", "claim": "归属于 GRU", "verdict": "supported", "reason": "片段支持"}]}
    return row, value


def test_schedule_same_questions_balanced_randomized_and_deterministic():
    ids = [f"q{i:03}" for i in range(20)]
    value = schedule(ids, 42)
    assert value == schedule(ids, 42) and value != schedule(ids, 43)
    assert len(value) == len({r["id"] for r in value}) == 80
    for qid in ids:
        assert {r["strategy"] for r in value if r["question_id"] == qid} == {"pure", "dense", "hybrid", "hybrid_reranker"}


def test_reference_neighbors_same_report_and_no_ground_truth_context_selection():
    question = {"evidence": [{"report_id": "r1", "page": 3}]}
    chunks = [{"report_id": "r1", "page": p} for p in (1, 2, 3, 4, 5)] + [{"report_id": "r2", "page": 3}]
    assert [c["page"] for c in select_references(question, chunks, 1)] == [2, 3, 4]


def test_citation_audit_preserves_bad_answers_without_success_filter():
    assert citation_audit("没有来源的回答", []) == {"labels": [], "unknown_labels": [], "malformed_groups": [], "unclosed": False}
    audit = citation_audit("事实 [S1, S2]；错 [S9]；坏 [s1]；未闭合 [S3", [{"label": "S1"}])
    assert audit["labels"] == ["S1", "S2", "S9", "s1"]
    assert audit["unknown_labels"] == ["S2", "S9", "s1"] and audit["unclosed"]
    assert audit["malformed_groups"] == ["s1"]


def test_review_schema_exact_quotes_and_reference_ids():
    row, value = sample()
    validate_review(value, row, ["归属"], ["c1"])
    for change, message in [
        (lambda v: v["claims"][0].update(text="不存在的原文"), "原文"),
        (lambda v: v["claims"][0].update(reference_chunk_ids=["fake"]), "编造"),
        (lambda v: v.update(citations=[]), "分母"),
        (lambda v: v["required_facts"][0].update(covered=1), "非法"),
    ]:
        modified = deepcopy(value)
        change(modified)
        with pytest.raises(ValueError, match=message):
            validate_review(modified, row, ["归属"], ["c1"])


def test_unknown_citation_cannot_be_supported():
    row, value = sample()
    row["sources"] = []
    with pytest.raises(ValueError, match="未知引用"):
        validate_review(value, row, ["归属"], ["c1"])


def test_metric_unknown_is_not_assumed_false_no_citation_na():
    _, value = sample()
    value["claims"] += [
        {"text": "a", "verdict": "contradicted"},
        {"text": "b", "verdict": "unverifiable"},
    ]
    value["citations"] = []
    scored = score_review(value)
    assert scored["accuracy"] == .5 and scored["completeness"] == 1
    assert scored["citation_correctness"] is None
    assert scored["hallucination_rate"] == pytest.approx(1 / 3)
    assert scored["unverifiable_rate"] == pytest.approx(1 / 3)
    value["claims"] = []
    assert score_review(value)["accuracy"] is None
    assert score_review(value)["hallucination_rate"] is None


def test_judge_prompt_masks_strategy_but_preserves_identical_reference():
    row, _ = sample()
    question = {"question": "谁？", "expected_answer": "标准"}
    reference = {"chunk_id": "c1", "report_id": "r1", "document_title": "报告", "vendor": "CISA",
                 "page": 1, "section": None, "source_url": "https://example.com", "text_sha256": "sha", "text": "参考"}
    prompt = {"system": "JSON", "user_template": "{payload}"}
    messages = review_messages(row, question, [reference], {"c1": reference}, ["归属"], prompt)
    assert "must_not_leak" not in json.dumps(messages)
    assert json.loads(messages[1]["content"])["reference_sources"] == [reference]


def test_deepseek_json_mode_optional_does_not_change_old_payload(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-not-secret")
    config, _, _, sources, spec, _, _, _, _, _ = protocol(ROOT, ROOT / "configs/ablation_benchmark.json")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": spec["model"], "system_fingerprint": "fp",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 0}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        messages = [{"role": "system", "content": "JSON"}, {"role": "user", "content": "test"}]
        DeepSeekProvider(spec, sources["generation"], client=client).generate(messages)
        DeepSeekProvider(spec, {**sources["generation"], **config["review"]}, client=client).generate(messages)
    assert "response_format" not in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert seen[1]["thinking"] == {"type": "disabled"} and seen[1]["max_tokens"] == 4096


def test_generation_resume_no_calls_and_changed_fingerprint_stops(tmp_path, monkeypatch):
    import apt_rag.evaluation.ablation_benchmark as module

    spec = {"model": "fake", "kind": "deepseek", "pricing": {
        "cache_hit_per_million": .006, "cache_miss_per_million": .3, "output_per_million": 1.2}}
    settings = {"temperature": 0, "max_output_tokens": 768, "timeout_seconds": 600}
    config = {"results_directory": "results"}
    controls = {"test": "frozen"}
    monkeypatch.setattr(module, "protocol", lambda *args: (
        config, {}, {}, {"generation": settings}, spec, {}, {}, {}, {}, controls))
    ordering = schedule(["q001"], 42)
    frozen = {"controls": controls, "inputs_fingerprint": "fixed", "schedule": ordering, "entries": {}}
    for planned in ordering:
        messages = [{"role": "system", "content": "相同 Prompt"}, {"role": "user", "content": planned["strategy"]}]
        frozen["entries"][planned["id"]] = {**planned, "category": "ttp", "sources": [], "messages": messages,
            "messages_sha256": fingerprint(messages), "token_budget": {"input_tokens": 100, "safety_margin": 256,
                "max_output_tokens": 768, "context_window_tokens": 8192}}

    class Fake:
        calls = 0
        fail_after = 2

        def identity(self):
            return {"model": "fake"}

        def generate(self, messages):
            if self.calls == self.fail_after:
                raise RuntimeError("模拟网络中断")
            self.calls += 1
            return GenerationResult("原始无引用答案", "fake", "stop", 1, 100, 10, 0, "fp")

    fake = Fake()
    with pytest.raises(RuntimeError, match="中断"):
        module.run_generations(tmp_path, tmp_path / "config.json", frozen, provider=fake)
    assert len(module.read(tmp_path / "results/answers.json")["answers"]) == 2
    fake.fail_after = 999
    result = module.run_generations(tmp_path, tmp_path / "config.json", frozen, provider=fake)
    assert fake.calls == 4 and len(result["answers"]) == 4
    assert module.run_generations(tmp_path, tmp_path / "config.json", frozen, provider=fake) == result
    assert fake.calls == 4
    modified = deepcopy(frozen)
    modified["inputs_fingerprint"] = "different"
    with pytest.raises(ValueError, match="指纹"):
        module.run_generations(tmp_path, tmp_path / "config.json", modified, provider=fake)
    assert fake.calls == 4


def test_saved_phase10_results_offline_when_present():
    from apt_rag.evaluation.ablation_benchmark import load_results, read

    path = ROOT / "results/phase10/summary-v2.json"
    if not path.exists():
        pytest.skip("完整阶段运行尚未结束")
    _, summary = load_results(ROOT, ROOT / "configs/ablation_benchmark.json",
                             review_config_path=ROOT / "configs/ablation_review_v2.json")
    assert summary == read(path)
    assert [r["answer_count"] for r in summary["results"]] == [20] * 4


def test_disk_checkpoint_recovers_exact_pending_prefix_not_duplicate_call(tmp_path):
    from apt_rag.evaluation.ablation_benchmark import _write_json, read_checkpoint

    path = tmp_path / "answers.json"
    first = {"controls": {"x": 1}, "completed": False, "answers": [{"id": "q1"}]}
    _write_json(path, first)
    pending = {**first, "answers": [*first["answers"], {"id": "q2"}]}
    _write_json(path.with_suffix(".json.tmp"), pending)
    assert read_checkpoint(path, "answers") == pending
    bad = {**pending, "controls": {"x": 2}}
    _write_json(path.with_suffix(".json.tmp"), bad)
    with pytest.raises(ValueError, match="协议不同"):
        read_checkpoint(path, "answers")
    _write_json(path.with_suffix(".json.tmp"), {**pending, "answers": [{"id": "changed"}]})
    with pytest.raises(ValueError, match="相同前缀"):
        read_checkpoint(path, "answers")


def test_judge_outside_shared_reference_downgraded_not_used_to_expand_truth():
    from apt_rag.evaluation.ablation_benchmark import project_shared_reference

    row, value = sample()
    value["claims"][0]["reference_chunk_ids"] = ["c2"]
    validate_review(value, row, ["归属"], ["c1"], provided_ids=["c2"])
    projected = project_shared_reference(value, ["c1"])
    assert projected["claims"][0]["verdict"] == "unverifiable"
    assert projected["reference_scope_downgrade_count"] == 1
    assert projected["claims"][0]["reference_chunk_ids"] == []
    assert value["claims"][0]["verdict"] == "supported"  # 原始判定未覆盖
    assert score_review(projected)["accuracy"] is None
    assert score_review(projected)["unverifiable_rate"] == 1
    with pytest.raises(ValueError, match="编造"):
        validate_review(value, row, ["归属"], ["c1"], provided_ids=["not_c2"])


def test_quote_matching_allows_only_presentation_not_factual_edits():
    from apt_rag.evaluation.ablation_benchmark import quote_in_answer

    assert quote_in_answer("至少五年", "驻留**至少五年** [S1]")
    assert quote_in_answer("NTDS.dit", "`NTDS.dit`")
    assert quote_in_answer("升级为供应链攻击目前属于潜在风险", '因此，"升级为供应链攻击"目前属于潜在风险。')
    assert not quote_in_answer("最多五年", "驻留**至少五年**")
    assert not quote_in_answer("123", "132")
    assert not quote_in_answer("**", "任何回答")


def test_windows_permission_error_retries_disk_only(monkeypatch, tmp_path):
    import apt_rag.evaluation.ablation_benchmark as module

    seen = []

    def writer(path, value):
        seen.append((path, value))
        if len(seen) < 3:
            raise PermissionError("模拟临时锁")

    monkeypatch.setattr(module, "write_json_once", writer)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    path, value = tmp_path / "answers.json", {"answer": "原始响应"}
    module._write_json(path, value)
    assert seen == [(path, value)] * 3


def test_unknown_label_machine_audit_overrides_raw_judge_without_changing_it():
    from apt_rag.evaluation.ablation_benchmark import validate_review_record
    from apt_rag.evaluation.llm_benchmark import estimated_cost

    row, value = sample()
    row.update(id="q001:pure", question_id="q001", sources=[], answer_sha256=fingerprint(row["answer"]))
    spec = {"model": "fake", "kind": "deepseek", "pricing": {
        "cache_hit_per_million": .006, "cache_miss_per_million": .3, "output_per_million": 1.2}}
    raw = GenerationResult(json.dumps(value, ensure_ascii=False), "fake", "stop", 1, 100, 200, 0, "fp")
    record = {**raw.to_dict(), "id": row["id"], "review_id": fingerprint(row["id"]),
              "answer_sha256": row["answer_sha256"], "answer_sha256_raw_review": fingerprint(raw.answer),
              "estimated_api_cost_usd": estimated_cost(raw, spec),
              "token_budget": {"input_tokens": 120, "safety_margin": 256, "max_output_tokens": 8192,
                               "context_window_tokens": 32768}}
    projected = validate_review_record(record, row, {"references": {"q001": [{"chunk_id": "c1"}]}},
                                       {"required_facts": {"q001": ["归属"]}}, spec)
    assert projected["citations"][0]["verdict"] == "unsupported"
    assert projected["unknown_citation_override_count"] == 1
    assert json.loads(record["answer"])["citations"][0]["verdict"] == "supported"
    assert score_review(projected)["citation_correctness"] == 0
