"""LangChain 强制编排与原管线契约等价；不加载模型或访问云端。"""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableBranch, RunnableSequence
from langsmith import get_tracing_context, tracing_context

from apt_rag.rag.budget import ContextOverflow
from test_rag import FakeBudget, FakeLLM, pipeline

ROOT = Path(__file__).resolve().parents[1]
LEGACY = runpy.run_path(str(ROOT / "tests/fixtures/legacy_rag_pipeline.py"))["RAGPipeline"]


def original(current):
    return LEGACY(chunks=list(current.chunks.values()), embedding=current.embedding,
        dense=current.dense, sparse=current.sparse, reranker=current.reranker, llm=current.llm,
        budget=current.budget, prompt=current.prompt, retrieval_config=current.config,
        top_k=current.top_k, max_question_characters=current.max_question_characters,
        public_corpus_verified=current.public_corpus_verified)


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {"timings_seconds", "total_seconds"}}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def outcome(active, question, *, dry_run=False):
    try:
        return "success", semantic(active.answer(question, dry_run=dry_run))
    except ValueError as error:
        return type(error).__name__, str(error), semantic(getattr(error, "record", None))


@pytest.mark.parametrize("question,options,llm_options,dry_run", [
    ("APT28 信息？", {}, {}, False),
    ("APT28 信息？", {}, {}, True),
    ("问题", {"empty": True}, {}, False),
    ("问题", {"empty": True}, {}, True),
    ("问题", {}, {"answer": "证据不足，无法依据当前报告回答。"}, False),
    ("问题", {}, {"answer": "事实 [S99]"}, False),
    ("问题", {}, {"answer": "未引用的事实"}, False),
    ("问题", {}, {"answer": "事实 [S1 S2]"}, False),
    ("问题", {}, {"finish": "length"}, False),
    ("问题", {"public": False}, {"kind": "deepseek"}, False),
    ("问题", {"public": False}, {"kind": "deepseek"}, True),
    (" ", {}, {}, False),
    ("长" * 101, {}, {}, False),
])
def test_original_and_langchain_outputs_errors_and_prompts_are_equivalent(question, options, llm_options, dry_run):
    current = pipeline(FakeLLM(**llm_options), **options)
    before = original(pipeline(FakeLLM(**llm_options), **options))
    assert outcome(current, question, dry_run=dry_run) == outcome(before, question, dry_run=dry_run)
    assert current.llm.calls == before.llm.calls


def test_retrieval_public_contract_and_ranking_are_equivalent():
    current, before = pipeline(), original(pipeline())
    result = current.retrieve("APT28 信息？")
    assert semantic(result) == semantic(before.retrieve("APT28 信息？"))
    assert not any(k.startswith("_") for k in result)


def test_chain_contains_real_separate_steps_not_a_whole_answer_wrapper():
    active = pipeline()
    assert isinstance(active.chain, RunnableSequence)
    assert isinstance(active.retrieval_chain, RunnableSequence)
    assert [step.name for step in active.retrieval_chain.steps] == [
        "validate_question", "embed_query", "dense_retrieval", "bm25_retrieval",
        "rrf_fusion", "rerank_top_k", "construct_context",
    ]
    branch = active.chain.steps[-1]
    assert isinstance(branch, RunnableBranch)
    prepared = branch.default
    assert [step.name for step in prepared.steps[:2]] == ["cloud_input_policy", "token_budget"]
    generation = prepared.steps[-1].default
    assert [step.name for step in generation.steps] == [
        "model_identity", "llm_generation", "validate_generation_usage", "resolve_citations", "accept_answer",
    ]


def test_calls_are_sequential_and_cloud_policy_precedes_budget_or_generation():
    active = pipeline()
    events = []
    for obj, method, label in [(active.embedding, "embed", "embedding"), (active.dense, "retrieve", "dense"),
        (active.sparse, "retrieve", "sparse"), (active.reranker, "score", "rerank"),
        (active.budget, "measure", "budget"), (active.llm, "identity", "identity"),
        (active.llm, "generate", "generate"), (active.budget, "verify_usage", "usage")]:
        previous = getattr(obj, method)
        def recorded(*args, _previous=previous, _label=label, **kwargs):
            events.append(_label)
            return _previous(*args, **kwargs)
        setattr(obj, method, recorded)
    active.answer("问题")
    assert events == ["embedding", "dense", "sparse", "rerank", "budget", "identity", "generate", "usage"]
    active.llm.spec["kind"], active.public_corpus_verified = "deepseek", False
    events.clear()
    with pytest.raises(ValueError, match="公开 Corpus"):
        active.answer("问题")
    assert events == ["embedding", "dense", "sparse", "rerank"]


def test_overflow_is_equivalent_and_never_generates():
    class Overflow(FakeBudget):
        def measure(self, messages):
            raise ContextOverflow("Context 超限")
    current = pipeline(budget=Overflow())
    before = original(pipeline(budget=Overflow()))
    assert outcome(current, "问题") == outcome(before, "问题")
    assert current.llm.calls == before.llm.calls == []


def test_repeated_calls_do_not_leak_request_state_or_change_fingerprint_policy():
    current, before = pipeline(), original(pipeline())
    for question in ("第一次", "第二次"):
        assert outcome(current, question) == outcome(before, question)
    current.llm.generate = before.llm.generate = lambda messages: SimpleNamespace(
        answer="事实 [S1]", response_model="test-model", finish_reason="stop", system_fingerprint="changed",
        to_dict=lambda: {"answer": "事实 [S1]"})
    current.backend_fingerprint = before.backend_fingerprint = "original"
    assert outcome(current, "第三次") == outcome(before, "第三次")


def test_framework_tracing_is_disabled_even_if_enabled_by_environment_or_parent(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    active = pipeline()
    previous = active.embedding.embed
    seen = []
    def embed(texts):
        seen.append(get_tracing_context()["enabled"])
        return previous(texts)
    active.embedding.embed = embed
    with tracing_context(enabled=True):
        active.retrieve("问题")
        active.answer("问题", dry_run=True)
        assert get_tracing_context()["enabled"] is True
    assert seen == [False, False]
