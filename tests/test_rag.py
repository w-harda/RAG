"""注入假组件测试完整编排；不加载神经模型、不读取密钥、不联网。"""

import hashlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apt_rag.generation.providers import GenerationResult, OllamaProvider
from apt_rag.rag.budget import ContextOverflow, TokenBudget
from apt_rag.rag.context import ABSTENTION, CitationError, construct_context, resolve_citations
from apt_rag.rag.pipeline import RAGPipeline, RAGResponseError
from apt_rag.rag.runtime import FaissRetriever, PreparationOnlyLLM, build_pipeline, load_settings
from apt_rag.reranking import RerankScores
from apt_rag.retrieval import BM25Retriever, DenseRetriever

ROOT = Path(__file__).resolve().parents[1]


def chunks():
    return [{"chunk_id": key, "report_id": "report", "document_title": "真实报告", "vendor": "CISA",
             "page": index, "section": "章节", "source_url": "https://example.com/report",
             "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for index, (key, text) in enumerate([("a", "APT28 first evidence"), ("b", "other evidence")], 1)]


class FakeLLM:
    def __init__(self, answer="事实 [S1, S2]", *, finish="stop", kind="ollama"):
        self.spec = {"model": "test-model", "kind": kind}
        self.answer, self.finish, self.calls = answer, finish, []

    def identity(self):
        return {"model": "test-model"}

    def generate(self, messages):
        self.calls.append(messages)
        return GenerationResult(self.answer, "test-model", self.finish, 0.1, 50, 10)


class FakeBudget:
    def measure(self, messages):
        return {"input_tokens": 50, "safety_margin": 5, "context_window_tokens": 100, "max_output_tokens": 20}

    def verify_usage(self, result, budget):
        pass


def pipeline(llm=None, budget=None, *, empty=False, public=True):
    texts = chunks()
    vectors = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    embedding = SimpleNamespace(embed=lambda questions: np.asarray([[1, 0]], dtype=np.float32))
    dense = DenseRetriever(vectors, [c["chunk_id"] for c in texts])
    sparse = BM25Retriever([c["text"] for c in texts], [c["chunk_id"] for c in texts], k1=1.2, b=0.75)
    if empty:
        dense = sparse = SimpleNamespace(retrieve=lambda *args: [])
    reranker = SimpleNamespace(score=lambda query, docs: RerankScores([float(i) for i in range(len(docs))],
                                                                                    [5] * len(docs), [5] * len(docs)))
    return RAGPipeline(chunks=texts, embedding=embedding, dense=dense, sparse=sparse, reranker=reranker,
                       llm=llm or FakeLLM(), budget=budget or FakeBudget(),
                       prompt={"system": "仅根据片段回答", "user_template": "{question}\n{context}"},
                       retrieval_config={"candidate_k": 2, "rrf": {"rank_constant": 60, "weights": [1, 1]}},
                       top_k=2, max_question_characters=100, public_corpus_verified=public)


def test_full_pipeline_reorders_sources_and_does_not_use_ground_truth():
    active = pipeline()
    result = active.answer("APT28 信息？")
    assert result["status"] == "answered"
    assert result["sources"][0]["chunk_id"] == "b"
    assert result["citations"][0]["page"] == 2
    assert result["citations"][0]["source_url"] == chunks()[1]["source_url"]
    assert "other evidence" in active.llm.calls[0][1]["content"]
    assert not any("expected_answer" in m["content"] for m in active.llm.calls[0])


@pytest.mark.parametrize("answer", ["事实 [S99]", "事实 [S0]", "事实 [S1 S2]", "事实 [S1] 和 [S999", "未引用的事实", "事实 [S1] [s999]"])
def test_bad_citations_are_not_accepted_as_answers(answer):
    with pytest.raises(RAGResponseError) as error:
        pipeline(FakeLLM(answer)).answer("APT28 信息？")
    assert error.value.record["status"] == "rejected"
    assert "answer" not in error.value.record
    assert error.value.record["generation"]["answer"] == answer


def test_context_overflow_stops_before_generation_and_dry_run_does_not_generate():
    class Overflow(FakeBudget):
        def measure(self, messages):
            raise ContextOverflow("Context 超限")

    llm = FakeLLM()
    with pytest.raises(ContextOverflow):
        pipeline(llm, Overflow()).answer("APT28 信息？")
    assert llm.calls == []
    assert pipeline(llm).answer("APT28 信息？", dry_run=True)["status"] == "prepared_without_generation"
    assert llm.calls == []


def test_no_evidence_and_canonical_abstention_do_not_fabricate_citations():
    llm = FakeLLM()
    result = pipeline(llm, empty=True).answer("问题")
    assert result["answer"] == ABSTENTION and result["citations"] == [] and llm.calls == []
    result = pipeline(FakeLLM(ABSTENTION)).answer("问题")
    assert result["abstained"] and result["citations"] == []


def test_private_corpus_never_reaches_cloud_and_cloud_requires_explicit_authority():
    llm = FakeLLM(kind="deepseek")
    with pytest.raises(ValueError, match="公开 Corpus"):
        pipeline(llm, public=False).answer("问题")
    assert llm.calls == []
    with pytest.raises(ValueError, match="allow_cloud"):
        build_pipeline(ROOT, ROOT / "configs/rag.json")


def test_truncated_generation_is_preserved_but_not_accepted():
    with pytest.raises(RAGResponseError, match="未正常完成"):
        pipeline(FakeLLM(finish="length")).answer("问题")


def test_context_is_complete_and_rejects_corrupt_text():
    data = chunks()
    context = construct_context("{question}", data, {"system": "s", "user_template": "{question}\n{context}"})
    assert all(c["text"] in context["messages"][1]["content"] for c in data)
    assert "{question}" in context["messages"][1]["content"]
    data[0]["text"] += "tamper"
    with pytest.raises(ValueError, match="指纹"):
        construct_context("q", data, {"system": "s", "user_template": "{context}"})


def test_shared_window_and_actual_usage_cannot_silently_overflow():
    counter = TokenBudget.__new__(TokenBudget)
    counter.spec = {"kind": "deepseek"}
    counter.config = {"context_window_tokens": 100, "token_safety_margin": 10}
    counter.output_tokens = 20
    counter.token_spec = {"tokenizer_sha256": "hash"}
    counter.tokenizer = SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=list(range(70))))
    budget = counter.measure([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    counter.verify_usage(GenerationResult("ok", "model", "stop", 1, 80, 10), budget)
    with pytest.raises(ContextOverflow):
        counter.verify_usage(GenerationResult("ok", "model", "stop", 1, 81, 10), budget)
    counter.tokenizer = SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=list(range(71))))
    with pytest.raises(ContextOverflow):
        counter.measure([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])


def test_faiss_boundary_ties_are_ordered_using_complete_corpus():
    from apt_rag.vectorstore.base import SearchHit
    calls = []

    def search(vector, k):
        calls.append(k)
        return [SearchHit(key, 1.0, {}) for key in ["c", "b", "a", "d"][:k]]

    hits = FaissRetriever(SimpleNamespace(search=search), ["a", "b", "c", "d"]).retrieve("q", None, 2)
    assert [h.chunk_id for h in hits] == ["a", "b"] and calls == [3, 4]


def test_strict_ollama_context_flags_do_not_change_phase6_defaults():
    class Recorder:
        requests = []

        def request(self, *args, **kwargs):
            self.requests.append(kwargs["json"])
            return SimpleNamespace(status_code=200, json=lambda: {"done": True, "done_reason": "stop", "model": "test",
                "message": {"content": "ok"}, "prompt_eval_count": 5, "eval_count": 2})

    spec = {"model": "test", "host_env": "TEST_RAG_OLLAMA_HOST", "default_host": "http://127.0.0.1:11434",
            "keep_alive": "1m", "num_ctx": 8192, "seed": 42}
    settings = {"timeout_seconds": 10, "temperature": 0, "max_output_tokens": 768}
    client = Recorder()
    OllamaProvider(spec, settings, client=client).generate([])
    assert "truncate" not in client.requests[-1]
    OllamaProvider(spec, {**settings, "strict_context": True}, client=client).generate([])
    assert client.requests[-1]["truncate"] is False and client.requests[-1]["shift"] is False


def test_selected_config_loads_without_secrets_or_models():
    config, decision, _ = load_settings(ROOT, ROOT / "configs/rag.json")
    assert config["context_window_tokens"] == 8192 and decision["phase"] == 8


def test_cloud_preparation_instance_cannot_generate_or_read_keys():
    provider = PreparationOnlyLLM({"kind": "deepseek"})
    assert provider.client is None
    with pytest.raises(ValueError, match="不允许生成"):
        provider.generate([])


def test_empty_and_excessive_question_stop_before_generation():
    llm = FakeLLM()
    for question in (" ", "长" * 101):
        with pytest.raises(ValueError, match="问题"):
            pipeline(llm).answer(question)
    assert llm.calls == []


def test_embedding_query_guard_preserves_old_default_without_loading_model(monkeypatch):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def get_embedding_dimension(self):
            return 2

        def tokenizer(self, texts, **kwargs):
            return {"input_ids": [list(range(len(text))) for text in texts]}

        def encode(self, texts, **kwargs):
            return np.asarray([[1, 0]] * len(texts), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))
    provider_class = runpy.run_path(str(ROOT / "src/apt_rag/embedding/sentence_transformer.py"))["SentenceTransformerProvider"]
    parameters = {"revision": "test", "device": "cpu", "batch_size": 1, "max_seq_length": 3}
    old = provider_class("test", **parameters)
    assert old.embed(["long"]).shape == (1, 2)
    strict = provider_class("test", **parameters, local_files_only=True, reject_truncated_inputs=True)
    assert strict._model.kwargs["local_files_only"]
    with pytest.raises(ValueError, match="禁止静默截断"):
        strict.embed(["long"])


def test_saved_real_integration_and_smoke_can_be_checked_offline():
    validator = runpy.run_path(str(ROOT / "scripts/validate_rag_results.py"))
    count, smoke = validator["validate_records"](ROOT, ROOT / "configs/rag.json")
    assert count == 20 and smoke >= 1
