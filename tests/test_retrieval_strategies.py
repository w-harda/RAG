from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from apt_rag.evaluation import retrieval_benchmark as experiment
from apt_rag.evaluation.retrieval_benchmark import rerank_order
from apt_rag.reranking import RerankScores
from apt_rag.retrieval import BM25Retriever, DenseRetriever, RankedChunk, fuse_rrf
from apt_rag.retrieval.strategies import lexical_tokens

ROOT = Path(__file__).resolve().parents[1]


def test_lexical_tokens_preserve_identifiers_and_do_not_translate():
    assert lexical_tokens("APT28 利用 CVE-2017-6742 / stage1.exe") == [
        "apt28", "利", "用", "cve-2017-6742", "stage1", "exe",
    ]
    assert "snow" not in lexical_tokens("雪")


def test_bm25_formula_stable_ties_and_no_zero_score_candidates():
    store = BM25Retriever(["apt28 apt28", "apt28", "other"], ["a", "b", "c"], k1=1.2, b=0.75)
    hits = store.retrieve("APT28 APT28", None, 10)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_a = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (4 / 3)))
    assert hits[0].chunk_id == "a"
    assert hits[0].score == pytest.approx(expected_a)
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert store.retrieve("missing", None, 10) == []
    ties = BM25Retriever(["x", "x"], ["z", "a"], k1=1.2, b=0.75)
    assert [h.chunk_id for h in ties.retrieve("x", None, 2)] == ["z", "a"]


def test_dense_uses_fixed_normalized_vectors_and_corpus_order():
    store = DenseRetriever(np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32), ["z", "a", "c"])
    assert [h.chunk_id for h in store.retrieve("任意文本", np.asarray([1, 0], dtype=np.float32), 2)] == ["z", "a"]
    with pytest.raises(ValueError, match="维数"):
        store.retrieve("q", None, 2)


def test_rrf_uses_ranks_not_incomparable_scores_and_falls_back_without_bm25():
    a = [RankedChunk("c", 100), RankedChunk("b", 1)]
    b = [RankedChunk("b", 0.01), RankedChunk("c", 0.001)]
    hits = fuse_rrf([a, b], ["b", "c", "other"], rank_constant=60, weights=[1, 1], top_k=2)
    assert [h.chunk_id for h in hits] == ["b", "c"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    fallback = fuse_rrf([a, []], ["b", "c", "other"], rank_constant=60, weights=[1, 1], top_k=2)
    assert [h.chunk_id for h in fallback] == ["c", "b"]
    with pytest.raises(ValueError, match="重复"):
        fuse_rrf([[a[0], a[0]]], ["b", "c"], rank_constant=60, weights=[1], top_k=2)


def test_reranker_only_reorders_existing_candidates():
    candidates = [RankedChunk("b", 0.2), RankedChunk("a", 0.1)]
    ordered = rerank_order(candidates, [-2, -2], ["a", "b", "c"])
    assert [h.chunk_id for h in ordered] == ["a", "b"]
    with pytest.raises(ValueError, match="有限分数"):
        rerank_order(candidates, [float("nan"), 1], ["a", "b"])


def test_checkpoint_reuse_ground_truth_exclusion_and_audit(monkeypatch, tmp_path):
    config = experiment.load_config(ROOT / "configs/retrieval_benchmark.json")
    config["results_directory"], config["reranker_cache_path"] = "results", "cache.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    benchmark = {"questions": [{"id": "q001", "category": "ioc", "question": "APT28 信息？",
                               "expected_answer": "不要向 Reranker 泄露 Ground Truth",
                               "evidence": [{"chunk_id": "a", "relevance": 3}]}]}
    chunks = [{"chunk_id": key, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "report_id": "test", "document_title": "测试", "vendor": "CISA", "page": 1,
               "section": None, "source_url": "https://example.com"} for key, text in
              [("a", "APT28 evidence"), ("b", "Other evidence")]]
    inputs = {"benchmark": benchmark, "chunks": chunks, "ids": ["a", "b"],
              "vectors": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
              "queries": np.asarray([[1, 0]], dtype=np.float32),
              "controls": {"benchmark_fingerprint": experiment.fingerprint(benchmark), "chunk_count": 2}}
    monkeypatch.setattr(experiment, "prepare_inputs", lambda root, settings: inputs)

    class FakeReranker:
        calls = []

        def score(self, query, passages):
            assert query == "APT28 信息？"
            assert passages == ["APT28 evidence", "Other evidence"]
            assert benchmark["questions"][0]["expected_answer"] not in query + "".join(passages)
            self.calls.append((query, passages))
            return RerankScores([1, 2], [5, 5], [5, 5])

    fake = FakeReranker()
    result = experiment.run_benchmark(tmp_path, config_path, reranker=fake)
    assert result["queries"][0]["rankings"]["hybrid_reranker"][0]["chunk_id"] == "b"
    assert len(fake.calls) == 1
    assert experiment.run_benchmark(tmp_path, config_path, reranker=fake) == result
    assert len(fake.calls) == 1
    bad = deepcopy(result)
    bad["queries"][0]["reranking"]["scores"][0] = float("nan")
    with pytest.raises(ValueError, match="非法"):
        experiment.summarize_results(bad, benchmark, config)
    bad = deepcopy(result)
    bad["queries"][0]["rankings"]["hybrid_reranker"][0]["chunk_id"] = "unknown"
    with pytest.raises(ValueError, match="重算"):
        experiment.summarize_results(bad, benchmark, config)
    # 模拟写入排名后丢失摘要，只重建摘要，不重新推理。
    (tmp_path / "results/summary.json").unlink()
    experiment.run_benchmark(tmp_path, config_path, reranker=fake)
    assert (tmp_path / "results/summary.json").exists()
    assert len(fake.calls) == 1
    # 仅移除测试生成的最终记录，验证逐题检查点能避免重复打分。
    (tmp_path / "results/rankings.json").unlink()
    replayed = experiment.run_benchmark(tmp_path, config_path, reranker=fake)
    assert replayed["queries"][0]["reranker_scores_reused"] is True
    assert len(fake.calls) == 1


def test_metrics_use_final_top_ten_not_candidate_pool(monkeypatch, tmp_path):
    config = experiment.load_config(ROOT / "configs/retrieval_benchmark.json")
    config["results_directory"], config["reranker_cache_path"] = "results", "cache.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    ids = [f"c{i:02d}" for i in range(12)]
    benchmark = {"questions": [{"id": "q001", "category": "ioc", "question": "missing",
                               "evidence": [{"chunk_id": ids[11], "relevance": 3}]}]}
    chunks = [{"chunk_id": key, "text": "text", "text_sha256": "hash", "report_id": "test",
               "document_title": "测试", "vendor": "CISA", "page": 1,
               "section": None, "source_url": "https://example.com"} for key in ids]
    inputs = {"benchmark": benchmark, "chunks": chunks, "ids": ids,
              "vectors": np.tile(np.asarray([[1, 0]], dtype=np.float32), (12, 1)),
              "queries": np.asarray([[1, 0]], dtype=np.float32),
              "controls": {"benchmark_fingerprint": experiment.fingerprint(benchmark), "chunk_count": 12}}
    monkeypatch.setattr(experiment, "prepare_inputs", lambda root, settings: inputs)

    class PreservingReranker:
        def score(self, query, passages):
            return RerankScores(list(range(12, 0, -1)), [5] * 12, [5] * 12)

    result = experiment.run_benchmark(tmp_path, path, reranker=PreservingReranker())
    summary = experiment.summarize_results(result, benchmark, config)
    for row in summary["results"]:
        assert row["candidate_recall"] == 1
        assert row["metrics"]["mrr@10"] == 0
        assert row["metrics"]["recall@10"] == 0


def test_real_records_recompute_without_model_or_processed_data():
    config = experiment.load_config(ROOT / "configs/retrieval_benchmark.json")
    directory = ROOT / config["results_directory"]
    result = json.loads((directory / "rankings.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "data/benchmark/apt_qa_v1.json").read_text(encoding="utf-8"))
    actual = experiment.summarize_results(result, benchmark, config)
    assert actual == json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert actual["scored_pairs"] == 600
    assert actual["results"][1]["candidate_recall"] == actual["results"][2]["candidate_recall"]
