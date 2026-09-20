from __future__ import annotations

import numpy as np
import pytest

from apt_rag.evaluation.embedding_benchmark import rank_embeddings
from apt_rag.evaluation.retrieval import evaluate_rankings


def test_rank_embeddings_uses_cosine_dot_product_order() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)

    rankings, scores = rank_embeddings(
        corpus,
        queries,
        ["c1", "c2", "c3"],
        ["q001"],
        maximum_k=3,
    )

    assert rankings["q001"] == ["c1", "c3", "c2"]
    assert scores["q001"] == pytest.approx([1.0, 0.8, 0.0])


def test_retrieval_metrics_include_graded_ndcg() -> None:
    metrics, per_query = evaluate_rankings(
        {"q001": ["secondary", "irrelevant", "primary"]},
        {"q001": {"primary": 3, "secondary": 1}},
        top_k=(1, 3),
    )

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 1.0
    assert 0.0 < metrics["ndcg@3"] < 1.0
    assert per_query[0]["first_relevant_rank"] == 1


def test_full_mrr_includes_evidence_below_top_ten() -> None:
    ranking = [f"irrelevant-{i}" for i in range(10)] + ["evidence"]
    metrics, _ = evaluate_rankings({"q": ranking}, {"q": {"evidence": 3}}, top_k=(1, 10))
    assert metrics["mrr"] == pytest.approx(1 / 11)
    assert metrics["recall@10"] == 0
    assert metrics["ndcg@10"] == 0


def test_duplicate_rankings_are_rejected() -> None:
    with pytest.raises(ValueError, match="重复"):
        evaluate_rankings({"q": ["c", "c"]}, {"q": {"c": 3}}, top_k=(1, 3))
