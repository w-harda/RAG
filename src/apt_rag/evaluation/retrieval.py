"""与检索后端无关的 Ranking 指标。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevance_by_question: Mapping[str, Mapping[str, int]],
    *,
    top_k: Sequence[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """计算宏平均 Recall@K、MRR 和 nDCG@max(K)。"""
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k 必须包含正整数")
    maximum_k = max(top_k)
    per_query: list[dict[str, Any]] = []

    for question_id, relevance in relevance_by_question.items():
        if not relevance:
            raise ValueError(f"{question_id} 没有 Ground Truth Evidence")
        ranking = list(rankings.get(question_id, ()))
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"{question_id} 的排名包含重复 Chunk")
        if any(value not in {1, 2, 3} for value in relevance.values()):
            raise ValueError("relevance 必须为 1、2 或 3")
        relevant_ids = set(relevance)
        recalls = {
            f"recall@{k}": len(relevant_ids.intersection(ranking[:k])) / len(relevant_ids)
            for k in top_k
        }
        first_rank = next(
            (index for index, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant_ids),
            None,
        )
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
        actual = [relevance.get(chunk_id, 0) for chunk_id in ranking[:maximum_k]]
        ideal = sorted(relevance.values(), reverse=True)[:maximum_k]
        ideal_dcg = _dcg(ideal)
        ndcg = 0.0 if ideal_dcg == 0 else _dcg(actual) / ideal_dcg
        per_query.append(
            {
                "question_id": question_id,
                **recalls,
                "reciprocal_rank": reciprocal_rank,
                f"ndcg@{maximum_k}": ndcg,
                "first_relevant_rank": first_rank,
                "retrieved_chunk_ids": ranking[:maximum_k],
            }
        )

    if not per_query:
        raise ValueError("没有可评测问题")
    metric_names = [f"recall@{k}" for k in top_k] + [
        "reciprocal_rank",
        f"ndcg@{maximum_k}",
    ]
    summary = {
        metric: sum(float(item[metric]) for item in per_query) / len(per_query)
        for metric in metric_names
    }
    summary["mrr"] = summary.pop("reciprocal_rank")
    return summary, per_query
