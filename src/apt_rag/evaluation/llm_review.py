"""验证逐题辅助复核，并将语义评分与自动格式检查分开。"""

from __future__ import annotations

from typing import Any

import numpy as np

from apt_rag.generation.providers import fingerprint


def summarize_review(
    runs: list[dict[str, Any]], rubric: dict[str, Any], review: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if review["rubric_fingerprint"] != fingerprint(rubric):
        raise ValueError("复核使用的事实清单指纹不匹配")
    if not review.get("reviewer") or review.get("independent_human_review") is not False:
        raise ValueError("当前辅助复核必须标明非独立人工评审")
    expected = {(run["provider_id"], row["question_id"]): row
                for run in runs for row in run["answers"]}
    reviewed = {(row["provider_id"], row["question_id"]): row for row in review["reviews"]}
    if len(reviewed) != len(review["reviews"]) or reviewed.keys() != expected.keys():
        raise ValueError("复核必须一一覆盖所有模型的全部问题，不能重复或漏题")
    scores: dict[str, list[dict[str, float | bool | None]]] = {}
    for key, row in reviewed.items():
        answer = expected[key]
        facts = rubric["required_facts"][key[1]]
        if row["answer_sha256"] != answer["answer_sha256"]:
            raise ValueError("复核绑定的答案指纹不匹配")
        covered = row["fact_coverage"]
        if len(covered) != len(facts) or any(type(v) is not bool for v in covered):
            raise ValueError("事实覆盖必须按清单顺序逐项填写布尔值")
        pairs = row["citation_supported"]
        if len(pairs) != len(answer["citation_check"]["occurrences"]) or any(type(v) is not bool for v in pairs):
            raise ValueError("必须逐次评估所有来源标签是否支持相邻断言")
        if not row.get("notes") or not isinstance(row["unsupported_claims"], list):
            raise ValueError("需要复核依据和无证据断言清单")
        if any(not isinstance(claim, str) or not claim for claim in row["unsupported_claims"]):
            raise ValueError("无证据断言必须提供具体文本或解释")
        scores.setdefault(key[0], []).append({
            "required_fact_coverage": sum(covered) / len(covered),
            "fully_covered": all(covered),
            "semantic_citation_support": sum(pairs) / len(pairs) if pairs else 0.0,
            "has_unsupported_claim": bool(row["unsupported_claims"]),
        })
    return {provider: {
        "status": "provisional_evidence_assisted_review",
        "independent_human_review": False,
        "question_count": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in (
            "required_fact_coverage", "fully_covered", "semantic_citation_support", "has_unsupported_claim",
        )},
    } for provider, rows in scores.items()}
