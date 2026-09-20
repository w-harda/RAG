"""校验 Phase 3 APT QA Benchmark 及其 Ground Truth Evidence。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_FIELDS = {
    "schema_version",
    "benchmark_id",
    "corpus_id",
    "created_at",
    "question_language",
    "questions",
}
QUESTION_FIELDS = {
    "id",
    "category",
    "question",
    "expected_answer",
    "evidence",
    "difficulty",
}
EVIDENCE_FIELDS = {"report_id", "page", "chunk_id", "relevance"}
CATEGORIES = {"background", "ttp", "attack_technique", "ioc"}
DIFFICULTIES = {"easy", "medium", "hard"}
QUESTION_ID_PATTERN = re.compile(r"^q\d{3}$")


class BenchmarkValidationError(ValueError):
    """Benchmark 或证据引用不满足数据契约。"""


def load_benchmark(path: Path) -> dict[str, Any]:
    """读取 Benchmark JSON，失败时提供明确错误。"""
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError(f"{path} 不是有效 JSON") from exc
    if not isinstance(benchmark, dict):
        raise BenchmarkValidationError("Benchmark 顶层必须是 JSON object")
    return benchmark


def _load_chunks(chunks_directory: Path) -> dict[str, dict[str, Any]]:
    if not chunks_directory.is_dir():
        raise BenchmarkValidationError(
            f"找不到处理后语料目录：{chunks_directory}；请先运行 Phase 2 处理流程"
        )

    chunks: dict[str, dict[str, Any]] = {}
    for path in sorted(chunks_directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkValidationError(
                        f"{path.name}:{line_number} 不是有效 JSON"
                    ) from exc
                chunk_id = chunk.get("chunk_id")
                if not isinstance(chunk_id, str):
                    raise BenchmarkValidationError(
                        f"{path.name}:{line_number} 缺少有效 chunk_id"
                    )
                if chunk_id in chunks:
                    raise BenchmarkValidationError(f"Chunk ID 重复：{chunk_id}")
                chunks[chunk_id] = chunk
    if not chunks:
        raise BenchmarkValidationError(f"{chunks_directory} 中没有 JSONL Chunk")
    return chunks


def validate_benchmark(
    benchmark: dict[str, Any],
    manifest: dict[str, Any],
    *,
    chunks_directory: Path | None = None,
) -> dict[str, Any]:
    """校验题目结构；传入 Chunk 目录时同时校验证据可追溯性。"""
    errors: list[str] = []
    if set(benchmark) != BENCHMARK_FIELDS:
        errors.append("Benchmark 顶层字段集合不符合契约")
    if benchmark.get("schema_version") != "1.0":
        errors.append("schema_version 必须为 1.0")
    if benchmark.get("corpus_id") != manifest.get("corpus_id"):
        errors.append("corpus_id 与报告 manifest 不一致")

    reports = {
        report["report_id"]: report for report in manifest.get("reports", [])
    }
    chunks = _load_chunks(chunks_directory) if chunks_directory is not None else None
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        errors.append("questions 必须是数组")
        questions = []
    elif len(questions) < 20:
        errors.append("Benchmark 至少需要 20 个问题")

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    referenced_reports: set[str] = set()
    categories: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()

    for index, question in enumerate(questions, start=1):
        prefix = f"questions[{index - 1}]"
        if not isinstance(question, dict) or set(question) != QUESTION_FIELDS:
            errors.append(f"{prefix} 字段集合不符合契约")
            continue
        question_id = question["id"]
        expected_id = f"q{index:03d}"
        if not isinstance(question_id, str) or not QUESTION_ID_PATTERN.fullmatch(question_id):
            errors.append(f"{prefix}.id 格式无效")
        elif question_id != expected_id:
            errors.append(f"{prefix}.id 应为 {expected_id}")
        if question_id in seen_ids:
            errors.append(f"问题 ID 重复：{question_id}")
        seen_ids.add(question_id)

        question_text = question["question"]
        if not isinstance(question_text, str) or not question_text.strip():
            errors.append(f"{prefix}.question 不能为空")
        elif question_text in seen_questions:
            errors.append(f"问题文本重复：{question_id}")
        seen_questions.add(question_text)
        if not isinstance(question["expected_answer"], str) or not question["expected_answer"].strip():
            errors.append(f"{prefix}.expected_answer 不能为空")

        category = question["category"]
        difficulty = question["difficulty"]
        if category not in CATEGORIES:
            errors.append(f"{prefix}.category 无效：{category}")
        else:
            categories[category] += 1
        if difficulty not in DIFFICULTIES:
            errors.append(f"{prefix}.difficulty 无效：{difficulty}")
        else:
            difficulties[difficulty] += 1

        evidence_items = question["evidence"]
        if not isinstance(evidence_items, list) or not evidence_items:
            errors.append(f"{prefix}.evidence 至少需要一条记录")
            continue
        seen_evidence: set[str] = set()
        for evidence_index, evidence in enumerate(evidence_items):
            evidence_prefix = f"{prefix}.evidence[{evidence_index}]"
            if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_FIELDS:
                errors.append(f"{evidence_prefix} 字段集合不符合契约")
                continue
            report_id = evidence["report_id"]
            chunk_id = evidence["chunk_id"]
            page = evidence["page"]
            relevance = evidence["relevance"]
            if report_id not in reports:
                errors.append(f"{evidence_prefix}.report_id 不在 manifest 中")
            else:
                referenced_reports.add(report_id)
            if not isinstance(page, int) or page < 1:
                errors.append(f"{evidence_prefix}.page 无效")
            if not isinstance(relevance, int) or relevance not in {1, 2, 3}:
                errors.append(f"{evidence_prefix}.relevance 必须为 1、2 或 3")
            if chunk_id in seen_evidence:
                errors.append(f"{prefix} 重复引用 Chunk：{chunk_id}")
            seen_evidence.add(chunk_id)
            if chunks is not None:
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    errors.append(f"{evidence_prefix}.chunk_id 不存在：{chunk_id}")
                elif chunk.get("report_id") != report_id or chunk.get("page") != page:
                    errors.append(f"{evidence_prefix} 与 Chunk Metadata 不一致")

    missing_categories = CATEGORIES - categories.keys()
    if missing_categories:
        errors.append("缺少问题类别：" + ", ".join(sorted(missing_categories)))
    if errors:
        raise BenchmarkValidationError("Benchmark 校验失败：\n- " + "\n- ".join(errors))

    return {
        "questions": len(questions),
        "categories": dict(sorted(categories.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "referenced_reports": len(referenced_reports),
        "evidence_verified": chunks is not None,
    }
