"""校验 Phase 2 生成的 Chunk 数据集。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from apt_rag.corpus.download import sha256_file

CHUNK_ID_PATTERN = re.compile(r"^(?P<report_id>.+)-p(?P<page>\d{4})-c(?P<page_index>\d{3})$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_FIELDS = {
    "chunk_id",
    "chunk_index",
    "report_id",
    "document_title",
    "vendor",
    "apt_group",
    "publish_date",
    "language",
    "page",
    "section",
    "source_url",
    "text",
    "char_count",
    "text_sha256",
}


class ProcessedDataValidationError(ValueError):
    """处理结果不满足 Chunk 数据契约。"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessedDataValidationError(
                    f"{path.name}:{line_number} 不是有效 JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ProcessedDataValidationError(
                    f"{path.name}:{line_number} 必须是 JSON object"
                )
            records.append(record)
    return records


def validate_processed_corpus(
    reports: list[dict[str, Any]],
    output_directory: Path,
    *,
    chunk_size: int,
) -> dict[str, int]:
    summary_path = output_directory / "summary.json"
    if not summary_path.is_file():
        raise ProcessedDataValidationError("缺少 summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_reports = {item["report_id"]: item for item in summary.get("reports", [])}

    errors: list[str] = []
    seen_ids: set[str] = set()
    total_chunks = 0
    total_pages = 0
    for report in reports:
        report_id = report["report_id"]
        path = output_directory / f"{report_id}.jsonl"
        if not path.is_file():
            errors.append(f"缺少输出文件：{path.name}")
            continue
        records = _load_jsonl(path)
        total_chunks += len(records)
        report_summary = summary_reports.get(report_id)
        if report_summary is None:
            errors.append(f"summary 缺少报告：{report_id}")
            continue
        total_pages += int(report_summary["pages"])
        if report_summary.get("chunks") != len(records):
            errors.append(f"{report_id} 的 Chunk 计数与 summary 不一致")
        if report_summary.get("output_sha256") != sha256_file(path):
            errors.append(f"{report_id} 的 JSONL SHA-256 与 summary 不一致")

        page_indexes: dict[int, list[int]] = {}
        for expected_index, record in enumerate(records):
            prefix = f"{path.name}:{expected_index + 1}"
            if set(record) != REQUIRED_FIELDS:
                errors.append(f"{prefix} 字段集合不符合契约")
                continue
            if record["chunk_index"] != expected_index:
                errors.append(f"{prefix} chunk_index 不连续")
            if record["report_id"] != report_id:
                errors.append(f"{prefix} report_id 不匹配")
            expected_metadata = {
                "document_title": report["title"],
                "vendor": report["vendor"],
                "apt_group": report["apt_group"],
                "publish_date": report["publish_date"],
                "language": report["language"],
                "source_url": report["url"],
            }
            if any(record[field] != value for field, value in expected_metadata.items()):
                errors.append(f"{prefix} 来源 Metadata 与 manifest 不一致")
            if not isinstance(record["page"], int) or record["page"] < 1:
                errors.append(f"{prefix} page 无效")
            text = record["text"]
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{prefix} text 为空")
                continue
            if record["char_count"] != len(text) or len(text) > chunk_size:
                errors.append(f"{prefix} char_count 或 Chunk 上限无效")
            actual_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if not SHA256_PATTERN.fullmatch(record["text_sha256"]) or record["text_sha256"] != actual_text_hash:
                errors.append(f"{prefix} text_sha256 无效")
            match = CHUNK_ID_PATTERN.fullmatch(record["chunk_id"])
            if (
                match is None
                or match.group("report_id") != report_id
                or int(match.group("page")) != record["page"]
            ):
                errors.append(f"{prefix} chunk_id 无效")
            else:
                page_indexes.setdefault(record["page"], []).append(
                    int(match.group("page_index"))
                )
            if record["chunk_id"] in seen_ids:
                errors.append(f"{prefix} chunk_id 重复")
            seen_ids.add(record["chunk_id"])

        for page, indexes in page_indexes.items():
            if indexes != list(range(len(indexes))):
                errors.append(f"{report_id} 第 {page} 页的页内 Chunk 序号不连续")

    if summary.get("report_count") != len(reports):
        errors.append("summary.report_count 不一致")
    if summary.get("total_chunks") != total_chunks:
        errors.append("summary.total_chunks 不一致")
    if summary.get("total_pages") != total_pages:
        errors.append("summary.total_pages 不一致")
    if errors:
        raise ProcessedDataValidationError("处理结果校验失败：\n- " + "\n- ".join(errors))
    return {"reports": len(reports), "pages": total_pages, "chunks": total_chunks}
