"""从已校验 PDF 构建带引用元数据的 Chunk JSONL。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber

from apt_rag.chunking import ChunkingConfig, chunk_paragraphs
from apt_rag.corpus.download import sha256_file, verify_report_file
from apt_rag.ingestion.cleaning import build_paragraphs, remove_repeated_boundaries
from apt_rag.ingestion.pdf import extract_pdf_pages

PIPELINE_VERSION = "1.0"


@dataclass(frozen=True)
class ProcessingConfig:
    x_tolerance: float
    y_tolerance: float
    dedupe_tolerance: float
    boundary_scan_lines: int
    boundary_min_fraction: float
    chunking: ChunkingConfig

    @classmethod
    def from_file(cls, path: Path) -> "ProcessingConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "1.0":
            raise ValueError("processing config schema_version 必须为 1.0")
        extractor = data.get("extractor", {})
        cleaning = data.get("cleaning", {})
        chunking = data.get("chunking", {})
        if extractor.get("name") != "pdfplumber":
            raise ValueError("extractor.name 当前只支持 pdfplumber")
        if chunking.get("strategy") != "page-section-character":
            raise ValueError("chunking.strategy 当前只支持 page-section-character")
        config = cls(
            x_tolerance=float(extractor["x_tolerance"]),
            y_tolerance=float(extractor["y_tolerance"]),
            dedupe_tolerance=float(extractor["dedupe_tolerance"]),
            boundary_scan_lines=int(cleaning["boundary_scan_lines"]),
            boundary_min_fraction=float(cleaning["boundary_min_fraction"]),
            chunking=ChunkingConfig(
                chunk_size=int(chunking["chunk_size"]),
                chunk_overlap=int(chunking["chunk_overlap"]),
                min_chunk_size=int(chunking["min_chunk_size"]),
            ),
        )
        if config.x_tolerance <= 0 or config.y_tolerance <= 0 or config.dedupe_tolerance < 0:
            raise ValueError("文本抽取 tolerance 配置无效")
        if config.boundary_scan_lines <= 0:
            raise ValueError("boundary_scan_lines 必须为正整数")
        if not 0 < config.boundary_min_fraction <= 1:
            raise ValueError("boundary_min_fraction 必须位于 (0, 1]")
        return config

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunking"] = asdict(self.chunking)
        return data


def _chunk_record(
    report: dict[str, Any],
    *,
    chunk_index: int,
    page_number: int,
    page_chunk_index: int,
    section: str,
    text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": (
            f"{report['report_id']}-p{page_number:04d}-c{page_chunk_index:03d}"
        ),
        "chunk_index": chunk_index,
        "report_id": report["report_id"],
        "document_title": report["title"],
        "vendor": report["vendor"],
        "apt_group": report["apt_group"],
        "publish_date": report["publish_date"],
        "language": report["language"],
        "page": page_number,
        "section": section,
        "source_url": report["url"],
        "text": text,
        "char_count": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def process_report(
    report: dict[str, Any],
    repository_root: Path,
    config: ProcessingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_path = verify_report_file(report, repository_root)
    extracted = extract_pdf_pages(
        pdf_path,
        x_tolerance=config.x_tolerance,
        y_tolerance=config.y_tolerance,
        dedupe_tolerance=config.dedupe_tolerance,
    )
    cleaned, repeated_keys = remove_repeated_boundaries(
        extracted,
        scan_lines=config.boundary_scan_lines,
        minimum_fraction=config.boundary_min_fraction,
    )
    paragraphs = build_paragraphs(cleaned, default_section=report["title"])
    text_chunks = chunk_paragraphs(paragraphs, config.chunking)
    records = [
        _chunk_record(
            report,
            chunk_index=index,
            page_number=chunk.page_number,
            page_chunk_index=chunk.page_chunk_index,
            section=chunk.section,
            text=chunk.text,
        )
        for index, chunk in enumerate(text_chunks)
    ]
    stats = {
        "report_id": report["report_id"],
        "source_sha256": report["sha256"],
        "pages": len(extracted),
        "empty_pages": sum(not page.lines for page in cleaned),
        "raw_char_count": sum(page.raw_char_count for page in extracted),
        "cleaned_char_count": sum(
            sum(len(line.text) for line in page.lines) for page in cleaned
        ),
        "chunk_char_count_with_overlap": sum(len(record["text"]) for record in records),
        "removed_boundary_lines": sum(page.removed_boundary_lines for page in cleaned),
        "repeated_boundary_templates": sorted(repeated_keys),
        "chunks": len(records),
    }
    return records, stats


def write_jsonl(records: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return sha256_file(path)


def process_corpus(
    reports: list[dict[str, Any]],
    repository_root: Path,
    config: ProcessingConfig,
    *,
    output_directory: Path,
    overwrite: bool,
) -> dict[str, Any]:
    targets = [output_directory / f"{report['report_id']}.jsonl" for report in reports]
    summary_path = output_directory / "summary.json"
    existing = [path for path in (*targets, summary_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(f"输出已存在（如 {names}），请使用 --overwrite 明确覆盖")

    report_summaries: list[dict[str, Any]] = []
    total_pages = 0
    total_chunks = 0
    for report, target in zip(reports, targets, strict=True):
        records, stats = process_report(report, repository_root, config)
        stats["output_path"] = str(target.relative_to(repository_root)).replace("\\", "/")
        stats["output_sha256"] = write_jsonl(records, target)
        report_summaries.append(stats)
        total_pages += stats["pages"]
        total_chunks += stats["chunks"]

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "pdfplumber_version": pdfplumber.__version__,
        "config": config.as_dict(),
        "report_count": len(reports),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "reports": report_summaries,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary
