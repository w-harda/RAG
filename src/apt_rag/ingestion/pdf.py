"""使用 pdfplumber 逐页抽取 PDF 文本。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pdfplumber


@dataclass(frozen=True)
class ExtractedLine:
    text: str
    font_size: float | None


@dataclass(frozen=True)
class ExtractedPage:
    """保留 PDF 物理页码的原始文本行。"""

    page_number: int
    lines: tuple[ExtractedLine, ...]
    raw_char_count: int
    body_font_size: float | None


def normalize_line(line: str) -> str:
    normalized = unicodedata.normalize("NFKC", line).replace("\u00ad", "")
    return " ".join(normalized.split())


def _is_visual_noise(line: str) -> bool:
    """过滤图表中按旋转路径抽取出的孤立字母。"""

    tokens = line.split()
    if len(tokens) == 1 and len(tokens[0]) == 1 and tokens[0].isalnum():
        return True
    if len(tokens) >= 2:
        single_alnum = sum(len(token) == 1 and token.isalnum() for token in tokens)
        if single_alnum / len(tokens) >= 0.7:
            return True
    return False


def extract_pdf_pages(
    path: Path,
    *,
    x_tolerance: float,
    y_tolerance: float,
    dedupe_tolerance: float,
) -> list[ExtractedPage]:
    """按 PDF 物理页顺序抽取文本，页码从 1 开始。"""

    pages: list[ExtractedPage] = []
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            deduped = page.dedupe_chars(
                tolerance=dedupe_tolerance,
                extra_attrs=("fontname", "size"),
            )
            extracted_lines = deduped.extract_text_lines(
                strip=True,
                return_chars=True,
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
            )
            lines: list[ExtractedLine] = []
            page_sizes: list[float] = []
            raw_text_parts: list[str] = []
            for extracted_line in extracted_lines:
                raw_text = extracted_line.get("text", "")
                raw_text_parts.append(raw_text)
                normalized = normalize_line(raw_text)
                if not normalized or _is_visual_noise(normalized):
                    continue
                sizes = [
                    float(char["size"])
                    for char in extracted_line.get("chars", [])
                    if char.get("text", "").strip() and char.get("size") is not None
                ]
                line_size = median(sizes) if sizes else None
                page_sizes.extend(sizes)
                lines.append(ExtractedLine(text=normalized, font_size=line_size))
            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    lines=tuple(lines),
                    raw_char_count=sum(len(part) for part in raw_text_parts),
                    body_font_size=median(page_sizes) if page_sizes else None,
                )
            )
    return pages
