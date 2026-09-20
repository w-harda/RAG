"""面向技术报告的确定性文本清洗。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from apt_rag.ingestion.pdf import ExtractedLine, ExtractedPage

NUMBER_PATTERN = re.compile(r"\d+")
BULLET_PATTERN = re.compile(r"^(?:[•●▪◦*-]|(?:\d+|[A-Za-z])[.)])\s+")
NUMBERED_HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*)\s+[A-Z]")
LETTER_PATTERN = re.compile(r"[A-Za-z]")
COMMON_HEADINGS = {
    "acknowledgements",
    "appendix",
    "background",
    "conclusion",
    "conclusions",
    "detection",
    "detection opportunities",
    "executive summary",
    "indicators of compromise",
    "introduction",
    "key findings",
    "key judgments",
    "mitigations",
    "overview",
    "recommendations",
    "references",
    "summary",
    "technical analysis",
    "technical details",
}


@dataclass(frozen=True)
class CleanedPage:
    page_number: int
    lines: tuple[ExtractedLine, ...]
    removed_boundary_lines: int
    raw_char_count: int
    body_font_size: float | None


@dataclass(frozen=True)
class Paragraph:
    page_number: int
    section: str
    text: str


def boundary_key(line: str) -> str:
    lowered = line.casefold()
    previous = None
    while lowered != previous:
        previous = lowered
        lowered = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", lowered)
    lowered = NUMBER_PATTERN.sub("#", lowered)
    return " ".join(lowered.split())


def find_repeated_boundary_keys(
    pages: list[ExtractedPage],
    *,
    scan_lines: int,
    minimum_fraction: float,
) -> set[str]:
    """查找出现在多页顶部或底部的页眉、页脚模板。"""

    populated = [page for page in pages if page.lines]
    if not populated:
        return set()
    threshold = max(3, math.ceil(len(populated) * minimum_fraction))
    counts: Counter[str] = Counter()
    for page in populated:
        boundary = (*page.lines[:scan_lines], *page.lines[-scan_lines:])
        counts.update({boundary_key(line.text) for line in boundary})
    return {key for key, count in counts.items() if key and count >= threshold}


def remove_repeated_boundaries(
    pages: list[ExtractedPage],
    *,
    scan_lines: int,
    minimum_fraction: float,
) -> tuple[list[CleanedPage], set[str]]:
    repeated = find_repeated_boundary_keys(
        pages,
        scan_lines=scan_lines,
        minimum_fraction=minimum_fraction,
    )
    cleaned: list[CleanedPage] = []
    for page in pages:
        boundary_indexes = set(range(min(scan_lines, len(page.lines))))
        boundary_indexes.update(
            range(max(0, len(page.lines) - scan_lines), len(page.lines))
        )
        kept: list[ExtractedLine] = []
        removed = 0
        for index, line in enumerate(page.lines):
            if index in boundary_indexes and boundary_key(line.text) in repeated:
                removed += 1
            else:
                kept.append(line)
        cleaned.append(
            CleanedPage(
                page_number=page.page_number,
                lines=tuple(kept),
                removed_boundary_lines=removed,
                raw_char_count=page.raw_char_count,
                body_font_size=page.body_font_size,
            )
        )
    return cleaned, repeated


def is_heading(
    line: str,
    *,
    font_size: float | None,
    body_font_size: float | None,
) -> bool:
    stripped = line.strip()
    words = stripped.split()
    if not stripped or len(stripped) > 120 or len(words) > 14:
        return False
    if BULLET_PATTERN.match(stripped) or stripped.endswith((".", ";", ",")):
        return False
    if stripped.casefold().rstrip(":") in COMMON_HEADINGS:
        return True
    if NUMBERED_HEADING_PATTERN.match(stripped):
        return True
    letters = LETTER_PATTERN.findall(stripped)
    if len(letters) >= 4 and sum(char.isupper() for char in letters) / len(letters) >= 0.8:
        return True
    return (
        font_size is not None
        and body_font_size is not None
        and font_size >= body_font_size * 1.35
        and len(words) >= 2
    )


def _join_wrapped_line(current: str, following: str) -> str:
    if current.endswith("-") and following[:1].islower():
        return current + following
    return f"{current} {following}"


def build_paragraphs(
    pages: list[CleanedPage],
    *,
    default_section: str,
) -> list[Paragraph]:
    """把视觉换行合并为段落，同时保留页码和保守推断的章节名。"""

    paragraphs: list[Paragraph] = []
    section = default_section
    for page in pages:
        current = ""
        current_section = section

        def flush() -> None:
            nonlocal current
            if current.strip():
                paragraphs.append(
                    Paragraph(
                        page_number=page.page_number,
                        section=current_section,
                        text=current.strip(),
                    )
                )
            current = ""

        for extracted_line in page.lines:
            line = extracted_line.text
            if is_heading(
                line,
                font_size=extracted_line.font_size,
                body_font_size=page.body_font_size,
            ):
                flush()
                section = line.rstrip(":")
                current_section = section
                current = line
                continue
            if BULLET_PATTERN.match(line):
                flush()
                current_section = section
                current = line
                continue
            if not current:
                current_section = section
                current = line
                continue
            if (
                len(current) >= 240
                and current.endswith((".", "!", "?"))
                and line[:1].isupper()
            ):
                flush()
                current_section = section
                current = line
            else:
                current = _join_wrapped_line(current, line)
        flush()
    return paragraphs
