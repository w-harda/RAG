"""页内、章节感知的字符切分。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from apt_rag.ingestion.cleaning import Paragraph


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        if self.min_chunk_size <= 0 or self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size 必须满足 0 < min_chunk_size <= chunk_size")


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    section: str
    page_chunk_index: int
    text: str


def _find_break(text: str, start: int, maximum_end: int) -> int:
    minimum_end = start + max(1, (maximum_end - start) // 2)
    candidates: list[int] = []
    for separator in ("\n\n", ". ", "; ", ", ", " "):
        position = text.rfind(separator, minimum_end, maximum_end)
        if position >= minimum_end:
            candidates.append(position + len(separator.rstrip()))
    return max(candidates, default=maximum_end)


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    """按字符上限切分，并在自然边界附近建立固定重叠。"""

    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        maximum_end = min(start + config.chunk_size, len(normalized))
        end = (
            len(normalized)
            if maximum_end == len(normalized)
            else _find_break(normalized, start, maximum_end)
        )
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - config.chunk_overlap)
        while next_start < end and not normalized[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return chunks


def chunk_paragraphs(
    paragraphs: list[Paragraph],
    config: ChunkingConfig,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_number, page_group in groupby(paragraphs, key=lambda item: item.page_number):
        page_paragraphs = list(page_group)
        provisional: list[tuple[str, str]] = []
        for section, section_group in groupby(page_paragraphs, key=lambda item: item.section):
            section_text = "\n\n".join(item.text for item in section_group)
            for text in split_text(section_text, config):
                provisional.append((section, text))

        merged: list[tuple[str, str]] = []
        index = 0
        while index < len(provisional):
            section, text = provisional[index]
            if len(text) < config.min_chunk_size and index + 1 < len(provisional):
                next_section, next_text = provisional[index + 1]
                combined = f"{text}\n\n{next_text}"
                if len(combined) <= config.chunk_size:
                    provisional[index + 1] = (section or next_section, combined)
                    index += 1
                    continue
            if len(text) < config.min_chunk_size and merged:
                previous_section, previous_text = merged[-1]
                combined = f"{previous_text}\n\n{text}"
                if len(combined) <= config.chunk_size:
                    merged[-1] = (previous_section, combined)
                    index += 1
                    continue
            merged.append((section, text))
            index += 1

        for page_chunk_index, (section, text) in enumerate(merged):
            chunks.append(
                TextChunk(
                    page_number=page_number,
                    section=section,
                    page_chunk_index=page_chunk_index,
                    text=text,
                )
            )
    return chunks
