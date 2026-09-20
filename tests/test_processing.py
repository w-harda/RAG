from __future__ import annotations

from pathlib import Path

import pytest

from apt_rag.chunking import ChunkingConfig, chunk_paragraphs, split_text
from apt_rag.corpus.process import ProcessingConfig
from apt_rag.ingestion.cleaning import (
    build_paragraphs,
    remove_repeated_boundaries,
)
from apt_rag.ingestion.pdf import ExtractedLine, ExtractedPage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_repeated_headers_and_page_numbers_are_removed() -> None:
    pages = [
        ExtractedPage(
            page_number=number,
            lines=(
                ExtractedLine("APT Report", 8.0),
                ExtractedLine(f"Body text for page {number}.", 10.0),
                ExtractedLine(f"Page {number} of 4", 8.0),
            ),
            raw_char_count=50,
            body_font_size=10.0,
        )
        for number in range(1, 5)
    ]

    cleaned, repeated = remove_repeated_boundaries(
        pages,
        scan_lines=1,
        minimum_fraction=0.5,
    )

    assert "apt report" in repeated
    assert "page # of #" in repeated
    assert all(
        tuple(line.text for line in page.lines) == (f"Body text for page {page.page_number}.",)
        for page in cleaned
    )


def test_wrapped_words_and_sections_are_preserved() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            lines=(
                ExtractedLine("Executive Summary", 20.0),
                ExtractedLine("state-", 10.0),
                ExtractedLine("sponsored activity continues.", 10.0),
            ),
            raw_char_count=60,
            body_font_size=10.0,
        )
    ]
    cleaned, _ = remove_repeated_boundaries(
        pages,
        scan_lines=1,
        minimum_fraction=1.0,
    )
    paragraphs = build_paragraphs(cleaned, default_section="Report")

    assert paragraphs[0].section == "Executive Summary"
    assert paragraphs[0].text == "Executive Summary state-sponsored activity continues."


def test_split_text_respects_size_and_overlap() -> None:
    config = ChunkingConfig(chunk_size=80, chunk_overlap=20, min_chunk_size=20)
    text = " ".join(f"token-{index}" for index in range(40))

    chunks = split_text(text, config)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= config.chunk_size for chunk in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_chunks_never_cross_pages() -> None:
    pages = [
        ExtractedPage(
            1,
            (ExtractedLine("Overview", 20.0), ExtractedLine("A" * 100, 10.0)),
            108,
            10.0,
        ),
        ExtractedPage(
            2,
            (ExtractedLine("Technical Details", 20.0), ExtractedLine("B" * 100, 10.0)),
            118,
            10.0,
        ),
    ]
    cleaned, _ = remove_repeated_boundaries(
        pages,
        scan_lines=1,
        minimum_fraction=1.0,
    )
    paragraphs = build_paragraphs(cleaned, default_section="Report")
    chunks = chunk_paragraphs(paragraphs, ChunkingConfig(80, 10, 20))

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(not ("A" in chunk.text and "B" in chunk.text) for chunk in chunks)


def test_processing_config_is_valid() -> None:
    config = ProcessingConfig.from_file(REPOSITORY_ROOT / "configs" / "processing.json")

    assert config.chunking.chunk_size == 1200
    assert config.chunking.chunk_overlap == 200
    assert config.chunking.min_chunk_size == 200


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100, min_chunk_size=20)
