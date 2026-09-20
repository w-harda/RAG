from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from apt_rag.benchmark import BenchmarkValidationError, load_benchmark, validate_benchmark
from apt_rag.corpus import load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPOSITORY_ROOT / "data" / "benchmark" / "apt_qa_v1.json"
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"


def _benchmark_using_one_evidence() -> dict:
    benchmark = load_benchmark(BENCHMARK_PATH)
    template = benchmark["questions"][0]
    categories = ["background", "ttp", "attack_technique", "ioc"]
    benchmark["questions"] = []
    for index in range(1, 21):
        question = deepcopy(template)
        question["id"] = f"q{index:03d}"
        question["question"] = f"测试问题 {index}"
        question["category"] = categories[(index - 1) % len(categories)]
        benchmark["questions"].append(question)
    return benchmark


def test_repository_benchmark_has_balanced_categories() -> None:
    result = validate_benchmark(
        load_benchmark(BENCHMARK_PATH),
        load_manifest(MANIFEST_PATH),
    )

    assert result["questions"] == 20
    assert result["categories"] == {
        "attack_technique": 5,
        "background": 5,
        "ioc": 5,
        "ttp": 5,
    }
    assert result["evidence_verified"] is False


def test_evidence_is_verified_against_chunk_metadata(tmp_path: Path) -> None:
    benchmark = _benchmark_using_one_evidence()
    manifest = load_manifest(MANIFEST_PATH)
    first_question = benchmark["questions"][0]

    evidence = first_question["evidence"][0]
    chunk_path = tmp_path / "chunks.jsonl"
    chunk_path.write_text(
        json.dumps(
            {
                "chunk_id": evidence["chunk_id"],
                "report_id": evidence["report_id"],
                "page": evidence["page"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_benchmark(benchmark, manifest, chunks_directory=tmp_path)
    assert result["evidence_verified"] is True


def test_mismatched_evidence_page_is_rejected(tmp_path: Path) -> None:
    benchmark = _benchmark_using_one_evidence()
    manifest = load_manifest(MANIFEST_PATH)
    evidence = benchmark["questions"][0]["evidence"][0]
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": evidence["chunk_id"],
                "report_id": evidence["report_id"],
                "page": evidence["page"] + 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError, match="Metadata 不一致"):
        validate_benchmark(benchmark, manifest, chunks_directory=tmp_path)
