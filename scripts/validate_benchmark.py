"""校验 APT QA Benchmark 结构和全部证据引用。"""

from pathlib import Path

from apt_rag.benchmark import (
    BenchmarkValidationError,
    load_benchmark,
    validate_benchmark,
)
from apt_rag.corpus import load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPOSITORY_ROOT / "data" / "benchmark" / "apt_qa_v1.json"
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"
CHUNKS_DIRECTORY = REPOSITORY_ROOT / "data" / "processed" / "chunks"


def main() -> int:
    try:
        benchmark = load_benchmark(BENCHMARK_PATH)
        manifest = load_manifest(MANIFEST_PATH)
        result = validate_benchmark(
            benchmark,
            manifest,
            chunks_directory=CHUNKS_DIRECTORY,
        )
    except (OSError, ValueError, BenchmarkValidationError) as exc:
        print(exc)
        return 1

    print(
        f"Benchmark 有效：{result['questions']} 题，"
        f"覆盖 {len(result['categories'])} 类、"
        f"{result['referenced_reports']} 份报告；Ground Truth Evidence 已核验"
    )
    for category, count in result["categories"].items():
        print(f"- {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
