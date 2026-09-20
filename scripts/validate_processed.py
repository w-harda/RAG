"""校验 Phase 2 生成的全部 Chunk 和 Metadata。"""

from pathlib import Path

from apt_rag.corpus import load_manifest
from apt_rag.corpus.process import ProcessingConfig
from apt_rag.corpus.validation import (
    ProcessedDataValidationError,
    validate_processed_corpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "processing.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data" / "processed" / "chunks"


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    config = ProcessingConfig.from_file(CONFIG_PATH)
    try:
        result = validate_processed_corpus(
            manifest["reports"],
            OUTPUT_DIRECTORY,
            chunk_size=config.chunking.chunk_size,
        )
    except (OSError, ValueError, ProcessedDataValidationError) as exc:
        print(exc)
        return 1
    print(
        f"处理结果有效：{result['reports']} 份报告，"
        f"{result['pages']} 页，{result['chunks']} 个 Chunk"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
