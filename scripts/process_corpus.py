"""将 manifest 中的 PDF 处理为带引用元数据的 JSONL Chunk。"""

from __future__ import annotations

import argparse
from pathlib import Path

from apt_rag.corpus import load_manifest
from apt_rag.corpus.process import ProcessingConfig, process_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifest" / "reports.json"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "processing.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data" / "processed" / "chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确覆盖已有的可重建处理结果",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(MANIFEST_PATH)
    config = ProcessingConfig.from_file(CONFIG_PATH)
    try:
        summary = process_corpus(
            manifest["reports"],
            REPOSITORY_ROOT,
            config,
            output_directory=OUTPUT_DIRECTORY,
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"处理失败：{exc}")
        return 1

    print(
        f"处理完成：{summary['report_count']} 份报告，"
        f"{summary['total_pages']} 页，{summary['total_chunks']} 个 Chunk"
    )
    for report in summary["reports"]:
        print(
            f"- {report['report_id']}: {report['pages']} 页，"
            f"{report['chunks']} 个 Chunk，移除 {report['removed_boundary_lines']} 条页眉/页脚"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
