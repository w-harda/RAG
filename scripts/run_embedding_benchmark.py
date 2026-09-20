"""运行 Phase 4 Embedding Benchmark。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from apt_rag.benchmark import load_benchmark, validate_benchmark
from apt_rag.corpus import load_manifest
from apt_rag.evaluation.embedding_benchmark import (
    EmbeddingBenchmarkConfig,
    load_chunks,
    run_model_benchmark,
    write_experiment_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "embedding_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-embeddings", action="store_true", help="校验并复用 Corpus 向量，重编码查询并评测")
    parser.add_argument(
        "--model",
        default="all",
        help="模型 ID；默认 all，按配置顺序运行全部模型",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EmbeddingBenchmarkConfig.from_file(CONFIG_PATH, REPOSITORY_ROOT)
    selected = list(config.models)
    if args.model != "all":
        selected = [model for model in config.models if model.id == args.model]
        if not selected:
            choices = ", ".join(model.id for model in config.models)
            print(f"未知模型 {args.model!r}；可选值：{choices}, all")
            return 2

    benchmark = load_benchmark(config.benchmark_path)
    validate_benchmark(benchmark, load_manifest(REPOSITORY_ROOT / "data/manifest/reports.json"),
                       chunks_directory=config.chunks_directory)
    chunks = load_chunks(config.chunks_directory)
    for spec in selected:
        print(f"开始评测 {spec.id} ({spec.model_name}@{spec.revision[:12]})")
        result = run_model_benchmark(spec, config, chunks, benchmark, reuse_embeddings=args.reuse_embeddings)
        print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
        gc.collect()

    summary = write_experiment_summary(config)
    print(
        f"已完成 {summary['completed_models']}/{summary['expected_models']} 个模型；"
        f"当前选择：{summary['selected_model_id'] or '待全部模型完成'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
