"""离线重算三策略指标；可选检查本地向量/Chunk 与候选输入。"""

import argparse
import json
from pathlib import Path

from apt_rag.benchmark import load_benchmark
from apt_rag.corpus import load_manifest
from apt_rag.evaluation.embedding_benchmark import EmbeddingBenchmarkConfig, _write_json
from apt_rag.evaluation.retrieval_benchmark import (
    SOURCE_FIELDS, candidates_fingerprint, load_config, prepare_candidates, prepare_inputs, summarize_results,
)

ROOT = Path(__file__).resolve().parents[1]


def load_results(config_path: Path | None = None):
    config = load_config(config_path or ROOT / "configs/retrieval_benchmark.json")
    embedding = EmbeddingBenchmarkConfig.from_file(ROOT / config["embedding_config"], ROOT)
    benchmark = load_benchmark(embedding.benchmark_path)
    directory = ROOT / config["results_directory"]
    result = json.loads((directory / "rankings.json").read_text(encoding="utf-8"))
    summary = summarize_results(result, benchmark, config)
    manifest = load_manifest(ROOT / config["manifest_path"])
    reports = {r["report_id"]: r for r in manifest["reports"]}
    for source in result["sources"].values():
        report = reports[source["report_id"]]
        if any(source[field] != report[key] for field, key in (
            ("document_title", "title"), ("vendor", "vendor"), ("source_url", "url"),
        )):
            raise ValueError("输出来源与固定报告 manifest 不一致")
    return config, directory, result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true", help="需要 Chunk/向量，但不加载 Reranker")
    args = parser.parse_args()
    config, directory, result, summary = load_results(args.config)
    if args.verify_inputs:
        inputs = prepare_inputs(ROOT, config)
        rows, _ = prepare_candidates(inputs, config)
        if inputs["controls"] != result["controls"] or candidates_fingerprint(rows) != result["candidates_fingerprint"]:
            raise ValueError("语料/向量不能复现相同候选")
        chunks = {c["chunk_id"]: c for c in inputs["chunks"]}
        for key, source in result["sources"].items():
            if source != {field: chunks[key][field] for field in SOURCE_FIELDS}:
                raise ValueError("保存的来源与真实 Metadata 不一致")
    path = directory / "summary.json"
    if args.write_summary:
        _write_json(path, summary)
    elif json.loads(path.read_text(encoding="utf-8")) != summary:
        raise ValueError("摘要与候选/分数/qrels 不一致")
    print(f"Phase 7 验证通过：三策略、{len(result['queries'])} 题、{summary['scored_pairs']} 个重排分数")
    for row in summary["results"]:
        print(row["strategy"], row["metrics"])


if __name__ == "__main__":
    main()
