"""离线复核 Phase 4 指标；全排名 MRR 依保存的首条相关排名，不假装重跑模型。"""

import argparse
import json
from pathlib import Path

import numpy as np

from apt_rag.benchmark import load_benchmark
from apt_rag.evaluation.embedding_benchmark import EmbeddingBenchmarkConfig, benchmark_fingerprint
from apt_rag.evaluation.retrieval import evaluate_rankings

ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT, config_path=None):
    config = EmbeddingBenchmarkConfig.from_file(config_path or root / "configs/embedding_benchmark.json", root)
    benchmark = load_benchmark(config.benchmark_path)
    summary = json.loads((config.results_directory / "summary.json").read_text(encoding="utf-8"))
    qrels = {q["id"]: {e["chunk_id"]: e["relevance"] for e in q["evidence"]} for q in benchmark["questions"]}
    results = []
    for spec in config.models:
        result = json.loads((config.results_directory / f"{spec.id}.json").read_text(encoding="utf-8"))
        controls = result["controlled_variables"]
        if (result["model"]["revision"] != spec.revision or result["model"]["model_name"] != spec.model_name
                or controls != summary["controlled_variables"]
                or controls["benchmark_fingerprint"] != benchmark_fingerprint(benchmark)
                or controls["top_k"] != list(config.top_k)
                or controls["max_seq_length"] != config.max_seq_length
                or controls["device"] != config.device or controls["batch_size"] != config.batch_size):
            raise ValueError("Phase 4 模型/控制变量不一致")
        rows = result["per_query"]
        if [r["question_id"] for r in rows] != list(qrels):
            raise ValueError("Phase 4 问题集合/顺序不一致")
        metrics, recalculated = evaluate_rankings(
            {r["question_id"]: r["retrieved_chunk_ids"] for r in rows}, qrels, top_k=config.top_k)
        reciprocal = []
        for saved, actual in zip(rows, recalculated, strict=True):
            rank = saved["first_relevant_rank"]
            if type(rank) is not int or not 1 <= rank <= controls["mrr_cutoff"]:
                raise ValueError("Phase 4 首条相关证据排名无效")
            if actual["first_relevant_rank"] is not None and rank != actual["first_relevant_rank"]:
                raise ValueError("Phase 4 首条相关证据与 Top-10 不一致")
            if actual["first_relevant_rank"] is None and rank <= max(config.top_k):
                raise ValueError("Phase 4 首条相关证据不在保存排名中")
            if not np.isclose(saved["reciprocal_rank"], 1 / rank, atol=1e-12, rtol=1e-9):
                raise ValueError("Phase 4 reciprocal_rank 不一致")
            reciprocal.append(1 / rank)
            for key in metrics.keys() - {"mrr"}:
                if not np.isclose(saved[key], actual[key], atol=1e-12, rtol=1e-9):
                    raise ValueError("Phase 4 逐题指标不一致")
        metrics["mrr"] = float(np.mean(reciprocal))
        if set(metrics) != set(result["metrics"]) or any(
                not np.isclose(result["metrics"][key], value, atol=1e-12, rtol=1e-9) for key, value in metrics.items()):
            raise ValueError("Phase 4 汇总指标不一致")
        results.append({"model_id": spec.id, "model_name": spec.model_name, "revision": spec.revision,
                        "dimension": result["model"]["dimension"], **metrics, **result["timings_seconds"]})
    ordered = sorted(results, key=lambda r: (r["mrr"], r[f"ndcg@{max(config.top_k)}"]), reverse=True)
    if (summary["results"] != ordered or summary["selected_model_id"] != ordered[0]["model_id"]
            or summary["completed_models"] != len(config.models) or summary["expected_models"] != len(config.models)):
        raise ValueError("Phase 4 比较摘要/选择不一致")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/embedding_benchmark.json")
    args = parser.parse_args()
    print(f"Phase 4 验证通过：{validate(config_path=args.config)} 个模型，Top-K 指标与保存的全排名 MRR 摘要一致；未调用模型")


if __name__ == "__main__":
    main()
