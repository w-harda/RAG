"""从已提交原始样本重算 Phase 5 汇总；不要求本地索引缓存。"""

import json
from pathlib import Path

import numpy as np

from apt_rag.benchmark import load_benchmark
from apt_rag.evaluation.embedding_benchmark import benchmark_fingerprint
from apt_rag.evaluation.retrieval import evaluate_rankings
from apt_rag.evaluation.vectorstore_benchmark import distribution, load_config

ROOT = Path(__file__).resolve().parents[1]


def check_statistics(actual: dict, expected: dict) -> None:
    if set(actual) != set(expected) or not all(
        np.isclose(actual[key], value, atol=1e-12, rtol=1e-9) for key, value in expected.items()
    ):
        raise ValueError("原始样本与汇总统计不一致")


def main() -> None:
    config = load_config(ROOT / "configs/vectorstore_benchmark.json")
    directory = ROOT / config["results_directory"]
    runs = json.loads((directory / "runs.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    benchmark = load_benchmark(ROOT / "data/benchmark/apt_qa_v1.json")
    controls = summary["controlled_variables"]
    if controls != runs["controlled_variables"] or controls["benchmark_fingerprint"] != benchmark_fingerprint(benchmark):
        raise ValueError("控制变量或 Benchmark 指纹不一致")
    question_ids = [q["id"] for q in benchmark["questions"]]
    relevance = {q["id"]: {e["chunk_id"]: e["relevance"] for e in q["evidence"]} for q in benchmark["questions"]}
    expected_runs = {(backend, size, repeat) for backend in config["backends"] for size in config["sizes"]
                     for repeat in range(config["build_repeats"])}
    actual_runs = {(row["backend"], row["size"], row["repeat"]) for row in runs["runs"]}
    if actual_runs != expected_runs or len(runs["runs"]) != len(expected_runs):
        raise ValueError("建库组数、规模或重复次数不完整")
    for row in runs["runs"]:
        for mode in ("unfiltered", "filtered"):
            data = row[mode]
            samples = data["latency_samples_ms"]
            if len(samples) != len(question_ids) * config["query_repeats"] or any(not np.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("延迟样本数量或取值无效")
            check_statistics(data["latency_ms"], distribution(samples))
            if [item["question_id"] for item in data["per_query"]] != question_ids:
                raise ValueError("查询集合或顺序不一致")
            recalls = []
            for query in data["per_query"]:
                k = min(config["top_k"], data["matching_chunks"])
                ids, reference = query["retrieved_chunk_ids"], query["exact_chunk_ids"]
                if len(ids) != k or len(set(ids)) != k or len(reference) != k or len(query["scores"]) != k:
                    raise ValueError("逐题 Top-K 无效")
                recall = len(set(ids) & set(reference)) / k
                if not np.isclose(recall, query["ann_recall_at_k"]):
                    raise ValueError("ANN Recall 无法由原始排名复现")
                recalls.append(recall)
            if not np.isclose(np.mean(recalls), data["ann_recall_at_k"]) or data["metadata_verified"] is not True:
                raise ValueError("ANN 汇总或 Metadata 校验状态无效")
        if row["size"] == controls["chunk_count"]:
            rankings = {q["question_id"]: q["retrieved_chunk_ids"] for q in row["unfiltered"]["per_query"]}
            metrics, _ = evaluate_rankings(rankings, relevance, top_k=(1, 3, 5, config["top_k"]))
            metrics[f"mrr@{config['top_k']}"] = metrics.pop("mrr")
            check_statistics(row["qa_metrics"], metrics)
    if len(summary["results"]) != len(config["backends"]) * len(config["sizes"]):
        raise ValueError("汇总组数不正确")
    for row in summary["results"]:
        selected = [item for item in runs["runs"] if item["backend"] == row["backend"] and item["size"] == row["size"]]
        check_statistics(row["build_seconds"], distribution([item["build_seconds"] for item in selected]))
        for mode in ("unfiltered", "filtered"):
            check_statistics(row[mode]["latency_ms"], distribution([x for item in selected for x in item[mode]["latency_samples_ms"]]))
            if not np.isclose(row[mode]["ann_recall_at_k"], np.mean([item[mode]["ann_recall_at_k"] for item in selected])):
                raise ValueError("汇总 ANN Recall 不一致")
        if "qa_metrics_mean" in row:
            check_statistics(row["qa_metrics_mean"], {key: float(np.mean([item["qa_metrics"][key] for item in selected]))
                                                       for key in selected[0]["qa_metrics"]})
    for backend in config["backends"]:
        if summary["persistence"][backend]["fresh_process_verified"] is not True or summary["persistence"][backend]["verified_metadata_records"] != controls["chunk_count"]:
            raise ValueError("完整索引的跨进程验证未通过")
    print(f"Phase 5 结果有效：{len(expected_runs)} 次建库，延迟、ANN Recall 和 QA 指标均可由原始样本重算")


if __name__ == "__main__":
    main()
