"""固定预计算向量的向量存储实验；子进程隔离每次建库。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apt_rag.evaluation.retrieval import evaluate_rankings
from apt_rag.vectorstore.local import ChromaStore, FaissStore, load_store

METADATA_FIELDS = (
    "report_id", "document_title", "vendor", "apt_group", "page", "section",
    "source_url", "language", "publish_date", "text_sha256",
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0" or set(config["backends"]) != {"faiss", "chroma"}:
        raise ValueError("Phase 5 配置必须指定 faiss 和 chroma")
    if len(config["backends"]) != 2:
        raise ValueError("backends 不能重复")
    for key in ("top_k", "build_repeats", "query_repeats", "warmup_rounds"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if sorted(set(config["sizes"])) != config["sizes"] or any(size <= 0 for size in config["sizes"]):
        raise ValueError("sizes 必须为升序、无重复的正整数")
    if set(config["hnsw"]) != {"max_neighbors", "ef_construction", "ef_search", "num_threads"}:
        raise ValueError("HNSW 参数集合不正确")
    if any(value <= 0 for value in config["hnsw"].values()):
        raise ValueError("HNSW 参数必须为正整数")
    if config["hnsw"]["ef_search"] < config["top_k"]:
        raise ValueError("ef_search 不能小于 top_k")
    if set(config["chroma"]) != {"batch_size", "sync_threshold"}:
        raise ValueError("Chroma 持久化参数集合不正确")
    if config["chroma"]["batch_size"] < 3 or config["chroma"]["sync_threshold"] < config["chroma"]["batch_size"]:
        raise ValueError("Chroma batch_size/sync_threshold 无效")
    if not config["filter"] or set(config["filter"]) - set(METADATA_FIELDS):
        raise ValueError("Metadata filter 必须使用已有字段")
    return config


def prepare_inputs(root: Path, config: dict[str, Any], run_directory: Path) -> dict[str, Any]:
    # 仅主进程加载 Phase 4 实验工具；查询和建库计时不包括模型或数据加载。
    from apt_rag.benchmark import load_benchmark, validate_benchmark
    from apt_rag.corpus import load_manifest
    from apt_rag.evaluation.embedding_benchmark import (
        EmbeddingBenchmarkConfig, benchmark_fingerprint, corpus_fingerprint,
        load_chunks, rank_embeddings, validate_vectors,
    )

    embedding_config = EmbeddingBenchmarkConfig.from_file(root / config["embedding_config"], root)
    if config["top_k"] != max(embedding_config.top_k):
        raise ValueError("Phase 5 的 top_k 必须等于 Phase 4 的最大 Top-K")
    model_id = config["embedding_model_id"]
    specs = [model for model in embedding_config.models if model.id == model_id]
    if len(specs) != 1:
        raise ValueError("Phase 5 模型不在 Phase 4 配置中")
    spec = specs[0]
    benchmark = load_benchmark(embedding_config.benchmark_path)
    validate_benchmark(benchmark, load_manifest(root / "data/manifest/reports.json"),
                       chunks_directory=embedding_config.chunks_directory)
    chunks = load_chunks(embedding_config.chunks_directory)
    if max(config["sizes"]) != len(chunks):
        raise ValueError("sizes 的最大规模必须等于完整 Corpus 的 Chunk 数")
    summary = json.loads((embedding_config.results_directory / "summary.json").read_text(encoding="utf-8"))
    if summary["selected_model_id"] != model_id:
        raise ValueError("模型与 Phase 4 的实测选择不一致")
    previous = json.loads((embedding_config.results_directory / f"{model_id}.json").read_text(encoding="utf-8"))
    metadata = json.loads((embedding_config.embedding_cache_directory / f"{model_id}.metadata.json").read_text(encoding="utf-8"))
    fingerprint = corpus_fingerprint(chunks)
    expected = {"revision": spec.revision, "model_name": spec.model_name,
                "corpus_fingerprint": fingerprint, "max_seq_length": embedding_config.max_seq_length,
                "normalized": True, "dtype": "float32", "count": len(chunks)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Corpus 向量缓存与 Phase 4 模型、语料或配置不一致")
    if previous["controlled_variables"]["benchmark_fingerprint"] != benchmark_fingerprint(benchmark):
        raise ValueError("Benchmark 与 Phase 4 实测不一致，请先重跑 Phase 4")
    if previous["controlled_variables"]["corpus_fingerprint"] != fingerprint:
        raise ValueError("Corpus 与 Phase 4 实测不一致")
    with np.load(embedding_config.embedding_cache_directory / f"{model_id}.npz", allow_pickle=False) as cache:
        vectors, ids = cache["vectors"], cache["chunk_ids"].tolist()
    with np.load(embedding_config.embedding_cache_directory / f"{model_id}.queries.npz", allow_pickle=False) as cache:
        queries, question_ids = cache["vectors"], cache["question_ids"].tolist()
    validate_vectors(vectors)
    validate_vectors(queries)
    if ids != [chunk["chunk_id"] for chunk in chunks] or question_ids != [q["id"] for q in benchmark["questions"]]:
        raise ValueError("缓存 ID 或顺序不一致")
    if vectors.shape != (len(chunks), metadata["dimension"]) or queries.shape != (len(question_ids), vectors.shape[1]):
        raise ValueError("缓存形状不一致")
    rankings, scores = rank_embeddings(vectors, queries, ids, question_ids, maximum_k=config["top_k"])
    # Query 缓存必须确实对应已提交的 Phase 4 查询，不能只检查题目 ID。
    for result in previous["per_query"]:
        qid = result["question_id"]
        if rankings[qid] != result["retrieved_chunk_ids"][:config["top_k"]] or not np.allclose(
            scores[qid], result["scores"][:config["top_k"]], atol=2e-5, rtol=0):
            raise ValueError(f"{qid} 的缓存查询向量无法复现 Phase 4 排名或分数")
    permutation = np.random.default_rng(config["seed"]).permutation(len(ids)).tolist()
    np.savez_compressed(run_directory / "inputs.npz", vectors=vectors, queries=queries)
    inputs = {
        "config": config, "ids": ids, "question_ids": question_ids,
        "metadata": [{field: chunk[field] for field in METADATA_FIELDS} for chunk in chunks],
        "permutation": permutation,
        "relevance": {q["id"]: {e["chunk_id"]: e["relevance"] for e in q["evidence"]}
                      for q in benchmark["questions"]},
    }
    write_json(run_directory / "inputs.json", inputs)
    return {
        "embedding_model_id": model_id, "model_name": spec.model_name, "revision": spec.revision,
        "dimension": vectors.shape[1], "corpus_fingerprint": fingerprint,
        "benchmark_fingerprint": benchmark_fingerprint(benchmark),
        "benchmark_id": benchmark["benchmark_id"], "chunk_count": len(ids),
        "question_count": len(question_ids), "top_k": config["top_k"],
        "dtype": "float32", "normalized": True, "metric": "inner_product_on_normalized_vectors",
        "corpus_vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "query_vectors_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
        "metadata_fields": list(METADATA_FIELDS), "hnsw": config["hnsw"],
        "seed": config["seed"], "build_repeats": config["build_repeats"],
        "query_repeats": config["query_repeats"], "warmup_rounds": config["warmup_rounds"],
        "filter": config["filter"], "insertion_order": "seeded nested permutation",
    }


def distribution(samples: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(samples)), "median": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95)), "min": min(samples), "max": max(samples)}


def _measure_queries(store: Any, queries: np.ndarray, question_ids: list[str],
                     vectors: np.ndarray, ids: list[str], metadata: list[dict[str, Any]],
                     config: dict[str, Any], repeat: int, where: dict[str, Any] | None) -> dict[str, Any]:
    expected_metadata = dict(zip(ids, metadata, strict=True))
    vector_positions = {key: i for i, key in enumerate(ids)}
    candidates = [i for i, record in enumerate(metadata)
                  if not where or all(record.get(key) == value for key, value in where.items())]
    if not candidates:
        raise ValueError("当前规模没有 Metadata filter 的匹配项")
    k = min(config["top_k"], len(candidates))
    exact_scores = queries @ vectors[candidates].T
    exact = [[ids[candidates[i]] for i in np.argsort(-row, kind="stable")[:k]] for row in exact_scores]
    for _ in range(config["warmup_rounds"]):
        for vector in queries:
            store.search(vector, config["top_k"], where)
    rng = np.random.default_rng(config["seed"] + repeat)
    samples: list[float] = []
    rankings: dict[str, list[str]] = {}
    score_rows: dict[str, list[float]] = {}
    stable = True
    for _ in range(config["query_repeats"]):
        for position in rng.permutation(len(queries)):
            qid = question_ids[position]
            started = time.perf_counter()
            hits = store.search(queries[position], config["top_k"], where)
            samples.append((time.perf_counter() - started) * 1000)
            found = [hit.chunk_id for hit in hits]
            if len(found) != k or len(found) != len(set(found)):
                raise ValueError("检索数量不正确或返回重复 Chunk")
            if qid in rankings and rankings[qid] != found:
                stable = False
            for hit in hits:
                if hit.chunk_id not in expected_metadata or hit.metadata != expected_metadata[hit.chunk_id]:
                    raise ValueError("返回的 Metadata 与输入不一致")
                if where and any(hit.metadata.get(key) != value for key, value in where.items()):
                    raise ValueError("Metadata 过滤泄漏")
                expected_score = float(queries[position] @ vectors[vector_positions[hit.chunk_id]])
                if abs(hit.score - expected_score) > 2e-5:
                    raise ValueError("检索相似度与原始向量点积不一致")
            rankings[qid] = found
            score_rows[qid] = [hit.score for hit in hits]
    overlap = [len(set(rankings[qid]) & set(exact[i])) / k for i, qid in enumerate(question_ids)]
    return {"latency_ms": distribution(samples), "latency_samples_ms": samples,
            "ann_recall_at_k": float(np.mean(overlap)), "ranking_stable": stable,
            "matching_chunks": len(candidates), "metadata_verified": True,
            "per_query": [{"question_id": qid, "retrieved_chunk_ids": rankings[qid],
                           "scores": score_rows[qid], "exact_chunk_ids": exact[i],
                           "ann_recall_at_k": overlap[i]} for i, qid in enumerate(question_ids)]}


def worker(run_directory: Path, backend: str, size: int, repeat: int, *, verify: bool = False) -> None:
    inputs = json.loads((run_directory / "inputs.json").read_text(encoding="utf-8"))
    config = inputs["config"]
    selected = inputs["permutation"][:size]
    with np.load(run_directory / "inputs.npz", allow_pickle=False) as archive:
        vectors, queries = archive["vectors"][selected], archive["queries"]
    ids = [inputs["ids"][i] for i in selected]
    metadata = [inputs["metadata"][i] for i in selected]
    store_directory = run_directory / f"size-{size}" / f"repeat-{repeat}" / backend
    if verify:
        original = json.loads((store_directory.parent / f"{backend}.json").read_text(encoding="utf-8"))
        started = time.perf_counter()
        restored = load_store(backend, store_directory)
        reload_seconds = time.perf_counter() - started
        for chunk_id, record in zip(ids, metadata, strict=True):
            if restored.get(chunk_id) != record:
                raise ValueError("跨进程重载的 Metadata 不一致")
        for mode, where in (("unfiltered", None), ("filtered", config["filter"])):
            for vector, row in zip(queries, original[mode]["per_query"], strict=True):
                hits = restored.search(vector, config["top_k"], where)
                if [hit.chunk_id for hit in hits] != row["retrieved_chunk_ids"] or not np.allclose(
                    [hit.score for hit in hits], row["scores"], atol=2e-5, rtol=0):
                    raise ValueError("跨进程重载的查询结果不一致")
        write_json(store_directory.parent / f"{backend}.persistence.json",
                   {"fresh_process_verified": True, "verified_metadata_records": len(ids),
                    "reload_seconds": reload_seconds, "query_modes": ["unfiltered", "filtered"]})
        return

    started = time.perf_counter()
    store = FaissStore(store_directory, config["hnsw"]) if backend == "faiss" else ChromaStore(
        store_directory, config["hnsw"], config["chroma"])
    store.build(vectors, ids, metadata)
    build_seconds = time.perf_counter() - started
    unfiltered = _measure_queries(store, queries, inputs["question_ids"], vectors, ids, metadata, config, repeat, None)
    filtered = _measure_queries(store, queries, inputs["question_ids"], vectors, ids, metadata, config, repeat, config["filter"])
    result = {"backend": backend, "size": size, "repeat": repeat, "build_seconds": build_seconds,
              "vectors_per_second": size / build_seconds, "unfiltered": unfiltered, "filtered": filtered}
    if size == len(inputs["ids"]):
        rankings = {row["question_id"]: row["retrieved_chunk_ids"] for row in unfiltered["per_query"]}
        metrics, rows = evaluate_rankings(rankings, inputs["relevance"], top_k=(1, 3, 5, config["top_k"]))
        metrics[f"mrr@{config['top_k']}"] = metrics.pop("mrr")
        result["qa_metrics"] = metrics
        result["qa_per_query"] = rows
    write_json(store_directory.parent / f"{backend}.json", result)


def run_benchmark(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    storage_directory = root / config["storage_directory"]
    storage_directory.mkdir(parents=True, exist_ok=True)
    run_directory = Path(tempfile.mkdtemp(prefix="phase5-", dir=storage_directory))
    controls = prepare_inputs(root, config, run_directory)
    records = []
    persistence = {}
    for size in config["sizes"]:
        for repeat in range(config["build_repeats"]):
            backends = config["backends"] if repeat % 2 == 0 else list(reversed(config["backends"]))
            for backend in backends:
                print(f"Phase 5: {backend}, {size} vectors, build {repeat + 1}/{config['build_repeats']}", flush=True)
                command = [sys.executable, "-m", "apt_rag.evaluation.vectorstore_benchmark", "--worker",
                           backend, "--run-directory", str(run_directory), "--size", str(size), "--repeat", str(repeat)]
                subprocess.run(command, check=True, timeout=180)
                directory = run_directory / f"size-{size}" / f"repeat-{repeat}"
                record = json.loads((directory / f"{backend}.json").read_text(encoding="utf-8"))
                if size == controls["chunk_count"] and repeat == 0:
                    subprocess.run(command + ["--verify"], check=True, timeout=180)
                    persistence[backend] = json.loads((directory / f"{backend}.persistence.json").read_text(encoding="utf-8"))
                record["disk_bytes"] = sum(path.stat().st_size for path in (directory / backend).rglob("*") if path.is_file())
                records.append(record)
    result = {
        "schema_version": "1.0", "experiment": "phase5-vectorstore-benchmark",
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "controlled_variables": controls,
        "backend_specific_parameters": {"chroma": config["chroma"],
                                        "faiss_filter": "metadata prescan + exact IndexFlatIP over all matches"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "faiss_cpu": importlib.metadata.version("faiss-cpu"),
                        "chromadb": importlib.metadata.version("chromadb")},
        "run_storage_path": run_directory.relative_to(root).as_posix(),
        "persistence": persistence, "runs": records,
    }
    summary_rows = []
    for size in config["sizes"]:
        for backend in config["backends"]:
            selected = [item for item in records if item["size"] == size and item["backend"] == backend]
            row = {"backend": backend, "size": size,
                   "build_seconds": distribution([item["build_seconds"] for item in selected]),
                   "disk_bytes_mean": float(np.mean([item["disk_bytes"] for item in selected]))}
            for mode in ("unfiltered", "filtered"):
                row[mode] = {"latency_ms": distribution([sample for item in selected for sample in item[mode]["latency_samples_ms"]]),
                             "ann_recall_at_k": float(np.mean([item[mode]["ann_recall_at_k"] for item in selected])),
                             "ranking_stable": all(item[mode]["ranking_stable"] for item in selected)}
            if "qa_metrics" in selected[0]:
                row["qa_metrics_mean"] = {key: float(np.mean([item["qa_metrics"][key] for item in selected]))
                                          for key in selected[0]["qa_metrics"]}
            summary_rows.append(row)
    summary = {key: value for key, value in result.items() if key != "runs"}
    summary["results"] = summary_rows
    write_json(root / config["results_directory"] / "runs.json", result)
    write_json(root / config["results_directory"] / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["faiss", "chroma"], required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    worker(args.run_directory, args.worker, args.size, args.repeat, verify=args.verify)


if __name__ == "__main__":
    main()
