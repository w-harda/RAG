"""Phase 7 固定向量的检索对比；不调用 LLM，不修改 Benchmark。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apt_rag.benchmark import load_benchmark, validate_benchmark
from apt_rag.corpus import load_manifest
from apt_rag.evaluation.embedding_benchmark import (
    EmbeddingBenchmarkConfig, _write_json, benchmark_fingerprint as fingerprint,
    corpus_fingerprint, load_chunks, rank_embeddings, validate_vectors,
)
from apt_rag.evaluation.retrieval import evaluate_rankings
from apt_rag.reranking import Reranker
from apt_rag.retrieval import BM25Retriever, DenseRetriever, RankedChunk, fuse_rrf

STRATEGIES = ("dense", "hybrid", "hybrid_reranker")
SOURCE_FIELDS = ("report_id", "document_title", "vendor", "page", "section", "source_url", "text_sha256")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["schema_version"] != "1.0" or tuple(config["strategies"]) != STRATEGIES:
        raise ValueError("必须比较 Dense、Hybrid、Hybrid + Reranker 三组")
    if config["dense_search"] != "numpy_exact_cosine" or config["tie_break"] != "corpus_order":
        raise ValueError("固定精确余弦检索与语料顺序并列规则")
    top_k = config["top_k"]
    if not top_k or sorted(set(top_k)) != top_k or any(type(k) is not int or k <= 0 for k in top_k):
        raise ValueError("top_k 必须为升序无重复正整数")
    if type(config["candidate_k"]) is not int or config["candidate_k"] < max(top_k):
        raise ValueError("候选池不能小于最大 Top-K")
    bm25 = config["bm25"]
    if (bm25["tokenizer"] != "ascii_identifiers_cjk_chars_v1" or
            bm25["query_term_frequency"] != "binary" or bm25["positive_scores_only"] is not True):
        raise ValueError("不支持的 BM25 分词或候选政策")
    if not np.isfinite(bm25["k1"]) or bm25["k1"] <= 0 or not 0 <= bm25["b"] <= 1:
        raise ValueError("非法 BM25 参数")
    rrf = config["rrf"]
    if type(rrf["rank_constant"]) is not int or rrf["rank_constant"] < 1:
        raise ValueError("RRF 常数必须为正整数")
    if len(rrf["weights"]) != 2 or any(not np.isfinite(w) or w <= 0 for w in rrf["weights"]):
        raise ValueError("需要两个有限正 RRF 权重")
    spec = config["reranker"]
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Reranker revision 必须固定为完整 commit SHA")
    if spec["dtype"] != "float32" or spec["device"] != "cpu" or spec["truncation"] != "only_second":
        raise ValueError("本阶段固定 CPU/float32 且只截断文档")
    for key in ("batch_size", "max_length", "cpu_threads"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"Reranker {key} 必须为正整数")
    return config


def prepare_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    embedding = EmbeddingBenchmarkConfig.from_file(root / config["embedding_config"], root)
    spec = next((s for s in embedding.models if s.id == config["embedding_model_id"]), None)
    if spec is None or tuple(config["top_k"]) != embedding.top_k:
        raise ValueError("Embedding 或 Top-K 与 Phase 4 不一致")
    benchmark = load_benchmark(embedding.benchmark_path)
    validate_benchmark(benchmark, load_manifest(root / config["manifest_path"]),
                       chunks_directory=embedding.chunks_directory)
    chunks = load_chunks(embedding.chunks_directory)
    metadata = json.loads((embedding.embedding_cache_directory / f"{spec.id}.metadata.json").read_text(encoding="utf-8"))
    previous = json.loads((embedding.results_directory / f"{spec.id}.json").read_text(encoding="utf-8"))
    expected = {"model_name": spec.model_name, "revision": spec.revision,
                "corpus_fingerprint": corpus_fingerprint(chunks), "max_seq_length": embedding.max_seq_length,
                "normalized": True, "dtype": "float32", "count": len(chunks)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Embedding 缓存与固定语料/模型不一致")
    if previous["controlled_variables"]["benchmark_fingerprint"] != fingerprint(benchmark):
        raise ValueError("问题集与 Phase 4 不一致")
    with np.load(embedding.embedding_cache_directory / f"{spec.id}.npz", allow_pickle=False) as data:
        vectors, ids = data["vectors"], data["chunk_ids"].tolist()
    with np.load(embedding.embedding_cache_directory / f"{spec.id}.queries.npz", allow_pickle=False) as data:
        queries, question_ids = data["vectors"], data["question_ids"].tolist()
    validate_vectors(vectors)
    validate_vectors(queries)
    if ids != [c["chunk_id"] for c in chunks] or question_ids != [q["id"] for q in benchmark["questions"]]:
        raise ValueError("缓存 ID/顺序不匹配")
    if vectors.shape != (len(ids), metadata["dimension"]) or queries.shape != (len(question_ids), vectors.shape[1]):
        raise ValueError("缓存维数不匹配")
    vector_hash = hashlib.sha256(vectors.tobytes()).hexdigest()
    query_hash = hashlib.sha256(queries.tobytes()).hexdigest()
    # 延续 Phase 5 的固定 Query 缓存，额外检查字节指纹，不重新调用 Embedding。
    phase5 = json.loads((root / "results/phase5/summary.json").read_text(encoding="utf-8"))["controlled_variables"]
    if phase5["corpus_vectors_sha256"] != vector_hash or phase5["query_vectors_sha256"] != query_hash:
        raise ValueError("向量字节与 Phase 5 不一致，不得静默重新编码")
    controls = {
        "benchmark_id": benchmark["benchmark_id"], "benchmark_fingerprint": fingerprint(benchmark),
        "corpus_fingerprint": expected["corpus_fingerprint"], "chunk_count": len(chunks),
        "question_count": len(question_ids), "embedding_model": previous["model"],
        "embedding_max_seq_length": embedding.max_seq_length,
        "corpus_vectors_sha256": vector_hash, "query_vectors_sha256": query_hash,
        "dense_search": config["dense_search"], "top_k": config["top_k"],
        "candidate_k": config["candidate_k"], "tie_break": config["tie_break"],
        "bm25": config["bm25"], "rrf": config["rrf"], "query_rewrite": False,
        "generation": "not_run: ranking-only benchmark; no LLM variable",
    }
    # 复核缓存能还原上一阶段 Dense Top-10，不把旧指标直接复制成新结果。
    rankings, scores = rank_embeddings(vectors, queries, ids, question_ids, maximum_k=max(config["top_k"]))
    for row in previous["per_query"]:
        if rankings[row["question_id"]] != row["retrieved_chunk_ids"] or not np.allclose(
            scores[row["question_id"]], row["scores"], atol=1e-6,
        ):
            raise ValueError("Dense 缓存不能还原 Phase 4 排名/分数")
    return {"controls": controls, "benchmark": benchmark, "chunks": chunks,
            "vectors": vectors, "queries": queries, "ids": ids}


def _records(hits: list[RankedChunk]) -> list[dict[str, Any]]:
    return [{"chunk_id": h.chunk_id, "score": h.score} for h in hits]


def _hits(records: list[dict[str, Any]]) -> list[RankedChunk]:
    return [RankedChunk(row["chunk_id"], row["score"]) for row in records]


def rerank_order(candidates: list[RankedChunk], scores: list[float], ids: list[str]) -> list[RankedChunk]:
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Reranker 必须为每个候选返回一个有限分数")
    positions = {key: index for index, key in enumerate(ids)}
    return [RankedChunk(candidate.chunk_id, float(score)) for candidate, score in sorted(
        zip(candidates, scores, strict=True), key=lambda item: (-item[1], positions[item[0].chunk_id]),
    )]


def prepare_candidates(inputs: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    dense = DenseRetriever(inputs["vectors"], inputs["ids"])
    started = time.perf_counter()
    sparse = BM25Retriever([c["text"] for c in inputs["chunks"]], inputs["ids"],
                          k1=config["bm25"]["k1"], b=config["bm25"]["b"])
    index_seconds = time.perf_counter() - started
    by_id = {c["chunk_id"]: c for c in inputs["chunks"]}
    rows = []
    for question, vector in zip(inputs["benchmark"]["questions"], inputs["queries"], strict=True):
        started = time.perf_counter()
        dense_hits = dense.retrieve(question["question"], vector, config["candidate_k"])
        dense_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bm25_hits = sparse.retrieve(question["question"], None, config["candidate_k"])
        sparse_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fused = fuse_rrf([dense_hits, bm25_hits], inputs["ids"],
                         **config["rrf"], top_k=config["candidate_k"])
        fusion_seconds = time.perf_counter() - started
        pairs = [{"chunk_id": hit.chunk_id, "text_sha256": by_id[hit.chunk_id]["text_sha256"]}
                 for hit in fused]
        rows.append({
            "question_id": question["id"], "category": question["category"],
            "query_sha256": hashlib.sha256(question["question"].encode("utf-8")).hexdigest(),
            "dense_candidates": _records(dense_hits), "bm25_candidates": _records(bm25_hits),
            "hybrid_candidates": _records(fused), "pair_fingerprint": fingerprint({
                "question": question["question"], "candidates": pairs,
            }),
            "timings_seconds": {"dense": dense_seconds, "bm25": sparse_seconds, "fusion": fusion_seconds},
        })
    return rows, index_seconds


def candidates_fingerprint(rows: list[dict[str, Any]]) -> str:
    return fingerprint({"queries": [{k: v for k, v in row.items() if k != "timings_seconds"} for row in rows]})


def run_benchmark(root: Path, config_path: Path, *, reranker: Reranker | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = prepare_inputs(root, config)
    rows, index_seconds = prepare_candidates(inputs, config)
    candidate_hash = candidates_fingerprint(rows)
    path = root / config["results_directory"] / "rankings.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing["config_fingerprint"] != fingerprint(config) or existing["controls"] != inputs["controls"]
                or existing["candidates_fingerprint"] != candidate_hash):
            raise ValueError("旧结果与配置/输入不同；请使用新实验目录")
        summary = summarize_results(existing, inputs["benchmark"], config)
        summary_path = path.parent / "summary.json"
        if not summary_path.exists():
            _write_json(summary_path, summary)
        elif json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("摘要与原始排名不一致；请显式重建摘要")
        print("Phase 7 已完成；复用真实记录，不重新加载 Reranker", flush=True)
        return existing
    cache_path = root / config["reranker_cache_path"]
    header = {"config_fingerprint": fingerprint(config), "candidates_fingerprint": candidate_hash,
              "reranker_spec": config["reranker"]}
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {**header, "queries": {}}
    if any(cache.get(key) != value for key, value in header.items()):
        raise ValueError("Reranker 检查点指纹不一致；新实验需独立缓存路径")
    qids = {row["question_id"] for row in rows}
    if set(cache["queries"]) - qids:
        raise ValueError("Reranker 检查点包含其他问题")
    active, load_seconds = reranker, 0.0
    questions = {q["id"]: q for q in inputs["benchmark"]["questions"]}
    chunks = {c["chunk_id"]: c for c in inputs["chunks"]}
    for row in rows:
        qid = row["question_id"]
        reused = qid in cache["queries"]
        if not reused:
            if active is None:
                from apt_rag.reranking.bge import BGEReranker
                started = time.perf_counter()
                active = BGEReranker(config["reranker"])
                load_seconds = time.perf_counter() - started
                cache["model_load_seconds"] = load_seconds
            started = time.perf_counter()
            scores = active.score(questions[qid]["question"], [
                chunks[c["chunk_id"]]["text"] for c in row["hybrid_candidates"]
            ])
            cache["queries"][qid] = {
                "pair_fingerprint": row["pair_fingerprint"], "scores": scores.scores,
                "original_tokens": scores.original_tokens, "input_tokens": scores.input_tokens,
                "seconds": time.perf_counter() - started,
            }
            validate_scores(cache["queries"][qid], row, config)
            _write_json(cache_path, cache)
        scored = cache["queries"][qid]
        validate_scores(scored, row, config)
        ordered = rerank_order(_hits(row["hybrid_candidates"]), scored["scores"], inputs["ids"])
        row["reranking"] = scored
        row["reranker_scores_reused"] = reused
        row["rankings"] = {"dense": _records(_hits(row["dense_candidates"])[:max(config["top_k"])]),
                           "hybrid": row["hybrid_candidates"][:max(config["top_k"])],
                           "hybrid_reranker": _records(ordered[:max(config["top_k"])])}
        print(f"{qid}: {len(ordered)} pairs, {scored['seconds']:.2f}s, reused={reused}", flush=True)
    selected = {r["chunk_id"] for row in rows for ranking in row["rankings"].values() for r in ranking}
    result = {
        "schema_version": "1.0", "experiment": "phase7-retrieval-benchmark",
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "config_fingerprint": fingerprint(config),
        "controls": inputs["controls"], "candidates_fingerprint": candidate_hash,
        "reranker_spec": config["reranker"], "corpus_ids": inputs["ids"], "queries": rows,
        "sources": {key: {field: chunks[key][field] for field in SOURCE_FIELDS} for key in sorted(selected)},
        "bm25_index_seconds": index_seconds, "reranker_model_load_seconds": cache.get("model_load_seconds", load_seconds),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__,
                        **{p: importlib.metadata.version(p) for p in ("torch", "transformers")}},
    }
    summary = summarize_results(result, inputs["benchmark"], config)
    _write_json(path, result)
    _write_json(path.parent / "summary.json", summary)
    return result


def validate_scores(scored: dict[str, Any], row: dict[str, Any], config: dict[str, Any]) -> None:
    count = len(row["hybrid_candidates"])
    if scored["pair_fingerprint"] != row["pair_fingerprint"]:
        raise ValueError("Reranker 的问题/有序候选输入已变化")
    if any(len(scored[key]) != count for key in ("scores", "original_tokens", "input_tokens")):
        raise ValueError("Reranker 分数或 Token 计数未覆盖所有候选")
    if not np.isfinite(scored["scores"]).all() or not np.isfinite(scored["seconds"]) or scored["seconds"] < 0:
        raise ValueError("Reranker 分数/耗时非法")
    for before, after in zip(scored["original_tokens"], scored["input_tokens"], strict=True):
        if type(before) is not int or type(after) is not int or after <= 0 or after != min(before, config["reranker"]["max_length"]):
            raise ValueError("Reranker Token 计数非法")


def summarize_results(result: dict[str, Any], benchmark: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if result["config_fingerprint"] != fingerprint(config) or result["reranker_spec"] != config["reranker"]:
        raise ValueError("排名记录与配置/模型不一致")
    if result["controls"]["benchmark_fingerprint"] != fingerprint(benchmark):
        raise ValueError("排名记录与问题集不一致")
    for key in ("top_k", "candidate_k", "dense_search", "tie_break", "bm25", "rrf"):
        if key in result["controls"] and result["controls"][key] != config[key]:
            raise ValueError("保存的控制变量与配置不同")
    rows, ids = result["queries"], result["corpus_ids"]
    if len(ids) != len(set(ids)) or len(ids) != result["controls"]["chunk_count"]:
        raise ValueError("语料 ID 审计记录非法")
    if [row["question_id"] for row in rows] != [q["id"] for q in benchmark["questions"]]:
        raise ValueError("必须按顺序覆盖完全相同的问题集")
    reconstructed = []
    for row, question in zip(rows, benchmark["questions"], strict=True):
        if row["category"] != question["category"] or row["query_sha256"] != hashlib.sha256(question["question"].encode("utf-8")).hexdigest():
            raise ValueError("问题文本/分类改变")
        for branch in ("dense_candidates", "bm25_candidates", "hybrid_candidates"):
            records = row[branch]
            if len(records) > config["candidate_k"] or len({r["chunk_id"] for r in records}) != len(records):
                raise ValueError("候选数超预算或重复")
            if any(r["chunk_id"] not in ids or not np.isfinite(r["score"]) for r in records):
                raise ValueError("候选 ID 或分数非法")
            positions = {key: index for index, key in enumerate(ids)}
            if records != sorted(records, key=lambda r: (-r["score"], positions[r["chunk_id"]])):
                raise ValueError("候选没有按分数及语料并列规则排序")
            if branch == "bm25_candidates" and any(r["score"] <= 0 for r in records):
                raise ValueError("BM25 不能加入无词法命中的零分候选")
        fused = fuse_rrf([_hits(row["dense_candidates"]), _hits(row["bm25_candidates"])], ids,
                         **config["rrf"], top_k=config["candidate_k"])
        if _records(fused) != row["hybrid_candidates"]:
            raise ValueError("Hybrid 候选不能由相同 Dense/BM25 输入重算")
        validate_scores(row["reranking"], row, config)
        ordered = rerank_order(fused, row["reranking"]["scores"], ids)
        expected = {"dense": row["dense_candidates"][:max(config["top_k"])],
                    "hybrid": _records(fused[:max(config["top_k"])]),
                    "hybrid_reranker": _records(ordered[:max(config["top_k"])])}
        if row["rankings"] != expected:
            raise ValueError("输出 Top-K 不能由既有候选/重排分数重算")
        reconstructed.append({k: row[k] for k in (
            "question_id", "category", "query_sha256", "dense_candidates", "bm25_candidates", "hybrid_candidates", "pair_fingerprint",
        )})
    if candidates_fingerprint(reconstructed) != result["candidates_fingerprint"]:
        raise ValueError("候选指纹不匹配")
    selected = {r["chunk_id"] for row in rows for ranking in row["rankings"].values() for r in ranking}
    if set(result["sources"]) != selected:
        raise ValueError("来源 Metadata 未覆盖所有输出")
    for source in result["sources"].values():
        if set(source) != set(SOURCE_FIELDS) or type(source["page"]) is not int or source["page"] < 1:
            raise ValueError("来源 Metadata 契约不匹配")
    relevance = {q["id"]: {e["chunk_id"]: e["relevance"] for e in q["evidence"]} for q in benchmark["questions"]}
    results = []
    for strategy in STRATEGIES:
        ranking = {r["question_id"]: [hit["chunk_id"] for hit in r["rankings"][strategy]] for r in rows}
        metrics, per_query = evaluate_rankings(ranking, relevance, top_k=config["top_k"])
        metrics[f"mrr@{max(config['top_k'])}"] = metrics.pop("mrr")
        latencies, candidate_recalls = [], []
        for row in rows:
            seconds = row["timings_seconds"]["dense"]
            if strategy != "dense":
                seconds += row["timings_seconds"]["bm25"] + row["timings_seconds"]["fusion"]
            if strategy == "hybrid_reranker":
                seconds += row["reranking"]["seconds"]
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError("检索阶段耗时非法")
            latencies.append(seconds * 1000)
            pool = row["dense_candidates"] if strategy == "dense" else row["hybrid_candidates"]
            qrels = set(relevance[row["question_id"]])
            candidate_recalls.append(len(qrels.intersection(hit["chunk_id"] for hit in pool)) / len(qrels))
        results.append({
            "strategy": strategy, "metrics": metrics, "per_query": per_query,
            "candidate_recall": float(np.mean(candidate_recalls)),
            "latency_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
            "category_metrics": {category: evaluate_rankings(
                ranking, {q["id"]: relevance[q["id"]] for q in benchmark["questions"] if q["category"] == category},
                top_k=config["top_k"],
            )[0] for category in sorted({q["category"] for q in benchmark["questions"]})},
        })
        for category in results[-1]["category_metrics"].values():
            category[f"mrr@{max(config['top_k'])}"] = category.pop("mrr")
    return {
        "schema_version": "1.0", "experiment": result["experiment"], "controls": result["controls"],
        "reranker_spec": result["reranker_spec"], "candidates_fingerprint": result["candidates_fingerprint"],
        "scored_pairs": sum(len(r["reranking"]["scores"]) for r in rows),
        "truncated_pairs": sum(a > b for r in rows for a, b in zip(
            r["reranking"]["original_tokens"], r["reranking"]["input_tokens"], strict=True)),
        "bm25_index_seconds": result["bm25_index_seconds"],
        "reranker_model_load_seconds": result["reranker_model_load_seconds"], "results": results,
    }
