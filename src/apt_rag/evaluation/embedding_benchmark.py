"""在固定语料与问题上运行可复现的 Embedding 检索实验。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apt_rag.embedding import EmbeddingProvider, SentenceTransformerProvider
from apt_rag.evaluation.retrieval import evaluate_rankings


@dataclass(frozen=True)
class EmbeddingModelSpec:
    id: str
    model_name: str
    revision: str
    loader: str
    license: str


@dataclass(frozen=True)
class EmbeddingBenchmarkConfig:
    benchmark_path: Path
    chunks_directory: Path
    embedding_cache_directory: Path
    results_directory: Path
    similarity: str
    top_k: tuple[int, ...]
    batch_size: int
    max_seq_length: int
    device: str
    models: tuple[EmbeddingModelSpec, ...]

    @classmethod
    def from_file(cls, path: Path, repository_root: Path) -> "EmbeddingBenchmarkConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "1.0":
            raise ValueError("embedding benchmark config schema_version 必须为 1.0")
        if data.get("similarity") != "cosine":
            raise ValueError("Phase 4 当前只支持 cosine similarity")
        top_k = tuple(int(value) for value in data["top_k"])
        if not top_k or sorted(set(top_k)) != list(top_k) or any(k <= 0 for k in top_k):
            raise ValueError("top_k 必须是升序、无重复的正整数")
        models = tuple(EmbeddingModelSpec(**model) for model in data["models"])
        if len(models) < 3 or len({model.id for model in models}) != len(models):
            raise ValueError("至少需要三个 ID 唯一的 Embedding 模型")
        allowed_loaders = {"sentence_transformers", "sentence_transformers_auto_mean"}
        if any(model.loader not in allowed_loaders for model in models):
            raise ValueError("存在不支持的模型 loader")
        config = cls(
            benchmark_path=repository_root / data["benchmark_path"],
            chunks_directory=repository_root / data["chunks_directory"],
            embedding_cache_directory=repository_root / data["embedding_cache_directory"],
            results_directory=repository_root / data["results_directory"],
            similarity=data["similarity"],
            top_k=top_k,
            batch_size=int(data["batch_size"]),
            max_seq_length=int(data["max_seq_length"]),
            device=data["device"],
            models=models,
        )
        if config.batch_size <= 0 or config.max_seq_length <= 0:
            raise ValueError("batch_size 和 max_seq_length 必须为正整数")
        return config


def load_chunks(directory: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            chunks.extend(json.loads(line) for line in source if line.strip())
    if not chunks:
        raise ValueError(f"{directory} 中没有 Chunk；请先运行 Phase 2")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("语料中存在重复 chunk_id")
    return chunks


def corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != chunk["text_sha256"]:
            raise ValueError(f"Chunk 内容指纹不匹配：{chunk['chunk_id']}")
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk["text_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def benchmark_fingerprint(benchmark: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(benchmark, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_vectors(vectors: NDArray[np.float32]) -> None:
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("向量必须为有限值 float32 二维矩阵")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("向量必须经过 L2 归一化")


def rank_embeddings(
    corpus_vectors: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    chunk_ids: list[str],
    question_ids: list[str],
    *,
    maximum_k: int,
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """用归一化向量点积执行确定性的精确余弦检索。"""
    if corpus_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("Embedding 必须是二维矩阵")
    if corpus_vectors.shape[0] != len(chunk_ids):
        raise ValueError("Corpus Embedding 数量与 chunk_ids 不一致")
    if query_vectors.shape[0] != len(question_ids):
        raise ValueError("Query Embedding 数量与 question_ids 不一致")
    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Corpus 与 Query Embedding 维数不一致")
    maximum_k = min(maximum_k, len(chunk_ids))
    scores = query_vectors @ corpus_vectors.T
    rankings: dict[str, list[str]] = {}
    ranking_scores: dict[str, list[float]] = {}
    for index, question_id in enumerate(question_ids):
        order = np.argsort(-scores[index], kind="stable")[:maximum_k]
        rankings[question_id] = [chunk_ids[position] for position in order]
        ranking_scores[question_id] = [float(scores[index, position]) for position in order]
    return rankings, ranking_scores


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_embeddings(
    directory: Path,
    spec: EmbeddingModelSpec,
    vectors: NDArray[np.float32],
    chunk_ids: list[str],
    *,
    fingerprint: str,
    max_seq_length: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{spec.id}.npz",
        vectors=np.asarray(vectors, dtype=np.float32),
        chunk_ids=np.asarray(chunk_ids),
    )
    _write_json(
        directory / f"{spec.id}.metadata.json",
        {
            "model_id": spec.id,
            "model_name": spec.model_name,
            "revision": spec.revision,
            "corpus_fingerprint": fingerprint,
            "max_seq_length": max_seq_length,
            "normalized": True,
            "dtype": "float32",
            "dimension": int(vectors.shape[1]),
            "count": int(vectors.shape[0]),
        },
    )


def build_provider(
    spec: EmbeddingModelSpec,
    config: EmbeddingBenchmarkConfig,
) -> SentenceTransformerProvider:
    return SentenceTransformerProvider(
        spec.model_name,
        revision=spec.revision,
        device=config.device,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length,
    )


def run_model_benchmark(
    spec: EmbeddingModelSpec,
    config: EmbeddingBenchmarkConfig,
    chunks: list[dict[str, Any]],
    benchmark: dict[str, Any],
    *,
    provider: EmbeddingProvider | None = None,
    reuse_embeddings: bool = False,
) -> dict[str, Any]:
    """运行一个模型；provider 参数用于无模型下载的确定性测试。"""
    load_started = time.perf_counter()
    active_provider = provider or build_provider(spec, config)
    model_load_seconds = time.perf_counter() - load_started

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    fingerprint = corpus_fingerprint(chunks)
    if reuse_embeddings:
        metadata = json.loads((config.embedding_cache_directory / f"{spec.id}.metadata.json").read_text(encoding="utf-8"))
        expected = {"model_name": spec.model_name, "revision": spec.revision,
                    "corpus_fingerprint": fingerprint, "max_seq_length": config.max_seq_length}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("向量缓存与当前配置或 Corpus 不一致")
        with np.load(config.embedding_cache_directory / f"{spec.id}.npz", allow_pickle=False) as cached:
            corpus_vectors = cached["vectors"]
            if cached["chunk_ids"].tolist() != chunk_ids:
                raise ValueError("缓存 Chunk 顺序不匹配")
        previous = json.loads((config.results_directory / f"{spec.id}.json").read_text(encoding="utf-8"))
        if previous["controlled_variables"]["corpus_fingerprint"] != fingerprint or previous["model"]["revision"] != spec.revision:
            raise ValueError("原始编码结果与缓存不一致")
        corpus_encoding_seconds = previous["timings_seconds"]["corpus_encoding"]
    else:
        corpus_started = time.perf_counter()
        corpus_vectors = active_provider.embed([chunk["text"] for chunk in chunks])
        corpus_encoding_seconds = time.perf_counter() - corpus_started
    validate_vectors(corpus_vectors)

    questions = benchmark["questions"]
    question_ids = [question["id"] for question in questions]
    query_started = time.perf_counter()
    query_vectors = active_provider.embed([question["question"] for question in questions])
    query_encoding_seconds = time.perf_counter() - query_started
    validate_vectors(query_vectors)

    search_started = time.perf_counter()
    rankings, ranking_scores = rank_embeddings(
        corpus_vectors,
        query_vectors,
        chunk_ids,
        question_ids,
        maximum_k=len(chunks),
    )
    search_seconds = time.perf_counter() - search_started
    relevance = {
        question["id"]: {
            evidence["chunk_id"]: evidence["relevance"]
            for evidence in question["evidence"]
        }
        for question in questions
    }
    metrics, per_query = evaluate_rankings(rankings, relevance, top_k=config.top_k)
    for item in per_query:
        item["scores"] = ranking_scores[item["question_id"]][:max(config.top_k)]

    fingerprint = corpus_fingerprint(chunks)
    _save_embeddings(
        config.embedding_cache_directory,
        spec,
        corpus_vectors,
        chunk_ids,
        fingerprint=fingerprint,
        max_seq_length=config.max_seq_length,
    )
    np.savez_compressed(config.embedding_cache_directory / f"{spec.id}.queries.npz",
                        vectors=query_vectors, question_ids=np.asarray(question_ids))
    result = {
        "schema_version": "1.0",
        "experiment": "phase4-embedding-benchmark",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_embeddings_reused": reuse_embeddings,
        "corpus_encoding_timing_note": "复用时保留首次实际语料编码耗时；其余耗时为本次测量",
        "model": {
            "id": spec.id,
            "model_name": spec.model_name,
            "revision": spec.revision,
            "loader": spec.loader,
            "license": spec.license,
            "dimension": active_provider.dimension,
        },
        "controlled_variables": {
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_fingerprint": benchmark_fingerprint(benchmark),
            "mrr_cutoff": len(chunks),
            "corpus_id": benchmark["corpus_id"],
            "corpus_fingerprint": fingerprint,
            "chunk_count": len(chunks),
            "question_count": len(questions),
            "similarity": config.similarity,
            "top_k": list(config.top_k),
            "max_seq_length": config.max_seq_length,
            "normalized": True,
            "dtype": "float32",
            "query_prefix": "",
            "device": config.device,
            "batch_size": config.batch_size,
        },
        "metrics": metrics,
        "timings_seconds": {
            "model_load": model_load_seconds,
            "corpus_encoding": corpus_encoding_seconds,
            "query_encoding": query_encoding_seconds,
            "search_total": search_seconds,
            "search_mean_per_query": search_seconds / len(questions),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "torch": importlib.metadata.version("torch"),
        },
        "per_query": per_query,
    }
    _write_json(config.results_directory / f"{spec.id}.json", result)
    return result


def write_experiment_summary(config: EmbeddingBenchmarkConfig) -> dict[str, Any]:
    results = []
    controls = None
    for spec in config.models:
        path = config.results_directory / f"{spec.id}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["model"]["revision"] != spec.revision:
                raise ValueError("汇总结果的模型 revision 与配置不一致")
            if controls is None:
                controls = result["controlled_variables"]
            elif controls != result["controlled_variables"]:
                raise ValueError("模型结果控制变量不一致，必须全部重跑或重新评测")
            results.append(
                {
                    "model_id": spec.id,
                    "model_name": spec.model_name,
                    "revision": spec.revision,
                    "dimension": result["model"]["dimension"],
                    **result["metrics"],
                    **result["timings_seconds"],
                }
            )
    if not results:
        raise ValueError("没有模型结果可汇总")
    ranked = sorted(
        results,
        key=lambda item: (item["mrr"], item[f"ndcg@{max(config.top_k)}"]),
        reverse=True,
    )
    summary = {
        "schema_version": "1.0",
        "experiment": "phase4-embedding-benchmark",
        "selection_rule": f"MRR descending, then nDCG@{max(config.top_k)} descending",
        "controlled_variables": controls,
        "completed_models": len(results),
        "expected_models": len(config.models),
        "selected_model_id": ranked[0]["model_id"] if len(results) == len(config.models) else None,
        "results": ranked,
    }
    _write_json(config.results_directory / "summary.json", summary)
    return summary
