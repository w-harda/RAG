"""固定向量 Dense、可解释 BM25 和 RRF；所有并列按原始语料顺序。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from apt_rag.evaluation.embedding_benchmark import validate_vectors


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, query_vector: np.ndarray | None, top_k: int) -> list[RankedChunk]: ...


def lexical_tokens(text: str) -> list[str]:
    """标识符保留内部 -/_，英文 casefold，中文逐字，不翻译或扩展问题。"""
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]", text.casefold())


def _validate_ids(ids: list[str]) -> None:
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("语料 ID 必须非空且无重复")


def rank_scores(scores: np.ndarray, ids: list[str], top_k: int, *, positive_only=False) -> list[RankedChunk]:
    values = np.asarray(scores)
    if top_k < 1 or values.shape != (len(ids),) or not np.isfinite(values).all():
        raise ValueError("Top-K 或检索分数非法")
    order = np.argsort(-values, kind="stable")
    if positive_only:
        order = order[values[order] > 0]
    return [RankedChunk(ids[i], float(values[i])) for i in order[:top_k]]


class DenseRetriever:
    def __init__(self, vectors: np.ndarray, ids: list[str]):
        _validate_ids(ids)
        validate_vectors(vectors)
        if vectors.shape[0] != len(ids):
            raise ValueError("语料向量与 ID 数量不同")
        self.vectors, self.ids = vectors, ids

    def retrieve(self, query: str, query_vector: np.ndarray | None, top_k: int) -> list[RankedChunk]:
        if query_vector is None or query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("Dense 检索需要维数匹配的 Query 向量")
        validate_vectors(query_vector[None, :])
        return rank_scores(self.vectors @ query_vector, self.ids, top_k)


class BM25Retriever:
    def __init__(self, texts: list[str], ids: list[str], *, k1: float, b: float):
        _validate_ids(ids)
        if len(texts) != len(ids) or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 语料或参数非法")
        counts = [Counter(lexical_tokens(text)) for text in texts]
        lengths = np.asarray([sum(c.values()) for c in counts], dtype=np.float64)
        average = float(lengths.mean()) or 1.0
        self.normalizer = k1 * (1 - b + b * lengths / average)
        postings = defaultdict(list)
        for index, terms in enumerate(counts):
            for term, frequency in terms.items():
                postings[term].append((index, frequency))
        self.postings = {}
        for term, values in postings.items():
            df = len(values)
            self.postings[term] = (
                np.asarray([i for i, _ in values]), np.asarray([tf for _, tf in values], dtype=float),
                math.log(1 + (len(ids) - df + 0.5) / (df + 0.5)),
            )
        self.ids, self.k1 = ids, k1

    def retrieve(self, query: str, query_vector: np.ndarray | None, top_k: int) -> list[RankedChunk]:
        scores = np.zeros(len(self.ids), dtype=np.float64)
        for term in sorted(set(lexical_tokens(query))):
            if term in self.postings:
                positions, frequencies, idf = self.postings[term]
                scores[positions] += idf * frequencies * (self.k1 + 1) / (
                    frequencies + self.normalizer[positions]
                )
        # 零分文档没有词法命中，不能依靠任意语料排序混入 Hybrid。
        return rank_scores(scores, self.ids, top_k, positive_only=True)


def fuse_rrf(
    branches: list[list[RankedChunk]], corpus_ids: list[str], *, rank_constant: int,
    weights: list[float], top_k: int,
) -> list[RankedChunk]:
    _validate_ids(corpus_ids)
    if rank_constant < 1 or len(weights) != len(branches) or not branches:
        raise ValueError("RRF 分支或参数非法")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("RRF 权重必须有限且为正")
    positions = {chunk_id: index for index, chunk_id in enumerate(corpus_ids)}
    scores = np.zeros(len(corpus_ids), dtype=np.float64)
    for weight, hits in zip(weights, branches, strict=True):
        if len({hit.chunk_id for hit in hits}) != len(hits):
            raise ValueError("RRF 分支有重复 ID")
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id not in positions:
                raise ValueError("RRF 候选不在语料中")
            scores[positions[hit.chunk_id]] += weight / (rank_constant + rank)
    return rank_scores(scores, corpus_ids, top_k, positive_only=True)
