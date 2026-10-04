"""与模型和框架无关的最小向量库契约。"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def build(self, vectors: np.ndarray, ids: list[str], metadata: list[dict[str, Any]]) -> None: ...

    def search(self, vector: np.ndarray, top_k: int,
               where: dict[str, Any] | None = None) -> list[SearchHit]: ...

    def get(self, chunk_id: str) -> dict[str, Any]: ...


def validate_batch(vectors: np.ndarray, ids: list[str], metadata: list[dict[str, Any]]) -> None:
    if vectors.ndim != 2 or not len(ids) or len(ids) != len(metadata) or len(ids) != len(vectors):
        raise ValueError("向量、ID 与 Metadata 长度必须相等且非空")
    if len(set(ids)) != len(ids):
        raise ValueError("Chunk ID 不能重复")
    validate_matrix(vectors)


def validate_matrix(vectors: np.ndarray) -> None:
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("向量必须是有限值 float32")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("余弦检索要求 L2 归一化向量")


def prepare_query(vector: np.ndarray, dimension: int, top_k: int) -> np.ndarray:
    if top_k <= 0 or vector.shape != (dimension,):
        raise ValueError("top_k 或查询向量维数无效")
    matrix = np.ascontiguousarray(vector.reshape(1, -1))
    validate_matrix(matrix)
    return matrix
