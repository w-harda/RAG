"""Embedding Provider 的最小框架无关契约。"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(Protocol):
    """把文本批量编码为 L2 归一化 float32 向量。"""

    @property
    def dimension(self) -> int:
        """返回向量维数。"""

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """编码文本，返回形状为 ``(len(texts), dimension)`` 的矩阵。"""
