"""可替换 Reranker 协议；导入时不加载 Torch 或模型。"""

from apt_rag.reranking.base import Reranker, RerankScores

__all__ = ["Reranker", "RerankScores"]
