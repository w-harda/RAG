"""可替换的文本 Embedding 接口。"""

from apt_rag.embedding.base import EmbeddingProvider
from apt_rag.embedding.sentence_transformer import SentenceTransformerProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerProvider"]
