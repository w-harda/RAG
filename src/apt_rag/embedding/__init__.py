"""可替换的文本 Embedding 接口。"""

from apt_rag.embedding.base import EmbeddingProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerProvider"]


def __getattr__(name: str):
    if name == "SentenceTransformerProvider":
        from apt_rag.embedding.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider
    raise AttributeError(name)
