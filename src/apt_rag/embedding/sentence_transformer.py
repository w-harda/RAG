"""基于 Sentence Transformers 的本地 Embedding Provider。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """加载固定 revision 的 Hugging Face 模型并输出归一化向量。"""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str,
        device: str,
        batch_size: int,
        max_seq_length: int,
        local_files_only: bool = False,
        reject_truncated_inputs: bool = False,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.reject_truncated_inputs = reject_truncated_inputs
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self._model.max_seq_length = max_seq_length

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"模型未声明向量维数：{self.model_name}")
        return int(dimension)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if self.reject_truncated_inputs:
            tokenized = self._model.tokenizer(texts, truncation=False, padding=False)
            if any(len(tokens) > self.max_seq_length for tokens in tokenized["input_ids"]):
                raise ValueError("问题超过 Embedding Token 上限；禁止静默截断问题")
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.dimension):
            raise RuntimeError(
                f"Embedding 形状异常：预期 {(len(texts), self.dimension)}，实际 {result.shape}"
            )
        return result
