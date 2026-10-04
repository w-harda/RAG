"""只对既有候选打分，不能引入 Ground Truth 或新候选。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankScores:
    scores: list[float]
    original_tokens: list[int]
    input_tokens: list[int]


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> RerankScores: ...
