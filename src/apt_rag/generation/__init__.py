"""与实验编排和具体框架解耦的 LLM 调用接口。"""

from apt_rag.generation.providers import GenerationResult, LLMProvider

__all__ = ["GenerationResult", "LLMProvider"]
