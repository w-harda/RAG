"""重构前管线的测试专用快照；不进入生产装配，不作为框架回退。"""

import math
import time

from apt_rag.generation.providers import fingerprint
from apt_rag.rag.context import ABSTENTION, construct_context, resolve_citations
from apt_rag.retrieval import RankedChunk, fuse_rrf


class RAGResponseError(ValueError):
    """保留失败请求的审计记录，但不把未经验证的文本展示为答案。"""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = {**record, "status": "rejected", "validation_error": message}


class RAGPipeline:
    def __init__(self, *, chunks, embedding, dense, sparse, reranker, llm, budget, prompt,
                 retrieval_config, top_k, max_question_characters, public_corpus_verified=False):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.ids = [c["chunk_id"] for c in chunks]
        if len(self.chunks) != len(self.ids):
            raise ValueError("重复 Chunk ID")
        self.embedding, self.dense, self.sparse = embedding, dense, sparse
        self.reranker, self.llm, self.budget = reranker, llm, budget
        self.prompt, self.config, self.top_k = prompt, retrieval_config, top_k
        self.max_question_characters = max_question_characters
        self.public_corpus_verified = public_corpus_verified
        self.backend_fingerprint = None
        if not 1 <= top_k <= retrieval_config["candidate_k"]:
            raise ValueError("Context Top-K 必须在候选池预算内")

    def retrieve(self, question: str) -> dict:
        if not question.strip() or len(question) > self.max_question_characters:
            raise ValueError("问题为空或超过配置长度上限")
        timings = {}
        started = time.perf_counter()
        vector = self.embedding.embed([question])[0]
        timings["embedding"] = time.perf_counter() - started
        started = time.perf_counter()
        dense = self.dense.retrieve(question, vector, self.config["candidate_k"])
        sparse = self.sparse.retrieve(question, None, self.config["candidate_k"])
        fused = fuse_rrf([dense, sparse], self.ids, **self.config["rrf"], top_k=self.config["candidate_k"])
        timings["retrieval_fusion"] = time.perf_counter() - started
        started = time.perf_counter()
        rerank = self.reranker.score(question, [self.chunks[h.chunk_id]["text"] for h in fused])
        if len(rerank.scores) != len(fused) or any(not math.isfinite(s) for s in rerank.scores):
            raise ValueError("Reranker 返回数量或分数非法")
        positions = {key: index for index, key in enumerate(self.ids)}
        ranked = sorted((RankedChunk(h.chunk_id, score) for h, score in zip(fused, rerank.scores, strict=True)),
                        key=lambda h: (-h.score, positions[h.chunk_id]))[:self.top_k]
        timings["reranking"] = time.perf_counter() - started
        context = construct_context(question, [self.chunks[h.chunk_id] for h in ranked], self.prompt)
        return {**context, "question": question, "rankings": [vars(h) for h in ranked],
                "candidate_ids": [h.chunk_id for h in fused], "reranking": {
                    "scores": rerank.scores, "original_tokens": rerank.original_tokens,
                    "input_tokens": rerank.input_tokens,
                }, "timings_seconds": timings}

    def answer(self, question: str, *, dry_run=False) -> dict:
        started = time.perf_counter()
        item = self.retrieve(question)
        if not item["sources"]:
            return {**item, "answer": ABSTENTION, "citations": [], "abstained": True, "status": "no_evidence"}
        # 云端政策必须在 Token 预检或任何 LLM 调用前执行。
        if getattr(self.llm, "spec", {}).get("kind") == "deepseek" and not self.public_corpus_verified:
            raise ValueError("云端只接受已核验的公开 Corpus；禁止发送未知来源")
        budget = self.budget.measure(item["messages"])
        item["token_budget"] = budget
        if dry_run:
            return {**item, "status": "prepared_without_generation"}
        identity = self.llm.identity()
        result = self.llm.generate(item["messages"])
        audit = {**item, "generation": result.to_dict(), "identity": identity}
        try:
            if not result.answer.strip() or result.response_model != self.llm.spec["model"] or result.finish_reason != "stop":
                raise ValueError("生成模型不符、空答案或输出未正常完成；不接受截断/被过滤的答案")
            if self.backend_fingerprint and result.system_fingerprint != self.backend_fingerprint:
                raise ValueError("同一 Pipeline 会话的云端 fingerprint 改变；禁止静默混合模型")
            self.budget.verify_usage(result, budget)
            citations = resolve_citations(result.answer, item["sources"])
        except ValueError as error:
            raise RAGResponseError(str(error), audit) from None
        self.backend_fingerprint = result.system_fingerprint
        return {**item, **citations, "answer": result.answer, "answer_sha256": fingerprint(result.answer),
                "generation": result.to_dict(), "identity": identity, "status": "answered",
                "total_seconds": time.perf_counter() - started}
