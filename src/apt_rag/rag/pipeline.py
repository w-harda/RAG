"""LangChain LCEL 强制编排；复用固定算法、Provider、预算和引用契约。"""

import math
import time

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence
from langsmith import tracing_context

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

        self.retrieval_chain = RunnableSequence(
            RunnableLambda(self._validate_question, name="validate_question"),
            RunnableLambda(self._embed_query, name="embed_query"),
            RunnableLambda(self._dense_retrieval, name="dense_retrieval"),
            RunnableLambda(self._sparse_retrieval, name="bm25_retrieval"),
            RunnableLambda(self._fuse_candidates, name="rrf_fusion"),
            RunnableLambda(self._rerank_candidates, name="rerank_top_k"),
            RunnableLambda(self._build_context, name="construct_context"),
            name="apt_rag_retrieval",
        )
        generation_chain = RunnableSequence(
            RunnableLambda(self._identify_model, name="model_identity"),
            RunnableLambda(self._generate, name="llm_generation"),
            RunnableLambda(self._validate_generation, name="validate_generation_usage"),
            RunnableLambda(self._bind_citations, name="resolve_citations"),
            RunnableLambda(self._accept_answer, name="accept_answer"),
            name="apt_rag_generation",
        )
        prepared_chain = (
            RunnableLambda(self._check_cloud_policy, name="cloud_input_policy")
            | RunnableLambda(self._measure_budget, name="token_budget")
            | RunnableBranch(
                (lambda item: item["_dry_run"], RunnableLambda(self._prepared_result, name="dry_run_result")),
                generation_chain,
            )
        )
        self.chain = RunnableSequence(
            self.retrieval_chain,
            RunnableBranch(
                (lambda item: not item["sources"], RunnableLambda(self._no_evidence, name="no_evidence")),
                prepared_chain,
            ),
            name="apt_rag",
        )

    @staticmethod
    def _public_item(item):
        return {key: value for key, value in item.items() if not key.startswith("_")}

    @staticmethod
    def _invoke(chain, request):
        # 框架跟踪不得因外部环境变量而把问题/报告 Context 上传到第三方。
        with tracing_context(enabled=False, parent=False):
            return chain.invoke(request)

    def retrieve(self, question: str) -> dict:
        return self._public_item(self._invoke(self.retrieval_chain, {"question": question}))

    def answer(self, question: str, *, dry_run=False) -> dict:
        return self._invoke(self.chain, {"question": question, "dry_run": dry_run,
                                         "started": time.perf_counter()})

    def _validate_question(self, request):
        question = request["question"]
        if not question.strip() or len(question) > self.max_question_characters:
            raise ValueError("问题为空或超过配置长度上限")
        return {"question": question, "_dry_run": request.get("dry_run", False),
                "_answer_started": request.get("started"), "_timings": {}}

    def _embed_query(self, item):
        started = time.perf_counter()
        item["_vector"] = self.embedding.embed([item["question"]])[0]
        item["_timings"]["embedding"] = time.perf_counter() - started
        return item

    def _dense_retrieval(self, item):
        item["_fusion_started"] = time.perf_counter()
        item["_dense"] = self.dense.retrieve(item["question"], item["_vector"], self.config["candidate_k"])
        return item

    def _sparse_retrieval(self, item):
        item["_sparse"] = self.sparse.retrieve(item["question"], None, self.config["candidate_k"])
        return item

    def _fuse_candidates(self, item):
        item["_fused"] = fuse_rrf([item["_dense"], item["_sparse"]], self.ids,
                                  **self.config["rrf"], top_k=self.config["candidate_k"])
        item["_timings"]["retrieval_fusion"] = time.perf_counter() - item["_fusion_started"]
        return item

    def _rerank_candidates(self, item):
        fused = item["_fused"]
        started = time.perf_counter()
        rerank = self.reranker.score(item["question"], [self.chunks[h.chunk_id]["text"] for h in fused])
        if len(rerank.scores) != len(fused) or any(not math.isfinite(s) for s in rerank.scores):
            raise ValueError("Reranker 返回数量或分数非法")
        positions = {key: index for index, key in enumerate(self.ids)}
        ranked = sorted((RankedChunk(h.chunk_id, score) for h, score in zip(fused, rerank.scores, strict=True)),
                        key=lambda h: (-h.score, positions[h.chunk_id]))[:self.top_k]
        item["_timings"]["reranking"] = time.perf_counter() - started
        item["_ranked"], item["_rerank"] = ranked, rerank
        return item

    def _build_context(self, item):
        ranked, rerank = item["_ranked"], item["_rerank"]
        context = construct_context(item["question"], [self.chunks[h.chunk_id] for h in ranked], self.prompt)
        return {**context, "question": item["question"], "rankings": [vars(h) for h in ranked],
                "candidate_ids": [h.chunk_id for h in item["_fused"]], "reranking": {
                    "scores": rerank.scores, "original_tokens": rerank.original_tokens,
                    "input_tokens": rerank.input_tokens,
                }, "timings_seconds": item["_timings"], "_dry_run": item["_dry_run"],
                "_answer_started": item["_answer_started"]}

    def _no_evidence(self, item):
        return {**self._public_item(item), "answer": ABSTENTION, "citations": [],
                "abstained": True, "status": "no_evidence"}

    def _check_cloud_policy(self, item):
        # 云端政策必须在 Token 预检或任何 LLM 调用前执行。
        if getattr(self.llm, "spec", {}).get("kind") == "deepseek" and not self.public_corpus_verified:
            raise ValueError("云端只接受已核验的公开 Corpus；禁止发送未知来源")
        return item

    def _measure_budget(self, item):
        item["token_budget"] = self.budget.measure(item["messages"])
        return item

    def _prepared_result(self, item):
        return {**self._public_item(item), "status": "prepared_without_generation"}

    def _identify_model(self, item):
        item["identity"] = self.llm.identity()
        return item

    def _generate(self, item):
        result = self.llm.generate(item["messages"])
        item["_generation"], item["generation"] = result, result.to_dict()
        return item

    def _validate_generation(self, item):
        result = item["_generation"]
        try:
            if not result.answer.strip() or result.response_model != self.llm.spec["model"] or result.finish_reason != "stop":
                raise ValueError("生成模型不符、空答案或输出未正常完成；不接受截断/被过滤的答案")
            if self.backend_fingerprint and result.system_fingerprint != self.backend_fingerprint:
                raise ValueError("同一 Pipeline 会话的云端 fingerprint 改变；禁止静默混合模型")
            self.budget.verify_usage(result, item["token_budget"])
        except ValueError as error:
            raise RAGResponseError(str(error), self._public_item(item)) from None
        return item

    def _bind_citations(self, item):
        try:
            item["_citations"] = resolve_citations(item["_generation"].answer, item["sources"])
        except ValueError as error:
            raise RAGResponseError(str(error), self._public_item(item)) from None
        return item

    def _accept_answer(self, item):
        result = item["_generation"]
        self.backend_fingerprint = result.system_fingerprint
        return {**self._public_item(item), **item["_citations"], "answer": result.answer,
                "answer_sha256": fingerprint(result.answer), "status": "answered",
                "total_seconds": time.perf_counter() - item["_answer_started"]}
