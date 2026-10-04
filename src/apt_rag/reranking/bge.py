"""固定 revision 的 BGE 分类模型，使用当前 Transformers API 输出原始 logits。"""

from __future__ import annotations

from typing import Any

from apt_rag.reranking.base import RerankScores


class BGEReranker:
    def __init__(self, spec: dict[str, Any]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if spec["device"] != "cpu" or spec["dtype"] != "float32":
            raise ValueError("本阶段固定现有 CPU/float32 环境，不改变旧阶段 Torch 安装")
        torch.set_num_threads(spec["cpu_threads"])
        torch.manual_seed(spec["seed"])
        torch.use_deterministic_algorithms(True)
        kwargs = {"revision": spec["revision"], "trust_remote_code": False,
                  "local_files_only": spec["local_files_only"]}
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec["model_name"], dtype=torch.float32, **kwargs,
        ).to(spec["device"])
        self.model.eval()
        if self.model.config.num_labels != 1:
            raise ValueError("BGE Reranker 必须是单 logit 分类模型")
        self.spec, self.torch = spec, torch

    def score(self, query: str, passages: list[str]) -> RerankScores:
        if not passages:
            return RerankScores([], [], [])
        scores, original_lengths, input_lengths = [], [], []
        for start in range(0, len(passages), self.spec["batch_size"]):
            documents = passages[start:start + self.spec["batch_size"]]
            queries = [query] * len(documents)
            original = self.tokenizer(queries, documents, truncation=False, padding=False)
            original_lengths.extend(len(tokens) for tokens in original["input_ids"])
            inputs = self.tokenizer(
                queries, documents, padding=True, truncation=self.spec["truncation"],
                max_length=self.spec["max_length"], return_tensors="pt",
            ).to(self.spec["device"])
            input_lengths.extend(inputs["attention_mask"].sum(dim=1).tolist())
            with self.torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1).float()
            scores.extend(logits.cpu().tolist())
        return RerankScores(scores, original_lengths, input_lengths)
