"""离线复核 Phase 9 接入与冒烟；可选核验本地 Chunk，不调用模型或 API。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from apt_rag.generation.providers import fingerprint
from apt_rag.rag.context import SOURCE_FIELDS, construct_context, resolve_citations
from apt_rag.rag.runtime import fixed_inputs, load_settings

ROOT = Path(__file__).resolve().parents[1]


def validate_records(root, config_path, *, verify_inputs=False):
    config, decision, sources = load_settings(root, config_path)
    directory = root / "results/phase9"
    report = json.loads((directory / "retrieval_integration.json").read_text(encoding="utf-8"))
    previous = json.loads((root / sources["retrieval"]["results_directory"] / "rankings.json").read_text(encoding="utf-8"))
    benchmark = sources["benchmark"]
    require(report["rag_config_fingerprint"] == fingerprint(config), "接入配置改变")
    require(report["controls"] == previous["controls"] and report["top30_verified"], "接入控制变量改变或未通过")
    require([r["question_id"] for r in report["queries"]] == [q["id"] for q in benchmark["questions"]], "接入问题集改变")
    for row, old in zip(report["queries"], previous["queries"], strict=True):
        require(row["dense_ids"] == row["exact_dense_ids"] == [h["chunk_id"] for h in old["dense_candidates"]], "Dense Top-30 改变")
        require(row["hybrid_ids"] == [h["chunk_id"] for h in old["hybrid_candidates"]], "Hybrid 候选改变")
        require(row["ann_recall@30"] == 1 and row["ordered_ids_equal"] and row["phase7_hybrid_ids_equal"], "排名验证记录不一致")
        require(row["scores_within_tolerance"] and np.allclose(row["faiss_scores"], row["exact_scores"], atol=2e-5, rtol=0), "分数不一致")
    require(set(report["token_budgets"]) == {decision["selection"][key] for key in
            ("primary_llm_provider_id", "local_llm_provider_id")}, "两后端的预算必须完整")
    for provider_id, rows in report["token_budgets"].items():
        require([r["question_id"] for r in rows] == [q["id"] for q in benchmark["questions"]], "预算问题不完整")
        for row in rows:
            require(row["context_window_tokens"] == config["context_window_tokens"] and
                    row["safety_margin"] == config["token_safety_margin"] and
                    row["max_output_tokens"] == sources["generation"]["max_output_tokens"] and
                    row["tokenizer_sha256"] == config["tokenizers"][provider_id]["tokenizer_sha256"], "预算参数改变")
            require(row["input_tokens"] + row["safety_margin"] + row["max_output_tokens"] <= row["context_window_tokens"], "预算超限")
    inputs = fixed_inputs(root, sources) if verify_inputs else None
    by_id = {c["chunk_id"]: c for c in inputs["chunks"]} if inputs else None
    reports = {r["report_id"]: r for r in sources["manifest"]["reports"]}
    if by_id is not None:
        for question, old in zip(benchmark["questions"], previous["queries"], strict=True):
            context = construct_context(question["question"], [by_id[h["chunk_id"]] for h in
                old["rankings"]["hybrid_reranker"]], sources["prompt"])
            for rows in report["token_budgets"].values():
                row = next(r for r in rows if r["question_id"] == question["id"])
                require(row["messages_sha256"] == context["messages_sha256"], "预算不是完整的检索 Context")
    smoke_paths = sorted(directory.glob("smoke-*.json"))
    require(bool(smoke_paths), "缺少真实冒烟记录")
    for path in smoke_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        require(result["controls"]["rag_config_fingerprint"] == fingerprint(config) and
                result["controls"]["corpus_fingerprint"] == report["controls"]["corpus_fingerprint"] and
                result["controls"]["context_mode"] == "retrieved_chunks", "冒烟不是相同管线/语料")
        require(result["status"] == "answered" and result["generation"]["finish_reason"] == "stop", "冒烟未成功")
        require(result["question"] in {q["question"] for q in benchmark["questions"]}, "冒烟不是公开固定问题")
        require(result["answer"] == result["generation"]["answer"] and result["answer_sha256"] == fingerprint(result["answer"]), "回答指纹改变")
        require(result["generation"]["response_model"] == result["controls"]["provider_spec"]["model"], "生成模型改变")
        for label, source in enumerate(result["sources"], 1):
            require(source["label"] == f"S{label}" and set(source) == set(SOURCE_FIELDS) | {"label"}, "来源契约改变")
            metadata = reports[source["report_id"]]
            require(all(source[field] == metadata[key] for field, key in (
                ("document_title", "title"), ("vendor", "vendor"), ("source_url", "url"))), "来源不是 Manifest 数据")
            if by_id is not None:
                require(all(source[field] == by_id[source["chunk_id"]][field] for field in SOURCE_FIELDS), "来源不是实际 Chunk")
        labels = resolve_citations(result["answer"], result["sources"])
        require(all(result[key] == value for key, value in labels.items()), "引用不能重算")
        require([r["chunk_id"] for r in result["rankings"]] == [s["chunk_id"] for s in result["sources"]], "排名与 Context 来源不同")
        reranking = result["reranking"]
        require(len(reranking["scores"]) == len(result["candidate_ids"]) and all(math.isfinite(s) for s in reranking["scores"]), "重排记录非法")
        if by_id is not None:
            positions = {c["chunk_id"]: i for i, c in enumerate(inputs["chunks"])}
            order = sorted(zip(result["candidate_ids"], reranking["scores"], strict=True), key=lambda h: (-h[1], positions[h[0]]))[:decision["policies"]["context_top_k"]]
            require([key for key, _ in order] == [r["chunk_id"] for r in result["rankings"]], "重排顺序不能复现")
            context = construct_context(result["question"], [by_id[s["chunk_id"]] for s in result["sources"]], sources["prompt"])
            require(context["messages_sha256"] == result["messages_sha256"], "完整检索 Context 指纹不同")
        budget, usage = result["token_budget"], result["generation"]
        require(0 < usage["prompt_tokens"] <= budget["input_tokens"] + budget["safety_margin"] and
                usage["prompt_tokens"] + budget["max_output_tokens"] <= budget["context_window_tokens"] and
                0 <= usage["output_tokens"] <= budget["max_output_tokens"], "实际用量超限")
    return len(report["queries"]), len(smoke_paths)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rag.json")
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    count, smoke = validate_records(ROOT, args.config, verify_inputs=args.verify_inputs)
    print(f"Phase 9 验证通过：{count} 题 Top-30/预算检查、{smoke} 条真实 RAG 冒烟；未调用模型/API")


if __name__ == "__main__":
    main()
