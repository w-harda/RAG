"""准备固定 RAG 索引并核验 Top-30；可选下载官方 tokenizer、预检 20 题预算。"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np

from apt_rag.evaluation.embedding_benchmark import _write_json
from apt_rag.generation.providers import build_provider, fingerprint
from apt_rag.rag.budget import TokenBudget
from apt_rag.rag.context import construct_context
from apt_rag.rag.runtime import FaissRetriever, fixed_inputs, load_settings, open_index
from apt_rag.retrieval import DenseRetriever, fuse_rrf

ROOT = Path(__file__).resolve().parents[1]


def download_tokenizer(root, config, provider_id):
    spec = config["tokenizers"][provider_id]
    response = httpx.get(spec["url"], timeout=60, follow_redirects=True)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != spec["archive_sha256"]:
        raise ValueError("官方 tokenizer 包改变；禁止静默更新")
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    directory = root / spec["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    # 只提取已知数据文件；不执行官方示例 Python 或 trust_remote_code。
    for name, key in (("tokenizer.json", "tokenizer_sha256"), ("tokenizer_config.json", "config_sha256")):
        content = archive.read("deepseek_v4_tokenizer/" + name)
        if hashlib.sha256(content).hexdigest() != spec[key]:
            raise ValueError("Tokenizer 文件指纹错误")
        path = directory / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("已有 tokenizer 改变，不覆盖")
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rag.json")
    parser.add_argument("--download-tokenizer", action="store_true")
    parser.add_argument("--check-budgets", action="store_true", help="复用已验证 Phase 7 排名，仅预检，不生成或重新重排")
    args = parser.parse_args()
    config, decision, sources = load_settings(ROOT, args.config)
    primary = decision["selection"]["primary_llm_provider_id"]
    if args.download_tokenizer:
        download_tokenizer(ROOT, config, primary)
    inputs = fixed_inputs(ROOT, sources)
    store, audit = open_index(ROOT, config, decision, sources, inputs, create=True)
    actual = FaissRetriever(store, inputs["ids"])
    reference = DenseRetriever(inputs["vectors"], inputs["ids"])
    old = json.loads((ROOT / sources["retrieval"]["results_directory"] / "rankings.json").read_text(encoding="utf-8"))
    records = []
    k = sources["retrieval"]["candidate_k"]
    for question, vector, previous in zip(inputs["benchmark"]["questions"], inputs["queries"], old["queries"], strict=True):
        hits = actual.retrieve(question["question"], vector, k)
        exact = reference.retrieve(question["question"], vector, k)
        ids, ref_ids = [h.chunk_id for h in hits], [h.chunk_id for h in exact]
        score_match = np.allclose([h.score for h in hits], [h.score for h in exact], atol=2e-5, rtol=0)
        bm25 = [type(hits[0])(**hit) for hit in previous["bm25_candidates"]]
        hybrid = fuse_rrf([hits, bm25], inputs["ids"], **sources["retrieval"]["rrf"], top_k=k)
        hybrid_ids = [h.chunk_id for h in hybrid]
        records.append({"question_id": question["id"], "dense_ids": ids, "exact_dense_ids": ref_ids,
                        "faiss_scores": [h.score for h in hits], "exact_scores": [h.score for h in exact],
                        "ann_recall@30": len(set(ids) & set(ref_ids)) / len(ref_ids),
                        "ordered_ids_equal": ids == ref_ids, "scores_within_tolerance": bool(score_match),
                        "hybrid_ids": hybrid_ids,
                        "phase7_hybrid_ids_equal": hybrid_ids == [h["chunk_id"] for h in previous["hybrid_candidates"]]})
    verified = all(r["ordered_ids_equal"] and r["scores_within_tolerance"] and r["phase7_hybrid_ids_equal"] for r in records)
    report = {"schema_version": "1.0", "experiment": "phase9-faiss-top30-integration",
              "controls": inputs["controls"], "rag_config_fingerprint": fingerprint(config),
              "index_insertion_order": "corpus_order", "top30_verified": verified, "queries": records,
              "reranker_for_budget": "phase7_scores_reused_only_when_identical_candidates_and_model"}
    if not verified:
        _write_json(ROOT / "data/processed/rag/top30-failure.json", report)
        raise ValueError("Top-30 接入存在排名偏差；未通过，需另存结果评估，不改 Phase 7")
    if args.check_budgets:
        by_id = {c["chunk_id"]: c for c in inputs["chunks"]}
        providers = {p["id"]: p for p in sources["generation"]["providers"]}
        budgets = {}
        for provider_id in (primary, decision["selection"]["local_llm_provider_id"]):
            spec = providers[provider_id]
            active = build_provider(spec, sources["generation"]) if spec["kind"] == "ollama" else None
            try:
                counter = TokenBudget(ROOT, config, spec, sources["generation"]["max_output_tokens"], provider=active)
                rows = []
                for question, previous in zip(inputs["benchmark"]["questions"], old["queries"], strict=True):
                    chunks = [by_id[h["chunk_id"]] for h in previous["rankings"]["hybrid_reranker"]]
                    context = construct_context(question["question"], chunks, sources["prompt"])
                    rows.append({"question_id": question["id"], "messages_sha256": context["messages_sha256"],
                                 **counter.measure(context["messages"])})
                budgets[provider_id] = rows
            finally:
                if active is not None:
                    active.client.close()
        report["token_budgets"] = budgets
    target = ROOT / "results/phase9/retrieval_integration.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        previous_base = {k: v for k, v in previous.items() if k != "token_budgets"}
        if previous_base != {k: v for k, v in report.items() if k != "token_budgets"}:
            raise ValueError("已保存接入记录改变；禁止覆盖")
        if "token_budgets" not in report and "token_budgets" in previous:
            report["token_budgets"] = previous["token_budgets"]
    _write_json(target, report)
    _write_json(ROOT / config["index_directory"] / "audit.json", {**audit, "top30_verified": True})
    print(f"Phase 9 接入通过：{len(records)} 题 FAISS Top-30 和 Hybrid 候选与精确/Phase 7 输入一致")
    if "token_budgets" in report:
        for provider_id, rows in report["token_budgets"].items():
            print(provider_id, "20 题输入 Token 预检最大值", max(r["input_tokens"] for r in rows))


if __name__ == "__main__":
    main()
