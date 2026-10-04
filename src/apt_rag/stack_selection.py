"""离线校验 Phase 8 决策、参数引用及实验范围；不创建 Pipeline 或调用模型。"""

import hashlib
import json
from pathlib import Path

REFERENCE_KEYS = {
    "processing", "embedding", "vectorstore", "generation", "retrieval", "prompt",
    "benchmark", "manifest", "phase4", "phase5", "phase6", "phase7",
}


def json_fingerprint(value: dict) -> str:
    # 与既有实验相同的 JSON 内容指纹，不受 Windows CRLF 或缩进影响。
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_selection(root: Path, decision: dict) -> dict:
    require(decision["schema_version"] == "1.0" and decision["phase"] == 8,
            "必须使用 Phase 8 决策版本")
    require(decision["status"] == "selected_not_integrated", "选型不能冒充已集成验证")
    require(decision["objective"] == "quality_first_research_not_latency_sla", "需明确质量优先研究目标")
    require(set(decision["references"]) == REFERENCE_KEYS, "选型依据必须完整覆盖配置、数据与 Phase 4–7")
    sources = {}
    for name, ref in decision["references"].items():
        path = (root / ref["path"]).resolve()
        require(path.is_relative_to(root.resolve()), "引用路径不能越出仓库")
        value = json.loads(path.read_text(encoding="utf-8"))
        require(json_fingerprint(value) == ref["json_sha256"], f"{name} 内容指纹改变；需重新审查选型")
        sources[name] = value

    selected = decision["selection"]
    models = {row["id"]: row for row in sources["embedding"]["models"]}
    require(selected["embedding_model_id"] in models, "Embedding 未参与既有实验")
    model = models[selected["embedding_model_id"]]
    require(selected["embedding_model_id"] == sources["phase4"]["selected_model_id"],
            "Embedding 不符合 Phase 4 的已声明选择规则")
    require(selected["vectorstore_backend"] in sources["vectorstore"]["backends"], "向量存储未参与实验")
    require(selected["retrieval_strategy"] in sources["retrieval"]["strategies"], "检索策略未参与实验")
    require(selected["embedding_model_id"] == sources["vectorstore"]["embedding_model_id"]
            == sources["retrieval"]["embedding_model_id"], "向量库/检索实验必须使用同一 Embedding")

    controls = [sources["phase4"]["controlled_variables"], sources["phase5"]["controlled_variables"],
                sources["phase6"]["controls"], sources["phase7"]["controls"]]
    for field in ("corpus_fingerprint", "benchmark_fingerprint", "benchmark_id", "question_count"):
        require(all(row[field] == controls[0][field] for row in controls), f"实验的 {field} 不一致")
    require(controls[0]["benchmark_fingerprint"] == json_fingerprint(sources["benchmark"]), "问题集不是同一版本")
    require(controls[2]["prompt_fingerprint"] == json_fingerprint(sources["prompt"]), "生成 Prompt 改变")
    require(model["revision"] == controls[1]["revision"]
            == controls[3]["embedding_model"]["revision"], "Embedding revision 不一致")
    for field in ("corpus_vectors_sha256", "query_vectors_sha256"):
        require(controls[1][field] == controls[3][field], "向量库与检索不是同一向量输入")
    require(sources["retrieval"]["reranker"] == sources["phase7"]["reranker_spec"], "Reranker 参数改变")
    require(sources["vectorstore"]["hnsw"] == controls[1]["hnsw"], "HNSW 参数改变")
    require(sources["embedding"]["max_seq_length"] == controls[0]["max_seq_length"]
            == controls[3]["embedding_max_seq_length"], "Embedding 截断预算改变")

    providers = {row["id"]: row for row in sources["generation"]["providers"]}
    evaluated_providers = {row["provider_id"] for row in sources["phase6"]["results"]}
    for field in ("primary_llm_provider_id", "local_llm_provider_id"):
        require(selected[field] in providers and selected[field] in evaluated_providers, "LLM 未参与实验")
    require(providers[selected["local_llm_provider_id"]]["kind"] == "ollama", "本地方案必须使用本地 Ollama")
    require(sources["generation"]["context_mode"] == "ground_truth_evidence",
            "Phase 6 应为 oracle context，不能误记为端到端实验")
    policies = decision["policies"]
    require(policies == {
        "context_mode": "retrieved_chunks", "context_top_k": max(sources["retrieval"]["top_k"]),
        "automatic_provider_fallback": False, "cloud_input_policy": "public_corpus_only",
        "query_rewrite": False, "on_context_overflow": "fail_without_silent_truncation",
        "on_unknown_citation": "reject_unknown_label",
    }, "Context/隐私/显式后端选择政策改变；需重新审查")
    return sources
