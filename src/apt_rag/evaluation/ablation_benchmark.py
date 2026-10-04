"""四组生成消融与可审计的 AI 辅助评审；不把辅助分数冒充人工结论。"""

from __future__ import annotations

import json
import math
import platform
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from apt_rag.evaluation.embedding_benchmark import _write_json as write_json_once
from apt_rag.evaluation.llm_benchmark import estimated_cost
from apt_rag.evaluation.retrieval_benchmark import summarize_results as validate_rankings
from apt_rag.generation.providers import GenerationResult, build_provider, fingerprint
from apt_rag.rag.budget import TokenBudget
from apt_rag.rag.context import SOURCE_FIELDS, construct_context
from apt_rag.rag.runtime import FaissRetriever, fixed_inputs, load_settings, open_index
from apt_rag.retrieval import BM25Retriever, fuse_rrf

STRATEGIES = ("pure", "dense", "hybrid", "hybrid_reranker")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    # Windows 杀毒/索引程序可能短暂占用目标。仅重试同一磁盘写入，不重试 API。
    for attempt in range(5):
        try:
            write_json_once(path, value)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.2 * (attempt + 1))


def read_checkpoint(path, list_key):
    saved = read(path) if path.exists() else None
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.exists():
        return saved
    pending = read(temporary)
    if saved is not None:
        ignored = {list_key, "completed", "finished_at"}
        require({k: v for k, v in saved.items() if k not in ignored}
                == {k: v for k, v in pending.items() if k not in ignored}, "待恢复检查点协议不同")
        require(saved[list_key] == pending[list_key][:len(saved[list_key])]
                and len(saved[list_key]) <= len(pending[list_key]) <= len(saved[list_key]) + 1,
                "待恢复检查点不是原记录的相同前缀")
    print("发现未替换的磁盘检查点；先完整验证，再恢复，不重发已有 API 请求", flush=True)
    return pending


def review_protocol(root, config_path, generation_settings, rag, spec):
    plan = read(config_path)
    for key in ("results_file", "summary_file", "plot_file", "pilot_file"):
        require(Path(plan[key]).name == plan[key] and plan[key] not in {".", ".."}, "评审文件名不能逃逸实验目录")
    settings = {**generation_settings, **plan["settings"]}
    require(settings["temperature"] == 0 and settings["response_format"] == {"type": "json_object"}
            and type(settings["max_output_tokens"]) is int and settings["max_output_tokens"] > 0
            and settings["max_output_tokens"] + rag["token_safety_margin"] < settings["context_window_tokens"]
            <= rag["tokenizers"][spec["id"]]["documented_context_tokens"], "评审独立预算非法")
    prompt = read(root / plan["prompt_path"])
    return plan, settings, prompt


def protocol(root, config_path):
    config = read(config_path)
    require(config["schema_version"] == "1.0" and tuple(config["strategies"]) == STRATEGIES,
            "必须保留完全相同的四组消融")
    require(type(config["schedule_seed"]) is int and config["reference_page_radius"] == 1,
            "不支持的调度/参考页协议")
    for key in ("inputs_path", "results_directory"):
        parent = "data/processed" if key == "inputs_path" else "results"
        require((root / config[key]).resolve().is_relative_to((root / parent).resolve()),
                "实验路径超出允许目录")
    rag, decision, sources = load_settings(root, root / config["rag_config_path"])
    require(config["provider_id"] == decision["selection"]["primary_llm_provider_id"],
            "消融固定为已选择的主 LLM，不同时改变模型")
    spec = next(p for p in sources["generation"]["providers"] if p["id"] == config["provider_id"])
    require(spec["kind"] == "deepseek", "本协议使用已选择的 DeepSeek 云端模型")
    prompt = read(root / config["prompt_path"])
    review_prompt = read(root / config["review_prompt_path"])
    rubric = read(root / config["rubric_path"])
    require(rubric["benchmark_id"] == sources["benchmark"]["benchmark_id"] and
            set(rubric["required_facts"]) == {q["id"] for q in sources["benchmark"]["questions"]},
            "评审清单与问题集不一致")
    review = config["review"]
    require(review["temperature"] == 0 and review["response_format"] == {"type": "json_object"}
            and type(review["max_output_tokens"]) is int and review["max_output_tokens"] > 0
            and sources["generation"]["max_output_tokens"] + rag["token_safety_margin"]
            < review["context_window_tokens"] <= rag["tokenizers"][spec["id"]]["documented_context_tokens"],
            "评审参数或预算非法")
    previous = read(root / sources["retrieval"]["results_directory"] / "rankings.json")
    validate_rankings(previous, sources["benchmark"], sources["retrieval"])
    controls = {
        "config_fingerprint": fingerprint(config), "rag_config_fingerprint": fingerprint(rag),
        "decision_fingerprint": fingerprint(decision), "prompt_fingerprint": fingerprint(prompt),
        "review_prompt_fingerprint": fingerprint(review_prompt), "rubric_fingerprint": fingerprint(rubric),
        "retrieval_results_fingerprint": fingerprint(previous), "retrieval": previous["controls"],
        "provider_spec": spec, "temperature": sources["generation"]["temperature"],
        "max_output_tokens": sources["generation"]["max_output_tokens"],
        "context_top_k": decision["policies"]["context_top_k"],
        "context_window_tokens": rag["context_window_tokens"], "token_safety_margin": rag["token_safety_margin"],
        "reranker_execution": "reuse_phase7_verified_scores_not_new_timing",
        "review_status": "ai_assisted_same_model_family_pending_independent_human_review",
    }
    return config, rag, decision, sources, spec, prompt, review_prompt, rubric, previous, controls


def schedule(question_ids, seed):
    rng = random.Random(seed)
    ordered = list(question_ids)
    rng.shuffle(ordered)
    rows = []
    for qid in ordered:
        strategies = list(STRATEGIES)
        rng.shuffle(strategies)
        rows.extend({"id": f"{qid}:{strategy}", "question_id": qid, "strategy": strategy}
                    for strategy in strategies)
    return rows


def select_references(question, chunks, radius):
    # 在观察新回答之前固定同报告物理页 +/-1；四组评审用完全相同的参考。
    return [c for c in chunks if any(c["report_id"] == e["report_id"] and
            abs(c["page"] - e["page"]) <= radius for e in question["evidence"])]


def persist_equal(path, value):
    if path.exists():
        require(read(path) == value, "冻结输入/配置已变化；使用新实验目录，不覆盖旧记录")
    else:
        _write_json(path, value)


def prepare(root, config_path):
    config, rag, decision, sources, spec, prompt, _, _, previous, controls = protocol(root, config_path)
    inputs = fixed_inputs(root, sources)
    require(inputs["controls"] == previous["controls"], "检索输入与 Phase 7 不一致")
    store, _ = open_index(root, rag, decision, sources, inputs)
    dense = FaissRetriever(store, inputs["ids"])
    sparse = BM25Retriever([c["text"] for c in inputs["chunks"]], inputs["ids"],
                          k1=sources["retrieval"]["bm25"]["k1"], b=sources["retrieval"]["bm25"]["b"])
    budget = TokenBudget(root, rag, spec, controls["max_output_tokens"])
    by_id = {c["chunk_id"]: c for c in inputs["chunks"]}
    entries, references = {}, {}
    for q, vector, old in zip(inputs["benchmark"]["questions"], inputs["queries"], previous["queries"], strict=True):
        qid = q["id"]
        dh = dense.retrieve(q["question"], vector, sources["retrieval"]["candidate_k"])
        bh = sparse.retrieve(q["question"], None, sources["retrieval"]["candidate_k"])
        hh = fuse_rrf([dh, bh], inputs["ids"], **sources["retrieval"]["rrf"],
                      top_k=sources["retrieval"]["candidate_k"])
        for hits, branch in ((dh, "dense_candidates"), (bh, "bm25_candidates"), (hh, "hybrid_candidates")):
            require([h.chunk_id for h in hits] == [h["chunk_id"] for h in old[branch]] and
                    np.allclose([h.score for h in hits], [h["score"] for h in old[branch]], atol=2e-5, rtol=0),
                    "FAISS/BM25/RRF 候选不能复现旧排名；不能复用旧重排分数")
        for strategy in STRATEGIES:
            chosen = [] if strategy == "pure" else [by_id[h["chunk_id"]] for h in
                old["rankings"][strategy][:controls["context_top_k"]]]
            context = construct_context(q["question"], chosen, prompt)
            entries[f"{qid}:{strategy}"] = {
                "id": f"{qid}:{strategy}", "question_id": qid, "strategy": strategy, "category": q["category"],
                **context, "token_budget": budget.measure(context["messages"]),
            }
        references[qid] = select_references(q, inputs["chunks"], config["reference_page_radius"])
    ordering = schedule([q["id"] for q in inputs["benchmark"]["questions"]], config["schedule_seed"])
    value = {"controls": controls, "schedule": ordering, "entries": entries, "references": references}
    value["inputs_fingerprint"] = fingerprint(value)
    # 整份报告正文仅在被忽略的本地缓存，Git 只保存 Metadata/指纹。
    manifest = {**value, "entries": {key: {k: v for k, v in row.items() if k != "messages"}
                                          for key, row in entries.items()},
                "references": {qid: [{k: c[k] for k in SOURCE_FIELDS} for c in rows]
                               for qid, rows in references.items()}}
    persist_equal(root / config["inputs_path"], value)
    persist_equal(root / config["results_directory"] / "inputs_manifest.json", manifest)
    return value


def citation_audit(answer, sources):
    # 消融保留无引用/未知引用答案；不使用生产管线的拒绝规则筛选好答案。
    labels = []
    malformed = []
    for group in re.findall(r"\[([^\]\n]*)\]", answer):
        if re.search(r"\bS\d+", group, re.IGNORECASE):
            labels.extend(re.findall(r"S\d+", group, re.IGNORECASE))
            if not re.fullmatch(r"S[1-9]\d*(?:\s*[,，]\s*S[1-9]\d*)*", group):
                malformed.append(group)
    unclosed = bool(re.search(r"\[\s*S\d+[^\]\n]*(?:\n|$)", answer, re.IGNORECASE))
    available = {s["label"] for s in sources}
    return {"labels": labels, "unknown_labels": [s for s in labels if s not in available],
            "malformed_groups": malformed, "unclosed": unclosed}


def generation_result(row):
    return GenerationResult(**{key: row[key] for key in GenerationResult.__dataclass_fields__})


def validate_usage(row, spec, budget):
    result = generation_result(row)
    require(result.response_model == spec["model"], "返回模型改变")
    require(result.prompt_tokens > 0 and 0 <= result.output_tokens <= budget["max_output_tokens"]
            and result.prompt_tokens <= budget["input_tokens"] + budget["safety_margin"]
            and result.prompt_tokens + budget["max_output_tokens"] <= budget["context_window_tokens"]
            and math.isfinite(result.latency_seconds) and result.latency_seconds > 0, "实际预算/耗时非法")
    require(row["answer_sha256"] == fingerprint(result.answer), "答案指纹不匹配")
    require(row["estimated_api_cost_usd"] == estimated_cost(result, spec), "费用不能复算")


def validate_generations(data, frozen, spec, *, complete=False):
    require(data["inputs_fingerprint"] == frozen["inputs_fingerprint"] and data["controls"] == frozen["controls"],
            "生成检查点输入指纹改变")
    expected = [r["id"] for r in frozen["schedule"]]
    require([r["id"] for r in data["answers"]] == expected[:len(data["answers"])] and
            len(data["answers"]) <= len(expected), "生成检查点顺序非法")
    if complete:
        require(data["completed"] and len(data["answers"]) == len(expected), "80 回答尚未全部完成")
    fps = set()
    for row in data["answers"]:
        item = frozen["entries"][row["id"]]
        for key in ("question_id", "category", "strategy", "messages_sha256", "sources", "token_budget"):
            require(row[key] == item[key], "生成记录与输入不同")
        validate_usage(row, spec, item["token_budget"])
        require(row["citation_audit"] == citation_audit(row["answer"], item["sources"]), "引用标签审计不一致")
        require(row["system_fingerprint"] is not None, "云端未返回 fingerprint，无法验证后端控制变量")
        fps.add(row["system_fingerprint"])
    require(len(fps) <= 1, "不能混合不同云端后端 fingerprint")
    return fps


def run_generations(root, config_path, frozen, *, allow_cloud=False, provider=None):
    config, _, _, sources, spec, _, _, _, _, controls = protocol(root, config_path)
    require(frozen["controls"] == controls, "当前协议与冻结输入不同")
    path = root / config["results_directory"] / "answers.json"
    data = read_checkpoint(path, "answers") or {
        "schema_version": "1.0", "controls": controls, "inputs_fingerprint": frozen["inputs_fingerprint"],
        "started_at": now(), "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "answers": [], "completed": False,
    }
    fps = validate_generations(data, frozen, spec)
    _write_json(path, data)
    if data["completed"]:
        validate_generations(data, frozen, spec, complete=True)
        print("80 个回答已完成；复用记录，不再次调用 API", flush=True)
        return data
    require(allow_cloud or provider is not None, "生成需要 --allow-cloud；公开片段将发送到云端并计费")
    active = provider or build_provider(spec, sources["generation"])
    try:
        identity = active.identity()
        require("identity" not in data or identity == data["identity"], "云端模型身份改变")
        data["identity"] = identity
        _write_json(path, data)
        for planned in frozen["schedule"][len(data["answers"]):]:
            item = frozen["entries"][planned["id"]]
            require(fingerprint(item["messages"]) == item["messages_sha256"], "生成 messages 被修改")
            result = active.generate(item["messages"])
            row = {**{k: v for k, v in item.items() if k != "messages"}, **result.to_dict(),
                   "generated_at": now(), "answer_sha256": fingerprint(result.answer),
                   "citation_audit": citation_audit(result.answer, item["sources"]),
                   "estimated_api_cost_usd": estimated_cost(result, spec)}
            # 先保存实际响应，之后才验证；截断/空回答/坏引用不从分母删除，不自动重试。
            data["answers"].append(row)
            _write_json(path, data)
            validate_generations(data, frozen, spec)
            fps.add(result.system_fingerprint)
            print(f"{len(data['answers'])}/80 {row['id']}: {result.latency_seconds:.2f}s "
                  f"{result.output_tokens} tokens ({result.finish_reason})", flush=True)
        data["completed"] = True
        data["finished_at"] = now()
        validate_generations(data, frozen, spec, complete=True)
        _write_json(path, data)
        return data
    finally:
        if provider is None:
            active.client.close()


def review_messages(row, question, references, chunks_by_id, facts, prompt):
    payload = {
        "question": question["question"], "expected_answer": question["expected_answer"],
        "required_facts": facts, "answer": row["answer"], "citation_labels": row["citation_audit"]["labels"],
        "provided_sources": [{**s, "text": chunks_by_id[s["chunk_id"]]["text"]} for s in row["sources"]],
        "reference_sources": [{k: c[k] for k in (*SOURCE_FIELDS, "text")} for c in references],
    }
    # 不含策略名、运行 ID、分数、召回标签或其他组回答；Context 数量仍可能暴露组别。
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_template"].format(
                payload=json.dumps(payload, ensure_ascii=False))}]


def quote_in_answer(quote, answer):
    def presentation_only(text):
        text = text.replace("**", "").replace("`", "")
        text = text.translate(str.maketrans("", "", '\"“”‘’'))
        return re.sub(r"\s+", "", text)
    normalized = presentation_only(quote)
    return bool(normalized) and normalized in presentation_only(answer)


def validate_review(value, row, facts, reference_ids, provided_ids=()):
    require(isinstance(value, dict) and set(value) == {"required_facts", "claims", "citations"}, "评审 JSON 字段非法")
    require(isinstance(value["required_facts"], list) and len(value["required_facts"]) == len(facts),
            "评审未逐项覆盖必答事实")
    for item in value["required_facts"]:
        require(set(item) == {"covered", "reason"} and type(item["covered"]) is bool
                and isinstance(item["reason"], str) and bool(item["reason"].strip()), "事实覆盖评审非法")
    require(isinstance(value["claims"], list), "断言评审必须为列表")
    for item in value["claims"]:
        require(set(item) == {"text", "verdict", "reference_chunk_ids", "reason"}
                and isinstance(item["text"], str) and quote_in_answer(item["text"], row["answer"])
                and item["verdict"] in {"supported", "contradicted", "unverifiable"}
                and isinstance(item["reason"], str) and bool(item["reason"].strip()), "断言评审/原文非法")
        ids = item["reference_chunk_ids"]
        require(isinstance(ids, list) and all(isinstance(k, str) for k in ids)
                and len(ids) == len(set(ids)) and set(ids) <= set(reference_ids) | set(provided_ids), "评审编造参考 Chunk")
        require(item["verdict"] == "unverifiable" or bool(ids), "判定支持或矛盾必须列真实依据")
    require(isinstance(value["citations"], list) and [c["label"] for c in value["citations"]]
            == row["citation_audit"]["labels"], "评审遗漏/合并引用，分母改变")
    available = {s["label"] for s in row["sources"]}
    for item in value["citations"]:
        require(set(item) == {"label", "claim", "verdict", "reason"}
                and isinstance(item["claim"], str) and quote_in_answer(item["claim"], row["answer"])
                and item["verdict"] in {"supported", "unsupported", "unverifiable"}
                and isinstance(item["reason"], str) and bool(item["reason"].strip()), "引用语义评审非法")
        require(item["label"] in available or item["verdict"] == "unsupported", "未知引用不能被判正确")


def run_reviews(root, config_path, frozen, answers, *, review_config_path, allow_cloud=False, provider=None):
    config, rag, _, sources, spec, _, _, rubric, _, controls = protocol(root, config_path)
    plan, settings, prompt = review_protocol(root, review_config_path, sources["generation"], rag, spec)
    validate_generations(answers, frozen, spec, complete=True)
    require(frozen["controls"] == controls, "当前评审协议与冻结输入不同")
    budget = TokenBudget(root, {**rag, "context_window_tokens": settings["context_window_tokens"]},
                         spec, settings["max_output_tokens"])
    chunks = {c["chunk_id"]: c for c in fixed_inputs(root, sources)["chunks"]}
    questions = {q["id"]: q for q in sources["benchmark"]["questions"]}
    path = root / config["results_directory"] / plan["results_file"]
    header = {"inputs_fingerprint": frozen["inputs_fingerprint"], "controls": controls,
              "answers_fingerprint": fingerprint(answers), "review_settings": settings,
              "review_protocol": plan, "review_prompt_fingerprint": fingerprint(prompt)}
    data = read_checkpoint(path, "reviews") or {**header, "started_at": now(), "reviews": [], "completed": False}
    require(all(data.get(k) == v for k, v in header.items()), "评审检查点与答案/协议不一致")
    # 不按生成策略分组展示；固定哈希排序，避免生成与评审顺序一致。
    ordered = sorted(answers["answers"], key=lambda r: fingerprint({"seed": config["schedule_seed"], "id": r["id"]}))
    require([r["id"] for r in data["reviews"]] == [r["id"] for r in ordered[:len(data["reviews"])]],
            "评审检查点顺序不一致")
    for index, row in enumerate(ordered[:len(data["reviews"])]):
        validate_review_record(data["reviews"][index], row, frozen, rubric, spec)
    require({r["system_fingerprint"] for r in data["reviews"]}
            <= {answers["answers"][0]["system_fingerprint"]}, "已有评审后端 fingerprint 改变")
    _write_json(path, data)
    if data["completed"]:
        require(len(data["reviews"]) == len(ordered), "评审尚未完成")
        print("80 份 AI 辅助评审已完成；复用记录，不再次调用 API", flush=True)
        return data
    require(allow_cloud or provider is not None, "辅助评审需要 --allow-cloud；会发送回答/参考证据并计费")
    plans = {}
    for row in ordered[len(data["reviews"]):]:
        qid = row["question_id"]
        messages = review_messages(row, questions[qid], frozen["references"][qid], chunks,
                                   rubric["required_facts"][qid], prompt)
        plans[row["id"]] = (messages, budget.measure(messages))
    print(f"辅助评审全部剩余输入预算预检通过：{len(plans)} 条", flush=True)
    active = provider or build_provider(spec, settings)
    try:
        identity = active.identity()
        require("identity" not in data or identity == data["identity"], "评审模型身份改变")
        data["identity"] = identity
        _write_json(path, data)
        for row in ordered[len(data["reviews"]):]:
            messages, measured = plans[row["id"]]
            result = active.generate(messages)
            record = {"id": row["id"], "review_id": fingerprint(row["id"]),
                      "answer_sha256": row["answer_sha256"], "messages_sha256": fingerprint(messages),
                      "token_budget": measured, "reviewed_at": now(), **result.to_dict(),
                      "answer_sha256_raw_review": fingerprint(result.answer),
                      "estimated_api_cost_usd": estimated_cost(result, spec)}
            # 保存原始 JSON 文本，即使解析失败也不丢弃/自动重试。
            data["reviews"].append(record)
            _write_json(path, data)
            validate_review_record(record, row, frozen, rubric, spec)
            fps = {r["system_fingerprint"] for r in data["reviews"]}
            require(fps == {answers["answers"][0]["system_fingerprint"]}, "生成/评审后端 fingerprint 改变")
            print(f"辅助评审 {len(data['reviews'])}/80 {record['review_id'][:12]}: "
                  f"{result.latency_seconds:.2f}s {result.output_tokens} tokens", flush=True)
        data["completed"] = True
        data["finished_at"] = now()
        _write_json(path, data)
        return data
    finally:
        if provider is None:
            active.client.close()


def validate_review_record(record, row, frozen, rubric, spec):
    require(record["answer_sha256"] == row["answer_sha256"] and record["review_id"] == fingerprint(row["id"]),
            "评审不是对应回答")
    raw = {**record, "answer_sha256": record["answer_sha256_raw_review"]}
    validate_usage(raw, spec, record["token_budget"])
    require(record["finish_reason"] == "stop", "辅助评审被截断；保留原始记录，需显式新评审运行")
    try:
        value = json.loads(record["answer"])
    except json.JSONDecodeError:
        raise ValueError("辅助评审不是有效 JSON；保留原始记录，不自动重试") from None
    # 无 Metadata 的标签必定无法成为正确的报告引用；覆盖 AI 的 supported/unverifiable 判定。
    available = {s["label"] for s in row["sources"]}
    value = json.loads(json.dumps(value, ensure_ascii=False))
    overrides = 0
    for item in value.get("citations", []):
        require(item["verdict"] in {"supported", "unsupported", "unverifiable"}, "引用评审枚举非法")
        if item["label"] not in available and item["verdict"] != "unsupported":
            overrides += 1
            item["verdict"] = "unsupported"
            item["reason"] = "自动审计：没有对应的提供来源 Metadata，该标签不是有效报告引用；" + item["reason"]
    reference_ids = [s["chunk_id"] for s in frozen["references"][row["question_id"]]]
    validate_review(value, row, rubric["required_facts"][row["question_id"]], reference_ids,
                    [s["chunk_id"] for s in row["sources"]])
    projected = project_shared_reference(value, reference_ids)
    projected["unknown_citation_override_count"] = overrides
    return projected


def project_shared_reference(value, reference_ids):
    # AI 可能把提供的生成片段误列为共享参考。保留原始评审，保守降级而非扩充参考/重评。
    projected = json.loads(json.dumps(value, ensure_ascii=False))
    changed = 0
    for item in projected["claims"]:
        if set(item["reference_chunk_ids"]) - set(reference_ids):
            changed += 1
            item["verdict"] = "unverifiable"
            item["reference_chunk_ids"] = []
            item["reason"] = "自动审计：依据超出固定共享参考范围，原始判定未采用；" + item["reason"]
    projected["reference_scope_downgrade_count"] = changed
    return projected


def score_review(value):
    counts = {label: sum(c["verdict"] == label for c in value["claims"])
              for label in ("supported", "contradicted", "unverifiable")}
    known = counts["supported"] + counts["contradicted"]
    total = sum(counts.values())
    citations = value["citations"]
    return {
        "accuracy": counts["supported"] / known if known else None,
        "completeness": statistics.mean(f["covered"] for f in value["required_facts"]),
        "citation_correctness": sum(c["verdict"] == "supported" for c in citations) / len(citations) if citations else None,
        "hallucination_rate": counts["contradicted"] / total if total else None,
        "unverifiable_rate": counts["unverifiable"] / total if total else None,
        "claim_counts": counts, "citation_count": len(citations),
        "reference_scope_downgrade_count": value.get("reference_scope_downgrade_count", 0),
        "unknown_citation_override_count": value.get("unknown_citation_override_count", 0),
    }


def summarize(answers, reviews, frozen, rubric, spec):
    validate_generations(answers, frozen, spec, complete=True)
    require(reviews["completed"] and len(reviews["reviews"]) == len(answers["answers"])
            and reviews["answers_fingerprint"] == fingerprint(answers)
            and reviews["inputs_fingerprint"] == frozen["inputs_fingerprint"]
            and reviews["controls"] == frozen["controls"], "评审不完整或答案指纹改变")
    by_id = {r["id"]: r for r in reviews["reviews"]}
    require(len(by_id) == len(reviews["reviews"]) and set(by_id) == {r["id"] for r in answers["answers"]}, "评审 ID 不完整")
    require({r["system_fingerprint"] for r in reviews["reviews"]}
            == {answers["answers"][0]["system_fingerprint"]}, "生成/评审混用云端后端")
    per_query = []
    for row in answers["answers"]:
        value = validate_review_record(by_id[row["id"]], row, frozen, rubric, spec)
        per_query.append({"id": row["id"], "question_id": row["question_id"], "strategy": row["strategy"],
                          "category": row["category"], **score_review(value)})

    def aggregate(rows):
        result = {"answer_count": len(rows)}
        for metric in ("accuracy", "completeness", "citation_correctness", "hallucination_rate", "unverifiable_rate"):
            values = [r[metric] for r in rows if r[metric] is not None]
            result[metric] = statistics.mean(values) if values else None
            result[metric + "_scored_answers"] = len(values)
        result["answers_with_citations"] = sum(r["citation_count"] > 0 for r in rows)
        result["reference_scope_downgrade_count"] = sum(r["reference_scope_downgrade_count"] for r in rows)
        result["unknown_citation_override_count"] = sum(r["unknown_citation_override_count"] for r in rows)
        result["claim_counts"] = {k: sum(r["claim_counts"][k] for r in rows)
                                  for k in ("supported", "contradicted", "unverifiable")}
        count = sum(r["citation_count"] for r in rows)
        supported = sum(round((r["citation_correctness"] or 0) * r["citation_count"]) for r in rows)
        result["citation_occurrence_count"] = count
        result["citation_correctness_micro"] = supported / count if count else None
        return result

    results = []
    for strategy in STRATEGIES:
        rows = [r for r in per_query if r["strategy"] == strategy]
        generated = [r for r in answers["answers"] if r["strategy"] == strategy]
        results.append({"strategy": strategy, **aggregate(rows),
                        "generation_latency_median_seconds": statistics.median(r["latency_seconds"] for r in generated),
                        "generation_prompt_tokens": sum(r["prompt_tokens"] for r in generated),
                        "generation_output_tokens": sum(r["output_tokens"] for r in generated),
                        "generation_estimated_cost_usd": sum(r["estimated_api_cost_usd"] for r in generated),
                        "truncated_answers": sum(r["finish_reason"] == "length" for r in generated),
                        "empty_answers": sum(not r["answer"].strip() for r in generated),
                        "citation_protocol_error_answers": sum(bool(r["citation_audit"]["unknown_labels"]
                            or r["citation_audit"]["malformed_groups"] or r["citation_audit"]["unclosed"])
                            for r in generated),
                        "by_category": {category: aggregate([r for r in rows if r["category"] == category])
                                        for category in sorted({r["category"] for r in rows})}})
    return {
        "experiment": "phase10-four-way-ablation", "controls": frozen["controls"],
        "inputs_fingerprint": frozen["inputs_fingerprint"], "answers_fingerprint": fingerprint(answers),
        "reviews_fingerprint": fingerprint(reviews), "results": results, "per_query": sorted(per_query, key=lambda r: r["id"]),
        "review_protocol": reviews["review_protocol"],
        "review_estimated_cost_usd": sum(r["estimated_api_cost_usd"] for r in reviews["reviews"]),
        "metric_policy": {
            "aggregation": "question_macro_except_explicit_micro; N/A denominators reported",
            "accuracy": "supported/(supported+contradicted); unverifiable excluded; not independent truth accuracy",
            "completeness": "fully covered required facts/all required facts; composite facts all-or-nothing",
            "citation_correctness": "supported citation occurrences/all citation occurrences; no citations=N/A",
            "hallucination_rate": "AI-labeled contradicted/all factual claims; reference-bounded proxy, NOT validated truth or guaranteed lower bound",
            "unverifiable_rate": "unverifiable/all factual claims; not assumed false; report alongside accuracy/hallucination",
            "reference_scope_audit": "real provided-source IDs outside shared references are downgraded to unverifiable; raw review retained",
            "unknown_citation_audit": "missing source Metadata forces unsupported regardless of raw AI verdict; raw review retained",
            "quote_validation": "continuous substring after whitespace/Markdown emphasis/backtick/quotation-delimiter removal only; no factual rewriting",
        },
        "limitations": ["同模型家族 AI 辅助评审，未独立人工复核；分数不是最终研究结论。",
                        "抽查发现 AI 混淆至少五年与确定最长时间，以及当前 Context 缺失与整份报告缺失；矛盾判定可能误报，不能称已确认幻觉下界。",
                        "隐藏策略名但 Context 长度可暴露组别，不能称完全盲评。",
                        "断言拆分/遗漏/语义关系仍需人工检查；参考资料并不覆盖所有真实知识。",
                        "20 题单次运行无显著性或外部泛化结论；云端权重未固定，temperature=0 不保证逐字复现。",
                        "复用固定重排分数，不报告本轮端到端检索/重排耗时；上下文长度随策略自然变化。"],
    }


def load_results(root, config_path, *, review_config_path, verify_inputs=False):
    config, rag, _, sources, spec, _, _, rubric, previous, controls = protocol(root, config_path)
    plan, settings, prompt = review_protocol(root, review_config_path, sources["generation"], rag, spec)
    directory = root / config["results_directory"]
    frozen = prepare(root, config_path) if verify_inputs else read(directory / "inputs_manifest.json")
    require(frozen["controls"] == controls, "结果协议与当前配置改变")
    require(frozen["schedule"] == schedule([q["id"] for q in sources["benchmark"]["questions"]], config["schedule_seed"]),
            "冻结调度改变")
    reports = {r["report_id"]: r for r in sources["manifest"]["reports"]}
    old_by_id = {r["question_id"]: r for r in previous["queries"]}
    questions = {q["id"]: q for q in sources["benchmark"]["questions"]}
    require(set(frozen["entries"]) == {r["id"] for r in frozen["schedule"]}, "冻结输入未完整覆盖四组")
    for key, item in frozen["entries"].items():
        require(item["id"] == key and key == f"{item['question_id']}:{item['strategy']}"
                and len(item["sources"]) == (0 if item["strategy"] == "pure" else controls["context_top_k"]), "输入/来源数量非法")
        ranking = [] if item["strategy"] == "pure" else old_by_id[item["question_id"]]["rankings"][item["strategy"]]
        require([s["chunk_id"] for s in item["sources"]] == [h["chunk_id"] for h in ranking]
                and item["category"] == questions[item["question_id"]]["category"], "生成 Context 不是固定策略 Top-10")
        for index, source in enumerate(item["sources"], 1):
            meta = reports[source["report_id"]]
            require(source["label"] == f"S{index}" and set(source) == set(SOURCE_FIELDS) | {"label"}
                    and source["document_title"] == meta["title"] and source["vendor"] == meta["vendor"]
                    and source["source_url"] == meta["url"], "来源 Metadata 非法")
            require({k: source[k] for k in previous["sources"][source["chunk_id"]]}
                    == previous["sources"][source["chunk_id"]], "生成来源与冻结检索 Metadata 不同")
        measured = item["token_budget"]
        require(measured["context_window_tokens"] == rag["context_window_tokens"]
                and measured["safety_margin"] == rag["token_safety_margin"]
                and measured["max_output_tokens"] == controls["max_output_tokens"]
                and measured["tokenizer_sha256"] == rag["tokenizers"][spec["id"]]["tokenizer_sha256"]
                and measured["input_tokens"] + measured["safety_margin"] + measured["max_output_tokens"]
                <= measured["context_window_tokens"], "输入预算与协议不同")
    answers, reviews = read(directory / "answers.json"), read(directory / plan["results_file"])
    require(reviews["review_settings"] == settings and reviews["review_protocol"] == plan
            and reviews["review_prompt_fingerprint"] == fingerprint(prompt), "评审协议/参数改变")
    for record in reviews["reviews"]:
        measured = record["token_budget"]
        require(measured["context_window_tokens"] == settings["context_window_tokens"]
                and measured["max_output_tokens"] == settings["max_output_tokens"]
                and measured["safety_margin"] == rag["token_safety_margin"]
                and measured["tokenizer_sha256"] == rag["tokenizers"][spec["id"]]["tokenizer_sha256"]
                and measured["input_tokens"] + measured["safety_margin"] + measured["max_output_tokens"]
                <= measured["context_window_tokens"], "评审预算改变或超限")
    summary = summarize(answers, reviews, frozen, rubric, spec)
    pilot_path = directory / plan["pilot_file"]
    if pilot_path.exists():
        pilot = read(pilot_path)
        require(pilot["inputs_fingerprint"] == frozen["inputs_fingerprint"]
                and pilot["answers_fingerprint"] == fingerprint(answers)
                and pilot["controls"] == controls, "试评记录不是当前回答/协议")
        generated_by_id = {r["id"]: r for r in answers["answers"]}
        for record in pilot["reviews"]:
            require(record["id"] in generated_by_id
                    and record["answer_sha256"] == generated_by_id[record["id"]]["answer_sha256"], "试评未绑定原始回答")
            validate_usage({**record, "answer_sha256": record["answer_sha256_raw_review"]}, spec, record["token_budget"])
        summary["pilot_review_count"] = len(pilot["reviews"])
        summary["pilot_reviews_fingerprint"] = fingerprint(pilot)
        summary["pilot_estimated_cost_usd"] = sum(r["estimated_api_cost_usd"] for r in pilot["reviews"])
    else:
        summary["pilot_review_count"] = 0
        summary["pilot_estimated_cost_usd"] = 0
    summary["total_estimated_cost_usd_including_pilot"] = (
        sum(r["generation_estimated_cost_usd"] for r in summary["results"])
        + summary["review_estimated_cost_usd"] + summary["pilot_estimated_cost_usd"])
    if verify_inputs:
        chunks = {c["chunk_id"]: c for c in fixed_inputs(root, sources)["chunks"]}
        questions = {q["id"]: q for q in sources["benchmark"]["questions"]}
        by_id = {r["id"]: r for r in answers["answers"]}
        budget = TokenBudget(root, {**rag, "context_window_tokens": settings["context_window_tokens"]},
                             spec, settings["max_output_tokens"])
        for record in reviews["reviews"]:
            row = by_id[record["id"]]
            messages = review_messages(row, questions[row["question_id"]], frozen["references"][row["question_id"]],
                                       chunks, rubric["required_facts"][row["question_id"]], prompt)
            require(fingerprint(messages) == record["messages_sha256"], "评审 messages 不能复现")
            require(budget.measure(messages) == record["token_budget"], "评审预算/完整模板不能复现")
    return directory, summary
