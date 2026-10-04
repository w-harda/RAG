"""Phase 6：冻结标准证据输入，比较生成，不执行检索或最终 RAG。"""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from apt_rag.benchmark import load_benchmark, validate_benchmark
from apt_rag.corpus import load_manifest
from apt_rag.evaluation.embedding_benchmark import (
    _write_json, benchmark_fingerprint, corpus_fingerprint, load_chunks,
)
from apt_rag.generation.providers import GenerationResult, build_provider, fingerprint


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["schema_version"] != "1.0" or config["context_mode"] != "ground_truth_evidence":
        raise ValueError("Phase 6 只支持标准证据 Context")
    if config["temperature"] != 0 or not 1 <= config["max_output_tokens"] <= 8192:
        raise ValueError("实验要求 temperature=0 和有效的输出上限")
    specs = config["providers"]
    if {p["kind"] for p in specs} != {"ollama", "deepseek"}:
        raise ValueError("需要 Ollama 和 DeepSeek 两种后端")
    if len({p["id"] for p in specs}) != len(specs):
        raise ValueError("provider ID 不得重复")
    if any(not re.fullmatch(r"[a-z0-9-]+", p["id"]) for p in specs):
        raise ValueError("provider ID 只能包含小写字母、数字和连字符")
    return config


def freeze_inputs(
    benchmark: dict[str, Any], chunks: list[dict[str, Any]], prompt: dict[str, str],
) -> list[dict[str, Any]]:
    by_id = {c["chunk_id"]: c for c in chunks}
    inputs = []
    for question in benchmark["questions"]:
        sources, blocks = [], []
        for index, evidence in enumerate(question["evidence"], start=1):
            chunk = by_id[evidence["chunk_id"]]
            if chunk["report_id"] != evidence["report_id"] or chunk["page"] != evidence["page"]:
                raise ValueError("Ground Truth 和实际 Chunk 的 Metadata 不一致")
            source = {field: chunk[field] for field in (
                "chunk_id", "report_id", "document_title", "vendor", "page", "section", "source_url",
            )}
            source["label"] = f"S{index}"
            source["text_sha256"] = chunk["text_sha256"]
            sources.append(source)
            blocks.append(
                f"[{source['label']}] {chunk['document_title']} | {chunk['vendor']} | "
                f"PDF 物理页码 {chunk['page']} | Chunk {chunk['chunk_id']}\n"
                f"URL: {chunk['source_url']}\n{chunk['text']}"
            )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_template"].format(
                question=question["question"], context="\n\n".join(blocks),
            )},
        ]
        inputs.append({
            "question_id": question["id"], "category": question["category"],
            "sources": sources, "messages": messages, "messages_sha256": fingerprint(messages),
        })
    return inputs


def prepare_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    benchmark = load_benchmark(root / config["benchmark_path"])
    validate_benchmark(benchmark, load_manifest(root / config["manifest_path"]),
                       chunks_directory=root / config["chunks_directory"])
    chunks = load_chunks(root / config["chunks_directory"])
    prompt = json.loads((root / config["prompt_path"]).read_text(encoding="utf-8"))
    controls = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_fingerprint": benchmark_fingerprint(benchmark),
        "corpus_fingerprint": corpus_fingerprint(chunks),
        "prompt_fingerprint": fingerprint(prompt), "context_mode": config["context_mode"],
        "temperature": config["temperature"], "max_output_tokens": config["max_output_tokens"],
        "thinking": False, "stream": False, "question_count": len(benchmark["questions"]),
    }
    data = {"controls": controls, "inputs": freeze_inputs(benchmark, chunks, prompt)}
    data["inputs_fingerprint"] = fingerprint(data)
    path = root / config["inputs_path"]
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != data:
        raise ValueError("冻结输入已变化；请用新的配置/结果目录启动新实验，不能覆盖原实验")
    _write_json(path, data)
    # Git 只保存输入指纹与来源映射，不复制报告全文或标准答案到运行结果。
    audit = {**data, "inputs": [{k: v for k, v in row.items() if k != "messages"}
                                for row in data["inputs"]]}
    audit_path = root / config["results_directory"] / "inputs_manifest.json"
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != audit:
        raise ValueError("输入审计记录已变化，禁止覆盖")
    _write_json(audit_path, audit)
    return data


def citation_check(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    # 仅检查来源标签的可解析性，绝不将它当作语义引用正确率。
    # 支持 [S1, S2]，但排除报告中的 ATT&CK Software ID（如 [S0183]）。
    groups = re.findall(r"\[(S[1-9]\d*(?:,\s*S[1-9]\d*)*)\]", answer)
    labels = [label for group in groups for label in re.findall(r"S[1-9]\d*", group)]
    known = {source["label"] for source in sources}
    return {
        "occurrences": labels,
        "unknown_labels": sorted(set(labels) - known),
        "has_citation": bool(labels),
        "label_validity": sum(label in known for label in labels) / len(labels) if labels else None,
    }


def estimated_cost(result: GenerationResult, spec: dict[str, Any]) -> float | None:
    if spec["kind"] == "ollama":
        return None  # 本地耗电/硬件折旧未知，不能写成真实成本为零。
    prices = spec["pricing"]
    hits = result.cache_hit_tokens
    if hits is None or not 0 <= hits <= result.prompt_tokens:
        raise ValueError("缺少或非法的缓存 Token 计数，无法按价格口径估算")
    return (hits * prices["cache_hit_per_million"]
            + (result.prompt_tokens - hits) * prices["cache_miss_per_million"]
            + result.output_tokens * prices["output_per_million"]) / 1e6


def run_provider(
    root: Path, config: dict[str, Any], frozen: dict[str, Any], spec: dict[str, Any], *, provider=None,
) -> dict[str, Any]:
    path = root / config["results_directory"] / f"{spec['id']}.json"
    header = {
        "schema_version": "1.0", "provider_id": spec["id"], "provider_spec": spec,
        "config_fingerprint": fingerprint(config), "controls": frozen["controls"],
        "inputs_fingerprint": frozen["inputs_fingerprint"],
    }
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        **header, "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "answers": [], "completed": False,
    }
    if any(data.get(key) != value for key, value in header.items()):
        raise ValueError("结果中的配置/输入指纹不匹配；不能混合不同实验")
    if [a["question_id"] for a in data["answers"]] != [
        i["question_id"] for i in frozen["inputs"][:len(data["answers"])]]:
        raise ValueError("已有结果顺序或题目 ID 错误")
    if data.get("completed"):
        validate_run(data, frozen)
        print(f"{spec['id']}: 已完成，复用记录，不再次调用模型", flush=True)
        return data
    if data["answers"]:
        # 先验证部分检查点，避免篡改/损坏的旧回答触发后续收费调用。
        prefix = {**frozen, "inputs": frozen["inputs"][:len(data["answers"])]}
        validate_run({**data, "completed": True}, prefix)
    active = provider or build_provider(spec, config)
    try:
        identity = active.identity()
        if "identity" in data and data["identity"] != identity:
            raise ValueError("模型身份变化；不能继续旧实验")
        data["identity"] = identity
        _write_json(path, data)
        for item in frozen["inputs"][len(data["answers"]):]:
            # 每次复制同一冻结 messages，适配层只改变模型调用参数。
            result = active.generate(json.loads(json.dumps(item["messages"], ensure_ascii=False)))
            if not result.answer.strip():
                raise ValueError("模型返回空答案；保留已完成记录但不计入成功样本")
            if result.response_model != spec["model"]:
                raise ValueError("实际返回模型与配置不同，停止实验而非静默替换")
            previous_fingerprints = {a["system_fingerprint"] for a in data["answers"]
                                     if a["system_fingerprint"] is not None}
            if result.system_fingerprint and previous_fingerprints and (
                result.system_fingerprint not in previous_fingerprints
            ):
                raise ValueError("云端后端 fingerprint 已变化；停止旧实验，不混合不同后端")
            if result.prompt_tokens <= 0 or result.output_tokens <= 0:
                raise ValueError("模型返回无效 Token 用量")
            if spec["kind"] == "ollama" and (
                result.prompt_tokens + config["max_output_tokens"] >= spec["num_ctx"]
            ):
                raise ValueError("本地 Context 接近窗口上限，存在截断风险；停止实验")
            row = {
                "question_id": item["question_id"], "category": item["category"],
                "messages_sha256": item["messages_sha256"], "sources": item["sources"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **result.to_dict(), "answer_sha256": fingerprint(result.answer),
                "citation_check": citation_check(result.answer, item["sources"]),
                "estimated_api_cost_usd": estimated_cost(result, spec),
            }
            data["answers"].append(row)
            # 每题检查点；失败不重试、不覆盖，避免隐藏成本或挑选较好答案。
            _write_json(path, data)
            print(f"{spec['id']} {item['question_id']}: {result.latency_seconds:.2f}s "
                  f"{result.output_tokens} tokens ({result.finish_reason})", flush=True)
        data["completed"] = True
        data["finished_at"] = datetime.now(timezone.utc).isoformat()
        validate_run(data, frozen)
        _write_json(path, data)
        return data
    finally:
        if provider is None:
            active.client.close()


def validate_run(data: dict[str, Any], frozen: dict[str, Any]) -> None:
    if not data["completed"] or len(data["answers"]) != len(frozen["inputs"]):
        raise ValueError("实验尚未完成全部问题")
    if data["inputs_fingerprint"] != frozen["inputs_fingerprint"] or data["controls"] != frozen["controls"]:
        raise ValueError("冻结输入指纹或控制变量不一致")
    for row, item in zip(data["answers"], frozen["inputs"], strict=True):
        for key in ("question_id", "category", "messages_sha256", "sources"):
            if row[key] != item[key]:
                raise ValueError(f"{row['question_id']} 的输入/来源不一致")
        if row["answer_sha256"] != fingerprint(row["answer"]):
            raise ValueError("答案指纹不匹配")
        if row["citation_check"] != citation_check(row["answer"], row["sources"]):
            raise ValueError("引用标签统计不匹配")
        result = GenerationResult(**{k: row[k] for k in GenerationResult.__dataclass_fields__})
        if result.latency_seconds <= 0 or result.prompt_tokens <= 0 or result.output_tokens <= 0:
            raise ValueError("用量/延迟必须为正")
        if row["estimated_api_cost_usd"] != estimated_cost(result, data["provider_spec"]):
            raise ValueError("费用估算不匹配")
        if row["response_model"] != data["provider_spec"]["model"]:
            raise ValueError("实际返回模型与配置不同，不能静默替换")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs or any(not r["completed"] for r in runs):
        raise ValueError("必须使用已完成的运行结果")
    if len({r["provider_id"] for r in runs}) != len(runs):
        raise ValueError("同一模型不能重复计入对比")
    if len({r["inputs_fingerprint"] for r in runs}) != 1:
        raise ValueError("不同输入的 LLM 不能一起比较")
    rows = []
    for run in runs:
        answers = run["answers"]
        latencies = [a["latency_seconds"] for a in answers]
        costs = [a["estimated_api_cost_usd"] for a in answers]
        rows.append({
            "provider_id": run["provider_id"], "question_count": len(answers),
            "latency_median_seconds": float(np.median(latencies)),
            "latency_p95_seconds": float(np.percentile(latencies, 95)),
            "latency_total_seconds": sum(latencies),
            "first_request_seconds": latencies[0],
            "prompt_tokens": sum(a["prompt_tokens"] for a in answers),
            "output_tokens": sum(a["output_tokens"] for a in answers),
            "truncated_count": sum(a["finish_reason"] == "length" for a in answers),
            "cited_answer_rate": sum(a["citation_check"]["has_citation"] for a in answers) / len(answers),
            "unknown_source_labels": sum(len(a["citation_check"]["unknown_labels"]) for a in answers),
            "estimated_api_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "semantic_quality": None,
        })
    return {"schema_version": "1.0", "controls": runs[0]["controls"],
            "inputs_fingerprint": runs[0]["inputs_fingerprint"], "results": rows}
