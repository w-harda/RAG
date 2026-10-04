"""运行完整 RAG；默认云端必须显式允许，不自动回退、不覆盖已有输出。"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from apt_rag.evaluation.embedding_benchmark import _write_json
from apt_rag.generation.providers import ProviderError
from apt_rag.rag.runtime import build_pipeline, load_settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rag.json")
    parser.add_argument("--provider", choices=["deepseek-flash", "ollama-qwen"])
    parser.add_argument("--allow-cloud", action="store_true", help="允许将本次问题和公开报告片段发给 DeepSeek，生成会计费")
    parser.add_argument("--dry-run", action="store_true", help="真实检索/重排及预算检查，不调用生成")
    parser.add_argument("--output", type=Path, help="不含报告全文的 JSON；默认不写文件")
    args = parser.parse_args()
    _, decision, sources = load_settings(ROOT, args.config)
    provider_id = args.provider or decision["selection"]["primary_llm_provider_id"]
    if provider_id == decision["selection"]["primary_llm_provider_id"] and not args.allow_cloud and not args.dry_run:
        parser.error("云端生成须显式 --allow-cloud；或使用 --provider ollama-qwen")
    if args.output:
        if args.output.exists():
            parser.error("输出已存在；不能重复收费或覆盖，请使用新路径")
        # 可提交冒烟只允许固定公开问题；任意用户问题不自动进入 Git。
        if args.output.resolve().is_relative_to(ROOT / "results"):
            if args.question not in {q["question"] for q in sources["benchmark"]["questions"]}:
                parser.error("results 只允许固定公开 Benchmark 问题；个人问题请保存到 data/processed")
    pipeline = None
    started, started_at = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    try:
        pipeline, controls = build_pipeline(ROOT, args.config, provider_id,
                                            allow_cloud=args.allow_cloud, dry_run=args.dry_run)
        result = pipeline.answer(args.question, dry_run=args.dry_run)
        audit = {"schema_version": "1.0", "controls": controls,
                 "started_at": started_at, "cli_total_seconds": time.perf_counter() - started,
                 **{key: value for key, value in result.items() if key != "messages"}}
        if args.output:
            _write_json(args.output, audit)
        if args.dry_run:
            print("真实检索/重排已完成；未执行 LLM 生成")
            print(json.dumps(result.get("token_budget", {"status": "no_evidence"}), ensure_ascii=False, indent=2))
        else:
            print(result["answer"])
            for source in result["citations"]:
                print(f"[{source['label']}] {source['document_title']} | 页码 {source['page']} | "
                      f"章节 {source['section']} | Chunk {source['chunk_id']}\n{source['source_url']}")
    except (ValueError, ProviderError) as error:
        if args.output and hasattr(error, "record"):
            _write_json(args.output, {"schema_version": "1.0", "controls": controls,
                                     "started_at": started_at, "cli_total_seconds": time.perf_counter() - started,
                                     **{key: value for key, value in error.record.items() if key != "messages"}})
        parser.exit(1, f"RAG 未产生通过验证的回答：{error}\n")
    finally:
        if pipeline is not None and pipeline.llm.client is not None:
            pipeline.llm.client.close()


if __name__ == "__main__":
    main()
