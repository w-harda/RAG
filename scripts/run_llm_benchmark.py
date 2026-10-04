"""Phase 6 非思考模式 LLM 对比；默认复用已完成记录，避免重复计费。"""

import argparse
import json
from pathlib import Path

from apt_rag.evaluation.embedding_benchmark import _write_json
from apt_rag.evaluation.llm_benchmark import load_config, prepare_inputs, run_provider, summarize_runs
from apt_rag.evaluation.llm_review import summarize_review
from apt_rag.generation.providers import ProviderError

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm_benchmark.json")
    parser.add_argument("--provider", choices=["all", "ollama", "deepseek"], default="all")
    parser.add_argument("--prepare-only", action="store_true", help="仅冻结输入，不调用任何 API")
    args = parser.parse_args()
    config = load_config(args.config)
    frozen = prepare_inputs(ROOT, config)
    print(f"已冻结 {len(frozen['inputs'])} 题；输入指纹 {frozen['inputs_fingerprint']}", flush=True)
    if args.prepare_only:
        return
    try:
        runs = [run_provider(ROOT, config, frozen, spec) for spec in config["providers"]
                if args.provider == "all" or spec["kind"] == args.provider]
    except ProviderError as error:
        raise SystemExit(str(error)) from None
    if args.provider == "all":
        path = ROOT / config["results_directory"] / "summary.json"
        summary = summarize_runs(runs)
        review_path = path.parent / "review.json"
        if review_path.exists():
            rubric = json.loads((ROOT / "data/benchmark/llm_rubric_v1.json").read_text(encoding="utf-8"))
            review = json.loads(review_path.read_text(encoding="utf-8"))
            scores = summarize_review(runs, rubric, review)
            for row in summary["results"]:
                row["semantic_quality"] = scores[row["provider_id"]]
        _write_json(path, summary)
        print(f"两个后端均完成；性能摘要：{path}；语义质量需逐题复核", flush=True)


if __name__ == "__main__":
    main()
