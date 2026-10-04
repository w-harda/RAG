"""离线重算 Phase 6 统计；不调用 LLM，不要求密钥或本地模型。"""

import argparse
import json
from pathlib import Path

from apt_rag.benchmark import load_benchmark
from apt_rag.evaluation.embedding_benchmark import _write_json, benchmark_fingerprint
from apt_rag.evaluation.llm_benchmark import load_config, prepare_inputs, summarize_runs, validate_run
from apt_rag.evaluation.llm_review import summarize_review
from apt_rag.generation.providers import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def load_results(root: Path = ROOT, *, config_path: Path | None = None):
    config = load_config(config_path or root / "configs/llm_benchmark.json")
    directory = root / config["results_directory"]
    frozen = json.loads((directory / "inputs_manifest.json").read_text(encoding="utf-8"))
    benchmark = load_benchmark(root / config["benchmark_path"])
    prompt = json.loads((root / config["prompt_path"]).read_text(encoding="utf-8"))
    if frozen["controls"]["benchmark_fingerprint"] != benchmark_fingerprint(benchmark):
        raise ValueError("当前 Benchmark 与冻结输入不同")
    if frozen["controls"]["prompt_fingerprint"] != fingerprint(prompt):
        raise ValueError("当前 Prompt 与冻结输入不同")
    if [row["question_id"] for row in frozen["inputs"]] != [q["id"] for q in benchmark["questions"]]:
        raise ValueError("输入审计未覆盖完整的问题集")
    runs = []
    for spec in config["providers"]:
        run = json.loads((directory / f"{spec['id']}.json").read_text(encoding="utf-8"))
        if run["config_fingerprint"] != fingerprint(config) or run["provider_spec"] != spec:
            raise ValueError("结果与当前配置不一致")
        validate_run(run, frozen)
        runs.append(run)
    summary = summarize_runs(runs)
    review_path = directory / "review.json"
    if review_path.exists():
        rubric = json.loads((root / "data/benchmark/llm_rubric_v1.json").read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
        scores = summarize_review(runs, rubric, review)
        for row in summary["results"]:
            row["semantic_quality"] = scores[row["provider_id"]]
    return directory, runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm_benchmark.json")
    parser.add_argument("--write-summary", action="store_true", help="重建摘要，不覆盖原始回答")
    parser.add_argument("--verify-inputs", action="store_true", help="要求本地 Chunk 并重建冻结输入验证完整 Context")
    args = parser.parse_args()
    if args.verify_inputs:
        prepare_inputs(ROOT, load_config(args.config))
    directory, runs, summary = load_results(config_path=args.config)
    path = directory / "summary.json"
    if args.write_summary:
        _write_json(path, summary)
    elif json.loads(path.read_text(encoding="utf-8")) != summary:
        raise ValueError("摘要与原始回答/复核不一致；请重建摘要")
    print(f"Phase 6 验证通过：{len(runs)} 个模型、{sum(len(r['answers']) for r in runs)} 个回答；无需 API 调用")
    if any(row["semantic_quality"] is None for row in summary["results"]):
        print("尚未填写语义复核；格式检查不能替代回答质量")


if __name__ == "__main__":
    main()
