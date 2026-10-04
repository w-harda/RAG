"""Phase 10：固定四组输入，保存 80 回答与 AI 辅助评审；云端首次运行计费。"""

import argparse
from pathlib import Path

from apt_rag.evaluation.ablation_benchmark import (
    load_results, prepare, read, run_generations, run_reviews,
)
from apt_rag.evaluation.embedding_benchmark import _write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation_benchmark.json")
    parser.add_argument("--review-config", type=Path, default=ROOT / "configs/ablation_review_v2.json")
    parser.add_argument("--prepare-only", action="store_true", help="仅冻结输入/预检预算，不读取密钥/调用 API")
    parser.add_argument("--generate-only", action="store_true", help="仅生成四组回答，不辅助评审")
    parser.add_argument("--allow-cloud", action="store_true", help="允许向 DeepSeek 发送公开数据并产生费用")
    args = parser.parse_args()
    frozen = prepare(ROOT, args.config)
    print(f"输入已冻结：{len(frozen['entries'])} 组；无 oracle Context，Top-10 全文未截断", flush=True)
    if args.prepare_only:
        return
    answers = run_generations(ROOT, args.config, frozen, allow_cloud=args.allow_cloud)
    if args.generate_only:
        return
    run_reviews(ROOT, args.config, frozen, answers, review_config_path=args.review_config,
                allow_cloud=args.allow_cloud)
    directory, summary = load_results(ROOT, args.config, review_config_path=args.review_config)
    plan = read(args.review_config)
    _write_json(directory / plan["summary_file"], summary)
    print("Phase 10 原始回答/辅助评审/摘要已保存；独立人工复核仍待完成", flush=True)


if __name__ == "__main__":
    main()
