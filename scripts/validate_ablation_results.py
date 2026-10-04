"""离线核验四组回答与 AI 辅助评审并重算摘要，不调用 API。"""

import argparse
from pathlib import Path

from apt_rag.evaluation.ablation_benchmark import load_results, read, require
from apt_rag.evaluation.embedding_benchmark import _write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation_benchmark.json")
    parser.add_argument("--review-config", type=Path, default=ROOT / "configs/ablation_review_v2.json")
    parser.add_argument("--verify-inputs", action="store_true", help="需要 Chunk/向量/索引/tokenizer，重建全部输入")
    parser.add_argument("--write-summary", action="store_true", help="重算摘要，不改原始回答/评审")
    args = parser.parse_args()
    directory, summary = load_results(ROOT, args.config, review_config_path=args.review_config,
                                      verify_inputs=args.verify_inputs)
    path = directory / read(args.review_config)["summary_file"]
    if args.write_summary:
        _write_json(path, summary)
    else:
        require(read(path) == summary, "摘要与原始回答/评审不同")
    print("Phase 10 验证通过：4 组 × 20 题、80 原始回答/辅助评审；未调用 API")
    print("AI 辅助评审待独立人工复核；不能将分数视为最终研究结论")


if __name__ == "__main__":
    main()
