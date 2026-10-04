"""固定语料和向量运行 Phase 7 三策略检索实验；已完成记录默认复用。"""

import argparse
from pathlib import Path

from apt_rag.evaluation.retrieval_benchmark import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval_benchmark.json")
    args = parser.parse_args()
    result = run_benchmark(ROOT, args.config)
    print(f"完成 {len(result['queries'])} 题三策略对比；结果目录由配置指定")


if __name__ == "__main__":
    main()
