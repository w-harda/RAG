"""运行固定 Embedding 的 Phase 5 FAISS/Chroma 实验。"""

import argparse
from pathlib import Path

from apt_rag.evaluation.vectorstore_benchmark import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vectorstore_benchmark.json")
    args = parser.parse_args()
    summary = run_benchmark(ROOT, args.config)
    print(f"完成 {len(summary['results'])} 组规模/后端对比；结果目录由配置指定")


if __name__ == "__main__":
    main()
