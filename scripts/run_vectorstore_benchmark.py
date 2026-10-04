"""运行固定 Embedding 的 Phase 5 FAISS/Chroma 实验。"""

from pathlib import Path

from apt_rag.evaluation.vectorstore_benchmark import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = run_benchmark(ROOT, ROOT / "configs/vectorstore_benchmark.json")
    print(f"完成 {len(summary['results'])} 组规模/后端对比；结果位于 results/phase5")


if __name__ == "__main__":
    main()
