"""绘制从原始排名重算的检索质量与阶段延迟，不调用模型。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validate_retrieval_results import load_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    _, directory, _, summary = load_results(args.config)
    rows = summary["results"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    for index, row in enumerate(rows):
        axes[0].bar(np.arange(len(metrics)) + (index - 1) * 0.25,
                    [row["metrics"][m] for m in metrics], 0.25, label=row["strategy"])
    axes[0].set_xticks(range(len(metrics)), metrics, rotation=25)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Fixed 20 queries and qrels; Top-10")
    axes[0].legend(fontsize=8)
    axes[1].bar([r["strategy"] for r in rows], [r["latency_ms"]["median"] for r in rows])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median component-sum latency (ms, log scale)")
    axes[1].set_title("One pass; CPU reranker; excludes embedding/load")
    figure.savefig(directory / "retrieval_comparison.png", dpi=160)
    plt.close(figure)
    print(f"图表已保存：{directory / 'retrieval_comparison.png'}")


if __name__ == "__main__":
    main()
