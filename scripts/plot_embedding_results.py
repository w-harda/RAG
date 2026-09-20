"""从真实 Phase 4 结果生成静态比较图。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / "results/phase4"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    width = 0.24
    for index, result in enumerate(summary["results"]):
        ax.bar(x + (index - 1) * width, [result[key] for key in metrics],
               width, label=result["model_id"])
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-average score")
    ax.set_title("APT QA v1.1: Chinese queries / English reports (20 questions)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
