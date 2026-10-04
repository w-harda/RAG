"""根据校验后的 Phase 10 原始记录绘图，不调用 API。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apt_rag.evaluation.ablation_benchmark import load_results, read, require

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation_benchmark.json")
    parser.add_argument("--review-config", type=Path, default=ROOT / "configs/ablation_review_v2.json")
    args = parser.parse_args()
    plan = read(args.review_config)
    directory, summary = load_results(ROOT, args.config, review_config_path=args.review_config)
    require(read(directory / plan["summary_file"]) == summary, "摘要与原始记录不同")
    rows = summary["results"]
    labels = ["Pure LLM", "Dense RAG", "Hybrid RAG", "Hybrid + Reranker"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, [
        ("accuracy", "Accuracy on verifiable claims"),
        ("completeness", "Required-fact completeness"),
        ("citation_correctness", "Citation correctness (macro)"),
        ("hallucination_rate", "AI-labeled contradiction rate (proxy)"),
        ("unverifiable_rate", "Unverifiable claim rate"),
        ("generation_latency_median_seconds", "Median generation-only latency (s)"),
    ], strict=True):
        values = [r[metric] for r in rows]
        ax.bar(range(4), [v or 0 for v in values], color=["#999999", "#4575b4", "#74add1", "#1b9e77"])
        ax.set_xticks(range(4), labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if metric != "generation_latency_median_seconds":
            ax.set_ylim(0, 1.12)
        for i, value in enumerate(values):
            ax.text(i, (value or 0) + 0.025, "N/A" if value is None else f"{value:.3f}",
                    ha="center", fontsize=9)
    fig.suptitle("20 fixed questions, one pass — same-family AI-assisted review; HUMAN REVIEW PENDING", fontsize=11)
    fig.savefig(directory / plan["plot_file"], dpi=170)
    plt.close(fig)
    print(f"图表已保存：{directory / plan['plot_file']}")


if __name__ == "__main__":
    main()
