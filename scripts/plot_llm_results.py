"""从已校验的 Phase 6 回答生成性能与辅助复核图。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validate_llm_results import load_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    directory, _, summary = load_results(config_path=args.config)
    rows = summary["results"]
    labels = [row["provider_id"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(labels, [row["latency_median_seconds"] for row in rows])
    axes[0].set_ylabel("Median wall time (s)")
    axes[0].set_title("20 fixed-context questions; one pass")
    if all(row["semantic_quality"] is not None for row in rows):
        width = 0.25
        for index, (metric, label) in enumerate([
            ("required_fact_coverage", "Required-fact coverage"),
            ("semantic_citation_support", "Citation support"),
            ("has_unsupported_claim", "Answers with unsupported claims"),
        ]):
            axes[1].bar([x + (index - 1) * width for x in range(len(rows))],
                        [row["semantic_quality"][metric] for row in rows], width, label=label)
        axes[1].set_xticks(range(len(labels)), labels)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title("Provisional assisted review; not independent")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "Semantic review pending", ha="center", va="center")
        axes[1].set_axis_off()
    figure.savefig(directory / "llm_comparison.png", dpi=160)
    plt.close(figure)
    print(f"图表已保存：{directory / 'llm_comparison.png'}")


if __name__ == "__main__":
    main()
