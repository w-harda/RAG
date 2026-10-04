"""由 Phase 5 实测结果生成规模、建库和延迟比较图。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / "results/phase5"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for backend in ("faiss", "chroma"):
        rows = [row for row in summary["results"] if row["backend"] == backend]
        x = [row["size"] for row in rows]
        axes[0].plot(x, [row["build_seconds"]["mean"] for row in rows], "o-", label=backend)
        for axis, mode in zip(axes[1:], ("unfiltered", "filtered"), strict=True):
            axis.plot(x, [row[mode]["latency_ms"]["p95"] for row in rows], "o-", label=backend)
    for axis, title, ylabel in zip(axes, ("Build incl. metadata + persistence", "Unfiltered query (p95)",
                                        "Metadata-filtered query (p95)"),
                                    ("Seconds", "Milliseconds", "Milliseconds"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Number of vectors")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_ylim(bottom=0)
    fig.suptitle("APT vector-store benchmark: fixed BGE-M3 vectors, HNSW, single thread")
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
