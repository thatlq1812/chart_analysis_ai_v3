"""
Thesis Figure Generator -- Geo-SLM Chart Analysis

Generates all matplotlib/seaborn figures for the thesis report.
Output: docs/thesis_capstone/figures/ (PDF format, 300dpi for raster elements)

Usage:
    .venv/Scripts/python.exe scripts/generate_thesis_figures.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("pgf")  # LaTeX-compatible backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "thesis_capstone" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Global matplotlib settings for academic figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
})

# Color palette (colorblind-friendly, academic)
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "accent1": "#3498DB",
    "accent2": "#2ECC71",
    "accent3": "#F39C12",
    "accent4": "#9B59B6",
    "accent5": "#1ABC9C",
    "accent6": "#E67E22",
    "gray": "#95A5A6",
    "light": "#ECF0F1",
}

CHART_TYPE_COLORS = {
    "line": "#3498DB",
    "bar": "#E74C3C",
    "scatter": "#2ECC71",
    "heatmap": "#F39C12",
    "box": "#9B59B6",
    "pie": "#1ABC9C",
    "histogram": "#E67E22",
    "area": "#95A5A6",
}


# ---------------------------------------------------------------------------
# Data (from verified project metrics)
# ---------------------------------------------------------------------------

# Chart type distribution in classified_charts corpus
CHART_TYPE_DIST = {
    "line": 10_036,
    "bar": 9_086,
    "box": 4_867,
    "scatter": 2_802,
    "pie": 2_421,
    "histogram": 2_060,
    "heatmap": 680,
    "area": 412,
}

# SLM training dataset v3 per-type distribution
SLM_V3_DIST = {
    "line": 108_419,
    "scatter": 52_163,
    "bar": 47_330,
    "heatmap": 33_373,
    "box": 13_948,
    "pie": 7_408,
    "histogram": 4_159,
    "area": 1_999,
}

# Stage 3 feature quality by chart type
FEATURE_QUALITY = {
    "histogram": {"axis_coverage": 99.0, "ocr_confidence": 0.932, "zero_text": 2.0},
    "bar":       {"axis_coverage": 96.8, "ocr_confidence": 0.905, "zero_text": 29.6},
    "line":      {"axis_coverage": 88.5, "ocr_confidence": 0.837, "zero_text": 5.0},
    "scatter":   {"axis_coverage": 73.3, "ocr_confidence": 0.786, "zero_text": 8.0},
    "box":       {"axis_coverage": 60.9, "ocr_confidence": 0.648, "zero_text": 15.0},
    "heatmap":   {"axis_coverage": 52.2, "ocr_confidence": 0.453, "zero_text": 18.0},
    "pie":       {"axis_coverage": 42.1, "ocr_confidence": 0.512, "zero_text": 10.0},
    "area":      {"axis_coverage": 34.2, "ocr_confidence": 0.387, "zero_text": 59.5},
}

# Processing time breakdown (estimated, seconds)
PROCESSING_TIME = {
    "Stage 1\n(Ingestion)": 0.4,
    "Stage 2\n(Detection)": 1.0,
    "Stage 3\n(Extraction)": 5.0,
    "Stage 4\n(Reasoning)": 1.5,
    "Stage 5\n(Reporting)": 0.3,
}

# Data pipeline funnel
DATA_PIPELINE_FUNNEL = [
    ("arXiv PDFs", 4_000),
    ("Page images", 150_000),
    ("Candidate charts", 70_000),
    ("Quality filtered", 46_910),
    ("Classified charts", 32_364),
]

# ResNet-18 integration test per-class accuracy
RESNET_PER_CLASS = {
    "line": 100.0,
    "scatter": 100.0,
    "pie": 100.0,
    "bar": 100.0,
    "box": 100.0,
    "histogram": 100.0,
    "heatmap": 100.0,
    "area": 50.0,
}


# ---------------------------------------------------------------------------
# Figure Generators
# ---------------------------------------------------------------------------

def fig_chart_type_distribution() -> None:
    """Bar chart of chart type distribution in the corpus."""
    types = list(CHART_TYPE_DIST.keys())
    counts = list(CHART_TYPE_DIST.values())
    colors = [CHART_TYPE_COLORS[t] for t in types]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(types, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar_item, pct in zip(bars, pcts):
        width = bar_item.get_width()
        ax.text(
            width + 150, bar_item.get_y() + bar_item.get_height() / 2,
            f"{width:,} ({pct:.1f}\\%)",
            va="center", fontsize=8,
        )

    ax.set_xlabel("Number of Charts")
    ax.set_title("Chart Type Distribution in Academic Corpus (N=32,364)")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig_chart_type_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_slm_dataset_distribution() -> None:
    """Bar chart comparing source charts vs SLM training samples per type."""
    types = list(SLM_V3_DIST.keys())
    slm_counts = [SLM_V3_DIST[t] for t in types]
    source_counts = [CHART_TYPE_DIST.get(t, 0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4))

    bars1 = ax1.bar(x - width / 2, source_counts, width, label="Source Charts",
                     color=COLORS["accent1"], edgecolor="white")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, slm_counts, width, label="SLM Samples (v3)",
                     color=COLORS["secondary"], alpha=0.8, edgecolor="white")

    ax1.set_xlabel("Chart Type")
    ax1.set_ylabel("Source Charts", color=COLORS["accent1"])
    ax2.set_ylabel("SLM Training Samples", color=COLORS["secondary"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=30, ha="right")
    ax1.set_title("Source Charts vs SLM Training Samples by Type")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    out = FIGURES_DIR / "fig_slm_dataset_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_feature_quality() -> None:
    """Grouped bar chart: axis coverage and OCR confidence by chart type."""
    types = list(FEATURE_QUALITY.keys())
    axis_cov = [FEATURE_QUALITY[t]["axis_coverage"] for t in types]
    ocr_conf = [FEATURE_QUALITY[t]["ocr_confidence"] * 100 for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.8))

    ax.bar(x - width / 2, axis_cov, width, label="Axis Info Coverage (\\%)",
           color=COLORS["accent1"], edgecolor="white")
    ax.bar(x + width / 2, ocr_conf, width, label="Mean OCR Confidence (\\%)",
           color=COLORS["accent3"], edgecolor="white")

    ax.set_xlabel("Chart Type")
    ax.set_ylabel("Percentage (\\%)")
    ax.set_title("Stage 3 Feature Quality by Chart Type")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(y=69.9, color=COLORS["gray"], linestyle="--", linewidth=0.8, label="Avg axis coverage")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig_feature_quality_by_type.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_processing_time() -> None:
    """Stacked horizontal bar showing processing time per stage."""
    stages = list(PROCESSING_TIME.keys())
    times = list(PROCESSING_TIME.values())
    total = sum(times)
    pcts = [t / total * 100 for t in times]

    palette = [COLORS["accent1"], COLORS["accent2"], COLORS["secondary"],
               COLORS["accent4"], COLORS["accent3"]]

    fig, ax = plt.subplots(figsize=(7, 2.0))

    left = 0.0
    for i, (stage, t) in enumerate(zip(stages, times)):
        bar = ax.barh(0, t, left=left, color=palette[i], edgecolor="white",
                       height=0.6, label=stage)
        cx = left + t / 2
        label_text = f"{t:.1f}s\n({pcts[i]:.0f}\\%)"
        ax.text(cx, 0, label_text, ha="center", va="center", fontsize=7,
                fontweight="bold", color="white" if t > 0.8 else COLORS["primary"])
        left += t

    ax.set_xlim(0, total * 1.05)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Per-Chart Processing Time Breakdown (Total: {total:.1f}s)")
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    out = FIGURES_DIR / "fig_processing_time_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_data_pipeline_funnel() -> None:
    """Horizontal funnel chart showing data pipeline reduction."""
    labels = [item[0] for item in DATA_PIPELINE_FUNNEL]
    values = [item[1] for item in DATA_PIPELINE_FUNNEL]
    max_val = max(values)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    palette = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"],
               COLORS["accent4"], COLORS["secondary"]]

    for i, (label, val) in enumerate(zip(labels, values)):
        bar_width = val / max_val * max_val
        bar = ax.barh(len(labels) - 1 - i, val, color=palette[i],
                       edgecolor="white", height=0.6)
        ax.text(val + max_val * 0.02, len(labels) - 1 - i,
                f"{val:,}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Count")
    ax.set_title("Data Pipeline Funnel: arXiv Papers to Classified Charts")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig_data_pipeline_funnel.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_resnet_per_class() -> None:
    """Bar chart of ResNet-18 per-class accuracy."""
    types = list(RESNET_PER_CLASS.keys())
    accs = list(RESNET_PER_CLASS.values())
    colors = [COLORS["accent2"] if a >= 90 else COLORS["secondary"] for a in accs]

    fig, ax = plt.subplots(figsize=(6, 3.2))

    bars = ax.bar(types, accs, color=colors, edgecolor="white", width=0.6)
    for bar_item, acc in zip(bars, accs):
        ax.text(bar_item.get_x() + bar_item.get_width() / 2, acc + 1.5,
                f"{acc:.0f}\\%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Chart Type")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_title("ResNet-18 Per-Class Classification Accuracy")
    ax.axhline(y=94.14, color=COLORS["accent1"], linestyle="--", linewidth=0.8)
    ax.text(7.5, 95.5, "Overall: 94.14\\%", fontsize=8, color=COLORS["accent1"], ha="right")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig_resnet_per_class_accuracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def fig_dataset_v2_vs_v3() -> None:
    """Side-by-side comparison of v2 vs v3 dataset metrics."""
    metrics = ["Samples", "Axis Coverage\n(\\%)", "Chart Types", "Stage3\nCoverage (\\%)"]
    v2_vals = [27_159, 4.0, 8, 60.0]
    v3_vals = [268_799, 69.9, 8, 100.0]

    # Normalize for visualization (log scale for samples)
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))

    for i, (metric, v2, v3) in enumerate(zip(metrics, v2_vals, v3_vals)):
        ax = axes[i]
        bars = ax.bar(["v2", "v3"], [v2, v3],
                       color=[COLORS["gray"], COLORS["accent1"]], edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=9)

        for bar_item in bars:
            height = bar_item.get_height()
            label = f"{height:,.0f}" if height > 100 else f"{height:.1f}"
            ax.text(bar_item.get_x() + bar_item.get_width() / 2, height,
                    label, ha="center", va="bottom", fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}K"))

    fig.suptitle("SLM Training Dataset: v2 vs v3 Comparison", fontsize=11, y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_dataset_v2_vs_v3.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    ("fig_chart_type_distribution", fig_chart_type_distribution),
    ("fig_slm_dataset_distribution", fig_slm_dataset_distribution),
    ("fig_feature_quality_by_type", fig_feature_quality),
    ("fig_processing_time_breakdown", fig_processing_time),
    ("fig_data_pipeline_funnel", fig_data_pipeline_funnel),
    ("fig_resnet_per_class_accuracy", fig_resnet_per_class),
    ("fig_dataset_v2_vs_v3", fig_dataset_v2_vs_v3),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info(f"Output directory: {FIGURES_DIR}")
    logger.info(f"Generating {len(ALL_FIGURES)} figures...")

    success = 0
    failed = 0
    for name, func in ALL_FIGURES:
        try:
            func()
            success += 1
        except Exception as e:
            logger.error(f"Failed: {name} | error={e}")
            failed += 1

    logger.info(f"Done | success={success} | failed={failed}")


if __name__ == "__main__":
    main()
