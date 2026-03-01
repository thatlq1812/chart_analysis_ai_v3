"""
Thesis LaTeX Table Generator -- Geo-SLM Chart Analysis

Generates all LaTeX table code for inclusion in thesis .tex files.
Output: docs/thesis_capstone/figures/tables/ (one .tex file per table)

Usage:
    .venv/Scripts/python.exe scripts/generate_thesis_tables.py
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "docs" / "thesis_capstone" / "figures" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write(name: str, content: str) -> None:
    out = TABLES_DIR / f"{name}.tex"
    out.write_text(content, encoding="utf-8")
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Table Generators
# ---------------------------------------------------------------------------

def tab_technology_stack() -> None:
    _write("tab_technology_stack", r"""
\begin{table}[htbp]
\centering
\caption{Core Technology Stack}
\label{tab:technology-stack}
\begin{tabular}{@{}llp{5cm}@{}}
\toprule
\textbf{Component} & \textbf{Technology} & \textbf{Rationale} \\
\midrule
Language         & Python 3.11+          & AI/ML ecosystem maturity \\
Object Detection & YOLOv8m (Ultralytics) & Fast, accurate, trainable \\
OCR              & EasyOCR / PaddleOCR   & Multi-language support \\
Image Processing & OpenCV + Pillow       & Industry standard \\
Geometric Calc.  & NumPy + Custom        & Precision arithmetic \\
Classification   & ResNet-18 (PyTorch)   & Lightweight, 94.14\% accuracy \\
Data Validation  & Pydantic v2           & Runtime schema enforcement \\
Configuration    & OmegaConf (YAML)      & Hierarchical ML config \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_ai_providers() -> None:
    _write("tab_ai_providers", r"""
\begin{table}[htbp]
\centering
\caption{AI Reasoning Providers and Fallback Strategy}
\label{tab:ai-providers}
\begin{tabular}{@{}lllll@{}}
\toprule
\textbf{Provider} & \textbf{Model} & \textbf{Use Case} & \textbf{Priority} & \textbf{Cost} \\
\midrule
Local SLM & Qwen-2.5-1.5B    & Default reasoning   & Primary    & Free \\
Gemini    & gemini-2.0-flash  & Complex charts       & Fallback 1 & API \\
OpenAI    & gpt-4o-mini       & Alternative          & Fallback 2 & API \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_chart_type_distribution() -> None:
    _write("tab_chart_type_distribution", r"""
\begin{table}[htbp]
\centering
\caption{Chart Type Distribution in Academic Corpus}
\label{tab:chart-type-dist}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Chart Type} & \textbf{Count} & \textbf{Share (\%)} \\
\midrule
line      & 10,036 & 31.0 \\
bar       &  9,086 & 28.1 \\
box       &  4,867 & 15.0 \\
scatter   &  2,802 &  8.7 \\
pie       &  2,421 &  7.5 \\
histogram &  2,060 &  6.4 \\
heatmap   &    680 &  2.1 \\
area      &    412 &  1.3 \\
\midrule
\textbf{Total} & \textbf{32,364} & \textbf{100.0} \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_dataset_splits() -> None:
    _write("tab_dataset_splits", r"""
\begin{table}[htbp]
\centering
\caption{SLM Training Dataset v3 Splits}
\label{tab:dataset-splits}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Split} & \textbf{Samples} & \textbf{Ratio (\%)} \\
\midrule
Train      & 228,494 & 85 \\
Validation &  26,888 & 10 \\
Test       &  13,417 &  5 \\
\midrule
\textbf{Total} & \textbf{268,799} & \textbf{100} \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_yolo_results() -> None:
    _write("tab_yolo_results", r"""
\begin{table}[htbp]
\centering
\caption{YOLOv8m Chart Detection Results}
\label{tab:yolo-results}
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Target} & \textbf{Status} \\
\midrule
mAP@0.5   & 93.5\% & $>$85\% & Exceeded \\
Precision  & $>$90\% & $>$80\% & Exceeded \\
Recall     & $>$90\% & $>$80\% & Exceeded \\
Strategy   & Single-class (``chart'') & -- & -- \\
Input size & 640$\times$640 & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_resnet_results() -> None:
    _write("tab_resnet_results", r"""
\begin{table}[htbp]
\centering
\caption{ResNet-18 Chart Classification Results}
\label{tab:resnet-results}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Test Accuracy       & 94.14\% \\
Integration Accuracy & 93.75\% (15/16) \\
Training Time       & 27 minutes (GPU) \\
ONNX Inference      & 6.90 ms/image (CPU) \\
Throughput          & 144.9 img/sec \\
Model Size (ONNX)   & 42.64 MB \\
Classes             & 8 \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_stage3_feature_quality() -> None:
    _write("tab_stage3_feature_quality", r"""
\begin{table}[htbp]
\centering
\caption{Stage 3 Feature Quality by Chart Type}
\label{tab:stage3-quality}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Type} & \textbf{Axis Coverage (\%)} & \textbf{OCR Confidence} & \textbf{Zero-Text (\%)} \\
\midrule
histogram & 99.0 & 0.932 &  2.0 \\
bar       & 96.8 & 0.905 & 29.6 \\
line      & 88.5 & 0.837 &  5.0 \\
scatter   & 73.3 & 0.786 &  8.0 \\
box       & 60.9 & 0.648 & 15.0 \\
heatmap   & 52.2 & 0.453 & 18.0 \\
pie       & 42.1 & 0.512 & 10.0 \\
area      & 34.2 & 0.387 & 59.5 \\
\midrule
\textbf{Average} & \textbf{69.9} & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_comparison_related_work() -> None:
    _write("tab_comparison_related_work", r"""
\begin{table}[htbp]
\centering
\caption{Comparison with Related Chart Analysis Methods}
\label{tab:comparison-related}
\begin{tabular}{@{}lp{2.5cm}rrlc@{}}
\toprule
\textbf{Method} & \textbf{Approach} & \textbf{Value Acc.} & \textbf{Types} & \textbf{Model Size} & \textbf{Explainable} \\
\midrule
DePlot \cite{liu2023deplot}     & Pix2Struct + LLM       & $\sim$85\% & 5+ & 282M+ & No \\
MatCha \cite{lee2023matcha}     & Chart pretraining      & $\sim$80\% & 5+ & 282M+ & No \\
ChartReader \cite{chen2022chartreader} & Detection + rules & $\sim$75\% & 3  & N/A   & Partial \\
ReVision \cite{savva2011revision}      & Hough transform   & $\sim$70\% & 1  & N/A   & Yes \\
\midrule
\textbf{Geo-SLM (Ours)} & \textbf{Hybrid neuro-symbolic} & \textbf{Target $>$95\%} & \textbf{8} & \textbf{1.5B} & \textbf{Yes} \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_pipeline_stages() -> None:
    _write("tab_pipeline_stages", r"""
\begin{table}[htbp]
\centering
\caption{Pipeline Stage Summary}
\label{tab:pipeline-stages}
\begin{tabular}{@{}clp{3cm}p{3cm}l@{}}
\toprule
\textbf{Stage} & \textbf{Name} & \textbf{Input} & \textbf{Output} & \textbf{Key Library} \\
\midrule
1 & Ingestion   & PDF, DOCX, PNG, JPG     & Clean page images        & PyMuPDF, Pillow \\
2 & Detection   & Page images             & Cropped charts + BBoxes  & Ultralytics YOLO \\
3 & Extraction  & Cropped chart           & Raw metadata (OCR, geometry) & EasyOCR, OpenCV \\
4 & Reasoning   & Metadata + chart image  & Refined structured data  & AI Router (SLM/Gemini) \\
5 & Reporting   & Refined data            & JSON + Markdown report   & Pydantic, Jinja2 \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_data_pipeline_scale() -> None:
    _write("tab_data_pipeline_scale", r"""
\begin{table}[htbp]
\centering
\caption{Data Collection Pipeline Scale}
\label{tab:data-pipeline-scale}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Stage} & \textbf{Count} \\
\midrule
arXiv PDFs processed      &  $\sim$4,000 \\
Raw page images           & $\sim$150,000 \\
Candidate chart regions   &  $\sim$70,000 \\
After quality filtering   &     46,910 \\
Final classified charts   &     32,364 \\
OCR cache entries         &     46,910 \\
SLM training samples (v3) &    268,799 \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_slm_training_config() -> None:
    _write("tab_slm_training_config", r"""
\begin{table}[htbp]
\centering
\caption{SLM Fine-tuning Configuration}
\label{tab:slm-config}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Base model       & Qwen2.5-1.5B-Instruct \\
Method           & QLoRA (4-bit NF4 + LoRA rank 16) \\
Trainable params & 11.27M (0.9\% of total) \\
VRAM requirement & $\sim$4 GB (RTX 3060 6GB) \\
Epochs           & 3 (3 sessions $\times$ 1 epoch $\times$ $\sim$14h) \\
Batch size       & 4 (effective 16 with gradient accumulation) \\
Learning rate    & $2 \times 10^{-4}$, cosine schedule \\
Max seq. length  & 512 tokens \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


def tab_test_suite() -> None:
    _write("tab_test_suite", r"""
\begin{table}[htbp]
\centering
\caption{Test Suite Results}
\label{tab:test-suite}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Suite} & \textbf{Passed} & \textbf{Failed} & \textbf{Total} \\
\midrule
Schemas  &  19 & 0 &  19 \\
Stage 3  & 139 & 1 & 140 \\
Stage 4  &  18 & 0 &  18 \\
\midrule
\textbf{Total} & \textbf{176} & \textbf{1} & \textbf{177} \\
\textbf{Pass Rate} & \multicolumn{3}{r}{\textbf{99.4\%}} \\
\bottomrule
\end{tabular}
\end{table}
""".strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TABLES = [
    ("tab_technology_stack", tab_technology_stack),
    ("tab_ai_providers", tab_ai_providers),
    ("tab_chart_type_distribution", tab_chart_type_distribution),
    ("tab_dataset_splits", tab_dataset_splits),
    ("tab_yolo_results", tab_yolo_results),
    ("tab_resnet_results", tab_resnet_results),
    ("tab_stage3_feature_quality", tab_stage3_feature_quality),
    ("tab_comparison_related_work", tab_comparison_related_work),
    ("tab_pipeline_stages", tab_pipeline_stages),
    ("tab_data_pipeline_scale", tab_data_pipeline_scale),
    ("tab_slm_training_config", tab_slm_training_config),
    ("tab_test_suite", tab_test_suite),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info(f"Output directory: {TABLES_DIR}")
    logger.info(f"Generating {len(ALL_TABLES)} LaTeX tables...")

    success = 0
    for name, func in ALL_TABLES:
        try:
            func()
            success += 1
        except Exception as e:
            logger.error(f"Failed: {name} | error={e}")

    logger.info(f"Done | success={success}/{len(ALL_TABLES)}")


if __name__ == "__main__":
    main()
