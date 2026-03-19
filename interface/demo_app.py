"""
Geo-SLM Chart Analysis -- Interactive Demo

Gradio Blocks app for demonstrating the chart analysis pipeline.
Supports multiple VLM backends, displays extracted data as table and JSON.

Usage:
    .venv\\Scripts\\python.exe interface/demo_app.py
    # Opens browser at http://localhost:7860

    # Or with Makefile:
    make demo
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from interface.demo_utils import get_sample_images, run_pipeline_for_demo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Run: .venv\\Scripts\\pip.exe install gradio")
        sys.exit(1)

    sample_images = get_sample_images()

    def analyze_chart(image_path, backend, device):
        """Main analysis function called by Gradio."""
        if image_path is None:
            return "{}", "Upload an image to start analysis.", "<p>No image provided</p>", ""

        result = run_pipeline_for_demo(
            image_path=image_path,
            backend=backend,
            device=device,
        )

        if result["error"]:
            error_msg = f"Error: {result['error']}"
            return "{}", error_msg, "<p>Analysis failed</p>", error_msg

        timing_str = (
            f"Extraction: {result['timing'].get('extraction_s', 0):.2f}s | "
            f"Classification: {result['timing'].get('classification_s', 0):.2f}s | "
            f"Total: {result['timing'].get('total_s', 0):.2f}s"
        )

        return (
            result["json_output"],
            result["summary"],
            result["table_html"],
            timing_str,
        )

    # Build Gradio interface
    with gr.Blocks(
        title="Geo-SLM Chart Analysis",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Geo-SLM Chart Analysis

            Upload a chart image to extract structured data using hybrid AI pipeline.

            **Pipeline:** EfficientNet Classification -> VLM Extraction (DePlot/MatCha) -> Structured Output

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                image_input = gr.Image(
                    label="Chart Image",
                    type="filepath",
                    height=400,
                )

                backend_dropdown = gr.Dropdown(
                    choices=["deplot", "matcha", "pix2struct"],
                    value="deplot",
                    label="VLM Backend",
                    info="DePlot is recommended for best accuracy",
                )

                device_dropdown = gr.Dropdown(
                    choices=["cpu", "cuda", "auto"],
                    value="cpu",
                    label="Device",
                )

                analyze_btn = gr.Button(
                    "Analyze Chart",
                    variant="primary",
                    size="lg",
                )

                # Sample gallery
                if sample_images:
                    gr.Markdown("### Sample Charts")
                    gr.Examples(
                        examples=[[img] for img in sample_images[:6]],
                        inputs=image_input,
                        label="Click to load sample",
                    )

            with gr.Column(scale=1):
                # Output section
                timing_output = gr.Textbox(
                    label="Processing Time",
                    interactive=False,
                )

                with gr.Tabs():
                    with gr.TabItem("Summary"):
                        summary_output = gr.Textbox(
                            label="Analysis Summary",
                            lines=8,
                            interactive=False,
                        )

                    with gr.TabItem("Extracted Table"):
                        table_output = gr.HTML(
                            label="Data Table",
                        )

                    with gr.TabItem("JSON Output"):
                        json_output = gr.Code(
                            label="Structured Data (JSON)",
                            language="json",
                            lines=20,
                        )

        # Wire up the button
        analyze_btn.click(
            fn=analyze_chart,
            inputs=[image_input, backend_dropdown, device_dropdown],
            outputs=[json_output, summary_output, table_output, timing_output],
        )

        gr.Markdown(
            """
            ---

            **Geo-SLM Chart Analysis v3** | Hybrid AI System for Chart Data Extraction

            Built with: YOLOv8 + EfficientNet-B0 + DePlot/MatCha + AI Router (Gemini/OpenAI/Local SLM)
            """
        )

    return app


def main():
    """Launch the Gradio demo server."""
    app = create_app()
    logger.info("Starting Gradio demo at http://localhost:7860")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
