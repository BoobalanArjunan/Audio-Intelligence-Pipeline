import gradio as gr
from prepare_lora_dataset import process_directory
import os

def run_pipeline(input_dir, output_dir, quarantine_dir, dataset_name, max_duration):
    if not all([input_dir, output_dir, quarantine_dir, dataset_name]):
        yield "❌ Error: Please fill in all directory and dataset name fields."
        return

    if not os.path.exists(input_dir):
        yield f"❌ Error: Source directory does not exist:\n  {input_dir}"
        return

    log_output = "🚀 Starting Audio Intelligence Pipeline...\n"
    yield log_output

    for msg in process_directory(input_dir, output_dir, quarantine_dir, dataset_name, max_duration):
        log_output += msg
        yield log_output


with gr.Blocks(title="Audio Intelligence DSP Pipeline — Boobalan Arjunan") as demo:

    gr.Markdown("""
# 🎛️ Audio Intelligence DSP Pipeline
### Product Owner & Lead Architect: **Boobalan Arjunan**

> *Enterprise-grade automated data engineering for AI music models.*
> *This engine uses advanced Digital Signal Processing (DSP) — including Krumhansl-Schmuckler Key Profiles, Onset-based One-Shot Classification, and Equal-Power Crossfade Loop Stitching — to prepare production-ready LoRA training datasets without any manual input.*
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Section 1: Directory Configuration")
            gr.Markdown("*Paste the absolute paths for your local folders below.*")
            input_dir    = gr.Textbox(label="Source Audio Directory (Raw WAVs)",
                                      placeholder="/Users/mymac/Desktop/Raw_Audio")
            output_dir   = gr.Textbox(label="Processed Output Directory (AI-Ready Dataset)",
                                      placeholder="/Users/mymac/Desktop/Processed_Dataset")
            quarantine_dir = gr.Textbox(label="🚨 Quarantine Directory (Bad / Corrupted Files)",
                                        placeholder="/Users/mymac/Desktop/Quarantine_Vault")

            gr.Markdown("### ⚙️ Section 2: Model Configuration")
            dataset_name = gr.Textbox(label="Dataset Name",
                                      placeholder="e.g. BollyHood Beats",
                                      value="BollyHood Beats")
            max_duration = gr.Slider(
                minimum=5, maximum=180, value=45, step=1,
                label="Target Context Window (Seconds)",
                info="Max audio duration fed into the DiT. 45s is required for ACE-Step 4B to avoid OOM."
            )
            run_btn = gr.Button("🚀 Initialize DSP Engine", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Section 3: Execution Engine & Live Console")
            gr.Markdown(
                "**Fully automated — no manual BPM input needed.** "
                "The engine uses beat tracking, Krumhansl-Schmuckler harmonic correlation, "
                "and spectral centroid analysis to label every file automatically."
            )
            console_output = gr.Textbox(
                label="Pipeline Log",
                lines=28, max_lines=32,
                interactive=False
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_dir, output_dir, quarantine_dir, dataset_name, max_duration],
        outputs=[console_output]
    )


if __name__ == "__main__":
    print("Launching Premium Audio Intelligence Pipeline UI...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7865,
        theme=gr.themes.Monochrome(primary_hue="zinc")
    )
