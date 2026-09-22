import gradio as gr
from prepare_lora_dataset import process_directory
import os

def run_pipeline(input_dir, output_dir, dataset_name, max_duration):
    if not input_dir or not output_dir or not dataset_name:
        yield "❌ Error: Please provide all required inputs (Input Directory, Output Directory, and Dataset Name)."
        return
        
    if not os.path.exists(input_dir):
        yield f"❌ Error: Input directory '{input_dir}' does not exist."
        return

    # Yield real-time logs from the backend
    log_output = ""
    yield "🚀 Starting Audio Intelligence Pipeline...\n"
    
    for log_msg in process_directory(input_dir, output_dir, dataset_name, max_duration):
        log_output += log_msg
        yield log_output

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="zinc"), title="Audio Intelligence DSP Pipeline") as demo:
    gr.Markdown(
        """
        # 🎛️ Audio Intelligence DSP Pipeline
        *Automated data engineering for AI music models. Trims silence, normalizes volume, extracts tempo & key, and quarantines bad audio.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Pipeline Configuration")
            input_dir = gr.Textbox(label="Source Directory", placeholder="/path/to/raw/audio/folder")
            output_dir = gr.Textbox(label="Destination Directory", placeholder="/path/to/save/processed/dataset")
            dataset_name = gr.Textbox(label="Dataset Name", placeholder="e.g. BollyHood Beats", value="BollyHood Beats")
            
            max_duration = gr.Slider(
                minimum=5, 
                maximum=180, 
                value=45, 
                step=1, 
                label="Maximum Duration (seconds)", 
                info="Audio longer than this will be truncated. Shorter rhythms will be looped up to this length."
            )
            
            run_btn = gr.Button("🚀 Start DSP Pipeline", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📜 Real-time Console Log")
            console_output = gr.Textbox(
                label="Pipeline Log (Shows Live Progress & Quarantined Files)", 
                lines=20, 
                max_lines=25, 
                interactive=False,
                show_copy_button=True
            )
            
    # Connect the button to the backend generator
    run_btn.click(
        fn=run_pipeline,
        inputs=[input_dir, output_dir, dataset_name, max_duration],
        outputs=[console_output]
    )

if __name__ == "__main__":
    print("Launching Audio Intelligence Pipeline UI...")
    demo.queue().launch(server_name="0.0.0.0", server_port=7865)
