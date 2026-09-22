import gradio as gr
from prepare_lora_dataset import process_directory
import os

def run_pipeline(input_dir, output_dir, quarantine_dir, dataset_name, max_duration):
    if not input_dir or not output_dir or not quarantine_dir or not dataset_name:
        yield "❌ Error: Please provide all required inputs (Source, Output, Quarantine Directories, and Dataset Name)."
        return
        
    if not os.path.exists(input_dir):
        yield f"❌ Error: Input directory '{input_dir}' does not exist."
        return

    # Yield real-time logs from the backend
    log_output = ""
    yield "🚀 Starting Audio Intelligence Pipeline...\n"
    
    for log_msg in process_directory(input_dir, output_dir, quarantine_dir, dataset_name, max_duration):
        log_output += log_msg
        yield log_output

# Build the Premium Gradio UI
with gr.Blocks(title="Audio Intelligence DSP Pipeline") as demo:
    gr.Markdown(
        """
        # 🎛️ Audio Intelligence DSP Pipeline
        **Product Owner & Lead Architect: Boobalan Arjunan**
        
        *Enterprise-grade automated data engineering for AI music models. This engine leverages Digital Signal Processing (DSP) to analyze audio files in bulk, extracting Chroma features (Key) and Spectral Centroids (Vibe), while simultaneously preventing model VRAM crashes through intelligent boundary truncation and audio normalization.*
        """
    )
    
    with gr.Row():
        # Section 1 & 2: Configuration
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Section 1: Directory Configuration\n*Please paste the absolute paths for your local folders below.*")
            input_dir = gr.Textbox(label="Source Audio Directory (Raw WAVs)", placeholder="/Users/mymac/Desktop/Raw_Audio")
            output_dir = gr.Textbox(label="Processed Output Directory (AI-Ready Dataset)", placeholder="/Users/mymac/Desktop/Processed_Dataset")
            quarantine_dir = gr.Textbox(label="Quarantine Directory (For Bad/Corrupted Data)", placeholder="/Users/mymac/Desktop/Quarantine_Vault")
            
            gr.Markdown("### ⚙️ Section 2: Model Configuration")
            dataset_name = gr.Textbox(label="Dataset Name", placeholder="e.g. BollyHood Beats", value="BollyHood Beats")
            
            max_duration = gr.Slider(
                minimum=5, 
                maximum=180, 
                value=45, 
                step=1, 
                label="Target Context Window (Seconds)", 
                info="Determines the max bounds for positional embeddings in the DiT."
            )
            
            run_btn = gr.Button("🚀 Initialize DSP Engine", variant="primary")
            
        # Section 3: Execution
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Section 3: Execution Engine & Live Console")
            gr.Markdown("Watch the engine dynamically process and quarantine audio in real time.")
            console_output = gr.Textbox(
                label="Pipeline Log", 
                lines=25, 
                max_lines=30, 
                interactive=False
            )
            
    # Connect the button to the backend generator
    run_btn.click(
        fn=run_pipeline,
        inputs=[input_dir, output_dir, quarantine_dir, dataset_name, max_duration],
        outputs=[console_output]
    )

if __name__ == "__main__":
    print("Launching Premium Audio Intelligence Pipeline UI...")
    demo.queue().launch(server_name="0.0.0.0", server_port=7865, theme=gr.themes.Monochrome(primary_hue="zinc"))
