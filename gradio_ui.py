import gradio as gr
import threading
import subprocess
import time
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def launch_training():
    # Launches the training script as a subprocess so it doesn't block the UI
    subprocess.Popen([
        "python", "train.py",
        "--data_dir", "/content/drive/MyDrive/Omega_20M_Final",
        "--output_dir", "/content/drive/MyDrive/Omega_20M_Final/checkpoints"
    ])

def launch_sft_training():
    subprocess.Popen([
        "python", "sft_trainer.py",
        "--data_file", "/content/drive/MyDrive/AI LMM TRAININGSDATEN DATA SET DRIN/phase2_coding_instruct.jsonl",
        "--output_dir", "/content/drive/MyDrive/Omega_20M_Final/checkpoints"
    ])

def get_latest_metrics():
    log_dir = "/content/drive/MyDrive/Omega_20M_Final/checkpoints/logs"
    if not os.path.exists(log_dir):
        return "Waiting for logs...", "N/A", "N/A", "N/A"

    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return "Waiting for events...", "N/A", "N/A", "N/A"

    latest_event_file = max(event_files, key=os.path.getctime)

    ea = EventAccumulator(latest_event_file)
    ea.Reload()

    step = "N/A"
    loss = "N/A"
    lr = "N/A"
    tps = "N/A"

    if "Train/Loss" in ea.Tags()['scalars']:
        events = ea.Scalars("Train/Loss")
        if events:
            latest = events[-1]
            step = str(latest.step)
            loss = f"{latest.value:.4f}"

    if "Train/LearningRate" in ea.Tags()['scalars']:
        events = ea.Scalars("Train/LearningRate")
        if events:
            lr = f"{events[-1].value:.2e}"

    if "Train/Throughput" in ea.Tags()['scalars']:
        events = ea.Scalars("Train/Throughput")
        if events:
            tps = f"{events[-1].value:.2f}"

    return step, loss, lr, tps

def chat_inference(message, history, mode):
    # Call the generate.py script to load the latest weights and generate a response
    checkpoint_dir = "/content/drive/MyDrive/Omega_20M_Final/checkpoints"

    prompt = message
    if mode == "Instruct Mode":
        prompt = f"Instruction: {message}\nInput: \nOutput: "

    if not os.path.exists(checkpoint_dir):
        return "Model has not saved any checkpoints yet. Please wait for training."

    try:
        result = subprocess.run([
            "python", "generate.py",
            "--checkpoint_dir", checkpoint_dir,
            "--prompt", prompt,
            "--max_new_tokens", "50"
        ], capture_output=True, text=True)

        output = result.stdout
        # Extract the generated text from the script's stdout
        if "--- Generated Text ---" in output:
            generated_text = output.split("--- Generated Text ---")[1].split("----------------------")[0].strip()
            # Remove the prompt from the generated text if it's there
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        else:
            return "Error generating response: " + output + "\n" + result.stderr
    except Exception as e:
        return str(e)

# Gradio UI Layout
with gr.Blocks(title="Omega 20M TPU Pre-training Dashboard") as demo:
    gr.Markdown("# 🚀 Omega 20M LLM Pre-training on TPU v5e")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Live Training Metrics")
            step_text = gr.Textbox(label="Current Step", value="Not Started", interactive=False)
            loss_text = gr.Textbox(label="Training Loss", value="N/A", interactive=False)
            lr_text = gr.Textbox(label="Learning Rate", value="N/A", interactive=False)
            tps_text = gr.Textbox(label="Tokens/second", value="N/A", interactive=False)

            start_btn = gr.Button("Start Pre-Training Background Thread", variant="primary")
            start_btn.click(fn=launch_training)

            sft_btn = gr.Button("Start Phase 2 SFT Thread", variant="secondary")
            sft_btn.click(fn=launch_sft_training)

            # Auto-refresh metrics every 5 seconds
            demo.load(
                fn=get_latest_metrics,
                inputs=None,
                outputs=[step_text, loss_text, lr_text, tps_text],
                every=5
            )

        with gr.Column(scale=2):
            gr.Markdown("## Live Evaluation Chat")
            gr.Markdown("*Chat with the most recently saved checkpoint from `/content/drive/MyDrive/Omega_20M_Final/checkpoints`.*")

            mode_radio = gr.Radio(["Pre-training Mode", "Instruct Mode"], value="Pre-training Mode", label="Inference Mode")

            chatbot = gr.ChatInterface(
                fn=chat_inference,
                additional_inputs=[mode_radio]
            )

if __name__ == "__main__":
    demo.launch(share=True, inline=True)
