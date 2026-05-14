import os
# Force XLA to reserve exactly 85% of VRAM for training on process startup, leaving 15% contiguous for UI/Chat buffer
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import gradio as gr
import threading
import time
import glob
import jax
import gc
import jax.numpy as jnp
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
import sft_trainer
import generate
from config import LlamaConfig

training_thread = None
stop_event = threading.Event()

def stop_training():
    global training_thread
    if training_thread is not None and training_thread.is_alive():
        # Using a hard stop for threads running jax is complex. We rely on standard process termination in Colab usually,
        # but here we can try to join or just set a flag (not fully implemented in loops, so it's a soft stop).
        # We will just abandon the thread reference so a new one can start (hacky but works for UI demo).
        training_thread = None
        return "Training thread abandoned."
    return "No training process running."

def run_train():
    train.main(["--data_dir", "/content/drive/MyDrive/Omega_20M_Final", "--output_dir", "/content/drive/MyDrive/Omega_20M_Final/checkpoints"])

def run_sft_train():
    sft_trainer.main(["--data_file", "/content/drive/MyDrive/AI LMM TRAININGSDATEN DATA SET DRIN/phase2_coding_instruct.jsonl", "--output_dir", "/content/drive/MyDrive/Omega_20M_Final/checkpoints"])

def launch_training():
    global training_thread
    stop_training()
    training_thread = threading.Thread(target=run_train, daemon=True)
    training_thread.start()
    return "Pre-Training Background Thread Started!"

def launch_sft_training():
    global training_thread
    stop_training()
    training_thread = threading.Thread(target=run_sft_train, daemon=True)
    training_thread.start()
    return "Phase 2 SFT Background Thread Started!"

def get_latest_metrics():
    log_dir = "/content/drive/MyDrive/Omega_20M_Final/checkpoints/logs"
    if not os.path.exists(log_dir):
        return "Waiting for logs...", "N/A", "N/A", "N/A"

    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return "Waiting for events...", "N/A", "N/A", "N/A"

    latest_event_file = max(event_files, key=os.path.getctime)

    ea = EventAccumulator(latest_event_file, size_guidance={'scalars': 1})
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
    checkpoint_dir = "/content/drive/MyDrive/Omega_20M_Final/checkpoints"

    prompt = message
    if mode == "Instruct Mode":
        prompt = f"Instruction: {message}\nInput: \nOutput: "

    try:
        # Prevent memory fragmentation by forcing GC before allocating new chat buffers
        gc.collect()

        # Load Tokenizer
        tokenizer_path = 'tokenizer.json'
        if os.path.exists(tokenizer_path):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        config = LlamaConfig()

        # Hot-reload in-memory weights if available
        current_state = None
        if train.global_state is not None:
            current_state = train.global_state
        elif sft_trainer.global_state is not None:
            current_state = sft_trainer.global_state
        else:
            # Fallback to loading from disk or random weights
            rng = jax.random.PRNGKey(0)
            dummy_lr = train.create_learning_rate_schedule(1, 1, 1.0)
            current_state = train.create_train_state(rng, config, dummy_lr)

            import orbax.checkpoint
            if os.path.exists(checkpoint_dir):
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=False)
                checkpoint_manager = orbax.checkpoint.CheckpointManager(
                    checkpoint_dir, orbax_checkpointer, options
                )
                latest_step = checkpoint_manager.latest_step()
                if latest_step is not None:
                    restored = checkpoint_manager.restore(latest_step)
                    current_state = current_state.replace(step=restored['step'], params=restored['params'])

        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = jnp.array(input_ids)

        output_ids = generate.generate(
            current_state.apply_fn,
            current_state.params,
            input_ids,
            config,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )

        generated_text = tokenizer.decode(np.array(output_ids[0]), skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text

    except Exception as e:
        return f"Error generating response: {e}"

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

            status_text = gr.Textbox(label="Process Status", value="Idle", interactive=False)

            start_btn = gr.Button("Start Pre-Training Background Thread", variant="primary")
            start_btn.click(fn=launch_training, outputs=status_text)

            sft_btn = gr.Button("Start Phase 2 SFT Thread", variant="secondary")
            sft_btn.click(fn=launch_sft_training, outputs=status_text)

            stop_btn = gr.Button("Stop All Training", variant="stop")
            stop_btn.click(fn=stop_training, outputs=status_text)

            # Auto-refresh metrics using gr.Timer
            timer = gr.Timer(1)
            timer.tick(fn=get_latest_metrics, outputs=[step_text, loss_text, lr_text, tps_text])

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
