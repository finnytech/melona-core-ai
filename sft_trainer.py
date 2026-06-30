import argparse
import os
import time
import jax
import subprocess
import jax.numpy as jnp
import optax
import flax
import orbax.checkpoint
import psutil
import gc
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from config import LlamaConfig
from model import LlamaForCausalLM
from train import TrainState, create_learning_rate_schedule

def export_to_safetensors(params, output_path):
    try:
        from safetensors.flax import save_file
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict) or isinstance(v, flax.core.FrozenDict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        flat_params = flatten_dict(flax.core.unfreeze(params))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(flat_params, output_path)
        print(f"Exported model weights to safetensors: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to export model to safetensors: {e}")

def get_sft_dataloader(data_file: str, tokenizer_path: str, batch_size: int, seq_length: int):
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # Find files: if data_file is a directory, glob all jsonl files in it
    import glob
    if os.path.isdir(data_file):
        data_files = glob.glob(os.path.join(data_file, "**/*.jsonl"), recursive=True)
    elif os.path.exists(data_file):
        data_files = [data_file]
    else:
        # Check if we can glob it directly (in case of wildcard strings)
        data_files = glob.glob(data_file)
        
    if not data_files:
        print(f"Warning: No training files found for path {data_file}. Using dummy dataset.")
        dataset = load_dataset("json", text='{"instruction": "dummy", "input": "dummy", "output": "dummy"}', split="train", streaming=True)
    else:
        print(f"Loading SFT dataset from {len(data_files)} files: {data_files[:5]}...")
        dataset = load_dataset("json", data_files=data_files, streaming=True, split="train")

    def process_and_tokenize(example):
        if "question" in example:
            instruction = "Answer this question."
            inp = example.get("question", "")
            output = example.get("text", "")
        elif "concept" in example:
            instruction = "Define this concept."
            inp = example.get("concept", "")
            output = example.get("text", "")
        else:
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")

        prompt = f"Instruction: {instruction}\nInput: {inp}\nOutput: "
        full_text = prompt + output + tokenizer.eos_token

        tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
        tokenized_full = tokenizer(full_text, truncation=True, max_length=seq_length, padding="max_length")

        input_ids = tokenized_full["input_ids"]
        prompt_len = len(tokenized_prompt["input_ids"])

        # Create loss mask: 0 for prompt, 1 for output
        loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

        # Ensure mask is exactly seq_length
        loss_mask = loss_mask[:seq_length]
        if len(loss_mask) < seq_length:
            loss_mask += [0] * (seq_length - len(loss_mask))

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask
        }

    tokenized_dataset = dataset.map(process_and_tokenize)

    def get_batch_iterator():
        while True: # Infinite Continuous Learning Loop
            batch_input_ids = []
            batch_loss_mask = []

            iterator = iter(tokenized_dataset)
            while True:
                try:
                    item = next(iterator)
                    batch_input_ids.append(item['input_ids'])
                    batch_loss_mask.append(item['loss_mask'])

                    if len(batch_input_ids) == batch_size:
                        yield {
                            "input_ids": jnp.array(batch_input_ids, dtype=jnp.int32),
                            "loss_mask": jnp.array(batch_loss_mask, dtype=jnp.float32)
                        }
                        batch_input_ids = []
                        batch_loss_mask = []
                except StopIteration:
                    print("SFT Dataset exhausted, looping back to the beginning for infinite learning.")
                    break
                except Exception as e:
                    print(f"Skipping corrupted SFT data line due to error: {e}")
                    continue

    return get_batch_iterator()

def sft_loss_fn(params, apply_fn, batch):
    input_ids = batch['input_ids']
    loss_mask = batch['loss_mask']

    # shift inputs and labels for causal LM
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    # Shift loss mask. The target token's loss is masked by the loss_mask corresponding to the target's position
    mask = loss_mask[:, 1:]

    logits, aux_loss = apply_fn({'params': params}, inputs)

    # Cast logits to fp32 to prevent NaNs
    logits = logits.astype(jnp.float32)

    # Cross entropy loss
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=targets_one_hot)

    # Apply Prompt Masking
    ce_loss = (ce_loss * mask).sum() / jnp.maximum(mask.sum(), 1e-5)

    # Combine SFT Cross Entropy Loss with Auxiliary Load Balancing Loss for MoE routing
    loss = ce_loss + 0.01 * aux_loss

    return loss

@jax.jit(donate_argnums=(0,))
def sft_train_step(state, batch):
    grad_fn = jax.value_and_grad(sft_loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def p_sft_train_step_fn(state, batch):
    grad_fn = jax.value_and_grad(sft_loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss

# Global state for hot-reloading in the UI
global_state = None

def create_sft_train_state(rng, config, learning_rate_fn):
    model = LlamaForCausalLM(config)
    dummy_input = jnp.ones((1, config.max_position_embeddings), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables['params']

    # Using a smaller learning rate for SFT
    tx = optax.adamw(learning_rate_fn, b1=0.9, b2=0.95, weight_decay=0.1)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def main(args_list=None):
    global global_state

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='/content/drive/MyDrive/Omega_20M_Final/knowledge_data.jsonl')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final/checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--peak_lr', type=float, default=2e-5) # lower for SFT
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--model_bucket', type=str, default=None, help='GCS bucket path for syncing checkpoints (e.g. finny-tech-ai-storage/finny-tech-ai-models)')
    parser.add_argument('--save_interval_secs', type=int, default=1200, help='Time interval in seconds between checkpoint saves (default: 1200s / 20 mins)')
    args = parser.parse_args(args_list)

    # Dynamic Batch Sizing based on available memory
    try:
        mem = psutil.virtual_memory()
        if mem.available > 30 * 1024**3:
            args.batch_size = max(args.batch_size, 16)
        elif mem.available > 15 * 1024**3:
            args.batch_size = max(args.batch_size, 8)
        else:
            args.batch_size = max(args.batch_size, 4)
    except Exception as e:
        print(f"Warning: Could not dynamically scale batch size: {e}")

    # Download SFT training dataset from GCS if it is a gs:// path
    if args.data_file.startswith("gs://"):
        try:
            print(f"Downloading SFT training dataset shards from GCS: {args.data_file} ...")
            local_data_dir = "./local_sft_data"
            os.makedirs(local_data_dir, exist_ok=True)
            # Sync GCS files to local directory
            subprocess.run(["gsutil", "-m", "cp", args.data_file, local_data_dir], check=True)
            args.data_file = local_data_dir
            print(f"Successfully downloaded training data to {local_data_dir}")
        except Exception as e:
            print(f"Error downloading training data from GCS: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs_sft'))
    config = LlamaConfig()

    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices
    print(f"Running SFT on {num_devices} devices: {jax.devices()}")

    dataloader = get_sft_dataloader(
        data_file=args.data_file,
        tokenizer_path=args.tokenizer_path,
        batch_size=global_batch_size,
        seq_length=config.max_position_embeddings
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    lr_schedule = create_learning_rate_schedule(args.max_steps, args.warmup_steps, args.peak_lr)
    state = create_sft_train_state(init_rng, config, lr_schedule)

    # Auto-restore from GCS bucket if local directory doesn't have checkpoints
    if args.model_bucket:
        try:
            print(f"Checking GCS bucket gs://{args.model_bucket} for existing checkpoints...")
            result = subprocess.run(["gsutil", "ls", f"gs://{args.model_bucket}"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                # Find all step numbers (GCS outputs them like gs://bucket/models/1000/)
                lines = result.stdout.strip().split("\n")
                steps = []
                for line in lines:
                    line = line.rstrip("/")
                    parts = line.split("/")
                    if parts[-1].isdigit():
                        steps.append(int(parts[-1]))
                
                if steps:
                    latest_gcs_step = max(steps)
                    print(f"Found latest checkpoint in GCS at step {latest_gcs_step}.")
                    local_step_dir = os.path.join(args.output_dir, str(latest_gcs_step))
                    if not os.path.exists(local_step_dir):
                        print(f"Downloading checkpoint for step {latest_gcs_step} from GCS to {args.output_dir}...")
                        os.makedirs(args.output_dir, exist_ok=True)
                        dl_result = subprocess.run([
                            "gsutil", "-m", "cp", "-r", 
                            f"gs://{args.model_bucket}/{latest_gcs_step}", 
                            args.output_dir
                        ], capture_output=True, text=True)
                        if dl_result.returncode == 0:
                            print(f"Successfully downloaded checkpoint {latest_gcs_step} from GCS.")
                        else:
                            print(f"Warning: Failed to download checkpoint: {dl_result.stderr}")
                    else:
                        print(f"Latest GCS checkpoint {latest_gcs_step} is already present locally.")
        except Exception as e:
            print(f"Warning: Failed to check or download GCS checkpoints: {e}")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        args.output_dir, orbax_checkpointer, options
    )

    start_step = 0
    if checkpoint_manager.latest_step() is not None:
        print(f"Restoring from checkpoint at step {checkpoint_manager.latest_step()}...")
        restored = checkpoint_manager.restore(checkpoint_manager.latest_step())
        # We restore params, but keep the new SFT optimizer state (which starts fresh)
        state = state.replace(params=restored['params'])
        # If we wanted to resume SFT exactly, we'd also load opt_state, but usually SFT starts fresh from PT params.
        start_step = restored.get('sft_step', 0)

    if num_devices > 1:
        state = flax.jax_utils.replicate(state)
        p_train_step = jax.pmap(p_sft_train_step_fn, axis_name='batch', donate_argnums=(0,))
    else:
        p_train_step = sft_train_step

    print("Starting SFT training...")
    step = start_step
    last_save_time = time.time()

    try:
        for batch in dataloader:
            # Removed `if step >= args.max_steps: break` for Infinite Continuous Learning

            try:
                if num_devices > 1:
                    batch = {k: v.reshape((num_devices, args.batch_size) + v.shape[1:]) for k, v in batch.items()}

                batch = jax.device_put(batch)
                state, loss = p_train_step(state, batch)

                if num_devices > 1:
                    loss = jnp.mean(loss)
            except Exception as e:
                print(f"SFT Training step failed. Error: {e}")
                continue

            global_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state

            # Clear stale XLA arrays
            gc.collect()

            if step % 10 == 0:
                current_lr = lr_schedule(step)
                print(f"SFT Step {step} | Loss: {loss:.4f} | LR: {current_lr:.2e}")
                writer.add_scalar("SFT/Loss", loss, step)
                writer.add_scalar("SFT/LearningRate", current_lr, step)

            time_to_save = (time.time() - last_save_time) >= args.save_interval_secs
            step_to_save = (step % args.save_every == 0 and step > start_step)

            if step_to_save or time_to_save:
                save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
                checkpoint_manager.save(step, {'step': save_state.step, 'sft_step': step, 'params': save_state.params, 'opt_state': save_state.opt_state})
                print(f"Saved SFT checkpoint at step {step}")
                
                # Export weights as safetensors inside the step directory
                safetensors_path = os.path.join(args.output_dir, str(step), "model.safetensors")
                export_to_safetensors(save_state.params, safetensors_path)
                
                last_save_time = time.time()

                if args.model_bucket:
                    print(f"Syncing SFT checkpoint {step} (including safetensors) to GCS bucket gs://{args.model_bucket} ...")
                    import threading
                    def upload_task(step_to_upload):
                        try:
                            cmd = ["gsutil", "-m", "cp", "-r", os.path.join(args.output_dir, str(step_to_upload)), f"gs://{args.model_bucket}/"]
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0:
                                print(f"SUCCESS: Uploaded checkpoint {step_to_upload} to GCS.")
                            else:
                                print(f"Warning: Failed to upload checkpoint to GCS: {res.stderr}")
                        except Exception as upload_err:
                            print(f"Error during GCS checkpoint upload: {upload_err}")

                    threading.Thread(target=upload_task, args=(step,), daemon=True).start()

            step += 1

    except KeyboardInterrupt:
        print("SFT Training interrupted manually. Saving final checkpoint...")
        save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
        checkpoint_manager.save(step, {'step': save_state.step, 'sft_step': step, 'params': save_state.params, 'opt_state': save_state.opt_state})
        
        # Export final weights to safetensors
        safetensors_path = os.path.join(args.output_dir, str(step), "model.safetensors")
        export_to_safetensors(save_state.params, safetensors_path)
        
        if args.model_bucket:
            print(f"Syncing final SFT checkpoint {step} (including safetensors) to GCS bucket gs://{args.model_bucket} ...")
            subprocess.run(["gsutil", "-m", "cp", "-r", os.path.join(args.output_dir, str(step)), f"gs://{args.model_bucket}/"])

    writer.close()
    print("SFT finished.")

if __name__ == "__main__":
    main()
