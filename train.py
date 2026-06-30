import argparse
import os
import time
import jax
import subprocess
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import orbax.checkpoint
import psutil
import gc
from torch.utils.tensorboard import SummaryWriter

from config import LlamaConfig
from model import LlamaForCausalLM
from dataloader import get_dataloader

class TrainState(train_state.TrainState):
    pass

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

def create_learning_rate_schedule(total_steps, warmup_steps, peak_lr):
    def schedule(step):
        warmup_lr = peak_lr * (step / warmup_steps)
        decay_steps = total_steps - warmup_steps
        cosine_decay_lr = peak_lr * 0.5 * (1 + jnp.cos(jnp.pi * (step - warmup_steps) / decay_steps))
        return jnp.where(step < warmup_steps, warmup_lr, cosine_decay_lr)
    return schedule

def create_train_state(rng, config, learning_rate_fn):
    model = LlamaForCausalLM(config)
    dummy_input = jnp.ones((1, config.max_position_embeddings), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables['params']

    tx = optax.adamw(learning_rate_fn, b1=0.9, b2=0.95, weight_decay=0.1)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def loss_fn(params, apply_fn, batch):
    input_ids = batch['input_ids']
    # shift inputs and labels for causal LM
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits, aux_loss = apply_fn({'params': params}, inputs)

    # Cast logits to fp32 to prevent NaNs in cross entropy
    logits = logits.astype(jnp.float32)

    # Cross entropy loss
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    ce_loss = optax.softmax_cross_entropy(logits=logits, labels=targets_one_hot)

    # Mask padding if necessary. Assuming we padded with 0.
    mask = (targets != 0)
    ce_loss = (ce_loss * mask).sum() / mask.sum()

    # Combine Cross Entropy Loss with Auxiliary Load Balancing Loss for MoE routing
    loss = ce_loss + 0.01 * aux_loss

    return loss

@jax.jit(donate_argnums=(0,))
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def p_train_step_fn(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss

# Global state for hot-reloading in the UI
global_state = None

def get_gpu_vram():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vrams = [int(x) for x in result.stdout.strip().split("\n") if x.strip().isdigit()]
        if vrams:
            return min(vrams)
    except Exception:
        pass
    return None

def main(args_list=None):
    global global_state

    # Optimize CPU core utilization for data loading, BLAS, and tokenizer preprocessing
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or 4
        os.environ["OMP_NUM_THREADS"] = str(physical_cores)
        os.environ["MKL_NUM_THREADS"] = str(physical_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(physical_cores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(physical_cores)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        print(f"Optimized CPU thread pools: Using {physical_cores} physical cores for preprocessing and dataloading.")
    except Exception as cpu_err:
        print(f"Warning: Could not optimize CPU threads: {cpu_err}")

    # Pre-allocate 85% of VRAM for training to leave room for inference
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final', help='Path to data directory containing .jsonl shards')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final/checkpoints', help='Path to save checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--max_steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--peak_lr', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--save_every', type=int, default=2000, help='Save checkpoint every N steps')
    parser.add_argument('--model_bucket', type=str, default=None, help='GCS bucket path for syncing checkpoints (e.g. finny-tech-ai-storage/finny-tech-ai-models)')
    parser.add_argument('--save_interval_secs', type=int, default=1200, help='Time interval in seconds between checkpoint saves (default: 1200s / 20 mins)')
    args = parser.parse_args(args_list)

    # Dynamic Batch Sizing based on available GPU VRAM
    try:
        gpu_vram_mb = get_gpu_vram()
        if gpu_vram_mb:
            print(f"Detected GPU with {gpu_vram_mb} MB VRAM.")
            # Scale batch size based on VRAM (for our ~520M model)
            if gpu_vram_mb >= 20000: # e.g. L4/A10G (24GB) or A100 (40GB/80GB)
                args.batch_size = max(args.batch_size, 32)
            elif gpu_vram_mb >= 14000: # e.g. T4 (16GB)
                args.batch_size = max(args.batch_size, 16)
            else: # Small GPUs
                args.batch_size = max(args.batch_size, 8)
            print(f"Dynamic batch size scaled to {args.batch_size} per device based on GPU VRAM.")
        else:
            # Fallback to system RAM scaling if no GPU is found
            mem = psutil.virtual_memory()
            if mem.available > 30 * 1024**3:
                args.batch_size = max(args.batch_size, 16)
            elif mem.available > 15 * 1024**3:
                args.batch_size = max(args.batch_size, 8)
            else:
                args.batch_size = max(args.batch_size, 4)
            print(f"No GPU detected by nvidia-smi. Dynamic batch size set to {args.batch_size} based on system RAM.")
    except Exception as e:
        print(f"Warning: Could not dynamically scale batch size: {e}")

    # Download training data from GCS if it is a gs:// path
    if args.data_dir.startswith("gs://"):
        try:
            print(f"Downloading pre-training dataset shards from GCS: {args.data_dir} ...")
            local_data_dir = "./local_pretrain_data"
            os.makedirs(local_data_dir, exist_ok=True)
            subprocess.run(["gsutil", "-m", "cp", args.data_dir, local_data_dir], check=True)
            args.data_dir = local_data_dir
            print(f"Successfully downloaded training data to {local_data_dir}")
        except Exception as e:
            print(f"Error downloading training data from GCS: {e}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Load Config
    config = LlamaConfig()

    # Adjust batch size for multiple devices if available (though targeted for single TPU)
    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices

    print(f"Running on {num_devices} devices: {jax.devices()}")
    print(f"Global batch size: {global_batch_size}")

    # Setup Dataloader
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        batch_size=global_batch_size,
        seq_length=config.max_position_embeddings
    )

    # Initialize model state
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    lr_schedule = create_learning_rate_schedule(args.max_steps, args.warmup_steps, args.peak_lr)
    state = create_train_state(init_rng, config, lr_schedule)

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

    # Setup Orbax CheckpointManager
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        args.output_dir, orbax_checkpointer, options
    )

    # Restore checkpoint if it exists
    start_step = 0
    if checkpoint_manager.latest_step() is not None:
        restored = checkpoint_manager.restore(checkpoint_manager.latest_step())
        # We assume restored has step, params, opt_state (we only save state as dict)
        state = state.replace(step=restored['step'], params=restored['params'], opt_state=restored['opt_state'])
        start_step = state.step

    # If using multiple devices, replicate the state (data parallelism)
    # Since prompt requested jax.pmap or shard_map, we'll use flax.jax_utils.replicate for pmap
    if num_devices > 1:
        state = flax.jax_utils.replicate(state)
        p_train_step = jax.pmap(p_train_step_fn, axis_name='batch', donate_argnums=(0,))
    else:
        p_train_step = train_step

    print(f"Starting training from step {start_step}...")

    step = start_step
    start_time = time.time()
    last_save_time = time.time()
    tokens_processed = 0

    try:
        for batch in dataloader:
            # Removed `if step >= args.max_steps: break` for Infinite Continuous Learning

            # If multi-device, reshape batch to (num_devices, batch_size_per_device, ...)
            try:
                if num_devices > 1:
                    batch = {k: v.reshape((num_devices, args.batch_size) + v.shape[1:]) for k, v in batch.items()}

                # Use jax.device_put to force TPU execution and catch VRAM issues
                batch = jax.device_put(batch)
                state, loss = p_train_step(state, batch)

                # If multi-device, average loss across devices
                if num_devices > 1:
                    loss = jnp.mean(loss)
            except Exception as e:
                print(f"Training step failed, possibly due to VRAM exhaustion. Skipping batch. Error: {e}")
                continue

            # Update global state for hot-reloading
            global_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state

            # Clear stale XLA arrays
            gc.collect()

            # Logging
            tokens_in_batch = global_batch_size * config.max_position_embeddings
            tokens_processed += tokens_in_batch

            if step % 10 == 0:
                elapsed = time.time() - start_time
                tps = tokens_processed / elapsed

                # Fetch learning rate for logging. If replicated, take from first device.
                current_lr = lr_schedule(step)

                print(f"Step {step} | Loss: {loss:.4f} | LR: {current_lr:.2e} | Tokens/s: {tps:.2f}")

                writer.add_scalar("Train/Loss", loss, step)
                writer.add_scalar("Train/LearningRate", current_lr, step)
                writer.add_scalar("Train/Throughput", tps, step)

                # Reset counters for next log interval
                start_time = time.time()
                tokens_processed = 0

            # Save Checkpoint based on step or time
            time_to_save = (time.time() - last_save_time) >= args.save_interval_secs
            step_to_save = (step % args.save_every == 0 and step > start_step)

            if step_to_save or time_to_save:
                # If multi-device, unreplicate state before saving
                save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
                checkpoint_manager.save(step, {'step': save_state.step, 'params': save_state.params, 'opt_state': save_state.opt_state})
                print(f"Saved checkpoint at step {step}")
                
                # Export weights as safetensors inside the step directory
                safetensors_path = os.path.join(args.output_dir, str(step), "model.safetensors")
                export_to_safetensors(save_state.params, safetensors_path)
                
                last_save_time = time.time()

                if args.model_bucket:
                    print(f"Syncing checkpoint {step} (including safetensors) to GCS bucket gs://{args.model_bucket} ...")
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
        print("Training interrupted manually. Saving checkpoint...")
        save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
        checkpoint_manager.save(step, {'step': save_state.step, 'params': save_state.params, 'opt_state': save_state.opt_state})
        
        # Export final weights to safetensors
        safetensors_path = os.path.join(args.output_dir, str(step), "model.safetensors")
        export_to_safetensors(save_state.params, safetensors_path)
        
        if args.model_bucket:
            print(f"Syncing final checkpoint {step} (including safetensors) to GCS bucket gs://{args.model_bucket} ...")
            subprocess.run(["gsutil", "-m", "cp", "-r", os.path.join(args.output_dir, str(step)), f"gs://{args.model_bucket}/"])

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
