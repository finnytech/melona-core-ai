import argparse
import os
import time
import jax
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

    logits = apply_fn({'params': params}, inputs)

    # Cast logits to fp32 to prevent NaNs in cross entropy
    logits = logits.astype(jnp.float32)

    # Cross entropy loss
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    loss = optax.softmax_cross_entropy(logits=logits, labels=targets_one_hot)

    # Mask padding if necessary. Assuming we padded with 0.
    mask = (targets != 0)
    loss = (loss * mask).sum() / mask.sum()

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

def main(args_list=None):
    global global_state

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
    args = parser.parse_args(args_list)

    # Dynamic Batch Sizing based on available memory
    try:
        mem = psutil.virtual_memory()
        # Scale batch size roughly based on available RAM (this is a simplified heuristic)
        if mem.available > 30 * 1024**3: # >30GB RAM
            args.batch_size = max(args.batch_size, 16)
        elif mem.available > 15 * 1024**3: # >15GB RAM
            args.batch_size = max(args.batch_size, 8)
        else:
            args.batch_size = max(args.batch_size, 4)
    except Exception as e:
        print(f"Warning: Could not dynamically scale batch size: {e}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Load Config
    config = LlamaConfig()

    # Adjust batch size for multiple devices if available (though targeted for single TPU)
    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices

    print(f"Running on {num_devices} devices.")
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

            # Save Checkpoint
            if step % args.save_every == 0 and step > start_step:
                # If multi-device, unreplicate state before saving
                save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
                checkpoint_manager.save(step, {'step': save_state.step, 'params': save_state.params, 'opt_state': save_state.opt_state})
                print(f"Saved checkpoint at step {step}")

            step += 1

    except KeyboardInterrupt:
        print("Training interrupted manually. Saving checkpoint...")
        save_state = flax.jax_utils.unreplicate(state) if num_devices > 1 else state
        checkpoint_manager.save(step, {'step': save_state.step, 'params': save_state.params, 'opt_state': save_state.opt_state})

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
