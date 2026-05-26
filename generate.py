import argparse
import os
import jax
import jax.numpy as jnp
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import orbax.checkpoint

from config import LlamaConfig
from model import LlamaForCausalLM
from train import create_train_state, create_learning_rate_schedule

def generate(model_apply_fn, params, input_ids, config, max_new_tokens=50, temperature=1.0, top_k=50):
    """
    Simple autoregressive generation loop.
    Note: For production inference, KV caching should be explicitly implemented,
    but for testing the trained weights, a naive loop works.
    """
    seq_len = input_ids.shape[1]

    # Compile the forward pass to avoid XLA memory fragmentation
    @jax.jit
    def _forward(params_inner, input_ids_inner):
        logits, aux_loss = model_apply_fn({'params': params_inner}, input_ids_inner)
        return logits

    # We do a naive JAX generation loop using a Python while loop
    # In a real library, use jax.lax.while_loop with KV cache

    for _ in range(max_new_tokens):
        # Forward pass
        logits = _forward(params, input_ids)
        next_token_logits = logits[:, -1, :]

        # Temperature
        next_token_logits = next_token_logits / temperature

        # Top-K
        if top_k > 0:
            indices_to_remove = next_token_logits < jax.lax.top_k(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits = jnp.where(indices_to_remove, -float("Inf"), next_token_logits)

        # Sample
        probs = jax.nn.softmax(next_token_logits, axis=-1)
        next_token = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs))
        next_token = next_token.reshape(1, 1)

        input_ids = jnp.concatenate([input_ids, next_token], axis=-1)

        if next_token[0, 0] == config.eos_token_id:
            break

    return input_ids

def main(args_list=None):
    # Only use 5% of VRAM to avoid crashing the background training process
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final/checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    args = parser.parse_args(args_list)

    # Load tokenizer
    if os.path.exists(args.tokenizer_path):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load config and model
    config = LlamaConfig()

    rng = jax.random.PRNGKey(0)
    dummy_lr = create_learning_rate_schedule(1, 1, 1.0)
    state = create_train_state(rng, config, dummy_lr)

    # Restore weights (only if we're not hot-reloading from memory)
    # Check if a global_state was passed via a hack or check checkpoints
    # In this standalone script mode we check disk:
    if os.path.exists(args.checkpoint_dir):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=False)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            args.checkpoint_dir, orbax_checkpointer, options
        )
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            restored = checkpoint_manager.restore(latest_step)
            state = state.replace(step=restored['step'], params=restored['params'])
            print(f"Loaded checkpoint from step {state.step}")
        else:
            print("Warning: No checkpoint found in directory, using random weights.")
    else:
        print("Warning: Checkpoint directory does not exist, using random weights.")

    # Encode prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    input_ids = jnp.array(input_ids)

    print("Generating...")
    output_ids = generate(
        state.apply_fn,
        state.params,
        input_ids,
        config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    generated_text = tokenizer.decode(np.array(output_ids[0]), skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    main()
