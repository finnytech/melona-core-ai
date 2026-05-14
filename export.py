import argparse
import os
import jax
import flax
import orbax.checkpoint
from safetensors.flax import save_file
import numpy as np

from config import LlamaConfig
from train import create_train_state, create_learning_rate_schedule

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, flax.core.FrozenDict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final/checkpoints')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/Omega_20M_Final/safetensors')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = LlamaConfig()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=False)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        args.checkpoint_dir, orbax_checkpointer, options
    )

    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        print("Warning: No checkpoint found. Ensure the directory contains orbax checkpoints.")
        return

    print(f"Restoring checkpoint step {latest_step} from {args.checkpoint_dir}...")
    restored = checkpoint_manager.restore(latest_step)

    print(f"Model step: {restored['step']}")

    # Extract params and convert to flat dict for safetensors
    params = restored['params']
    # Safetensors requires flax arrays or numpy arrays, flattened with '.'
    flat_params = flatten_dict(flax.core.unfreeze(params))

    # Save using safetensors
    output_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"Saving to {output_path}...")
    save_file(flat_params, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
