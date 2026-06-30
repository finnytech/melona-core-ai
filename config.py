from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp

@dataclass
class LlamaConfig:
    # Optimized hidden dimensions for 1x/2x NVIDIA L4 GPUs (24GB VRAM each) to run extremely fast without OOM
    vocab_size: int = 50000 # Typical German BPE size
    dtype: Any = jnp.bfloat16 # Explicitly set bfloat16 for VRAM efficiency
    hidden_size: int = 1024
    intermediate_size: int = 2048  # usually multiple of 256
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4 # GQA
    max_position_embeddings: int = 4096 # Perfect context length for SFT Q&A
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # MoE Config
    num_experts: int = 4
    num_experts_per_tok: int = 2
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    tie_word_embeddings: bool = False
