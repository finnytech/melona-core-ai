from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp

@dataclass
class LlamaConfig:
    # Maximize the hidden dimensions for 47GB VRAM (TPU v5e) without OOM
    vocab_size: int = 50000 # Typical German BPE size
    dtype: Any = jnp.bfloat16 # Explicitly set bfloat16 for VRAM efficiency
    hidden_size: int = 3072
    intermediate_size: int = 8192  # usually multiple of 256
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    tie_word_embeddings: bool = False
