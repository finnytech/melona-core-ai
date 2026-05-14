import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple
from config import LlamaConfig

class LlamaRMSNorm(nn.Module):
    config: LlamaConfig

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.config.hidden_size,), self.config.dtype)
        # RMSNorm should be computed in fp32 for stability
        x_fp32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        x_normed = x_fp32 * jax.lax.rsqrt(variance + self.config.rms_norm_eps)
        x_normed = x_normed.astype(x.dtype)
        return weight * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = xq[..., 0::2] + 1j * xq[..., 1::2]
    xk_ = xk[..., 0::2] + 1j * xk[..., 1::2]

    # Broadcast freqs_cis
    # freqs_cis is (seq_len, head_dim/2)
    # xq_, xk_ are (batch, seq_len, num_heads, head_dim/2)
    freqs_cis = jnp.expand_dims(freqs_cis, axis=(0, 2))

    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    xq_out = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

class LlamaAttention(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.k_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.v_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.o_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)

    def __call__(self, hidden_states, freqs_cis, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # GQA: repeat K, V
        if self.num_key_value_groups > 1:
            k = jnp.repeat(k, self.num_key_value_groups, axis=2)
            v = jnp.repeat(v, self.num_key_value_groups, axis=2)

        q = jnp.transpose(q, (0, 2, 1, 3)) # (batch, num_heads, seq_len, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if attention_mask is not None:
            # attention_mask is typically (batch, 1, seq_len, seq_len)
            scores = scores + attention_mask.astype(scores.dtype)

        # Softmax in fp32 for numerical stability
        attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
        output = jnp.matmul(attn_weights, v)

        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_length, -1)

        return self.o_proj(output)

class LlamaMLP(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.up_proj = nn.Dense(self.config.intermediate_size, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.self_attn = LlamaAttention(self.config)
        self.mlp = LlamaMLP(self.config)
        self.input_layernorm = LlamaRMSNorm(self.config)
        self.post_attention_layernorm = LlamaRMSNorm(self.config)

    def __call__(self, hidden_states, freqs_cis, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size, dtype=self.config.dtype, param_dtype=self.config.dtype)
        self.layers = [LlamaDecoderLayer(self.config, name=f"layers.{i}") for i in range(self.config.num_hidden_layers)]
        self.norm = LlamaRMSNorm(self.config)

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        freqs_cis = precompute_freqs_cis(self.config.hidden_size // self.config.num_attention_heads, seq_length, self.config.rope_theta)

        if attention_mask is None:
            # Causal mask
            mask = jnp.tril(jnp.ones((seq_length, seq_length)))
            mask = jnp.where(mask == 0, -1e9, 0.0)
            attention_mask = mask.reshape(1, 1, seq_length, seq_length)

        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.model = LlamaModel(self.config)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.config.dtype, param_dtype=self.config.dtype)

    def __call__(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
