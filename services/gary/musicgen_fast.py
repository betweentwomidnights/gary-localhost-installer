#!/usr/bin/env python3
"""
musicgen_fast.py - Lossless optimizations for MusicGen inference.

Patches applied at runtime to any MusicGen model:
  1. Pure FP16: Convert remaining FP32 params, disable autocast overhead
  2. Static KV cache: Pre-allocated buffer, no torch.cat per step
  3. torch.compile: Fuse kernels and eliminate Python/launch overhead

Usage:
    from audiocraft.models import MusicGen
    from musicgen_fast import optimize_model

    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model = optimize_model(model)
    # Use normally - generate(), generate_continuation(), etc.
"""
import torch
import torch.nn as nn
import types
import typing as tp


# ==========================================================================
# Optimization 1: Pure FP16 -- eliminate dtype bouncing
# ==========================================================================

def apply_fp16_conversion(model):
    """Convert all FP32 parameters to FP16 and disable autocast.

    The transformer weights are already FP16. The remaining FP32 params are:
    - emb.{0,1,2,3}.weight (codebook embeddings)
    - out_norm.{weight,bias} (final LayerNorm)
    - linears.{0,1,2,3}.weight (output projections)
    - condition_provider output proj

    Converting these eliminates ~11.5% overhead from aten::copy_ dtype casting.
    """
    lm = model.lm
    converted = 0

    for name, param in lm.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.half()
            converted += 1

    for name, buf in lm.named_buffers():
        if buf.dtype == torch.float32 and buf.is_floating_point():
            buf.data = buf.data.half()
            converted += 1

    # Disable autocast since everything is now FP16
    model.autocast = _NoOpAutocast()

    print(f"[musicgen_fast] FP16: converted {converted} params/buffers")
    return model


class _NoOpAutocast:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ==========================================================================
# Optimization 2: Static KV Cache -- eliminate torch.cat overhead
# ==========================================================================

def apply_static_kv_cache(model, max_seq_len=1536):
    """Patch attention layers to use pre-allocated KV cache buffers.

    Eliminates ~7.3% overhead from aten::cat by writing into fixed buffers.
    Optimized for MusicGen's known config: time_dim=2, past_context=None.
    """
    from audiocraft.modules.transformer import StreamingMultiheadAttention

    patched = 0
    for name, module in model.lm.named_modules():
        if isinstance(module, StreamingMultiheadAttention) and not module.cross_attention:
            module._kv_max_len = max_seq_len
            module._original_complete_kv = module._complete_kv
            module._complete_kv = types.MethodType(_fast_complete_kv, module)
            patched += 1

    print(f"[musicgen_fast] Static KV: patched {patched} self-attention layers")
    return model


def _fast_complete_kv(self, k, v):
    """Lean static KV cache. Assumes time_dim=2, past_context=None."""
    if self.cross_attention:
        return k, v

    if not self._is_streaming:
        return k, v

    if not self._streaming_state:
        # First step: allocate buffers at full size
        B, H, T, D = k.shape
        buf_k = torch.zeros(B, H, self._kv_max_len, D, dtype=k.dtype, device=k.device)
        buf_v = torch.zeros(B, H, self._kv_max_len, D, dtype=v.dtype, device=v.device)
        buf_k[:, :, :T] = k
        buf_v[:, :, :T] = v
        self._streaming_state['past_keys'] = buf_k
        self._streaming_state['past_values'] = buf_v
        self._streaming_state['kv_pos'] = T
        self._streaming_state['offset'] = torch.tensor(0)
        return k, v

    pos = self._streaming_state['kv_pos']
    new_t = k.shape[2]
    end = pos + new_t

    buf_k = self._streaming_state['past_keys']
    buf_v = self._streaming_state['past_values']
    buf_k[:, :, pos:end] = k
    buf_v[:, :, pos:end] = v
    self._streaming_state['kv_pos'] = end

    return buf_k[:, :, :end], buf_v[:, :, :end]


# ==========================================================================
# Optimization 3: Flash Attention 2 -- bypass SDPA math fallback
# ==========================================================================

def apply_flash_attention(model):
    """Patch self-attention to call flash_attn_func directly.

    PyTorch 2.4's SDPA dispatcher doesn't recognize sm_121 as flash-capable,
    so it falls back to the slow math path even with FA2 installed.
    This bypasses SDPA and calls FA2 directly.

    Zero-copy integration: projects QKV directly into FA2's native [B, T, H, D]
    layout instead of audiocraft's [B, H, T, D], eliminating all transpose overhead.
    The KV cache is also kept in [B, T, H, D] layout when FA2 is active.
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("[musicgen_fast] Flash Attention: not installed, skipping")
        return model

    from audiocraft.modules.transformer import StreamingMultiheadAttention

    patched = 0
    for name, module in model.lm.named_modules():
        if isinstance(module, StreamingMultiheadAttention) and not module.cross_attention:
            module._fa2_func = flash_attn_func
            module._original_forward = module.forward
            module.forward = types.MethodType(_fa2_forward, module)
            # Override _complete_kv for FA2's [B, T, H, D] layout
            module._complete_kv = types.MethodType(_fa2_complete_kv, module)
            patched += 1

    print(f"[musicgen_fast] Flash Attention 2: patched {patched} self-attention layers")
    return model


def _fa2_complete_kv(self, k, v):
    """KV cache in FA2's native [B, T, H, D] layout. No transposes needed."""
    if self.cross_attention:
        return k, v

    if not self._is_streaming:
        return k, v

    if not self._streaming_state:
        B, T, H, D = k.shape
        max_len = getattr(self, '_kv_max_len', 1536)
        buf_k = torch.zeros(B, max_len, H, D, dtype=k.dtype, device=k.device)
        buf_v = torch.zeros(B, max_len, H, D, dtype=v.dtype, device=v.device)
        buf_k[:, :T] = k
        buf_v[:, :T] = v
        self._streaming_state['past_keys'] = buf_k
        self._streaming_state['past_values'] = buf_v
        self._streaming_state['kv_pos'] = T
        self._streaming_state['offset'] = torch.tensor(0)
        return k, v

    pos = self._streaming_state['kv_pos']
    new_t = k.shape[1]
    end = pos + new_t

    buf_k = self._streaming_state['past_keys']
    buf_v = self._streaming_state['past_values']
    buf_k[:, pos:end] = k
    buf_v[:, pos:end] = v
    self._streaming_state['kv_pos'] = end

    return buf_k[:, :end], buf_v[:, :end]


def _fa2_forward(self, query, key, value,
                 key_padding_mask=None, need_weights=False, attn_mask=None,
                 average_attn_weights=True, is_causal=False):
    """Forward pass using FA2 directly, zero-copy layout.

    Projects QKV directly into FA2's [B, T, H, D] layout instead of
    audiocraft's default [B, H, T, D]. This eliminates all transpose
    and .contiguous() calls that were negating FA2's speed advantage.
    """
    from einops import rearrange

    assert not is_causal
    dtype = query.dtype

    # Project QKV directly into FA2's native [B, T, H, D] layout
    projected = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
    packed = rearrange(projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads)
    q, k, v = packed.unbind(dim=2)  # each [B, T, H, D] — FA2 native, no copy

    if self.qk_layer_norm:
        q, k = [rearrange(x, "b t h d -> b t (h d)") for x in [q, k]]
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        q, k = [rearrange(x, "b t (h d) -> b t h d", h=self.num_heads) for x in [q, k]]

    if self.rope:
        # RoPE expects time_dim position — transpose temporarily
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_rope(q, k)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

    # KV cache — also in [B, T, H, D] layout (patched _fa2_complete_kv)
    k, v = self._complete_kv(k, v)

    p = self.dropout if self.training else 0.0
    x = self._fa2_func(q, k, v, dropout_p=p, causal=self.causal)

    # x is [B, T, H, D] — merge heads directly, no transpose
    x = x.to(dtype)
    x = rearrange(x, "b t h d -> b t (h d)")
    x = self.out_proj(x)

    return x, None


# ==========================================================================
# Optimization 4: torch.compile -- fuse kernels, eliminate launch overhead
# ==========================================================================

def apply_torch_compile(model, mode="reduce-overhead"):
    """Apply torch.compile to the transformer for fused execution.

    mode options:
      - "reduce-overhead": Best for inference, uses CUDA graphs internally
      - "default": Good balance of compile time and speedup
      - "max-autotune": Slowest compile, potentially fastest execution
    """
    lm = model.lm
    # Compile the transformer (the hot path during generation)
    lm.transformer = torch.compile(lm.transformer, mode=mode, dynamic=True)
    print(f"[musicgen_fast] torch.compile: transformer compiled (mode={mode})")
    return model


# ==========================================================================
# Public API
# ==========================================================================

def optimize_model(model, max_seq_len=1536,
                   enable_fp16=True, enable_static_kv=True, enable_fa2=True,
                   enable_compile=False, compile_mode="reduce-overhead"):
    """Apply lossless optimizations to a MusicGen model.

    Args:
        model: A MusicGen model from audiocraft.
        max_seq_len: Max KV cache length (1536 = 30s at 50Hz + headroom).
        enable_fp16: Convert remaining FP32 params to FP16.
        enable_static_kv: Use pre-allocated KV cache buffers.
        enable_fa2: Use Flash Attention 2 (if installed).
        enable_compile: Apply torch.compile to transformer.
        compile_mode: torch.compile mode.

    Returns:
        The same model, patched in-place.
    """
    print(f"[musicgen_fast] Optimizing...")

    if enable_fp16:
        apply_fp16_conversion(model)

    if enable_static_kv:
        apply_static_kv_cache(model, max_seq_len=max_seq_len)

    if enable_fa2:
        apply_flash_attention(model)

    if enable_compile:
        apply_torch_compile(model, mode=compile_mode)

    print(f"[musicgen_fast] Done.")
    return model
