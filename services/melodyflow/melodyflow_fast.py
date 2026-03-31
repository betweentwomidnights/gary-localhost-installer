#!/usr/bin/env python3
"""
melodyflow_fast.py - Optional Flash Attention 2 optimization for MelodyFlow.

This patches MelodyFlow's DiT self-attention blocks at runtime so they can call
flash_attn_func directly on CUDA when the flash-attn wheel is installed.

We intentionally leave cross-attention on the existing AudioCraft path because
those blocks can carry custom masks, while FA2's simple API is a better fit for
the large self-attention passes over audio latents.
"""

import types

import torch
import torch.nn as nn
from einops import rearrange


def optimize_model(model):
    """Patch MelodyFlow self-attention layers to use Flash Attention 2."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("[melodyflow_fast] Flash Attention 2: not installed, skipping")
        return model

    if not torch.cuda.is_available():
        print("[melodyflow_fast] Flash Attention 2: CUDA unavailable, skipping")
        return model

    try:
        from audiocraft.modules.transformer import StreamingMultiheadAttention
    except Exception as exc:
        print(f"[melodyflow_fast] Flash Attention 2: import failed ({exc}), skipping")
        return model

    patched = 0
    for module in model.lm.modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        if module.cross_attention:
            continue
        if getattr(module, "_melodyflow_fa2_patched", False):
            continue
        module._fa2_func = flash_attn_func
        module._original_forward = module.forward
        module._original_complete_kv = module._complete_kv
        module.forward = types.MethodType(_fa2_forward, module)
        module._complete_kv = types.MethodType(_fa2_complete_kv, module)
        module._melodyflow_fa2_patched = True
        patched += 1

    if patched == 0:
        print("[melodyflow_fast] Flash Attention 2: no self-attention layers patched")
    else:
        print(f"[melodyflow_fast] Flash Attention 2: patched {patched} self-attention layers")

    return model


def _can_use_fa2(self, q, k, v, attn_mask):
    if self.cross_attention:
        return False
    if attn_mask is not None:
        return False
    if q.device.type != "cuda":
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    head_dim = q.shape[-1]
    if head_dim > 256:
        return False
    return True


def _fa2_complete_kv(self, k, v):
    if self.cross_attention:
        return k, v

    if not self._is_streaming:
        return k, v

    if self._streaming_state:
        pk = self._streaming_state["past_keys"]
        nk = torch.cat([pk, k], dim=1)
        if v is k:
            nv = nk
        else:
            pv = self._streaming_state["past_values"]
            nv = torch.cat([pv, v], dim=1)
    else:
        nk = k
        nv = v

    assert nk.shape[1] == nv.shape[1]
    offset = 0
    if self.past_context is not None:
        offset = max(0, nk.shape[1] - self.past_context)
    if self._is_streaming:
        self._streaming_state["past_keys"] = nk[:, offset:]
        if v is not k:
            self._streaming_state["past_values"] = nv[:, offset:]
        if "offset" in self._streaming_state:
            self._streaming_state["offset"] += offset
        else:
            self._streaming_state["offset"] = torch.tensor(0, device=nk.device)
    return nk, nv


def _fa2_forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=False,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    """Self-attention forward pass using Flash Attention 2 when it is safe."""
    assert not is_causal

    if need_weights or key_padding_mask is not None:
        return self._original_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    if not _is_specialized_self_attention(query, key, value):
        return self._original_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    dtype = query.dtype
    projected = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
    packed = rearrange(projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads)
    q, k, v = packed.unbind(dim=2)

    if self.qk_layer_norm:
        q = rearrange(q, "b t h d -> b t (h d)")
        k = rearrange(k, "b t h d -> b t (h d)")
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)

    if self.rope:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_rope(q, k)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

    k, v = self._complete_kv(k, v)

    if self.add_zero_attn:
        zero_shape = (k.shape[0], 1, k.shape[2], k.shape[3])
        zero_k = torch.zeros(zero_shape, dtype=k.dtype, device=k.device)
        zero_v = torch.zeros(zero_shape, dtype=v.dtype, device=v.device)
        k = torch.cat([zero_k, k], dim=1)
        v = torch.cat([zero_v, v], dim=1)

    if not _can_use_fa2(self, q, k, v, attn_mask):
        return self._original_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    dropout_p = self.dropout if self.training else 0.0
    x = self._fa2_func(q, k, v, dropout_p=dropout_p, causal=self.causal)
    x = x.to(dtype)
    x = rearrange(x, "b t h d -> b t (h d)")
    x = self.out_proj(x)
    return x, None


def _is_specialized_self_attention(query, key, value):
    return query is key and value is key
