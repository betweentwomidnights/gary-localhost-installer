"""VRAM optimizations shared by the SA3 LoRA trainers.

Both helpers operate on a DiffusionCond model (the thing with `.model`,
`.conditioner`, `.pretransform`) after LoRA has been applied. They target the two
big non-DiT VRAM consumers that pre-encoded training doesn't actually need hot:
the VAE/pretransform (never called when training from pre-encoded latents) and the
T5/Gemma text encoder (only used to turn a small fixed set of deterministic
captions into embeddings). Freeing both gives ~2 GB of headroom, the difference
between fitting and thrashing on an 8 GB card.
"""
from __future__ import annotations

import torch


def free_pretransform(model) -> bool:
    """Drop the VAE/pretransform weights. Pre-encoded training never calls
    pretransform.encode, so the weights are dead VRAM for the whole run. Returns
    True if something was freed."""
    if getattr(model, "pretransform", None) is None:
        return False
    model.pretransform = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[memory] freed VAE/pretransform from VRAM (unused in pre-encoded training)", flush=True)
    return True


def offload_text_conditioner(model, dataloader, device) -> int:
    """Cache the text-encoder (T5/Gemma) outputs, then move its weights to CPU.

    Text conditioning is deterministic per clip (tag_keys=["prompt"], shuffle off),
    so one warm pass over the dataset encodes every caption once (encoder still on
    GPU, fp16); afterwards the cached embeddings are reused and the ~GB of encoder
    weights leave VRAM. Tiny NumberEmbedder conditioners stay live so per-crop
    seconds_total still varies. Correctness is never traded for memory: on a cache
    miss after offload (shouldn't happen with deterministic prompts) the encoder is
    moved back to GPU. Returns the param count offloaded, or 0."""
    conditioner = getattr(model, "conditioner", None)
    if conditioner is None or not hasattr(conditioner, "conditioners"):
        return 0

    target = None
    for name, cond in conditioner.conditioners.items():
        inner = getattr(cond, "model", None)
        if inner is None:
            continue
        n_params = sum(p.numel() for p in inner.parameters())
        if n_params >= 5_000_000:  # the T5/Gemma encoder; skip tiny embedders
            target = (name, cond, n_params)
            break
    if target is None:
        return 0
    name, cond, n_params = target

    state = {"offloaded": False}
    cache: dict = {}
    orig_forward = cond.forward

    def _key(inputs):
        return tuple(
            ("tok", tuple(x["input_ids"].reshape(-1).tolist())) if isinstance(x, dict)
            else ("str", str(x))
            for x in inputs
        )

    def cached_forward(inputs, dev):
        key = _key(inputs)
        hit = cache.get(key)
        if hit is not None:
            emb, mask = hit
            return emb.to(dev), mask.to(dev)
        if state["offloaded"]:
            if getattr(cond, "model", None) is not None:
                cond.model.to(dev)
            if hasattr(cond, "proj_out"):
                cond.proj_out.to(dev)
            state["offloaded"] = False
            print(f"[memory] text-encoder cache miss; moved '{name}' back to GPU", flush=True)
        emb, mask = orig_forward(inputs, dev)
        cache[key] = (emb.detach().to("cpu"), mask.detach().to("cpu"))
        return emb, mask

    cond.forward = cached_forward

    try:
        with torch.no_grad():
            for batch in dataloader:
                _, metadata = batch
                conditioner(list(metadata), device)
    except Exception as exc:  # never let the optimization break the run
        cond.forward = orig_forward
        print(f"[memory] text-encoder warm pass failed ({exc}); leaving it on GPU", flush=True)
        return 0

    if getattr(cond, "model", None) is not None:
        cond.model.to("cpu")
    if hasattr(cond, "proj_out"):
        cond.proj_out.to("cpu")
    cond._device_initialized = True  # keep forward() from moving it back on its own
    state["offloaded"] = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[memory] cached {len(cache)} caption embedding(s); offloaded text encoder "
          f"'{name}' ({n_params / 1e6:.0f}M params) to CPU", flush=True)
    return n_params
