"""State-dict helpers vendored from stable-audio-tools.

Pure PyTorch utilities — backend-agnostic. Used by both the SAT-dev and SA3
adapters, and by analysis scripts that need to load checkpoints directly.
"""
import torch
from safetensors.torch import load_file
from torch.nn.utils import remove_weight_norm


def copy_state_dict(model, state_dict):
    """Load state_dict into model for keys matching exactly in name and shape."""
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        else:
            print(f"Key {key} not found in target state_dict or shape mismatch. Skipping.")
    model.load_state_dict(model_state_dict, strict=False)


def load_ckpt_state_dict(ckpt_path):
    """Load a checkpoint's state_dict. Handles three input shapes:
      - path ending in .safetensors  → safetensors.torch.load_file
      - path ending in .pt/.ckpt/.bin → torch.load (weights_only)
      - extensionless path (HF cache blob, etc.) → peek at the file header
        to detect format.

    The HF-cache fallback exists because content-addressed blobs in
    ~/.cache/huggingface/hub/<repo>/blobs/<hash> have no extension; without
    sniffing, a safetensors blob would get fed to torch.load and explode.
    """
    path = str(ckpt_path)
    if path.endswith(".safetensors"):
        return load_file(path)
    if path.endswith((".pt", ".ckpt", ".bin")):
        return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    # Unknown extension — sniff first bytes. safetensors layout: 8-byte
    # little-endian header length, then JSON ('{...}'). torch.save uses
    # pickle (\x80) or zip (PK).
    with open(path, "rb") as f:
        head = f.read(16)
    if len(head) >= 9 and head[8:9] == b"{":
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def stream_checkpoint_into_model(model, ckpt_path, *, device, dtype=None,
                                  remap_keys=True):
    """Load a safetensors checkpoint tensor-by-tensor into `model`, copying
    each to `device` and dropping the CPU side immediately. Avoids holding
    the full state_dict in CPU RAM.

    `remap_keys=True` applies SA3's drop-one-part heuristic: when a source
    key doesn't match any model key, try dropping each path component to see
    if a shorter key matches (e.g. `pretransform.model.encoder.foo` ->
    `pretransform.encoder.foo`).

    Returns `(matched, skipped)`. Returns `None` for non-safetensors paths
    (caller should fall back to a bulk load).
    """
    from pathlib import Path as _Path
    if _Path(ckpt_path).suffix.lower() != ".safetensors":
        return None
    from safetensors import safe_open
    from accelerate.utils import set_module_tensor_to_device

    model_state_keys = set(model.state_dict().keys())
    matched = skipped = 0
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for src_key in f.keys():
            tgt_key = src_key
            if tgt_key not in model_state_keys:
                if not remap_keys:
                    skipped += 1
                    continue
                parts = src_key.split(".")
                tgt_key = None
                for i in range(1, len(parts)):
                    candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1:])
                    if candidate in model_state_keys:
                        tgt_key = candidate
                        break
                if tgt_key is None:
                    skipped += 1
                    continue

            tensor = f.get_tensor(src_key)
            cast = dtype if (dtype is not None and tensor.is_floating_point()) else None
            try:
                set_module_tensor_to_device(model, tgt_key, device,
                                            value=tensor, dtype=cast)
                matched += 1
            except Exception as e:
                print(f"  warn: couldn't set {tgt_key!r}: {e}", flush=True)
                skipped += 1
            del tensor
    return (matched, skipped)


def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)
    return model


WRAPPER_PREFIXES = {
    "diffusion_uncond": "diffusion.",
    "diffusion_cond": "diffusion.",
    "diffusion_cond_inpaint": "diffusion.",
    "diffusion_autoencoder": "diffusion.",
    "autoencoder": "autoencoder.",
    "lm": "lm.",
    "clap": "clap.",
    "captioner": "model.",
}


def unwrap_state_dict(state_dict, model_type):
    """Detect and strip Lightning training-wrapper prefixes from a state_dict.

    Wrapped checkpoints have keys like 'diffusion.model.xxx' or
    'diffusion_ema.ema_model.xxx'. Returns the unwrapped dict (or original if
    already unwrapped / unknown model_type).
    """
    prefix = WRAPPER_PREFIXES.get(model_type)
    if prefix is None:
        return state_dict

    has_wrapper_prefix = any(k.startswith(prefix) for k in state_dict.keys())
    if not has_wrapper_prefix:
        return state_dict

    ema_prefix = prefix.replace(".", "_ema.ema_model.")
    has_ema = any(k.startswith(ema_prefix) for k in state_dict.keys())
    ema_wraps_whole_model = model_type in ("autoencoder",)

    unwrapped = {}
    if has_ema:
        for k, v in state_dict.items():
            if k.startswith(ema_prefix):
                suffix = k[len(ema_prefix):]
                new_key = suffix if ema_wraps_whole_model else "model." + suffix
                unwrapped[new_key] = v
        if not ema_wraps_whole_model:
            conditioner_prefix = prefix + "conditioner."
            pretransform_prefix = prefix + "pretransform."
            for k, v in state_dict.items():
                if k.startswith(conditioner_prefix) or k.startswith(pretransform_prefix):
                    unwrapped[k[len(prefix):]] = v
    else:
        for k, v in state_dict.items():
            if k.startswith(prefix):
                unwrapped[k[len(prefix):]] = v
    return unwrapped
