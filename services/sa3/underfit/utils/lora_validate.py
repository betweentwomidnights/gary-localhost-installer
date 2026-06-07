"""Validate user-uploaded .safetensors LoRA checkpoints before they seed a run.

We only accept .safetensors (never .ckpt / .pt — those are torch.load, which is
a pickle and a remote-code-execution vector). The validator opens the file via
`safetensors.safe_open` (header-only by default), extracts the embedded
`lora_config` metadata, samples a few keys for sanity, and returns a structured
result the dashboard can use to populate (and freeze) the New Finetune form.

The validator is intentionally lightweight — it does NOT load the base model to
check shape compatibility. If the user's LoRA was made against a different
model size, training will error out at load_state_dict time. The "did this
.safetensors come from our training stack" check is structural:
  - has the expected `parametrizations.weight.<N>.<lora_kind>` key pattern
  - has a `lora_config` metadata blob
  - rank / alpha / adapter_type fields are present and parseable

Returns one of two dataclass-ish dicts so callers can render both success and
informative failure pop-ups.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_LORA_KEY_RE = re.compile(
    r"\.parametrizations\.weight\.\d+\."
    r"(lora_A|lora_B|magnitude|magnitude_r|magnitude_c|M_xs|U|V)$"
)

# Adapter-type inference from the saved-key set. The training stack writes
# different buffers per variant; the *intersection* with the saved keys is
# what tells us which class was used.
_ADAPTER_FINGERPRINTS = [
    # (adapter_type, must_have_keys, must_not_have_keys)
    ("lora-xs",    {"M_xs"},                          {"magnitude", "magnitude_r", "magnitude_c"}),
    ("bora",       {"magnitude_r", "magnitude_c"},    set()),
    ("dora-rows",  {"magnitude"},                     {"magnitude_r", "magnitude_c"}),
    # plain LoRA / DoRA-cols both have lora_A/lora_B; cols also has magnitude.
    # We default to "lora" if nothing else fits — the metadata blob is the
    # source of truth, and we always prefer that.
    ("lora",       {"lora_A", "lora_B"},              set()),
]


def _infer_adapter_type(keys: list[str]) -> str | None:
    suffixes = set()
    for k in keys:
        m = _LORA_KEY_RE.search(k)
        if m:
            suffixes.add(m.group(1))
    for name, must, must_not in _ADAPTER_FINGERPRINTS:
        if must.issubset(suffixes) and not (must_not & suffixes):
            return name
    return None


def validate_lora_safetensors(path: str | Path) -> dict[str, Any]:
    """Inspect a .safetensors LoRA checkpoint without loading weights.

    Always returns a dict with these fields:
      ok          : True if the file looks like a usable LoRA checkpoint.
      path        : Echo of the input path (so the caller has a stable ref).
      config      : The parsed `lora_config` JSON blob (or {} if absent).
      partial     : Best-effort facts we extracted even on failure
                    (key count, sample keys, adapter-type guess, ...).
      error       : Human-readable explanation of why validation failed,
                    or None on success.

    On success, `config` contains at least {rank, alpha, adapter_type, exclude,
    include?, step?, epoch?}.
    """
    result: dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "config": {},
        "partial": {},
        "error": None,
    }

    p = Path(path)
    if not p.exists():
        result["error"] = f"file not found: {p}"
        return result
    if p.suffix.lower() != ".safetensors":
        result["error"] = (
            f"only .safetensors is accepted (got '{p.suffix}'). "
            f".ckpt/.pt files use pickle and are a security risk — convert first."
        )
        return result

    try:
        from safetensors import safe_open
    except ImportError:
        result["error"] = "safetensors package not available in dashboard venv"
        return result

    try:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            metadata = f.metadata() or {}
            # Sample shape of one lora_A key to recover rank (last-resort fallback
            # when the metadata blob is missing rank).
            sample_shape = None
            for k in keys:
                if k.endswith("lora_A") or k.endswith("M_xs"):
                    try:
                        # Slice header only — don't materialize tensors.
                        sample_shape = list(f.get_slice(k).get_shape())
                    except Exception:
                        pass
                    break
            # Per-layer (name, fan_in, fan_out) fingerprint, used by the
            # base-model heuristic when metadata's base_model is missing.
            # lora_A shape = (rank, fan_in); lora_B shape = (fan_out, rank).
            layer_dims = {}  # base_name -> {"fan_in": int, "fan_out": int}
            for k in keys:
                m = _LORA_KEY_RE.search(k)
                if not m:
                    continue
                kind = m.group(1)
                base = k[: m.start()]
                try:
                    shape = list(f.get_slice(k).get_shape())
                except Exception:
                    continue
                d = layer_dims.setdefault(base, {})
                if kind == "lora_A" and len(shape) >= 2:
                    d["fan_in"] = int(shape[1])
                elif kind == "lora_B" and len(shape) >= 2:
                    d["fan_out"] = int(shape[0])
    except Exception as e:
        result["error"] = f"could not open as safetensors: {type(e).__name__}: {e}"
        return result

    result["partial"]["num_keys"] = len(keys)
    result["partial"]["sample_keys"] = keys[:8]
    result["partial"]["metadata_keys"] = list(metadata.keys())
    inferred_type = _infer_adapter_type(keys)
    if inferred_type:
        result["partial"]["inferred_adapter_type"] = inferred_type

    # No LoRA-shaped keys at all? Not a LoRA file.
    if not any(_LORA_KEY_RE.search(k) for k in keys):
        result["error"] = (
            f"no LoRA keys found in file ({len(keys)} keys total). "
            f"Expected names ending in '.parametrizations.weight.<N>.lora_A' etc. "
            f"This might be a base-model checkpoint or VAE, not a LoRA."
        )
        return result

    # Parse the embedded lora_config metadata.
    if "lora_config" not in metadata:
        result["error"] = (
            "missing 'lora_config' metadata. This LoRA was saved without the "
            "config block — we can't be sure of its rank/alpha/adapter_type."
        )
        if inferred_type:
            result["partial"]["note"] = (
                f"Adapter type appears to be '{inferred_type}' from key fingerprints, "
                f"but rank/alpha are not recoverable without the config."
            )
        return result

    try:
        config = json.loads(metadata["lora_config"])
    except json.JSONDecodeError as e:
        result["error"] = f"lora_config metadata is not valid JSON: {e}"
        return result
    if not isinstance(config, dict):
        result["error"] = f"lora_config metadata is not a dict (got {type(config).__name__})"
        return result

    # Required fields. Training won't run without these — surface clearly.
    for field in ("rank", "adapter_type"):
        if field not in config:
            result["error"] = f"lora_config is missing required field '{field}'"
            result["partial"]["config_so_far"] = config
            return result

    # Best-effort fill-ins for fields that older saves may omit.
    config.setdefault("alpha", config["rank"])
    config.setdefault("include", [])
    config.setdefault("exclude", [])

    # Per-layer (name, fan_in, fan_out) list — what the server-side base-
    # model heuristic compares against each registry's lora_layer_template.
    layers = []
    for base, dims in layer_dims.items():
        if "fan_in" in dims and "fan_out" in dims:
            layers.append({"name": base, "fan_in": dims["fan_in"], "fan_out": dims["fan_out"]})
    result["layers"] = layers

    # Cross-check: does the saved adapter_type match what the key fingerprints say?
    # If they disagree the metadata wins (it's authoritative), but we note the
    # discrepancy in the partial info so the user can spot a tampered file.
    if inferred_type and inferred_type != config["adapter_type"]:
        result["partial"]["adapter_type_mismatch_note"] = (
            f"keys look like {inferred_type!r} but config says {config['adapter_type']!r}"
        )

    # Fallback rank recovery from the lora_A slice shape, if the metadata's
    # rank looks off (e.g. zero). Shape is [rank, in_features] for lora_A.
    if sample_shape and isinstance(sample_shape, list) and len(sample_shape) >= 2:
        sample_rank = sample_shape[0]
        result["partial"]["sample_rank_from_shape"] = sample_rank
        if config.get("rank", 0) <= 0 and sample_rank > 0:
            config["rank"] = sample_rank

    result["ok"] = True
    result["config"] = config
    return result
