"""Helpers for loading pre-trained LoRA checkpoints onto a live model."""
import os
from functools import partial

import torch

from .model import LoRAParametrization, add_lora
from .utils import (
    get_lora_layers,
    infer_global_rank,
    load_lora_checkpoint,
    prepare_dora_state_dict,
    remap_lora_state_dict,
    resolve_adapter_type,
)
from ...verbose import vprint


def _resolve_lora_targets(model, model_type, target):
    """Resolve the submodules adapted by a checkpoint's declared target."""
    if target in ("decoder", "encoder"):
        # Full serving model: model.pretransform.model.<half>. Standalone
        # autoencoder (training/eval): model.<half>.
        pretransform = getattr(model, "pretransform", None)
        autoencoder = (
            getattr(pretransform, "model", None)
            if pretransform is not None
            else None
        )
        half = getattr(autoencoder, target, None) if autoencoder is not None else None
        if half is None:
            half = getattr(model, target, None)
        if half is None:
            raise ValueError(
                f"LoRA target {target!r} needs either an autoencoder pretransform "
                f"(model.pretransform.model.{target}) or a standalone autoencoder "
                f"(model.{target}); neither was found."
            )
        return [half], [f"{target}."]

    if target != "dit":
        raise ValueError(
            f"unknown LoRA target {target!r}; expected 'dit', 'decoder' or "
            f"'encoder'"
        )

    if model_type in ("diffusion_cond", "diffusion_cond_inpaint"):
        return [model.model, model.conditioner], ["model.", "conditioner."]
    return [model], ["model."]


def load_and_apply_loras(model, lora_ckpt_paths, model_type, svd_bases_path=None):
    """Load LoRA checkpoints from disk and attach them to `model`.

    - Uses a two-pass approach: first resolves each LoRA's adapter type, then
      loads SVD bases (only if any LoRA is -XS), then applies each LoRA.
    - Handles legacy "dora" adapter_type via `resolve_adapter_type`.
    - Only passes svd_bases to -XS LoRAs to avoid spurious "no -XS layers"
      messages for non-XS LoRAs.
    - Sets `model.use_lora = True` and `model.lora_names = [...]` as a
      convenience for downstream UI code.

    Args:
        model: The loaded model. For `diffusion_cond` / `diffusion_cond_inpaint`
            types, LoRAs are applied to `model.model` and `model.conditioner`;
            otherwise they are applied to `model` directly.
        lora_ckpt_paths: List of paths to `.ckpt` or `.safetensors` files.
        model_type: String from the model config, e.g. "diffusion_cond".
        svd_bases_path: Optional path to a precomputed SVD bases .pt file
            (used only by -XS adapter types).

    Returns:
        List of display names (file stems) in the same order as the input paths.
    """
    if not lora_ckpt_paths:
        model.use_lora = False
        model.lora_names = []
        return []

    # Pass 1: load each checkpoint and resolve adapter type.
    lora_entries = []  # (path, state_dict, config_dict, adapter_type)
    for i, lora_path in enumerate(lora_ckpt_paths):
        print(f"Loading LoRA {i} from {lora_path}")
        state_dict, config_dict = load_lora_checkpoint(lora_path)
        adapter_type_raw = config_dict.get("adapter_type", "lora")
        adapter_type = resolve_adapter_type(adapter_type_raw, state_dict)
        if adapter_type != adapter_type_raw:
            vprint(f"Resolved legacy '{adapter_type_raw}' -> {adapter_type}")
        lora_entries.append((lora_path, state_dict, config_dict, adapter_type))

    # Load SVD bases only if at least one LoRA is -XS.
    svd_bases_all = None
    any_xs = any(t.endswith("-xs") for _, _, _, t in lora_entries)
    if any_xs:
        if svd_bases_path is not None:
            vprint(f"Loading SVD bases from {svd_bases_path}")
            svd_bases_all = torch.load(svd_bases_path, map_location="cpu", weights_only=True)
            vprint(f"  {len(svd_bases_all)} bases loaded")
        else:
            print("WARNING: -XS adapter type present without svd_bases_path -- SVD will be computed on current device")

    def _bases_for(prefix):
        if svd_bases_all is None:
            return None
        return {
            key[len(prefix):]: value
            for key, value in svd_bases_all.items()
            if key.startswith(prefix)
        }

    # Pass 2: apply each LoRA.
    lora_names = []
    # Parametrization slots are local to a target. A decoder adapter loaded
    # after two DiT adapters is still decoder slot 0, not global slot 2.
    attach_pos = {}
    for i, (lora_path, state_dict, config_dict, adapter_type) in enumerate(lora_entries):
        rank = config_dict.get("rank", infer_global_rank(state_dict))
        alpha = config_dict.get("alpha", rank)
        include = config_dict.get("include", None)
        exclude = config_dict.get("exclude", None)
        is_xs = adapter_type.endswith("-xs")

        lora_config = {
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha, adapter_type=adapter_type, lora_index=i),
            },
            torch.nn.Conv1d: {
                "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, adapter_type=adapter_type, lora_index=i),
            },
        }
        # Missing target preserves every existing checkpoint's DiT behaviour.
        target = config_dict.get("target", "dit")
        targets, bases_prefixes = _resolve_lora_targets(model, model_type, target)
        if target != "dit":
            vprint(f"  target: {target}")

        pos = attach_pos.get(id(targets[0]), 0)
        for module, prefix in zip(targets, bases_prefixes):
            add_lora(
                module,
                lora_config,
                include=include,
                exclude=exclude,
                svd_bases=_bases_for(prefix) if is_xs else None,
            )
        for module in targets:
            attach_pos[id(module)] = attach_pos.get(id(module), 0) + 1

        prepare_dora_state_dict(state_dict)
        remapped_sd = remap_lora_state_dict(state_dict, pos)
        loaded_any = False
        for module in targets:
            result = module.load_state_dict(remapped_sd, strict=False)
            unexpected = set(getattr(result, "unexpected_keys", []) or [])
            if any(key not in unexpected for key in remapped_sd):
                loaded_any = True
        if not loaded_any and remapped_sd:
            print(
                f"WARNING: LoRA {lora_path} matched NO parameters on its "
                f"target(s) at parametrization position {pos} -- the adapter "
                f"is attached but will have no effect. Checkpoint key layout "
                f"probably does not match the target module."
            )
        lora_names.append(os.path.splitext(os.path.basename(lora_path))[0])

    vprint("lora layers:", len(get_lora_layers(model)))
    model.use_lora = True
    model.lora_names = lora_names
    return lora_names
