"""Refresh dashboard model registries from on-disk base safetensors.

Each `dashboard/models/<key>/registry.json` carries a `lora_layer_template`
that the dashboard uses to:
  1. estimate VRAM and parameter counts in the New Finetune UI;
  2. fingerprint uploaded seed-LoRAs against known base models.

Template entries (`fi`/`fo`) are easy to get wrong by hand — and one wrong
fan_out silently breaks the seed-LoRA heuristic, manifesting as
"Incompatible LoRA" popups. This tool re-derives them by opening the actual
model.safetensors and reading layer shapes.

USAGE
    underfit-update-registries                      # update all registries (in-place)
    underfit-update-registries --check              # dry-run; print would-be diffs
    underfit-update-registries sa3-sm-sfx           # only that model
    underfit-update-registries --models-dir /path   # override MODELS_DIR

Models whose `paths.base_ckpt` doesn't resolve to a file on disk are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    """Repo root — parent of this file's underfit/cli/ dir."""
    return Path(__file__).resolve().parent.parent.parent


def _registries_dir() -> Path:
    return _project_root() / "dashboard" / "models"


def _resolve_base_ckpt(reg_path: str, models_dir: Path) -> Path:
    """Resolve a registry's paths.base_ckpt placeholder ({models_dir}) to a
    real filesystem path."""
    return Path(str(reg_path).replace("{models_dir}", str(models_dir)))


def _check_entry(safe_open_f, base_key: str, entry: dict) -> tuple[int, int] | None:
    """Look up `base_key` in the safetensors file; return (new_fi, new_fo) when
    the entry's stored dims disagree, else None. Skips entries whose key is
    missing or whose tensor isn't 2D (e.g. Conv1d that doesn't fit the
    Linear-shaped LoRA template)."""
    try:
        shape = list(safe_open_f.get_slice(base_key).get_shape())
    except Exception:
        return None
    if len(shape) != 2:
        return None
    actual_fo, actual_fi = shape[0], shape[1]
    if actual_fi == entry.get("fi") and actual_fo == entry.get("fo"):
        return None
    return actual_fi, actual_fo


def update_registry(reg_path: Path, models_dir: Path, dry_run: bool = False) -> int:
    """Walk the lora_layer_template entries, patch any (fi, fo) mismatches
    from the actual base safetensors. Returns the number of changes (or
    would-be changes when dry_run=True). Skips silently when the base ckpt
    isn't downloaded."""
    from safetensors import safe_open

    reg = json.loads(reg_path.read_text())
    key = reg.get("key", reg_path.parent.name)
    paths = reg.get("paths", {})
    base_ckpt = _resolve_base_ckpt(paths.get("base_ckpt", ""), models_dir)
    if not base_ckpt.is_file():
        print(f"  skip {key}: base ckpt not on disk ({base_ckpt})")
        return 0

    template = reg.get("ui", {}).get("lora_layer_template")
    if not template:
        print(f"  skip {key}: no lora_layer_template in registry")
        return 0

    n_changes = 0
    print(f"  checking {key} against {base_ckpt} ...")
    with safe_open(str(base_ckpt), framework="pt", device="cpu") as f:
        # Per-block entries: prefix=per_block_prefix.format(i)+suffix.
        # LoRA dimensions are layer-uniform across blocks, so checking block 0
        # is sufficient. Conv1d-style 'suffix' entries (preprocess_conv etc.)
        # are skipped by _check_entry (3D tensors).
        pb_prefix = template.get("per_block_prefix", "")
        block0_prefix = "model." + pb_prefix.replace("{i}", "0")
        for entry in template.get("per_block", []) or []:
            base_key = block0_prefix + entry["suffix"] + ".weight"
            patch = _check_entry(f, base_key, entry)
            if patch is None:
                continue
            new_fi, new_fo = patch
            print(f"    per_block{entry['suffix']}: "
                  f"fi={entry['fi']},fo={entry['fo']}  →  fi={new_fi},fo={new_fo}")
            if not dry_run:
                entry["fi"], entry["fo"] = new_fi, new_fo
            n_changes += 1

        # prefix / suffix entries — name is the full path past 'model.'.
        for section in ("prefix", "suffix"):
            for entry in template.get(section, []) or []:
                base_key = "model." + entry["name"] + ".weight"
                patch = _check_entry(f, base_key, entry)
                if patch is None:
                    continue
                new_fi, new_fo = patch
                print(f"    {section} {entry['name']}: "
                      f"fi={entry['fi']},fo={entry['fo']}  →  fi={new_fi},fo={new_fo}")
                if not dry_run:
                    entry["fi"], entry["fo"] = new_fi, new_fo
                n_changes += 1

    if not dry_run and n_changes:
        reg_path.write_text(json.dumps(reg, indent=2) + "\n")
        print(f"    wrote {n_changes} change(s) to {reg_path}")
    elif n_changes == 0:
        print(f"    no changes")
    return n_changes


def _resolve_models_dir(arg_value: str | None) -> Path:
    """Mirror the dashboard's resolution: $UNDERFIT_MODELS_DIR or
    <repo>/state/models."""
    if arg_value:
        return Path(arg_value).expanduser()
    env = os.environ.get("UNDERFIT_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return _project_root() / "state" / "models"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="underfit-update-registries",
        description="Re-derive lora_layer_template dims from base safetensors.",
    )
    p.add_argument(
        "models",
        nargs="*",
        help="Specific model keys to update (default: all in dashboard/models/).",
    )
    p.add_argument(
        "--check", "--dry-run",
        action="store_true",
        help="Don't write anything — just print would-be changes.",
    )
    p.add_argument(
        "--models-dir",
        default=None,
        help="Override the models directory (defaults to $UNDERFIT_MODELS_DIR "
             "or <repo>/state/models).",
    )
    args = p.parse_args(argv)

    models_dir = _resolve_models_dir(args.models_dir)
    reg_root = _registries_dir()
    print(f"models dir: {models_dir}")
    print(f"registries: {reg_root}")
    print()

    all_keys = sorted(p.name for p in reg_root.iterdir() if (p / "registry.json").is_file())
    if args.models:
        unknown = set(args.models) - set(all_keys)
        if unknown:
            print(f"unknown model keys: {', '.join(sorted(unknown))}")
            print(f"known: {', '.join(all_keys)}")
            return 1
        targets = [k for k in all_keys if k in set(args.models)]
    else:
        targets = all_keys

    total = 0
    for key in targets:
        total += update_registry(reg_root / key / "registry.json", models_dir, dry_run=args.check)

    print()
    if args.check:
        print(f"dry-run summary: {total} field(s) would change. Re-run without --check to apply.")
    else:
        print(f"summary: {total} field(s) updated across {len(targets)} registry/registries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
