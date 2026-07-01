#!/usr/bin/env python3
"""Gary-native SA3 LoRA training job.

This is a thin Windows-friendly wrapper around the vendored underfit trainer.
It keeps the control center API small: Tauri launches this one process, then
polls the status JSON written here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
MODEL_KEY = "sa3-medium"
BASE_REPO = "stabilityai/stable-audio-3-medium-base"
TRAINING_DEPENDENCIES = {
    "accelerate": "accelerate>=0.30",
    "dill": "dill>=0.3.8",
    "audio_metadata": "audio-metadata>=0.11",
}


class Cancelled(RuntimeError):
    pass


def slugify(raw: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:64] or "sa3-lora"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_status(args, **updates) -> None:
    payload = read_json(args.status_path, {})
    payload.update(
        {
            "jobId": args.job_id,
            "name": args.name,
            "pid": os.getpid(),
            "runDir": str(args.run_dir),
            "logPath": str(args.log_path),
            "cancelPath": str(args.cancel_path),
            "updatedAt": time.time(),
        }
    )
    payload.update(updates)
    write_json(args.status_path, payload)
    write_json(args.current_job_path, {"jobId": args.job_id, "statusPath": str(args.status_path)})


def cancel_requested(args) -> bool:
    return args.cancel_path.exists()


def check_cancel(args) -> None:
    if cancel_requested(args):
        raise Cancelled("Training cancelled.")


def missing_training_dependencies() -> list[str]:
    return [
        requirement
        for module, requirement in TRAINING_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]


def ensure_training_dependencies(args) -> None:
    missing = missing_training_dependencies()
    if not missing:
        return

    print(
        "[environment-setup] Installing missing SA3 LoRA training dependencies: "
        + ", ".join(missing),
        flush=True,
    )
    try:
        run_step(
            args,
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *missing,
            ],
            "environment-setup",
            "Installing missing SA3 LoRA training dependencies",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not install the missing SA3 LoRA training dependencies automatically. "
            "Use 'rebuild env' for SA3 and try again."
        ) from exc
    importlib.invalidate_caches()

    unresolved = missing_training_dependencies()
    if unresolved:
        raise RuntimeError(
            "SA3 LoRA training dependencies are still missing after automatic repair: "
            + ", ".join(unresolved)
            + ". Use 'rebuild env' for SA3 and try again."
        )


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    else:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def stage_base_model(args) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    check_cancel(args)
    update_status(args, status="running", phase="staging-model", message="Staging SA3 base model")
    config_src = Path(hf_hub_download(repo_id=BASE_REPO, filename="model_config.json"))
    check_cancel(args)
    ckpt_src = Path(hf_hub_download(repo_id=BASE_REPO, filename="model.safetensors"))
    check_cancel(args)

    base_dir = args.models_dir / MODEL_KEY / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    config_dst = base_dir / "model_config.json"
    ckpt_dst = base_dir / "model.safetensors"
    shutil.copy2(config_src, config_dst)
    link_or_copy(ckpt_src, ckpt_dst)
    return config_dst, ckpt_dst


def torch_backend_label(torch_module) -> str:
    parts = [
        f"torch={torch_module.__version__}",
        f"hip={getattr(torch_module.version, 'hip', None)}",
        f"cuda_build={getattr(torch_module.version, 'cuda', None)}",
        f"cuda_available={torch_module.cuda.is_available()}",
        f"device_count={torch_module.cuda.device_count()}",
    ]
    if torch_module.cuda.is_available():
        try:
            props = torch_module.cuda.get_device_properties(0)
            device = torch_module.cuda.get_device_name(0)
            gcn_arch = getattr(props, "gcnArchName", None)
            if getattr(torch_module.version, "hip", None) and gcn_arch:
                device += f" gcn={gcn_arch}"
            parts.append(f"device0={device}")
        except Exception as exc:
            parts.append(f"device_error={type(exc).__name__}: {exc}")
    return "; ".join(parts)


def require_accelerator(args) -> None:
    check_cancel(args)
    update_status(args, status="running", phase="checking-gpu", message="Checking GPU accelerator")
    import torch

    backend = torch_backend_label(torch)
    print(f"[checking-gpu] {backend}", flush=True)
    if not torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            raise RuntimeError(
                "SA3 LoRA training found a ROCm/HIP PyTorch build, but torch/HIP cannot "
                "see an AMD GPU. Confirm the Radeon driver supports this device and run "
                "scripts/rocm/windows-pytorch-preflight.ps1."
            )
        raise RuntimeError("SA3 LoRA training requires a CUDA/HIP GPU accelerator.")


def run_step(args, command: list[str], phase: str, message: str, cwd: Path | None = None) -> None:
    check_cancel(args)
    update_status(args, status="running", phase=phase, message=message)
    print(f"\n[{phase}] {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["UNDERFIT_BACKEND"] = "sa3"
    env["UNDERFIT_STATE_DIR"] = str(args.training_root)
    env["UNDERFIT_MODELS_DIR"] = str(args.models_dir)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONPATH"] = str(SERVICE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd or SERVICE_DIR),
        env=env,
        creationflags=creationflags,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    update_status(args, status="running", phase=phase, message=message, childPid=proc.pid)
    try:
        while True:
            code = proc.poll()
            if code is not None:
                if code != 0:
                    raise RuntimeError(f"{phase} failed with exit code {code}")
                return
            if cancel_requested(args):
                print(f"\n[{phase}] cancellation requested; stopping child process {proc.pid}", flush=True)
                terminate_process_tree(proc)
                update_status(
                    args,
                    status="cancelled",
                    phase="cancelled",
                    message="Training cancelled.",
                    error=None,
                    childPid=None,
                )
                raise Cancelled("Training cancelled.")
            time.sleep(0.5)
    finally:
        if proc.poll() is not None:
            update_status(args, childPid=None)


def latent_crop_length(seconds: float) -> int:
    # SA3 medium uses 44.1 kHz audio and a 4096x latent downsampling ratio.
    tokens = round(max(1.0, seconds) * 44100 / 4096)
    tokens = max(64, min(4096, tokens))
    return int((tokens + 15) // 16 * 16)


def build_dataset_config(args, latent_dir: Path) -> Path:
    prompt_config = {
        "use_tags": True,
        "use_paths": not bool(args.fixed_prompt.strip()),
        "use_fixed": bool(args.fixed_prompt.strip()),
        "fixed_text": args.fixed_prompt.strip(),
        "balance": {"tags": 40, "paths": 30, "fixed": 60},
        "tag_keys": ["prompt", "title", "artist", "genre", "bpm"],
        "hide_tag_names": False,
        "shuffle": True,
    }
    payload = {
        "dataset_type": "pre_encoded",
        "datasets": [
            {
                "id": args.name,
                "path": str(latent_dir),
                "custom_metadata_module": str(SERVICE_DIR / "dataset_processing" / "prompt_templates.py"),
            }
        ],
        "latent_crop_length": latent_crop_length(args.latent_crop_seconds),
        "random_crop": True,
        "prompt_config": prompt_config,
    }
    path = args.run_dir / f"{args.job_id}_dataset.json"
    write_json(path, payload)
    return path


def build_model_config(args) -> Path:
    template_path = SERVICE_DIR / "dashboard" / "models" / MODEL_KEY / "training_template.json"
    payload = read_json(template_path, None)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Could not read training template: {template_path}")

    training = payload.setdefault("training", {})
    training["base_precision"] = "fp16"
    training["cfg_dropout_prob"] = 0.1
    training.setdefault("demo", {})["demo_every"] = 0
    lora = training.setdefault("lora_config", {})
    lora["rank"] = args.rank
    lora["alpha"] = args.alpha if args.alpha > 0 else args.rank
    lora["adapter_type"] = args.adapter_type
    if args.lora_include.strip():
        lora["include"] = [item.strip() for item in args.lora_include.split(",") if item.strip()]
    if args.lora_exclude.strip():
        lora["exclude"] = [item.strip() for item in args.lora_exclude.split(",") if item.strip()]
    if args.learning_rate > 0:
        opt = training.setdefault("optimizer_configs", {}).setdefault("diffusion", {}).setdefault("optimizer", {})
        opt.setdefault("type", "AdamW")
        opt.setdefault("config", {})["lr"] = args.learning_rate
    payload["base_model"] = MODEL_KEY

    path = args.run_dir / f"{args.job_id}_model.json"
    write_json(path, payload)
    return path


def newest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def register_lora(args, checkpoint: Path) -> Path:
    check_cancel(args)
    final_dir = args.lora_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{args.name}.safetensors"
    shutil.copy2(checkpoint, final_path)

    catalog = read_json(args.catalog_path, {})
    if not isinstance(catalog, dict):
        catalog = {}
    catalog[args.name] = {
        "path": str(final_path),
        "promptsPath": str(args.dataset_dir),
        "strength": 1.0,
    }
    write_json(args.catalog_path, catalog)
    return final_path


def maybe_build_prompts(args) -> None:
    check_cancel(args)
    txts = list(args.dataset_dir.rglob("*.txt"))
    if not txts:
        return
    run_step(
        args,
        [
            sys.executable,
            str(SERVICE_DIR / "build_lora_prompts.py"),
            "--name",
            args.name,
            "--captions-dir",
            str(args.dataset_dir),
            "--out-dir",
            str(args.prompts_dir),
            "--force",
        ],
        "building-prompts",
        "Building prompt dice pool",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fixed-prompt", default="")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--current-job-path", type=Path, required=True)
    parser.add_argument("--cancel-path", type=Path)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--adapter-type", default="dora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--latent-crop-seconds", type=float, default=47.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--per-track-target-latent-rms", type=float, default=0.0)
    parser.add_argument("--lora-include", default="")
    parser.add_argument("--lora-exclude", default="")
    args = parser.parse_args()
    args.name = slugify(args.name)
    args.cancel_path = args.cancel_path or (args.run_dir / "cancel.requested")

    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if args.cancel_path.exists():
            args.cancel_path.unlink()
        update_status(args, status="running", phase="starting", message="Starting SA3 LoRA training")

        ensure_training_dependencies(args)
        require_accelerator(args)
        _, base_ckpt = stage_base_model(args)

        encoded_root = args.run_dir / "encoded"
        pre_encode_command = [
            sys.executable,
            str(SERVICE_DIR / "dataset_processing" / "pre_encode.py"),
            "--input-dir",
            str(args.dataset_dir),
            "--model",
            MODEL_KEY,
            "--output-dir",
            str(encoded_root),
            "--num-gpus",
            "1",
            "--half",
            "--batch-size",
            "1",
        ]
        if args.per_track_target_latent_rms > 0:
            pre_encode_command.extend(
                [
                    "--per-track-target-latent-rms",
                    str(args.per_track_target_latent_rms),
                ]
            )

        run_step(
            args,
            pre_encode_command,
            "pre-encoding",
            "Pre-encoding audio to SA3 latents",
        )

        latent_dir = encoded_root / "latents" / MODEL_KEY
        check_cancel(args)
        dataset_config = build_dataset_config(args, latent_dir)
        model_config = build_model_config(args)
        demos_dir = args.run_dir / "demos"
        demos_dir.mkdir(parents=True, exist_ok=True)
        runs_root = args.training_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        run_step(
            args,
            [
                sys.executable,
                "-u",
                str(SERVICE_DIR / "lora_train.py"),
                "--name",
                args.job_id,
                "--config-file",
                str(SERVICE_DIR / "defaults.ini"),
                "--save-dir",
                str(runs_root),
                "--model-config",
                str(model_config),
                "--dataset-config",
                str(dataset_config),
                "--pretrained-ckpt-path",
                str(base_ckpt),
                "--num-workers",
                "1",
                "--precision",
                "16-mixed",
                "--batch-size",
                str(args.batch_size),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--max-steps",
                str(args.max_steps),
                "--gradient-clip-val",
                "1.0",
            ],
            "training",
            "Training LoRA",
            cwd=demos_dir,
        )

        checkpoint = newest_checkpoint(runs_root / args.job_id)
        if checkpoint is None:
            raise RuntimeError("Training finished but no .safetensors checkpoint was found")
        final_path = register_lora(args, checkpoint)
        maybe_build_prompts(args)
        update_status(
            args,
            status="completed",
            phase="completed",
            message="Training complete",
            finalCheckpointPath=str(final_path),
            childPid=None,
        )
        return 0
    except Cancelled:
        update_status(
            args,
            status="cancelled",
            phase="cancelled",
            message="Training cancelled.",
            error=None,
            childPid=None,
        )
        return 0
    except Exception as exc:
        update_status(args, status="failed", phase="failed", message=str(exc), error=str(exc), childPid=None)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
