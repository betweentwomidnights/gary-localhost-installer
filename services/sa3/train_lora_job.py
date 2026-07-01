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
# Train the adapter against the *base* (pre-ARC-distillation) checkpoint and apply
# it to `stable-audio-3-medium` at inference — this is what the Spark does
# (`train_lora.py --model medium-base`) and what every shipped LoRA (kev, koan,
# keygen, succession) was trained on. Training a rectified-flow LoRA against the
# ARC-distilled `medium` weights is an objective/weights mismatch; the adapter
# tries to correct for it at the output projection, which shows up as
# postprocess_conv inflation and the low-frequency "drone".
# medium-base carries model.safetensors, model_config.json AND the full
# t5gemma-b-b-ul2/ conditioner, so this single snapshot still covers everything.
MODEL_REPO = "stabilityai/stable-audio-3-medium-base"
T5GEMMA_SUBFOLDER = "t5gemma-b-b-ul2"
MODEL_SNAPSHOT_FILES = (
    "model_config.json",
    "model.safetensors",
    f"{T5GEMMA_SUBFOLDER}/config.json",
    f"{T5GEMMA_SUBFOLDER}/model.safetensors",
    f"{T5GEMMA_SUBFOLDER}/tokenizer.json",
    f"{T5GEMMA_SUBFOLDER}/tokenizer_config.json",
)
TRAINING_DEPENDENCIES = {
    "accelerate": "accelerate>=0.30",
    "dill": "dill>=0.3.8",
    "audio_metadata": "audio-metadata>=0.11",
    # SA3 LoRA training runs on the official Lightning DiffusionCondTrainingWrapper.
    # Auto-installed at training start (checked via importlib below) so auto-updater
    # users never have to run "rebuild env" to pick it up. All heavy deps (torch,
    # numpy) are already present; this only pulls small pure-Python wheels.
    "pytorch_lightning": "pytorch-lightning>=2.2,<2.6",
}


class Cancelled(RuntimeError):
    pass


def slugify(raw: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:64] or "sa3-lora"


def write_json(path: Path, payload: dict, *, attempts: int = 8) -> None:
    """Atomically write JSON, tolerating Windows' transient replace failures.

    os.replace raises PermissionError (WinError 5) when the destination is
    momentarily held open — the control center polling the status file, Defender
    scanning the freshly written .tmp, or a search indexer. Retrying briefly
    clears it. The staging name is unique per process+call so two writers can
    never collide on one .tmp (same reason install_managed_sa3_checkpoint stages
    with a pid+nonce on the Rust side).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    delay = 0.05
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # transient on Windows; back off and retry
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 0.5)
        except OSError as exc:  # not a lock (bad path, full disk); don't spin
            last_error = exc
            break

    try:
        tmp.unlink()  # don't leave staging litter behind
    except OSError:
        pass
    if last_error is not None:
        raise last_error


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_if_changed(path: Path, payload: dict) -> None:
    """write_json, but skip the replace entirely when the file already matches.

    Every replace is a chance to collide with a reader that has the file open
    without FILE_SHARE_DELETE (gary4juce polling from a DAW does exactly this,
    which is what surfaces as WinError 5). Pointer files whose contents are
    constant for a whole run should therefore be written once, not on every
    status tick."""
    if read_json(path, None) == payload:
        return
    write_json(path, payload)


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
    # These two files are UI plumbing. update_status runs at every phase (and on
    # every run_step), so a single unlucky write must not abort a training run
    # that is otherwise healthy — warn and carry on instead.
    try:
        write_json(args.status_path, payload)
        # This pointer is identical for the whole run, so write it once instead of
        # on every tick — one replace per job rather than one per status update.
        write_json_if_changed(
            args.current_job_path,
            {"jobId": args.job_id, "statusPath": str(args.status_path)},
        )
    except OSError as exc:
        print(f"[status] could not update status files ({exc}); continuing", flush=True)


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


def link_or_copy(src: Path, dst: Path, *, replace: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        if not replace:
            return

    staged = dst.with_name(dst.name + ".staging") if replace else dst
    if staged.exists():
        staged.unlink()
    try:
        os.link(src, staged)
    except OSError:
        shutil.copy2(src, staged)
    if replace:
        os.replace(staged, dst)


def resolve_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": MODEL_REPO,
        "allow_patterns": list(MODEL_SNAPSHOT_FILES),
    }
    try:
        snapshot_dir = Path(snapshot_download(**kwargs, local_files_only=True))
    except Exception:
        snapshot_dir = Path(snapshot_download(**kwargs))

    missing = [name for name in MODEL_SNAPSHOT_FILES if not (snapshot_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"The cached {MODEL_REPO} snapshot is incomplete; missing: {', '.join(missing)}"
        )
    return snapshot_dir


def stage_base_model(args) -> tuple[Path, Path, Path]:
    check_cancel(args)
    update_status(args, status="running", phase="staging-model", message="Staging SA3 base model")
    snapshot_dir = resolve_model_snapshot()
    check_cancel(args)

    base_dir = args.models_dir / MODEL_KEY / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in MODEL_SNAPSHOT_FILES:
        check_cancel(args)
        link_or_copy(
            snapshot_dir / relative_path,
            base_dir / relative_path,
            replace=True,
        )
    return (
        base_dir / "model_config.json",
        base_dir / "model.safetensors",
        base_dir / T5GEMMA_SUBFOLDER,
    )


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


def resolve_precision(args) -> None:
    """Pick bf16 when the GPU supports it (Ampere+), else fall back to fp16.

    SA3 was trained in bf16 and the official trainer uses bf16 throughout. Its
    DoRA parametrization normalizes over the full weight (magnitude/direction
    decomposition = a norm-division), which fp16's narrow 5-bit exponent range
    corrupts — that's the "metallic"/flattened adapter that degrades as you turn
    strength up. bf16 keeps fp32's exponent range and fixes it. Turing (RTX 20xx)
    has no native bf16, so we fall back to fp16 there rather than forcing slow
    emulation. Stored on args and consumed by build_model_config (base weights)
    and the lora_train.py --precision arg (autocast)."""
    import torch

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args.base_precision = "bf16" if use_bf16 else "fp16"
    args.amp_precision = "bf16-mixed" if use_bf16 else "16-mixed"
    print(
        f"[precision] {args.base_precision} base / {args.amp_precision} AMP "
        + (
            "(GPU supports bf16)"
            if use_bf16
            else "(bf16 unsupported on this GPU; falling back to fp16)"
        ),
        flush=True,
    )


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
    # We deliberately diverge from underfit's dashboard here in favor of a
    # trainer that's trivial to reason about:
    #   * NO path/filename prompts. Feeding relpath (e.g.
    #     "01 - Montanita [XrXqKoCPvE0].npy") as text conditioning teaches the
    #     encoder garbage. Unconditional generation is covered by
    #     cfg_dropout_prob (see build_model_config), not by path prompts.
    #   * tag_keys is just ["prompt"] — the caption our auto-labeller writes.
    #     No title/artist/genre/bpm re-labeling, so it's obvious from the
    #     sidecar exactly what a clip trains on.
    #   * A shared trigger word, when set, is PREPENDED to every caption
    #     ("<trigger>, <caption>"), so the token gets tied to the style while the
    #     model still learns the caption. Blank leaves captions untouched.
    #     This replaces the old use_fixed behaviour, which *substituted* the
    #     phrase for the caption on 60% of steps — training those steps on a
    #     bare trigger with no descriptive content.
    trigger = args.fixed_prompt.strip()
    prompt_config = {
        "use_tags": True,
        "use_paths": False,
        "use_fixed": False,
        "fixed_text": "",
        "balance": {"tags": 40},
        "tag_keys": ["prompt"],
        "hide_tag_names": True,
        "shuffle": False,
        "trigger": trigger,
        "trigger_pct": 100 if trigger else 0,
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
    # Make the prompt policy explicit in the log — the trigger is applied per-step
    # at training time (prompt_templates), not baked into the sidecars, so without
    # this there is nothing on screen showing it is in effect.
    sample = ""
    for meta_path in sorted(latent_dir.rglob("*.json")):
        if meta_path.name.startswith(".") or meta_path.name == "details.json":
            continue
        sample = " ".join(str(read_json(meta_path, {}).get("prompt") or "").split())
        if sample:
            break
    if trigger:
        print(f'[prompts] trigger word "{trigger}" is prepended to every caption', flush=True)
        if sample:
            print(f'[prompts]   e.g. "{trigger}, {sample}"', flush=True)
    else:
        print("[prompts] no trigger word set; captions are used as-is", flush=True)
        if sample:
            print(f'[prompts]   e.g. "{sample}"', flush=True)

    path = args.run_dir / f"{args.job_id}_dataset.json"
    write_json(path, payload)
    return path


def build_model_config(args, t5gemma_dir: Path) -> Path:
    template_path = SERVICE_DIR / "dashboard" / "models" / MODEL_KEY / "training_template.json"
    payload = read_json(template_path, None)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Could not read training template: {template_path}")

    training = payload.setdefault("training", {})
    training["base_precision"] = getattr(args, "base_precision", "fp16")
    training["cfg_dropout_prob"] = 0.1
    training.setdefault("demo", {})["demo_every"] = 0
    lora = training.setdefault("lora_config", {})
    lora["rank"] = args.rank
    lora["alpha"] = args.alpha if args.alpha > 0 else args.rank
    lora["adapter_type"] = args.adapter_type
    if args.lora_include.strip():
        lora["include"] = [item.strip() for item in args.lora_include.split(",") if item.strip()]
    # Always exclude the seconds_total conditioner from the adapter (228 modules,
    # not 229). SA3's own docs recommend this on small datasets ("conditioner
    # hijacking"), and our pre-encoded pipeline makes it worse: seconds_total is
    # stored as the full clip duration and is NOT updated when a shorter window is
    # cropped at train time (see stable_audio_3/data/dataset.py). So every crop
    # feeds a duration that doesn't match the latent length; letting the adapter
    # learn that conditioner bakes in a length-dependent artifact (reverb that
    # gets worse the longer you generate). kev, our best DGX LoRA, barely adapted
    # this layer anyway (bottom ~10% of adaptation energy).
    # Match the Spark recipe: --include/--exclude default to None, i.e. train all
    # 229 modules (seconds_total conditioner INCLUDED). The shipped Spark LoRAs
    # (kev etc.) are all 229 and clean; excluding seconds_total was a divergence
    # we're reverting. Only honor an explicit user-supplied exclude.
    exclude = [item.strip() for item in args.lora_exclude.split(",") if item.strip()]
    if exclude:
        lora["exclude"] = list(dict.fromkeys(exclude))  # dedupe, keep order
    if args.learning_rate > 0:
        opt = training.setdefault("optimizer_configs", {}).setdefault("diffusion", {}).setdefault("optimizer", {})
        opt.setdefault("type", "AdamW")
        opt.setdefault("config", {})["lr"] = args.learning_rate

    t5gemma_found = False
    conditioning = payload.get("model", {}).get("conditioning", {})
    for conditioner in conditioning.get("configs", []):
        if conditioner.get("type") != "t5gemma":
            continue
        config = conditioner.setdefault("config", {})
        config["model_path"] = str(t5gemma_dir)
        config.pop("repo_id", None)
        config.pop("subfolder", None)
        t5gemma_found = True
    if not t5gemma_found:
        raise RuntimeError("Training template does not define a T5Gemma conditioner")
    payload["base_model"] = MODEL_KEY

    path = args.run_dir / f"{args.job_id}_model.json"
    write_json(path, payload)
    return path


CHECKPOINT_FILENAME_RE = re.compile(r"-step=(\d+)-epoch=(\d+)\.safetensors$", re.IGNORECASE)


def training_checkpoints(run_dir: Path) -> list[dict]:
    checkpoints = []
    for path in run_dir.rglob("*.safetensors"):
        match = CHECKPOINT_FILENAME_RE.search(path.name)
        if not match:
            continue
        checkpoints.append(
            {
                "step": int(match.group(1)),
                "epoch": int(match.group(2)),
                "path": str(path.resolve()),
            }
        )
    checkpoints.sort(key=lambda item: (item["step"], item["epoch"], item["path"]))
    return checkpoints


def register_lora(args, checkpoint: Path, checkpoints: list[dict]) -> Path:
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
        "trainingJobId": args.job_id,
        "trainingCheckpoints": checkpoints,
        "selectedTrainingStep": checkpoints[-1]["step"],
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
        resolve_precision(args)
        _, base_ckpt, t5gemma_dir = stage_base_model(args)

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
        model_config = build_model_config(args, t5gemma_dir)
        demos_dir = args.run_dir / "demos"
        demos_dir.mkdir(parents=True, exist_ok=True)
        runs_root = args.training_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        run_step(
            args,
            [
                sys.executable,
                "-u",
                str(SERVICE_DIR / "lora_train_lightning.py"),
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
                "0",
                "--precision",
                getattr(args, "amp_precision", "16-mixed"),
                "--batch-size",
                str(args.batch_size),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--max-steps",
                str(args.max_steps),
                # The Spark trains with NO gradient clipping: scripts/train_lora.py
                # never defines --gradient_clip_val, so its `hasattr` guard sets it
                # to None. 0 here is converted to None by the entry point.
                "--gradient-clip-val",
                "0",
            ],
            "training",
            "Training LoRA",
            cwd=demos_dir,
        )

        checkpoints = training_checkpoints(runs_root / args.job_id)
        if not checkpoints:
            raise RuntimeError("Training finished but no .safetensors checkpoint was found")
        checkpoint = Path(checkpoints[-1]["path"])
        final_path = register_lora(args, checkpoint, checkpoints)
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
