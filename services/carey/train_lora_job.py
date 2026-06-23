#!/usr/bin/env python3
"""Gary-native ACE-Step LoRA training job.

This process owns dataset preparation, optional ``understand_music``
captioning, preprocessing, training, status updates, and cancellation. The
Tauri service manager remains the sole owner of Carey's service process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ace_training_dataset import (
    audio_duration_seconds,
    build_dataset_json,
    discover_audio_files,
    load_sidecar_metadata,
    write_canonical_sidecar,
)
from bpm_analysis import choose_bpm, estimate_bpm
from key_analysis import choose_key, estimate_key

SERVICE_DIR = Path(__file__).resolve().parent
TRAIN_ENTRY = SERVICE_DIR / "train.py"
MODEL_MAP = {
    "base": {
        "variant": "base",
        "folder": "acestep-v15-base",
        "family": "standard",
    },
    "xl-base": {
        "variant": "acestep-v15-xl-base",
        "folder": "acestep-v15-xl-base",
        "family": "xl",
    },
}
CAPTION_LM_MODELS = (
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
)
# Torch wheels must come from Carey's platform-specific build path. Installing
# them from generic PyPI during job startup could silently replace CUDA Torch.
CORE_RUNTIME_MODULES = ("torch", "torchaudio")
JOB_DEPENDENCIES = {
    "transformers": "transformers>=4.51.0,<4.58.0",
    "diffusers": "diffusers",
    "accelerate": "accelerate>=1.12.0",
    "einops": "einops>=0.8.1",
    "safetensors": "safetensors==0.7.0",
    "httpx": "httpx>=0.27.0",
    "scipy": "scipy>=1.10.1",
    "soundfile": "soundfile>=0.13.1",
    "peft": "peft==0.18.1",
    "lightning": "lightning>=2.0.0",
    "tensorboard": "tensorboard>=2.0.0",
}
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass(frozen=True)
class PreparedCaptionAudio:
    path: Path
    cleanup_path: Path | None
    duration: float
    offset: float


class Cancelled(RuntimeError):
    pass


def slugify(raw: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:64] or "ace-lora"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_status(args: argparse.Namespace, **updates: Any) -> None:
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
    write_json(
        args.current_job_path,
        {"jobId": args.job_id, "statusPath": str(args.status_path)},
    )


def cancel_requested(args: argparse.Namespace) -> bool:
    return args.cancel_path.exists()


def check_cancel(args: argparse.Namespace) -> None:
    if cancel_requested(args):
        raise Cancelled("Training cancelled.")


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


def run_step(
    args: argparse.Namespace,
    command: list[str],
    phase: str,
    message: str,
    *,
    cwd: Path | None = None,
) -> None:
    check_cancel(args)
    failure_report = (cwd or SERVICE_DIR) / ".training-failure.txt"
    if phase == "training":
        failure_report.unlink(missing_ok=True)
    update_status(args, status="running", phase=phase, message=message)
    print(f"\n[{phase}] {subprocess.list2cmdline(command)}", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SERVICE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd or SERVICE_DIR),
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    update_status(args, status="running", phase=phase, message=message, childPid=proc.pid)
    try:
        while True:
            code = proc.poll()
            if code is not None:
                if code != 0:
                    if phase == "training" and failure_report.is_file():
                        reason = failure_report.read_text(encoding="utf-8").strip()
                        if reason:
                            raise RuntimeError(reason)
                    raise RuntimeError(f"{phase} failed with exit code {code}")
                return
            if cancel_requested(args):
                terminate_process_tree(proc)
                raise Cancelled("Training cancelled.")
            time.sleep(0.5)
    finally:
        if proc.poll() is not None:
            update_status(args, childPid=None)


def missing_core_runtime_modules() -> list[str]:
    return [
        module
        for module in CORE_RUNTIME_MODULES
        if importlib.util.find_spec(module) is None
    ]


def missing_job_dependencies() -> list[str]:
    return [
        requirement
        for module, requirement in JOB_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]


def ensure_job_dependencies(args: argparse.Namespace) -> None:
    """Repair safe missing packages before captioning or training starts."""
    missing_core = missing_core_runtime_modules()
    if missing_core:
        raise RuntimeError(
            "Carey's core CUDA environment is incomplete (missing "
            + ", ".join(missing_core)
            + "). Use 'rebuild env' for Carey and try again."
        )

    missing = missing_job_dependencies()
    if not missing:
        return

    print(
        "[environment-setup] Installing missing Carey captioning/training dependencies: "
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
            "Installing missing Carey captioning/training dependencies",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not install the missing Carey captioning/training dependencies "
            "automatically. Use 'rebuild env' for Carey and try again."
        ) from exc

    importlib.invalidate_caches()
    unresolved = missing_job_dependencies()
    if unresolved:
        raise RuntimeError(
            "Carey captioning/training dependencies are still missing after "
            "automatic repair: "
            + ", ".join(unresolved)
            + ". Use 'rebuild env' for Carey and try again."
        )


def require_training_environment(args: argparse.Namespace) -> None:
    ensure_job_dependencies(args)
    check_cancel(args)
    update_status(args, status="running", phase="checking-gpu", message="Checking CUDA GPU")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ACE-Step LoRA training requires an NVIDIA CUDA GPU.")


def require_model_checkpoint(args: argparse.Namespace) -> None:
    model = MODEL_MAP[args.model]
    model_dir = args.checkpoint_dir / model["folder"]
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise RuntimeError(f"ACE-Step model checkpoint is incomplete: {model_dir}")


def build_preprocess_command(
    args: argparse.Namespace,
    dataset_json: Path,
    tensors_dir: Path,
    output_dir: Path,
) -> list[str]:
    model = MODEL_MAP[args.model]
    return [
        sys.executable,
        "-u",
        str(TRAIN_ENTRY),
        "--plain",
        "-y",
        "fixed",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--model-variant",
        model["variant"],
        "--base-model",
        "base",
        "--dataset-dir",
        str(tensors_dir),
        "--output-dir",
        str(output_dir),
        "--preprocess",
        "--dataset-json",
        str(dataset_json),
        "--tensor-output",
        str(tensors_dir),
        "--max-duration",
        str(args.max_duration),
        "--device",
        "cuda:0",
        "--precision",
        "bf16",
        "--num-workers",
        "0",
        "--prefetch-factor",
        "0",
        "--no-persistent-workers",
    ]


def build_train_command(
    args: argparse.Namespace,
    tensors_dir: Path,
    output_dir: Path,
) -> list[str]:
    model = MODEL_MAP[args.model]
    command = [
        sys.executable,
        "-u",
        str(TRAIN_ENTRY),
        "--plain",
        "-y",
        "fixed",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--model-variant",
        model["variant"],
        "--base-model",
        "base",
        "--dataset-dir",
        str(tensors_dir),
        "--output-dir",
        str(output_dir),
        "--max-duration",
        str(args.max_duration),
        "--adapter-type",
        "lora",
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--module-profile",
        str(getattr(args, "module_profile", "balanced")),
        "--lr",
        str(args.learning_rate),
        "--optimizer-type",
        "adamw",
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--epochs",
        str(args.epochs),
        "--save-every",
        str(args.save_every),
        "--save-best-after",
        str(getattr(args, "save_best_after", 25)),
        "--cfg-ratio",
        str(args.cfg_ratio),
        "--timestep-mu",
        str(resolve_timestep_mu(args)),
        "--loss-weighting",
        str(args.loss_weighting),
        "--snr-gamma",
        str(args.snr_gamma),
        "--shift",
        "1.0",
        "--num-inference-steps",
        "50",
        "--gradient-checkpointing",
        "--offload-encoder",
        "--vram-preflight",
        "--device",
        "cuda:0",
        "--precision",
        "bf16",
        "--num-workers",
        "0",
        "--prefetch-factor",
        "0",
        "--no-persistent-workers",
    ]
    command.append(
        "--save-best" if getattr(args, "save_best", True) else "--no-save-best"
    )
    if getattr(args, "adapter_type", "dora") == "dora":
        command.append("--use-dora")
    return command


def resolve_timestep_mu(args: argparse.Namespace) -> float:
    """Resolve the explicit override or the ACE-Step/Side-Step default."""
    explicit = getattr(args, "timestep_mu", None)
    if explicit is not None:
        return float(explicit)
    return -0.4


def is_complete_peft_adapter(path: Path) -> bool:
    return (
        (path / "adapter_model.safetensors").is_file()
        and (path / "adapter_config.json").is_file()
    )


def parse_carey_endpoint(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def build_caption_server_command(args: argparse.Namespace) -> list[str]:
    host, port = parse_carey_endpoint(args.carey_url)
    return [
        sys.executable,
        str(SERVICE_DIR / "acestep" / "api_server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--no-init",
        "--init-llm",
        "--lm-model-path",
        args.caption_lm_model,
    ]


def build_caption_server_env(
    args: argparse.Namespace,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SERVICE_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    force_dit_offload = args.caption_lm_model != "acestep-5Hz-lm-0.6B"

    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "ACESTEP_CONFIG_PATH": MODEL_MAP[args.model]["folder"],
            "ACESTEP_INIT_LLM": "true",
            "ACESTEP_LM_MODEL_PATH": args.caption_lm_model,
            "ACESTEP_LM_BACKEND": "pt",
            "ACESTEP_LM_OFFLOAD_TO_CPU": "true",
            "ACESTEP_NO_INIT": "true",
            "ACESTEP_OFFLOAD_TO_CPU": "true",
            "ACESTEP_OFFLOAD_DIT_TO_CPU": "true" if force_dit_offload else "false",
            "ACESTEP_UNDERSTAND_MAX_NEW_TOKENS": "1024",
            "ACESTEP_UNDERSTAND_TEMPERATURE": "0.3",
            "ACESTEP_USE_FLASH_ATTENTION": "false",
            "ACESTEP_COMPILE_MODEL": "false",
            "ACESTEP_API_WORKERS": "1",
        }
    )
    for key in ("ACESTEP_CONFIG_PATH2", "ACESTEP_CONFIG_PATH3"):
        env.pop(key, None)
    return env


def start_caption_server(args: argparse.Namespace) -> subprocess.Popen[Any]:
    update_status(
        args,
        status="running",
        phase="starting-caption-service",
        message=f"Starting temporary ACE captioner with {args.caption_lm_model}",
    )
    kwargs: dict[str, Any] = {
        "cwd": str(SERVICE_DIR),
        "env": build_caption_server_env(args),
        "stdout": sys.stdout,
        "stderr": sys.stderr,
    }
    if CREATE_NO_WINDOW:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    process = subprocess.Popen(build_caption_server_command(args), **kwargs)
    update_status(args, childPid=process.pid)
    return process


def stop_caption_server(args: argparse.Namespace, process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        update_status(args, childPid=None)
        return

    update_status(
        args,
        status="running",
        phase="stopping-caption-service",
        message="Stopping temporary ACE captioner and releasing GPU memory",
    )
    if os.name == "nt" and process.pid:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    update_status(args, childPid=None)


def prepare_caption_audio(
    args: argparse.Namespace,
    audio_path: Path,
    *,
    caption_window_seconds: float | None = None,
) -> PreparedCaptionAudio:
    window_seconds = float(
        (
            caption_window_seconds
            if caption_window_seconds is not None
            else getattr(args, "caption_window_seconds", 0.0)
        )
        or 0.0
    )
    try:
        actual_duration = audio_duration_seconds(audio_path)
    except Exception as exc:
        print(
            f"[captioning] Could not read duration for {audio_path.name}; "
            f"using full file for captioning: {exc}",
            flush=True,
        )
        actual_duration = 0.0
    if window_seconds <= 0 or actual_duration <= window_seconds + 1.0:
        return PreparedCaptionAudio(
            path=audio_path,
            cleanup_path=None,
            duration=max(1.0, float(actual_duration or 0.0)),
            offset=0.0,
        )

    try:
        import soundfile as sf

        start_seconds = max(0.0, (actual_duration - window_seconds) / 2.0)
        with sf.SoundFile(str(audio_path)) as source:
            sample_rate = int(source.samplerate)
            start_frame = int(round(start_seconds * sample_rate))
            frames = int(round(window_seconds * sample_rate))
            source.seek(start_frame)
            audio = source.read(frames, always_2d=True, dtype="float32")
        if getattr(audio, "size", 0) == 0:
            raise RuntimeError("excerpt read returned no samples")

        temp = tempfile.NamedTemporaryFile(
            prefix="gary_ace_caption_",
            suffix=".wav",
            delete=False,
        )
        temp_path = Path(temp.name)
        temp.close()
        sf.write(str(temp_path), audio, sample_rate)
        excerpt_duration = float(len(audio) / sample_rate)
        return PreparedCaptionAudio(
            path=temp_path,
            cleanup_path=temp_path,
            duration=max(1.0, excerpt_duration),
            offset=start_seconds,
        )
    except Exception as exc:
        print(
            f"[captioning] Could not create {window_seconds:.0f}s caption excerpt "
            f"for {audio_path.name}; using full file instead: {exc}",
            flush=True,
        )
        return PreparedCaptionAudio(
            path=audio_path,
            cleanup_path=None,
            duration=max(1.0, float(actual_duration or 0.0)),
            offset=0.0,
        )


def wait_for_carey(
    args: argparse.Namespace,
    client: Any,
    process: subprocess.Popen[Any] | None = None,
) -> None:
    deadline = time.monotonic() + args.caption_startup_timeout
    last_error = "not reachable"
    while time.monotonic() < deadline:
        check_cancel(args)
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Temporary ACE captioner exited before becoming ready "
                f"(code {process.returncode}). Check the job log for details."
            )
        try:
            response = client.get(f"{args.carey_url}/health", timeout=5)
            if response.is_success:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Carey analysis backend did not become ready: {last_error}")


def ensure_carey_model_loaded(args: argparse.Namespace, client: Any) -> None:
    """Load the selected DiT when the analysis backend started with no-init."""
    health = client.get(f"{args.carey_url}/health", timeout=10)
    health.raise_for_status()
    data = health.json().get("data") or {}
    model = MODEL_MAP[args.model]["folder"]
    if data.get("initialized") and data.get("current_model") == model:
        return

    response = client.post(
        f"{args.carey_url}/v1/load",
        params={"config_path": model},
        timeout=args.model_load_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 200) not in (0, 200):
        raise RuntimeError(f"Carey model load failed: {payload}")


def caption_with_understand_music(args: argparse.Namespace) -> int:
    import httpx

    audio_files = discover_audio_files(args.dataset_dir)
    pending = [
        audio
        for audio in audio_files
        if args.overwrite_captions or not audio.with_suffix(".txt").is_file()
    ]
    if not pending:
        message = (
            f"All {len(audio_files)} track"
            f"{'' if len(audio_files) == 1 else 's'} already "
            "have sidecars; skipping understand_music. Enable overwrite "
            "captions to recaption existing .txt files."
        )
        setattr(args, "_caption_skip_message", message)
        print(f"[captioning] {message}", flush=True)
        update_status(
            args,
            status="running",
            phase="captioning-skipped",
            message=message,
            currentFile=0,
            totalFiles=len(audio_files),
            captionedCount=0,
            captionLmModel=args.caption_lm_model,
        )
        return 0

    update_status(
        args,
        status="running",
        phase="captioning",
        message=f"Captioning with {args.caption_lm_model}",
        currentFile=0,
        totalFiles=len(pending),
        captionLmModel=args.caption_lm_model,
    )
    caption_server = start_caption_server(args)
    try:
        with httpx.Client(timeout=httpx.Timeout(args.caption_timeout)) as client:
            wait_for_carey(args, client, caption_server)
            update_status(
                args,
                status="running",
                phase="loading-caption-model",
                message=f"Loading {MODEL_MAP[args.model]['folder']} for audio analysis",
            )
            ensure_carey_model_loaded(args, client)
            for index, audio_path in enumerate(pending, 1):
                check_cancel(args)
                update_status(
                    args,
                    status="running",
                    phase="captioning",
                    message=f"Analyzing {audio_path.name}",
                    currentFile=index,
                    totalFiles=len(pending),
                )
                result = request_valid_music_analysis(args, client, audio_path)
                bpm_decision = decide_sidecar_bpm(args, audio_path, result)
                key_decision = decide_sidecar_key(args, audio_path, result)
                lyrics = str(result.get("lyrics") or "")
                is_instrumental = analysis_is_instrumental(
                    result,
                    default=args.instrumental,
                )
                language = str(result.get("language") or "")
                if is_instrumental:
                    language = "unknown"
                write_canonical_sidecar(
                    audio_path.with_suffix(".txt"),
                    caption=str(result.get("prompt") or result.get("caption") or ""),
                    genre=str(result.get("genre") or result.get("genres") or ""),
                    lyrics=lyrics,
                    bpm=bpm_decision.bpm,
                    bpm_source=bpm_decision.source,
                    lm_bpm=bpm_decision.lm_bpm,
                    local_bpm=bpm_decision.local_bpm,
                    filename_bpm=bpm_decision.filename_bpm,
                    keyscale=key_decision.keyscale,
                    key_source=key_decision.source,
                    lm_keyscale=key_decision.lm_keyscale,
                    local_keyscale=key_decision.local_keyscale,
                    timesignature=auto_timesignature(args, result),
                    language=language,
                    is_instrumental=is_instrumental,
                    custom_tag=args.trigger,
                )
                print(
                    f"[captioning] {index}/{len(pending)} {audio_path.name} "
                    f"bpm={bpm_decision.bpm or 'n/a'} ({bpm_decision.source}) "
                    f"key={key_decision.keyscale or 'n/a'} ({key_decision.source})",
                    flush=True,
                )
    finally:
        stop_caption_server(args, caption_server)
    return len(pending)


def request_music_analysis(
    args: argparse.Namespace,
    client: Any,
    audio_path: Path,
    *,
    caption_window_seconds: float | None = None,
) -> dict[str, Any]:
    prepared = prepare_caption_audio(
        args,
        audio_path,
        caption_window_seconds=caption_window_seconds,
    )
    upload_name = (
        f"{audio_path.stem}_caption_excerpt.wav"
        if prepared.cleanup_path
        else audio_path.name
    )
    if prepared.cleanup_path:
        print(
            f"[captioning] Using {prepared.duration:.1f}s excerpt from "
            f"{prepared.offset:.1f}s for {audio_path.name}",
            flush=True,
        )

    try:
        mime = mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        with prepared.path.open("rb") as audio_file:
            response = client.post(
                f"{args.carey_url}/release_task",
                files={"ctx_audio": (upload_name, audio_file, mime)},
                data={
                    "full_analysis_only": "true",
                    "thinking": "false",
                    "audio_duration": str(
                        resolve_music_analysis_duration(
                            args,
                            audio_path,
                            prepared_duration=prepared.duration,
                        )
                    ),
                    "lm_backend": "pt",
                    "lm_model_path": args.caption_lm_model,
                },
            )
        response.raise_for_status()
        task_id = response.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Carey did not return a task id for {audio_path.name}")

        started_at = time.monotonic()
        last_status_update = 0.0
        deadline = started_at + args.caption_timeout
        while time.monotonic() < deadline:
            check_cancel(args)
            query = client.post(
                f"{args.carey_url}/query_result",
                json={"task_id_list": [task_id]},
            )
            query.raise_for_status()
            records = query.json().get("data") or []
            if records:
                record = records[0]
                status = int(record.get("status", 0))
                if status == 1:
                    return normalize_analysis_result(record.get("result"))
                if status == 2:
                    raise RuntimeError(
                        f"Carey analysis failed for {audio_path.name}: "
                        f"{record.get('error') or record.get('result') or 'unknown error'}"
                    )
            now = time.monotonic()
            if now - last_status_update >= 10:
                update_status(
                    args,
                    status="running",
                    phase="captioning",
                    message=(
                        f"Analyzing {audio_path.name} with {args.caption_lm_model} "
                        f"({int(now - started_at)}s)"
                    ),
                )
                last_status_update = now
            time.sleep(2)
        raise RuntimeError(f"Carey analysis timed out for {audio_path.name}")
    finally:
        if prepared.cleanup_path:
            try:
                prepared.cleanup_path.unlink()
            except OSError:
                pass


def request_valid_music_analysis(
    args: argparse.Namespace,
    client: Any,
    audio_path: Path,
) -> dict[str, Any]:
    primary_window = float(getattr(args, "caption_window_seconds", 0.0) or 0.0)
    fallback_window = float(
        getattr(args, "caption_fallback_window_seconds", 120.0) or 0.0
    )
    attempts: list[float | None] = [primary_window]
    if primary_window <= 0 and fallback_window > 0:
        attempts.append(fallback_window)

    last_error: RuntimeError | None = None
    for index, window in enumerate(attempts):
        result = request_music_analysis(
            args,
            client,
            audio_path,
            caption_window_seconds=window,
        )
        try:
            validate_caption_analysis_result(audio_path, result)
            return result
        except RuntimeError as exc:
            last_error = exc
            if index + 1 >= len(attempts):
                raise
            print(
                f"[captioning] Full-track metadata failed quality checks for "
                f"{audio_path.name}; retrying with "
                f"{fallback_window:.0f}s excerpt. Reason: {exc}",
                flush=True,
            )

    raise last_error or RuntimeError(f"Carey analysis failed for {audio_path.name}")


def resolve_music_analysis_duration(
    args: argparse.Namespace,
    audio_path: Path,
    *,
    prepared_duration: float | None = None,
) -> float:
    """Return the duration hint to send with ACE understand_music requests.

    A positive CLI value is an explicit override. The Gary UI sends 0 so the
    captioner gets the actual source duration for each track.
    """
    explicit = float(getattr(args, "analysis_duration", 0.0) or 0.0)
    if explicit > 0:
        return max(1.0, explicit)

    if prepared_duration is not None and prepared_duration > 0:
        return max(1.0, float(prepared_duration))

    actual = audio_duration_seconds(audio_path)
    return max(1.0, float(actual or 0.0))


def normalize_analysis_result(value: Any) -> dict[str, Any]:
    while isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise RuntimeError(f"Carey returned an invalid analysis result: {value[:200]}")
    if isinstance(value, list):
        value = value[0] if value else {}
    if isinstance(value, dict) and isinstance(value.get("result"), (dict, list, str)):
        return normalize_analysis_result(value["result"])
    if not isinstance(value, dict):
        raise RuntimeError("Carey returned an empty analysis result")
    if value.get("error"):
        raise RuntimeError(str(value["error"]))
    return value


def caption_text_quality_error(value: Any, *, field: str) -> str | None:
    text = str(value or "")
    if not text.strip():
        return f"{field} is empty"
    if "<|audio_code_" in text:
        return f"{field} contains raw ACE audio-code tokens"
    if "\ufffd" in text or "ï¿½" in text:
        return f"{field} contains invalid replacement characters"
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 24 and len(set(compact)) <= 3:
        return f"{field} is dominated by repeated characters"
    if re.search(r"([!?._=-])\1{20,}", compact):
        return f"{field} is dominated by repeated punctuation"
    if field == "caption":
        alpha_count = sum(1 for char in text if char.isalpha())
        if len(text.strip()) >= 80 and alpha_count / max(1, len(text)) < 0.15:
            return f"{field} has too little word content"
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
        if len(text.strip()) >= 40 and len(words) < 3:
            return f"{field} has too little descriptive text"
    if field == "caption" and re.search(
        r"(?:^|[^a-z])(?:bpm|duration|genres?|keyscale|language|timesignature)\s*:",
        text,
        re.IGNORECASE,
    ):
        return f"{field} appears to contain embedded metadata fields"
    return None


def validate_caption_analysis_result(
    audio_path: Path,
    result: dict[str, Any],
    *,
    require_caption: bool = True,
) -> None:
    caption = result.get("prompt") or result.get("caption")
    genre = result.get("genre") or result.get("genres")
    lyrics = result.get("lyrics")
    checks = (
        ("caption", caption),
        ("genre", genre),
        ("lyrics", lyrics),
    )
    for field, value in checks:
        if field == "caption" and not require_caption and not str(value or "").strip():
            continue
        if field != "caption" and not str(value or "").strip():
            continue
        reason = caption_text_quality_error(value, field=field)
        if reason:
            raise RuntimeError(
                f"ACE understand_music returned unusable metadata for "
                f"{audio_path.name}: {reason}. Re-run with a smaller LM, "
                "overwrite captions, or edit the sidecar manually."
            )


def validate_dataset_sidecars(dataset_dir: Path) -> None:
    bad: list[str] = []
    for audio_path in discover_audio_files(dataset_dir):
        meta = load_sidecar_metadata(audio_path)
        if not meta:
            continue
        try:
            validate_caption_analysis_result(audio_path, meta, require_caption=False)
        except RuntimeError as exc:
            bad.append(str(exc))
    if bad:
        preview = "\n".join(f"- {item}" for item in bad[:5])
        extra = "\n..." if len(bad) > 5 else ""
        raise RuntimeError(
            "One or more ACE sidecars look corrupted and were not used to "
            f"build dataset.json:\n{preview}{extra}"
        )


def analysis_is_instrumental(
    result: dict[str, Any],
    *,
    default: bool = False,
) -> bool:
    explicit = result.get("is_instrumental")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower() in {"1", "true", "yes", "y", "on"}
    lyrics = str(result.get("lyrics") or "").strip()
    return (
        default
        or not lyrics
        or "[instrumental]" in lyrics.lower()
        or lyrics_are_structural_only(lyrics)
    )


def lyrics_are_structural_only(lyrics: str) -> bool:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return True
    return all(re.fullmatch(r"\[[^\]]+\]", line) for line in lines)


def decide_sidecar_bpm(
    args: argparse.Namespace,
    audio_path: Path,
    result: dict[str, Any],
) -> Any:
    filename_bpm = _filename_bpm(audio_path.name)
    local_estimate = None
    if getattr(args, "bpm_analysis", True):
        try:
            local_estimate = estimate_bpm(audio_path)
        except Exception as exc:
            print(f"[captioning] Local BPM analysis failed for {audio_path.name}: {exc}", flush=True)
    return choose_bpm(
        filename_bpm=filename_bpm,
        lm_bpm=result.get("bpm"),
        local_estimate=local_estimate,
        disagreement_threshold=getattr(args, "bpm_disagreement_threshold", 5.0),
        minimum_local_confidence=getattr(args, "bpm_min_confidence", 1.2),
    )


def decide_sidecar_key(
    args: argparse.Namespace,
    audio_path: Path,
    result: dict[str, Any],
) -> Any:
    local_estimate = None
    if getattr(args, "key_analysis", True):
        try:
            local_estimate = estimate_key(audio_path)
        except Exception as exc:
            print(f"[captioning] Local key analysis failed for {audio_path.name}: {exc}", flush=True)
    return choose_key(
        lm_keyscale=result.get("keyscale") or result.get("key_scale"),
        local_estimate=local_estimate,
        minimum_local_confidence=getattr(args, "key_min_confidence", 0.15),
    )


def ensure_carey_stopped(args: argparse.Namespace) -> None:
    """Wait for Tauri (or the user) to release the managed Carey process."""
    import httpx

    deadline = time.monotonic() + args.carey_stop_timeout
    announced = False
    urls = [getattr(args, "inference_carey_url", "http://127.0.0.1:8003")]
    if args.carey_url not in urls:
        urls.append(args.carey_url)
    while time.monotonic() < deadline:
        check_cancel(args)
        running_urls = []
        for url in urls:
            try:
                response = httpx.get(f"{url}/health", timeout=2)
                if response.is_success:
                    running_urls.append(url)
            except Exception:
                pass
        if not running_urls:
            return
        if not announced:
            update_status(
                args,
                status="running",
                phase="waiting-for-carey-stop",
                message=(
                    "Waiting for Carey inference processes to stop and release CUDA memory"
                ),
            )
            announced = True
        time.sleep(2)
    raise RuntimeError(
        "Carey is still running. Stop it through gary4local before "
        "preprocessing or training so the full CUDA context is released."
    )


def write_plan(
    args: argparse.Namespace,
    dataset_json: Path,
    tensors_dir: Path,
    output_dir: Path,
) -> Path:
    plan_path = args.run_dir / "training_plan.json"
    write_json(
        plan_path,
        {
            "model": args.model,
            "modelFamily": MODEL_MAP[args.model]["family"],
            "datasetJson": str(dataset_json),
            "tensorsDir": str(tensors_dir),
            "outputDir": str(output_dir),
            "fisher": False,
            "adapterType": args.adapter_type,
            "moduleProfile": args.module_profile,
            "timestepMu": resolve_timestep_mu(args),
            "saveBest": args.save_best,
            "saveBestAfter": args.save_best_after,
            "vramPreflight": True,
            "optimizer": "adamw",
            "quantization": "disabled",
            "preprocessCommand": build_preprocess_command(
                args, dataset_json, tensors_dir, output_dir
            ),
            "trainCommand": build_train_command(args, tensors_dir, output_dir),
        },
    )
    return plan_path


def _parse_caption_and_genre(sidecar: Path) -> tuple[str | None, str | None]:
    caption = None
    genre = None
    try:
        for line in sidecar.read_text(encoding="utf-8-sig").splitlines():
            lower = line.lower()
            if caption is None and lower.startswith("caption:"):
                caption = line.split(":", 1)[1].strip() or None
            elif genre is None and lower.startswith("genre:"):
                genre = line.split(":", 1)[1].strip() or None
            if caption and genre:
                break
    except OSError:
        pass
    return caption, genre


def _genre_variants(genre: str) -> list[str]:
    tokens = [token.strip() for token in genre.split(",") if token.strip()]
    if not tokens:
        return []
    variants = [", ".join(tokens)]
    for size in (2, 3):
        if len(tokens) >= size:
            variants.extend(", ".join(combo) for combo in combinations(tokens, size))
    return variants


def collect_caption_pool(dataset_dir: Path) -> list[str]:
    seen: set[str] = set()
    pool: list[str] = []
    for sidecar in sorted(dataset_dir.glob("*.txt")):
        if ".v4bak" in sidecar.name or sidecar.name.endswith(".v4bak"):
            continue
        caption, genre = _parse_caption_and_genre(sidecar)
        entries: list[str] = []
        if caption:
            entries.append(caption)
        if genre:
            entries.extend(_genre_variants(genre))
        for entry in entries:
            if entry not in seen:
                seen.add(entry)
                pool.append(entry)
    return pool


def register_trained_lora(args: argparse.Namespace, final_checkpoint: Path) -> None:
    family = MODEL_MAP[args.model]["family"]
    metadata = {
        "path": str(final_checkpoint),
        "captionsPath": str(args.dataset_dir),
        "scale": 1.0,
        "backends": ["base", "turbo"],
        "modelFamily": family,
        "adapterType": args.adapter_type,
        "moduleProfile": args.module_profile,
        "timestepMu": resolve_timestep_mu(args),
    }

    if args.lora_catalog_path:
        catalog = read_json(args.lora_catalog_path, {})
        if not isinstance(catalog, dict):
            catalog = {}
        catalog[args.name] = metadata
        write_json(args.lora_catalog_path, catalog)

    if args.lora_registry_path:
        registry = read_json(args.lora_registry_path, {})
        if not isinstance(registry, dict):
            registry = {}
        registry[args.name] = {
            "path": str(final_checkpoint),
            "scale": 1.0,
            "backends": ["base", "turbo"],
            "model_family": family,
            "adapter_type": args.adapter_type,
            "module_profile": args.module_profile,
            "timestep_mu": resolve_timestep_mu(args),
        }
        write_json(args.lora_registry_path, registry)

    if args.captions_json_path:
        pools = read_json(args.captions_json_path, {})
        if not isinstance(pools, dict):
            pools = {}
        pools[args.name] = collect_caption_pool(args.dataset_dir)
        write_json(args.captions_json_path, pools)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--current-job-path", type=Path, required=True)
    parser.add_argument("--cancel-path", type=Path)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_MAP, default="base")
    parser.add_argument("--instrumental", action="store_true")
    parser.add_argument("--trigger", default="")
    parser.add_argument(
        "--tag-position",
        choices=("prepend", "append", "replace"),
        default="prepend",
    )
    parser.add_argument("--genre-ratio", type=int, default=20)
    parser.add_argument(
        "--caption",
        choices=("understand_music", "skip"),
        default="skip",
    )
    parser.add_argument(
        "--carey-url",
        default="http://127.0.0.1:8013",
        help="Temporary ACE api_server URL used only for understand_music captioning.",
    )
    parser.add_argument(
        "--inference-carey-url",
        default="http://127.0.0.1:8003",
        help="Normal Carey inference service URL that must be stopped before training.",
    )
    parser.add_argument("--caption-startup-timeout", type=float, default=900.0)
    parser.add_argument("--caption-timeout", type=float, default=900.0)
    parser.add_argument(
        "--caption-window-seconds",
        type=float,
        default=0.0,
        help="Maximum audio excerpt sent to the LM captioner; 0 sends the full file.",
    )
    parser.add_argument(
        "--caption-fallback-window-seconds",
        type=float,
        default=120.0,
        help=(
            "Center excerpt length used when full-track understand_music output "
            "fails quality checks; 0 disables fallback."
        ),
    )
    parser.add_argument(
        "--caption-lm-model",
        choices=CAPTION_LM_MODELS,
        default="acestep-5Hz-lm-0.6B",
        help="ACE-Step understand_music LM; larger models improve captions but need more VRAM",
    )
    parser.add_argument("--model-load-timeout", type=float, default=900.0)
    parser.add_argument("--carey-stop-timeout", type=float, default=180.0)
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=0.0,
        help="Duration hint for understand_music; 0 uses each source track duration.",
    )
    parser.add_argument(
        "--no-bpm-analysis",
        action="store_false",
        dest="bpm_analysis",
        help="Do not override understand_music BPM with local tempo analysis",
    )
    parser.add_argument("--bpm-disagreement-threshold", type=float, default=5.0)
    parser.add_argument("--bpm-min-confidence", type=float, default=1.2)
    parser.add_argument(
        "--include-auto-timesignature",
        action="store_true",
        help="Write understand_music time signatures instead of leaving them editable/blank",
    )
    parser.add_argument(
        "--no-key-analysis",
        action="store_false",
        dest="key_analysis",
        help="Do not override understand_music key with local chroma analysis",
    )
    parser.add_argument("--key-min-confidence", type=float, default=0.15)
    parser.add_argument("--overwrite-captions", action="store_true")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument(
        "--module-profile",
        choices=("attention", "balanced"),
        default="balanced",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cfg-ratio", type=float, default=0.15)
    parser.add_argument(
        "--timestep-mu",
        type=float,
        default=None,
        help="Advanced training schedule override. Default: -0.4.",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("none", "min_snr"),
        default="min_snr",
    )
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-best-after", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--adapter-type", choices=("lora", "dora"), default="dora")
    parser.add_argument("--fisher", action="store_true")
    parser.add_argument("--lora-catalog-path", type=Path)
    parser.add_argument("--lora-registry-path", type=Path)
    parser.add_argument("--captions-json-path", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.rank <= 0 or args.alpha <= 0:
        raise ValueError("rank and alpha must be greater than zero")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("batch size and gradient accumulation must be greater than zero")
    if args.epochs <= 0 or args.save_every <= 0:
        raise ValueError("epochs and save interval must be greater than zero")
    if args.save_best_after <= 0:
        raise ValueError("best-checkpoint start epoch must be greater than zero")
    if not 0 <= args.genre_ratio <= 100:
        raise ValueError("genre ratio must be between 0 and 100")
    if not 0 <= args.cfg_ratio < 1:
        raise ValueError("cfg ratio must be in [0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be greater than zero")
    if args.snr_gamma <= 0:
        raise ValueError("SNR gamma must be greater than zero")
    if not math.isfinite(resolve_timestep_mu(args)):
        raise ValueError("timestep mu must be finite")
    if args.fisher:
        raise RuntimeError(
            "Fisher/Preprocessing++ is not enabled in this trainer slice yet. "
            "A true adaptive rank map must be integrated from the pinned MIT "
            "Side-Step snapshot before this option can be used."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.name = slugify(args.name)
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.cancel_path = args.cancel_path or (args.run_dir / "cancel.requested")

    dataset_json = args.run_dir / "dataset.json"
    output_dir = args.run_dir / "output"
    tensors_dir = output_dir / "tensors"
    final_checkpoint = output_dir / "final"
    best_checkpoint = output_dir / "best"

    try:
        validate_args(args)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if args.cancel_path.exists():
            args.cancel_path.unlink()
        update_status(
            args,
            status="running",
            phase="starting",
            message="Starting ACE-Step LoRA training",
            error=None,
            childPid=None,
            finalCheckpointPath=None,
            resultPath=None,
            captionLmModel=(
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            adapterType=args.adapter_type,
        )

        # Caption/prepare and training share the Carey environment. Repair
        # safe missing packages before either path imports the ML stack.
        ensure_job_dependencies(args)

        captioned = 0
        if args.caption == "understand_music" and not args.dry_run:
            captioned = caption_with_understand_music(args)

        update_status(
            args,
            status="running",
            phase="building-dataset",
            message="Building ACE-Step dataset metadata",
        )
        validate_dataset_sidecars(args.dataset_dir)
        dataset_result = build_dataset_json(
            args.dataset_dir,
            dataset_json,
            name=args.name,
            trigger=args.trigger,
            tag_position=args.tag_position,
            genre_ratio=args.genre_ratio,
            instrumental_default=args.instrumental,
        )
        plan_path = write_plan(args, dataset_json, tensors_dir, output_dir)

        if args.prepare_only or args.dry_run:
            prepare_message = "ACE-Step training dataset prepared"
            caption_skip_message = getattr(args, "_caption_skip_message", "")
            if caption_skip_message:
                prepare_message = (
                    "ACE-Step training dataset prepared from existing sidecars. "
                    f"{caption_skip_message}"
                )
            update_status(
                args,
                status="completed",
                phase="prepared",
                message=prepare_message,
                datasetJsonPath=str(dataset_json),
                trainingPlanPath=str(plan_path),
                sampleCount=dataset_result["samples"],
                captionedCount=captioned,
                captionLmModel=(
                    args.caption_lm_model if args.caption == "understand_music" else None
                ),
                modelFamily=MODEL_MAP[args.model]["family"],
                error=None,
                childPid=None,
            )
            return 0

        ensure_carey_stopped(args)
        require_training_environment(args)
        require_model_checkpoint(args)
        tensors_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_step(
            args,
            build_preprocess_command(args, dataset_json, tensors_dir, output_dir),
            "preprocessing",
            "Pre-encoding audio with Carey's two-pass pipeline",
        )
        if not any(tensors_dir.glob("*.pt")):
            raise RuntimeError("Preprocessing completed without producing training tensors")

        run_step(
            args,
            build_train_command(args, tensors_dir, output_dir),
            "training",
            "Training ACE-Step LoRA",
            cwd=output_dir,
        )
        if not is_complete_peft_adapter(final_checkpoint):
            raise RuntimeError(
                f"Training finished without a complete PEFT adapter in {final_checkpoint}"
            )

        selected_checkpoint = (
            best_checkpoint
            if args.save_best and is_complete_peft_adapter(best_checkpoint)
            else final_checkpoint
        )

        result_path = args.run_dir / "result.json"
        result = {
            "jobId": args.job_id,
            "name": args.name,
            # Kept for UI compatibility: this is the checkpoint selected for
            # registration, normally best/ rather than the final epoch.
            "finalCheckpointPath": str(selected_checkpoint),
            "bestCheckpointPath": (
                str(best_checkpoint) if is_complete_peft_adapter(best_checkpoint) else None
            ),
            "lastEpochCheckpointPath": str(final_checkpoint),
            "captionsPath": str(args.dataset_dir),
            "modelFamily": MODEL_MAP[args.model]["family"],
            "adapterType": args.adapter_type,
            "moduleProfile": args.module_profile,
            "timestepMu": resolve_timestep_mu(args),
            "checkpointSelection": (
                "best_ma5" if selected_checkpoint == best_checkpoint else "final_epoch"
            ),
            "captionLmModel": (
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            "backends": ["base", "turbo"],
            "scale": 1.0,
        }
        write_json(selected_checkpoint / "metadata.json", result)
        write_json(result_path, result)
        register_trained_lora(args, selected_checkpoint)
        update_status(
            args,
            status="completed",
            phase="completed",
            message="ACE-Step LoRA training complete",
            finalCheckpointPath=str(selected_checkpoint),
            captionsPath=str(args.dataset_dir),
            resultPath=str(result_path),
            datasetJsonPath=str(dataset_json),
            sampleCount=dataset_result["samples"],
            # Caption counts belong to the caption/prepare result. A completed
            # training run should describe training, not report that it
            # intentionally captioned zero tracks.
            captionedCount=None,
            captionLmModel=(
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            modelFamily=MODEL_MAP[args.model]["family"],
            adapterType=args.adapter_type,
            registeredLoraName=args.name,
            error=None,
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
        update_status(
            args,
            status="failed",
            phase="failed",
            message=str(exc),
            error=str(exc),
            childPid=None,
        )
        raise


def _filename_bpm(filename: str) -> int | None:
    match = re.search(r"(?:^|[_-])bpm[_-]?(\d{2,3})(?:[_-]|\.|$)", filename, re.I)
    if not match:
        return None
    bpm = int(match.group(1))
    return bpm if 1 <= bpm <= 400 else None


def auto_timesignature(args: argparse.Namespace, result: dict[str, Any]) -> str:
    if not getattr(args, "include_auto_timesignature", False):
        return ""
    return str(result.get("timesignature") or result.get("time_signature") or "")


if __name__ == "__main__":
    raise SystemExit(main())
