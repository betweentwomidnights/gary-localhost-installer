#!/usr/bin/env python3
"""Endpoint-level ACE-Step caption and LoRA training smoke test.

The harness owns a dedicated ``api_server.py`` process on an isolated port.
It loads the base model with sequential CPU offload, runs
``full_analysis_only`` using the selected LM, terminates the complete process tree,
then optionally invokes the Gary-native training job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CAREY_DIR = REPO_ROOT / "services" / "carey"
CAREY_PYTHON = CAREY_DIR / "env" / "Scripts" / "python.exe"
API_SERVER = CAREY_DIR / "acestep" / "api_server.py"
TRAIN_JOB = CAREY_DIR / "train_lora_job.py"
CHECKPOINT_DIR = CAREY_DIR / "checkpoints"
DEFAULT_AUDIO = REPO_ROOT / "keygen_music_for_installer.wav"

sys.path.insert(0, str(CAREY_DIR))

from ace_training_dataset import write_canonical_sidecar  # noqa: E402
from train_lora_job import (  # noqa: E402
    CAPTION_LM_MODELS,
    analysis_is_instrumental,
    normalize_analysis_result,
)
from key_analysis import normalize_keyscale  # noqa: E402


class VramSampler:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples_mib: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def peak_mib(self) -> int:
        return max(self.samples_mib, default=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = current_vram_mib()
            if value is not None:
                self.samples_mib.append(value)
            self._stop.wait(self.interval)


def current_vram_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def make_clip(source: Path, destination: Path, seconds: float) -> None:
    import soundfile as sf

    audio, sample_rate = sf.read(str(source), always_2d=True)
    frame_count = min(len(audio), max(1, int(sample_rate * seconds)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio[:frame_count], sample_rate, subtype="PCM_16")


def wait_for_health(client: Any, base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not reachable"
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{base_url}/health", timeout=5)
            if response.is_success:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"caption backend did not become ready: {last_error}")


def start_caption_backend(args: argparse.Namespace, log_path: Path) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(CAREY_DIR),
            "ACESTEP_CONFIG_PATH": "acestep-v15-base",
            "ACESTEP_DEVICE": "cuda",
            "ACESTEP_NO_INIT": "true",
            "ACESTEP_INIT_LLM": "true",
            "ACESTEP_LM_MODEL_PATH": args.lm_model,
            "ACESTEP_LM_BACKEND": "pt",
            "ACESTEP_LM_DEVICE": "cuda",
            "ACESTEP_LM_OFFLOAD_TO_CPU": "true",
            "ACESTEP_OFFLOAD_TO_CPU": "true",
            "ACESTEP_OFFLOAD_DIT_TO_CPU": "true",
            "ACESTEP_UNDERSTAND_MAX_NEW_TOKENS": "1024",
            "ACESTEP_USE_FLASH_ATTENTION": "false",
            "ACESTEP_COMPILE_MODEL": "false",
            "ACESTEP_API_WORKERS": "1",
        }
    )
    env.pop("ACESTEP_QUANTIZATION", None)
    command = [
        str(CAREY_PYTHON),
        "-u",
        str(API_SERVER),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-init",
        "--init-llm",
        "--lm-model-path",
        args.lm_model,
    ]
    proc = subprocess.Popen(
        command,
        cwd=str(CAREY_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return proc, log_file


def run_analysis(args: argparse.Namespace, audio_path: Path) -> dict[str, Any]:
    import httpx

    base_url = f"http://127.0.0.1:{args.port}"
    log_path = args.output_dir / "caption_backend.log"
    proc, log_file = start_caption_backend(args, log_path)
    sampler = VramSampler()
    baseline_mib = current_vram_mib()
    sampler.start()
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
            health = wait_for_health(client, base_url, args.startup_timeout)
            print(f"[carey-smoke] health={json.dumps(health, sort_keys=True)}")

            load = client.post(
                f"{base_url}/v1/load",
                params={"config_path": "acestep-v15-base"},
                timeout=args.model_load_timeout,
            )
            load.raise_for_status()
            print(f"[carey-smoke] load={json.dumps(load.json(), sort_keys=True)}")

            with audio_path.open("rb") as audio_file:
                accepted = client.post(
                    f"{base_url}/release_task",
                    files={"ctx_audio": (audio_path.name, audio_file, "audio/wav")},
                    data={
                        "full_analysis_only": "true",
                        "thinking": "false",
                        "audio_duration": str(args.clip_seconds),
                        "lm_backend": "pt",
                        "lm_model_path": args.lm_model,
                    },
                )
            accepted.raise_for_status()
            task_id = accepted.json().get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"analysis did not return a task id: {accepted.text}")

            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                query = client.post(
                    f"{base_url}/query_result",
                    json={"task_id_list": [task_id]},
                )
                query.raise_for_status()
                records = query.json().get("data") or []
                if records:
                    record = records[0]
                    status = int(record.get("status", 0))
                    if status == 1:
                        result = normalize_analysis_result(record.get("result"))
                        break
                    if status == 2:
                        raise RuntimeError(
                            f"analysis failed: {record.get('error') or record.get('result')}"
                        )
                    print(
                        f"[carey-smoke] analysis status={status} "
                        f"progress={record.get('progress_text', '')}",
                        flush=True,
                    )
                time.sleep(args.poll_interval)
            else:
                raise TimeoutError(f"analysis timed out after {args.timeout}s")

        elapsed = time.perf_counter() - started
        return {
            "result": result,
            "lmModel": args.lm_model,
            "elapsedSeconds": round(elapsed, 3),
            "baselineVramMiB": baseline_mib,
            "peakVramMiB": sampler.peak_mib,
            "backendLog": str(log_path),
        }
    finally:
        sampler.stop()
        terminate_process_tree(proc)
        log_file.close()


def wait_for_vram_release(target_mib: int, timeout: float = 90.0) -> int | None:
    deadline = time.monotonic() + timeout
    last = current_vram_mib()
    while time.monotonic() < deadline:
        last = current_vram_mib()
        if last is not None and last <= target_mib:
            return last
        time.sleep(1)
    return last


def run_training_job(args: argparse.Namespace, dataset_dir: Path) -> int:
    run_dir = args.output_dir / "training-run"
    log_path = args.output_dir / "training_job.log"
    command = [
        str(CAREY_PYTHON),
        "-u",
        str(TRAIN_JOB),
        "--job-id",
        "carey-smoke",
        "--name",
        "carey-smoke",
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint-dir",
        str(CHECKPOINT_DIR),
        "--run-dir",
        str(run_dir),
        "--status-path",
        str(run_dir / "status.json"),
        "--current-job-path",
        str(args.output_dir / "current_job.json"),
        "--log-path",
        str(log_path),
        "--caption",
        "skip",
        "--instrumental",
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.rank * 2),
        "--epochs",
        str(args.epochs),
        "--save-every",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "1",
    ]
    if args.phase == "prepare":
        command.append("--prepare-only")
    print(f"[carey-smoke] training command: {subprocess.list2cmdline(command)}")
    return subprocess.run(command, cwd=str(CAREY_DIR), check=False).returncode


def validate_checkpoints(args: argparse.Namespace) -> None:
    required = [
        CHECKPOINT_DIR / "acestep-v15-base" / "model.safetensors",
        CHECKPOINT_DIR / "acestep-v15-base" / "silence_latent.pt",
        CHECKPOINT_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        CHECKPOINT_DIR / "Qwen3-Embedding-0.6B" / "model.safetensors",
        CHECKPOINT_DIR / args.lm_model / "model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Carey checkpoints:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("caption", "prepare", "full"), default="caption")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "carey-training-smoke",
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--model-load-timeout", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--lm-model",
        choices=CAPTION_LM_MODELS,
        default="acestep-5Hz-lm-0.6B",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    validate_checkpoints(args)
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = (args.output_dir / "dataset").resolve()
    if args.output_dir not in dataset_dir.parents:
        raise RuntimeError(
            f"Refusing to replace dataset outside {args.output_dir}: {dataset_dir}"
        )
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    clip_path = dataset_dir / "smoke_clip.wav"
    make_clip(args.audio, clip_path, args.clip_seconds)

    if args.phase in {"caption", "full"}:
        analysis = run_analysis(args, clip_path)
        result = analysis["result"]
        write_canonical_sidecar(
            clip_path.with_suffix(".txt"),
            caption=str(result.get("prompt") or result.get("caption") or ""),
            genre=str(result.get("genre") or result.get("genres") or ""),
            lyrics=str(result.get("lyrics") or ""),
            bpm=result.get("bpm"),
            keyscale=normalize_keyscale(result.get("keyscale")),
            timesignature=str(result.get("timesignature") or ""),
            language=str(result.get("language") or ""),
            is_instrumental=analysis_is_instrumental(result),
        )
        released_mib = wait_for_vram_release(
            max(256, int(analysis.get("baselineVramMiB") or 0) + 128)
        )
        analysis["releasedVramMiB"] = released_mib
        result_path = args.output_dir / "analysis_result.json"
        result_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        print(
            f"[carey-smoke] analysis complete in {analysis['elapsedSeconds']}s; "
            f"peak={analysis['peakVramMiB']} MiB; released={released_mib} MiB"
        )
        print(f"[carey-smoke] result={result_path}")

    if args.phase in {"prepare", "full"}:
        code = run_training_job(args, dataset_dir)
        if code != 0:
            raise RuntimeError(f"training job failed with exit code {code}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[carey-smoke] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
