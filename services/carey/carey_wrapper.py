"""
ACE-Step wrapper service — lego (stem) + complete (continuation) + cover (remix) modes.

Localhost version: launches api_server.py as a subprocess, then serves the
JUCE-compatible wrapper API on port 8003.

Async submit/poll architecture matching the existing JUCE frontend pattern.

Endpoints:
  POST /lego                     Submit a lego stem job -> returns task_id
  GET  /lego/status/{task_id}    Poll lego progress/completion

  POST /complete                 Submit a continuation job -> returns task_id
  GET  /complete/status/{task_id} Poll continuation progress/completion

  POST /cover                    Submit a cover/remix job -> returns task_id
  GET  /cover/status/{task_id}   Poll cover progress/completion

  GET  /health                   Wrapper + backend health
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACESTEP_PORT = int(os.getenv("ACESTEP_PORT", "8001"))
ACESTEP_URL = os.getenv("ACESTEP_URL", f"http://localhost:{ACESTEP_PORT}").rstrip("/")
WRAPPER_PORT = int(os.getenv("WRAPPER_PORT", "8003"))
MAX_CONCURRENT = int(os.getenv("ACESTEP_MAX_CONCURRENT", "1"))
API_KEY = os.getenv("ACESTEP_API_KEY", "")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_STARTUP_CONFIG = (os.getenv("ACESTEP_CONFIG_PATH") or "acestep-v15-base").strip()
ACESTEP_BASE_CONFIG = (os.getenv("ACESTEP_BASE_CONFIG_PATH") or DEFAULT_STARTUP_CONFIG).strip()
ACESTEP_TURBO_CONFIG = (os.getenv("ACESTEP_TURBO_CONFIG_PATH") or "acestep-v15-turbo").strip()
MANAGE_MODEL_LIFECYCLE = _env_bool("ACESTEP_MANAGE_MODEL_LIFECYCLE", True)
BACKEND_STARTS_LOADED = not _env_bool("ACESTEP_NO_INIT", False)
EFFECTIVE_MAX_CONCURRENT = 1 if MANAGE_MODEL_LIFECYCLE else MAX_CONCURRENT

# Generation constants
INFERENCE_STEPS = 50
POLL_INTERVAL = 1.5
GENERATION_TIMEOUT = int(os.getenv("CAREY_GENERATION_TIMEOUT", "600"))
JOB_TTL = 3600
COVER_INFERENCE_STEPS = 8
COVER_GUIDANCE_SCALE = 1.0

# Default captions per track type (lego mode only)
TRACK_CAPTIONS = {
    "vocals":         "soulful indie vocalist, warm, wordless melody, expressive, intimate",
    "backing_vocals": "background vocals, close harmony, wordless, warm, following the lead vocal",
    "drums":          "live acoustic drum kit, tight kick and snare, brushed hi-hats, warm",
    "bass":           "electric bass, warm fingerstyle, rhythmic, supportive",
    "guitar":         "acoustic guitar, fingerpicked, warm, rhythmic",
    "piano":          "piano, expressive, warm, melodic",
    "strings":        "string ensemble, lush, warm, cinematic",
    "synth":          "analog synth pad, warm, atmospheric",
    "keyboard":       "electric piano, warm, smooth",
    "percussion":     "percussion, shaker, tambourine, tight groove",
    "brass":          "brass section, warm, expressive",
    "woodwinds":      "woodwind ensemble, warm, airy, melodic",
}

ALLOWED_TRACKS = set(TRACK_CAPTIONS.keys())


# ---------------------------------------------------------------------------
# Subprocess: ace-step api_server
# ---------------------------------------------------------------------------

_backend_process: Optional[subprocess.Popen] = None
_current_model: Optional[str] = DEFAULT_STARTUP_CONFIG if BACKEND_STARTS_LOADED else None


def _sync_checkpoint_overrides(script_dir: Optional[Path] = None) -> list[Path]:
    """Copy bundled checkpoint code overrides into existing runtime checkpoint dirs."""
    root = script_dir or Path(__file__).parent
    overrides_root = root / "checkpoint_overrides"
    checkpoints_root = root / "checkpoints"
    synced: list[Path] = []

    if not overrides_root.exists() or not checkpoints_root.exists():
        return synced

    for model_dir in sorted(overrides_root.iterdir()):
        if not model_dir.is_dir():
            continue
        target_dir = checkpoints_root / model_dir.name
        if not target_dir.exists():
            continue
        for src in sorted(model_dir.glob("*.py")):
            dst = target_dir / src.name
            if dst.exists() and src.read_bytes() == dst.read_bytes():
                continue
            shutil.copy2(src, dst)
            synced.append(dst)

    return synced


def _start_backend():
    """Launch acestep/api_server.py as a subprocess."""
    global _backend_process
    script_dir = Path(__file__).parent
    api_server = script_dir / "acestep" / "api_server.py"
    synced_overrides = _sync_checkpoint_overrides(script_dir)
    if synced_overrides:
        synced_names = ", ".join(path.parent.name + "/" + path.name for path in synced_overrides)
        print(f"[wrapper] Synced checkpoint overrides: {synced_names}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    python_exe = sys.executable
    _backend_process = subprocess.Popen(
        [python_exe, str(api_server)],
        cwd=str(script_dir),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"[wrapper] Started api_server.py (pid={_backend_process.pid}) on port {ACESTEP_PORT}")


def _stop_backend():
    """Stop the api_server subprocess."""
    global _backend_process
    if _backend_process and _backend_process.poll() is None:
        print(f"[wrapper] Stopping api_server.py (pid={_backend_process.pid})...")
        _backend_process.terminate()
        try:
            _backend_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _backend_process.kill()
        print("[wrapper] api_server.py stopped.")
    _backend_process = None


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    COMPRESSING = "compressing"
    SUBMITTING = "submitting"
    GENERATING = "generating"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    task_id: str
    task_type: str               # "lego", "complete", or "cover"
    bpm: int
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    progress_text: str = "queued"
    ace_task_id: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_format: str = "wav"
    duration: Optional[float] = None       # actual source duration
    target_duration: Optional[float] = None # user-requested output duration (complete)
    track_name: Optional[str] = None       # lego only
    error: Optional[str] = None


_jobs: dict[str, Job] = {}
_generation_semaphore: asyncio.Semaphore | None = None


def _cleanup_old_jobs():
    now = time.time()
    expired = [
        tid for tid, job in _jobs.items()
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        and (now - job.created_at) > JOB_TTL
    ]
    for tid in expired:
        del _jobs[tid]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LegoRequest(BaseModel):
    """Lego mode: generate a single stem over existing audio."""
    audio_data: str = Field(..., description="Base64-encoded audio")
    track_name: str = Field(..., description="Track type: vocals, drums, bass, etc.")
    bpm: int = Field(..., description="BPM of the source audio")
    caption: str = Field("", description="Override default caption for track type")
    lyrics: str = Field("", description="Optional lyrics with structure tags like [Verse 1]")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    guidance_scale: float = Field(7.0, description="CFG scale. 7-9 recommended")
    inference_steps: int = Field(50, description="Diffusion steps. 50 default")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


class CompleteRequest(BaseModel):
    """Complete mode: continue/extend audio with full arrangement."""
    audio_data: str = Field(..., description="Base64-encoded source audio")
    bpm: int = Field(..., description="BPM of the source audio")
    audio_duration: float = Field(..., description="Target output duration in seconds")
    caption: str = Field("", description="Style caption")
    lyrics: str = Field("", description="Optional lyrics with structure tags")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    key_scale: str = Field("", description="Optional key/scale e.g. 'F minor', 'C major'")
    guidance_scale: float = Field(7.0, description="CFG scale. 7-9 recommended")
    inference_steps: int = Field(50, description="Diffusion steps. 50 default")
    use_src_as_ref: bool = Field(False, description="Pass source as ref_audio for timbre anchoring")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


class CoverRequest(BaseModel):
    """Cover/remix mode: restyle audio guided by caption while preserving structure."""
    audio_data: str = Field(..., description="Base64-encoded source audio")
    bpm: int = Field(..., description="BPM of the source audio")
    caption: str = Field(..., description="Style caption driving the remix")
    lyrics: str = Field("", description="Optional lyrics with structure tags")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    key_scale: str = Field("", description="Optional key/scale e.g. 'F minor', 'C major'")
    cover_noise_strength: float = Field(0.2, description="0=pure noise, 1=closest to source. Recommended 0.2")
    audio_cover_strength: float = Field(0.3, description="Fraction of steps using semantic codes. 0.3 instrumental, 0.5-0.7 vocals")
    guidance_scale: float = Field(1.0, description="Cover mode is locked to CFG 1.0 for turbo generation")
    inference_steps: int = Field(8, description="Cover mode is locked to 8 diffusion steps for turbo generation")
    use_src_as_ref: bool = Field(False, description="Pass source as ref_audio for subtler transformation")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ACE-Step Wrapper (localhost)", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _init():
    global _generation_semaphore
    _generation_semaphore = asyncio.Semaphore(EFFECTIVE_MAX_CONCURRENT)
    _start_backend()

    # Wait for backend to be ready
    print("[wrapper] Waiting for api_server to be ready...")
    async with httpx.AsyncClient(timeout=5) as client:
        for _ in range(60):
            try:
                r = await client.get(f"{ACESTEP_URL}/health")
                if r.status_code == 200:
                    print("[wrapper] api_server is ready!")
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
    print("[wrapper] WARNING: api_server did not become ready within 120s")


@app.on_event("shutdown")
async def _shutdown():
    _stop_backend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_duration(path: str) -> Optional[float]:
    """Get audio duration using soundfile (no ffprobe needed on Windows)."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_FRACTION_RE = re.compile(r'(\d+)\s*/\s*(\d+)')
_PERCENT_RE = re.compile(r'(\d+)%')
_GENERIC_PROGRESS_STAGES = {"queued", "running", "succeeded", "failed"}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and control chars from text."""
    stripped = _ANSI_RE.sub('', text)
    return stripped.strip()


def _parse_fraction_from_text(text: str) -> Optional[tuple[int, int]]:
    if not text:
        return None
    m = _FRACTION_RE.search(text)
    if not m:
        return None
    current, total = int(m.group(1)), int(m.group(2))
    if 0 < total <= 200:
        return current, total
    return None


def _parse_progress_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    fraction = _parse_fraction_from_text(text)
    if fraction is not None:
        current, total = fraction
        return min(int((current / total) * 100), 99)
    m = _PERCENT_RE.search(text)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return min(pct, 99)
    return None


def _parse_acestep_result_entry(result_payload: object) -> dict:
    if isinstance(result_payload, str):
        try:
            result_payload = json.loads(result_payload)
        except Exception:
            return {}
    if isinstance(result_payload, list) and result_payload:
        first = result_payload[0]
        return first if isinstance(first, dict) else {}
    return result_payload if isinstance(result_payload, dict) else {}


def _coerce_progress_ratio(value: object) -> Optional[float]:
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= progress <= 1.0:
        return progress
    return None


def _map_fraction_to_percent(current: int, total: int, start: int, end: int) -> int:
    if total <= 0:
        return start
    ratio = max(0.0, min(current / total, 1.0))
    return min(int(round(start + ((end - start) * ratio))), end)


def _is_decode_stage(stage_text: str, progress_text: str) -> bool:
    combined = f"{stage_text} {progress_text}".lower()
    return "decoding audio" in combined or "decoding latents" in combined


def _is_finalize_stage(stage_text: str, progress_text: str) -> bool:
    combined = f"{stage_text} {progress_text}".lower()
    return "preparing audio data" in combined or "preparing audio tensors" in combined


def _resolve_progress_label(stage_text: str, progress_text: str) -> str:
    stage_text = stage_text.strip()
    progress_text = progress_text.strip()
    label = progress_text or stage_text or "generating..."
    if stage_text and stage_text.lower() not in _GENERIC_PROGRESS_STAGES:
        label = stage_text
    fraction = _parse_fraction_from_text(progress_text)
    if (
        fraction is not None
        and label
        and progress_text
        and progress_text != label
    ):
        current, total = fraction
        label = f"{label} ({current}/{total})"
    return label or "generating..."


def _resolve_generation_progress(result_item: dict) -> tuple[Optional[int], str]:
    progress_text = _strip_ansi(result_item.get("progress_text") or "")
    result_entry = _parse_acestep_result_entry(result_item.get("result"))
    stage_text = _strip_ansi(str(result_entry.get("stage") or ""))
    raw_progress = _coerce_progress_ratio(result_entry.get("progress"))

    progress_pct = None
    if raw_progress is not None and raw_progress > 0:
        progress_pct = min(int(round(raw_progress * 100)), 99)

    fraction = _parse_fraction_from_text(progress_text)
    if _is_decode_stage(stage_text, progress_text):
        if fraction is not None:
            current, total = fraction
            progress_pct = max(progress_pct or 80, _map_fraction_to_percent(current, total, 80, 94))
        elif progress_pct is None or progress_pct < 80:
            progress_pct = 80
    elif _is_finalize_stage(stage_text, progress_text):
        progress_pct = max(progress_pct or 96, 96)
    elif progress_pct is None:
        progress_pct = _parse_progress_from_text(progress_text)

    return progress_pct, _resolve_progress_label(stage_text, progress_text)


def _acestep_headers() -> dict[str, str]:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def _required_model_for_task(task_type: str) -> str:
    if task_type == "cover":
        return ACESTEP_TURBO_CONFIG
    return ACESTEP_BASE_CONFIG


def _effective_guidance_scale(task_type: str, requested_guidance_scale: float) -> float:
    if task_type == "cover":
        return COVER_GUIDANCE_SCALE
    return requested_guidance_scale


def _effective_inference_steps(task_type: str, requested_inference_steps: int) -> int:
    if task_type == "cover":
        return COVER_INFERENCE_STEPS
    return requested_inference_steps


async def _load_model(client: httpx.AsyncClient, config_path: str) -> str:
    resp = await client.post(
        f"{ACESTEP_URL}/v1/load",
        params={"config_path": config_path},
        headers=_acestep_headers(),
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/load failed for {config_path}: {resp.text}")
    body = resp.json()
    return ((body.get("data") or {}).get("model") or config_path).strip()


async def _unload_model(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        f"{ACESTEP_URL}/v1/unload",
        headers=_acestep_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/unload failed: {resp.text}")


async def _ensure_required_model(client: httpx.AsyncClient, job: Job) -> None:
    global _current_model

    if not MANAGE_MODEL_LIFECYCLE:
        return

    required_model = _required_model_for_task(job.task_type)
    if _current_model == required_model:
        return

    job.status = JobStatus.LOADING
    job.progress = max(job.progress, 3)
    job.progress_text = f"loading {required_model}..."

    if _current_model is not None:
        await _unload_model(client)
        _current_model = None

    _current_model = await _load_model(client, required_model)


# ---------------------------------------------------------------------------
# Generalized background generation
# ---------------------------------------------------------------------------

def _build_form_data(job: Job, req, send_path: str) -> dict:
    """Build the multipart form data dict for /release_task."""
    effective_guidance_scale = _effective_guidance_scale(job.task_type, req.guidance_scale)
    effective_inference_steps = _effective_inference_steps(job.task_type, req.inference_steps)

    if job.task_type == "lego":
        effective_caption = req.caption.strip() or TRACK_CAPTIONS.get(req.track_name, "")
        audio_duration = str(job.duration)
    elif job.task_type == "cover":
        effective_caption = req.caption.strip()
        audio_duration = str(job.duration)
    else:  # complete
        effective_caption = req.caption.strip()
        audio_duration = str(req.audio_duration)

    data = {
        "task_type":        job.task_type,
        "caption":          effective_caption,
        "lyrics":           req.lyrics,
        "language":         req.language,
        "bpm":              str(req.bpm),
        "time_signature":   req.time_signature,
        "guidance_scale":   str(effective_guidance_scale),
        "thinking":         "false",
        "use_cot_caption":  "false",
        "use_cot_language": "false",
        "batch_size":       str(req.batch_size),
        "audio_duration":   audio_duration,
        "audio_format":     req.audio_format,
    }

    if job.task_type == "lego":
        data["track_name"] = req.track_name
        data["repainting_start"] = "0.0"
        data["repainting_end"] = "-1"
        data["inference_steps"] = str(effective_inference_steps)
    elif job.task_type == "cover":
        data["cover_noise_strength"] = str(req.cover_noise_strength)
        data["audio_cover_strength"] = str(req.audio_cover_strength)
        data["inference_steps"] = str(effective_inference_steps)
    else:  # complete
        data["inference_steps"] = str(effective_inference_steps)

    if hasattr(req, 'key_scale') and req.key_scale.strip():
        data["key_scale"] = req.key_scale.strip()

    return data


async def _run_generation(job: Job, req):
    """Background task: handles the full generation lifecycle for any task type."""
    raw_audio_path = None

    async with _generation_semaphore:
        try:
            # --- Decode audio ---
            job.status = JobStatus.COMPRESSING
            job.progress = max(job.progress, 8)
            job.progress_text = "preparing audio..."
            try:
                audio_bytes = base64.b64decode(req.audio_data)
            except Exception:
                raise RuntimeError("Invalid base64 audio data")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                raw_audio_path = tmp.name

            job.duration = _probe_duration(raw_audio_path)
            print(f"[wrapper] Audio duration: {job.duration}s, path: {raw_audio_path}", flush=True)
            if job.duration is None:
                raise RuntimeError("Could not determine audio duration")

            if job.task_type == "complete":
                job.target_duration = req.audio_duration

            send_path = raw_audio_path

            async with httpx.AsyncClient() as client:
                await _ensure_required_model(client, job)

                # --- Submit to ace-step ---
                job.status = JobStatus.SUBMITTING
                job.progress = max(job.progress, 15)
                job.progress_text = "submitting to ace-step..."

                filename = Path(send_path).name
                mime = "audio/flac" if filename.endswith(".flac") else \
                        "audio/ogg" if filename.endswith(".ogg") else "audio/wav"
                form_data = _build_form_data(job, req, send_path)

                use_ref = getattr(req, 'use_src_as_ref', False)

                with open(send_path, "rb") as fh:
                    files = [("ctx_audio", (filename, fh, mime))]

                    ref_fh = None
                    if use_ref:
                        ref_fh = open(send_path, "rb")
                        files.append(("ref_audio", (filename, ref_fh, mime)))

                    try:
                        resp = await client.post(
                            f"{ACESTEP_URL}/release_task",
                            headers=_acestep_headers(),
                            data=form_data,
                            files=files,
                            timeout=120,
                        )
                    finally:
                        if ref_fh:
                            ref_fh.close()

                if resp.status_code != 200:
                    raise RuntimeError(f"/release_task failed: {resp.text}")

                body = resp.json()
                print(f"[wrapper] /release_task response: {json.dumps(body)[:200]}", flush=True)
                job.ace_task_id = body["data"]["task_id"]

                # --- Poll ace-step for progress ---
                job.status = JobStatus.GENERATING
                job.progress = max(job.progress, 18)
                job.progress_text = "starting generation..."

                gen_start_time = time.time()
                inference_steps = _effective_inference_steps(
                    job.task_type,
                    getattr(req, 'inference_steps', INFERENCE_STEPS),
                )
                est_seconds_per_step = 0.35

                deadline = time.time() + GENERATION_TIMEOUT
                while time.time() < deadline:
                    resp = await client.post(
                        f"{ACESTEP_URL}/query_result",
                        headers=_acestep_headers(),
                        json={"task_id_list": [job.ace_task_id]},
                        timeout=15,
                    )
                    resp_body = resp.json()
                    result = resp_body["data"][0]
                    ace_status = result["status"]
                    result_entry = _parse_acestep_result_entry(result.get("result"))
                    stage_text = _strip_ansi(str(result_entry.get("stage") or ""))
                    raw_progress = _coerce_progress_ratio(result_entry.get("progress"))
                    progress_text = _strip_ansi(result.get("progress_text") or "")
                    print(
                        f"[wrapper] poll: status={ace_status} stage={stage_text or '-'} "
                        f"progress={raw_progress} text={progress_text[:80]}",
                        flush=True,
                    )

                    resolved_progress, resolved_label = _resolve_generation_progress(result)
                    if resolved_label:
                        job.progress_text = resolved_label
                    if resolved_progress is not None:
                        job.progress = min(max(job.progress, resolved_progress), 99)

                    # Time-based estimate for cover mode only
                    if resolved_progress is None and job.status == JobStatus.GENERATING \
                            and job.task_type == "cover":
                        elapsed = time.time() - gen_start_time
                        est_total = inference_steps * est_seconds_per_step
                        if est_total > 0:
                            est_pct = min(int((elapsed / est_total) * 75), 75)
                            if est_pct > 0:
                                job.progress = min(max(job.progress, est_pct), 99)
                                job.progress_text = f"~{est_pct}% ({inference_steps} steps)"

                    if ace_status == 1:
                        break
                    if ace_status == 2:
                        error_msg = result.get("error") or result.get("progress_text") or "generation failed"
                        print(f"[wrapper] ace-step FAILED, full result: {json.dumps(result, default=str)}", flush=True)
                        raise RuntimeError(error_msg)

                    await asyncio.sleep(POLL_INTERVAL)
                else:
                    raise RuntimeError("Generation timed out")

                # --- Download audio ---
                job.status = JobStatus.DOWNLOADING
                job.progress = max(job.progress, 99)
                job.progress_text = "downloading audio..."

                files_list = json.loads(result["result"])
                if not files_list:
                    raise RuntimeError("No audio files in result")

                file_path = files_list[0]["file"]
                resp = await client.get(
                    f"{ACESTEP_URL}{file_path}",
                    headers=_acestep_headers(),
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Failed to download audio: {resp.status_code}")

                job.audio_b64 = base64.b64encode(resp.content).decode("utf-8")
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.progress_text = "complete"

        except Exception as e:
            import traceback
            print(f"[wrapper] Generation FAILED: {e}", flush=True)
            traceback.print_exc()
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.progress_text = f"failed: {e}"

        finally:
            if raw_audio_path:
                try:
                    os.unlink(raw_audio_path)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Shared status response builder
# ---------------------------------------------------------------------------

def _build_status_response(job: Job) -> JSONResponse:
    """Build JUCE-compatible status response for any task type."""

    if job.status in (
        JobStatus.QUEUED, JobStatus.LOADING, JobStatus.COMPRESSING,
        JobStatus.SUBMITTING, JobStatus.GENERATING, JobStatus.DOWNLOADING,
    ):
        status_messages = {
            JobStatus.QUEUED: "queued",
            JobStatus.LOADING: "loading model...",
            JobStatus.COMPRESSING: "preparing audio...",
            JobStatus.SUBMITTING: "submitting...",
            JobStatus.GENERATING: job.progress_text or "generating...",
            JobStatus.DOWNLOADING: "downloading result...",
        }

        return JSONResponse({
            "success": True,
            "generation_in_progress": True,
            "transform_in_progress": False,
            "progress": job.progress,
            "status": "processing",
            "queue_status": {
                "status": "queued" if job.status == JobStatus.QUEUED else "ready",
                "message": status_messages.get(job.status, "processing..."),
                "position": 0,
                "estimated_seconds": 0,
                "estimated_time": "",
            },
        })

    if job.status == JobStatus.COMPLETED:
        resp = {
            "success": True,
            "generation_in_progress": False,
            "transform_in_progress": False,
            "status": "completed",
            "audio_data": job.audio_b64,
            "progress": 100,
            "bpm": job.bpm,
            "duration": job.target_duration or job.duration,
            "audio_format": job.audio_format,
            "task_type": job.task_type,
        }
        if job.track_name:
            resp["track_name"] = job.track_name
        return JSONResponse(resp)

    return JSONResponse({
        "success": False,
        "generation_in_progress": False,
        "transform_in_progress": False,
        "status": "failed",
        "error": job.error or "Unknown error",
        "progress": 0,
    })


# ---------------------------------------------------------------------------
# Lego endpoints
# ---------------------------------------------------------------------------

@app.post("/lego")
async def lego_submit(req: LegoRequest):
    if req.track_name not in ALLOWED_TRACKS:
        raise HTTPException(400, f"track_name must be one of {sorted(ALLOWED_TRACKS)}")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="lego",
        bpm=req.bpm,
        track_name=req.track_name,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/lego/status/{task_id}")
async def lego_status(task_id: str):
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Complete endpoints
# ---------------------------------------------------------------------------

@app.post("/complete")
async def complete_submit(req: CompleteRequest):
    if req.audio_duration < 5:
        raise HTTPException(400, "audio_duration must be at least 5 seconds")
    if req.audio_duration > 300:
        raise HTTPException(400, "audio_duration must be at most 300 seconds (5 min)")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="complete",
        bpm=req.bpm,
        target_duration=req.audio_duration,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/complete/status/{task_id}")
async def complete_status(task_id: str):
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Cover endpoints
# ---------------------------------------------------------------------------

@app.post("/cover")
async def cover_submit(req: CoverRequest):
    if req.cover_noise_strength < 0 or req.cover_noise_strength > 1:
        raise HTTPException(400, "cover_noise_strength must be 0.0-1.0")
    if req.audio_cover_strength < 0 or req.audio_cover_strength > 1:
        raise HTTPException(400, "audio_cover_strength must be 0.0-1.0")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="cover",
        bpm=req.bpm,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/cover/status/{task_id}")
async def cover_status(task_id: str):
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    ace_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ACESTEP_URL}/health")
            ace_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.ConnectError:
        ace_status = "unreachable"
    except Exception as e:
        ace_status = f"error: {type(e).__name__}"

    active_jobs = sum(
        1 for j in _jobs.values()
        if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED)
    )

    return {
        "status": "ok",
        "acestep_url": ACESTEP_URL,
        "acestep_status": ace_status,
        "manage_model_lifecycle": MANAGE_MODEL_LIFECYCLE,
        "current_model": _current_model,
        "base_model": ACESTEP_BASE_CONFIG,
        "turbo_model": ACESTEP_TURBO_CONFIG,
        "active_jobs": active_jobs,
        "max_concurrent": EFFECTIVE_MAX_CONCURRENT,
        "configured_max_concurrent": MAX_CONCURRENT,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("carey_wrapper:app", host="0.0.0.0", port=WRAPPER_PORT, reload=False)
