"""
ACE-Step lego wrapper service.

Exposes a single simplified POST /lego endpoint for the VST and iOS clients.
Internally handles:
  - GPU token acquisition from gpu-queue-service
  - Model load / unload lifecycle on the ACE-Step container
  - Generation submission and polling
  - Audio download and streaming back to the caller

Environment variables:
  ACESTEP_URL        URL of the ACE-Step api_server  (default http://localhost:8001)
  QUEUE_URL          URL of the gpu-queue-service    (default http://gpu-queue-service:8085)
  QUEUE_TOKENS       GPU tokens to acquire per job   (default 1000)
  WRAPPER_PORT       Port this service listens on    (default 8002)
  ACESTEP_API_KEY    API key for ACE-Step if set     (optional)
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACESTEP_URL   = os.getenv("ACESTEP_URL",   "http://localhost:8001").rstrip("/")
QUEUE_URL     = os.getenv("QUEUE_URL",     "http://gpu-queue-service:8085").rstrip("/")
QUEUE_TOKENS  = int(os.getenv("QUEUE_TOKENS", "1000"))
WRAPPER_PORT  = int(os.getenv("WRAPPER_PORT", "8002"))
API_KEY       = os.getenv("ACESTEP_API_KEY", "")

INFERENCE_STEPS   = 50
DEFAULT_BATCH     = 1
POLL_INTERVAL     = 3      # seconds between status polls
GENERATION_TIMEOUT = 300   # seconds before giving up on a job

TRACK_CAPTIONS = {
    "vocals":          "soulful indie vocalist, warm, wordless melody, expressive, intimate",
    "backing_vocals":  "background vocals, close harmony, wordless, warm, following the lead vocal",
    "drums":           "live acoustic drum kit, tight kick and snare, brushed hi-hats, warm",
}

ALLOWED_TRACKS = set(TRACK_CAPTIONS.keys())

app = FastAPI(title="ACE-Step Wrapper")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acestep_headers() -> dict:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def _probe_duration(path: str) -> Optional[float]:
    """Use ffprobe to get audio duration in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return None


async def _acquire_gpu_token(client: httpx.AsyncClient, session_id: str) -> bool:
    """Post a task to the gpu-queue-service to claim GPU tokens."""
    try:
        resp = await client.post(
            f"{QUEUE_URL}/tasks",
            json={"session_id": session_id, "tokens": QUEUE_TOKENS},
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


async def _release_gpu_token(client: httpx.AsyncClient, session_id: str) -> None:
    """Release GPU tokens back to the queue."""
    try:
        await client.post(
            f"{QUEUE_URL}/task/status",
            json={"session_id": session_id, "status": "completed"},
            timeout=10,
        )
    except Exception:
        pass


async def _load_model(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        f"{ACESTEP_URL}/v1/load",
        headers=_acestep_headers(),
        timeout=120,  # first load from disk can take a moment
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"ACE-Step /v1/load failed: {resp.text}")


async def _unload_model(client: httpx.AsyncClient) -> None:
    try:
        await client.post(
            f"{ACESTEP_URL}/v1/unload",
            headers=_acestep_headers(),
            timeout=30,
        )
    except Exception:
        pass  # best-effort; never block the response


async def _submit_lego(
    client: httpx.AsyncClient,
    audio_path: str,
    track_name: str,
    caption: str,
    bpm: int,
    key_scale: str,
    audio_duration: float,
    batch_size: int,
) -> str:
    with open(audio_path, "rb") as fh:
        files = {"ctx_audio": (Path(audio_path).name, fh, "audio/wav")}
        data = {
            "task_type":       "lego",
            "track_name":      track_name,
            "caption":         caption,
            "bpm":             str(bpm),
            "time_signature":  "4",
            "inference_steps": str(INFERENCE_STEPS),
            "thinking":        "false",
            "use_cot_caption": "false",
            "repainting_start": "0.0",
            "repainting_end":  "-1",
            "batch_size":      str(batch_size),
            "audio_duration":  str(audio_duration),
        }
        if key_scale:
            data["key_scale"] = key_scale
        resp = await client.post(
            f"{ACESTEP_URL}/release_task",
            headers=_acestep_headers(),
            data=data,
            files=files,
            timeout=60,
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"ACE-Step /release_task failed: {resp.text}")
    body = resp.json()
    task_id = body["data"]["task_id"]
    return task_id


async def _poll_until_done(client: httpx.AsyncClient, task_id: str) -> dict:
    deadline = time.time() + GENERATION_TIMEOUT
    while time.time() < deadline:
        resp = await client.post(
            f"{ACESTEP_URL}/query_result",
            headers=_acestep_headers(),
            json={"task_id_list": [task_id]},
            timeout=15,
        )
        data = resp.json()["data"][0]
        status = data["status"]
        if status == 1:
            return data
        if status == 2:
            raise HTTPException(502, f"ACE-Step generation failed: {data.get('error')}")
        await asyncio.sleep(POLL_INTERVAL)
    raise HTTPException(504, "ACE-Step generation timed out")


async def _download_first_audio(client: httpx.AsyncClient, result_data: dict) -> bytes:
    import json
    files = json.loads(result_data["result"])
    first_file_path = files[0]["file"]
    resp = await client.get(
        f"{ACESTEP_URL}{first_file_path}",
        headers=_acestep_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        raise HTTPException(502, "Failed to download generated audio")
    return resp.content


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/lego")
async def lego(
    audio_file:     UploadFile = File(...),
    track_type:     str        = Form(...),
    bpm:            int        = Form(...),
    key_scale:      str        = Form(""),
    batch_size:     int        = Form(DEFAULT_BATCH),
    caption:        str        = Form(""),
):
    """
    Generate a stem over the provided audio using ACE-Step lego mode.

    track_type: vocals | backing_vocals | drums
    bpm:        integer BPM of the source audio
    key_scale:  optional, e.g. "F# minor" (model handles ambiguous keys well)
    batch_size: 1 or 2 (T4 default: 1)
    caption:    override the default caption for the track type
    """
    import asyncio

    if track_type not in ALLOWED_TRACKS:
        raise HTTPException(400, f"track_type must be one of {sorted(ALLOWED_TRACKS)}")

    effective_caption = caption.strip() or TRACK_CAPTIONS[track_type]

    # Save uploaded audio to a temp file
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio_file.read())
        audio_path = tmp.name

    try:
        audio_duration = _probe_duration(audio_path)
        if audio_duration is None:
            raise HTTPException(400, "Could not determine audio duration — is ffprobe installed?")

        session_id = f"acestep-{int(time.time() * 1000)}"

        async with httpx.AsyncClient() as client:
            # 1. Acquire GPU token
            ok = await _acquire_gpu_token(client, session_id)
            if not ok:
                raise HTTPException(503, "GPU queue unavailable")

            try:
                # 2. Load model onto GPU
                await _load_model(client)

                # 3. Submit generation
                task_id = await _submit_lego(
                    client, audio_path, track_type, effective_caption,
                    bpm, key_scale, audio_duration, batch_size,
                )

                # 4. Poll until done
                result_data = await _poll_until_done(client, task_id)

                # 5. Download first audio candidate
                audio_bytes = await _download_first_audio(client, result_data)

            finally:
                # 6. Always unload model and release token
                await _unload_model(client)
                await _release_gpu_token(client, session_id)

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": f'attachment; filename="{track_type}.mp3"'},
        )

    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass


@app.get("/health")
async def health():
    """Check wrapper and ACE-Step health."""
    ace_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ACESTEP_URL}/health")
            ace_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "wrapper": "ok",
        "acestep": "ok" if ace_ok else "unreachable",
        "acestep_url": ACESTEP_URL,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    uvicorn.run("main:app", host="0.0.0.0", port=WRAPPER_PORT, reload=False)
