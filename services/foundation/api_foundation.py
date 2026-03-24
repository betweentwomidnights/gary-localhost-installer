#!/usr/bin/env python3
"""
Foundation-1 API Service
Structured text-to-sample inference for the RoyalCities Foundation-1 model.
Designed for the DGX Spark compose network alongside gary4juce.

Async generation with polling — matches the gary4juce poll_status contract.
"""

from flask import Flask, request, jsonify
import torch
import torchaudio
import json
import io
import base64
import uuid
import os
import time
import threading
import gc
import ctypes
import numpy as np

# RC stable-audio-tools imports
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

# RC prompt engine — imported at module level; lazy-loaded in the
# /randomize endpoint so startup doesn't fail if the path needs fixing.
_rc_prompt = None

def _load_rc_prompt():
    global _rc_prompt
    if _rc_prompt is not None:
        return _rc_prompt
    import importlib, sys
    # The RC fork is installed as an editable package, but the prompts
    # sub-package may not be on the path in every container build.
    # Fall back to a direct path import if the package route fails.
    try:
        mod = importlib.import_module(
            "stable_audio_tools.interface.prompts.master_prompt_map"
        )
    except ModuleNotFoundError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "master_prompt_map",
            os.path.join(
                os.path.dirname(__file__),
                "RC-stable-audio-tools",
                "stable_audio_tools",
                "interface",
                "prompts",
                "master_prompt_map.py",
            ),
        )
        if spec is None:
            # Inside the container the repo is at /app/RC-stable-audio-tools
            spec = importlib.util.spec_from_file_location(
                "master_prompt_map",
                "/app/RC-stable-audio-tools/stable_audio_tools/interface/prompts/master_prompt_map.py",
            )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _rc_prompt = mod
    return mod

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FOUNDATION_MODEL_DIR = os.environ.get("FOUNDATION_MODEL_DIR", "/models/foundation-1")
FOUNDATION_CKPT_PATH = os.environ.get(
    "FOUNDATION_CKPT_PATH",
    os.path.join(FOUNDATION_MODEL_DIR, "Foundation_1.safetensors"),
)
FOUNDATION_CONFIG_PATH = os.environ.get(
    "FOUNDATION_CONFIG_PATH",
    os.path.join(FOUNDATION_MODEL_DIR, "model_config.json"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_BPMS = [100, 110, 120, 128, 130, 140, 150]
SUPPORTED_BARS = [4, 8]
VALID_KEY_ROOTS = [
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
]
VALID_KEY_MODES = ["major", "minor"]

FALLBACK_PROMPT = "Synth, Pad, Warm, Wide, Chord Progression, 4 Bars, 120 BPM, C minor"

# ---------------------------------------------------------------------------
# Session store  (session_id -> job state)
# ---------------------------------------------------------------------------

sessions = {}
sessions_lock = threading.Lock()
# Only one generation at a time on a single GPU
generation_semaphore = threading.Semaphore(1)


def create_session(session_id: str, meta: dict):
    with sessions_lock:
        sessions[session_id] = {
            "status": "queued",           # queued -> generating -> stretching -> completed / failed
            "generation_in_progress": True,
            "transform_in_progress": False,
            "progress": 0,                # 0-100
            "step": 0,
            "total_steps": meta.get("steps", 100),
            "audio_data": None,           # base64 WAV on completion
            "error": None,
            "meta": meta,                 # prompt, seed, bars, bpm, etc.
            "created_at": time.time(),
        }


def update_session(session_id: str, **kwargs):
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id].update(kwargs)


def get_session(session_id: str) -> dict | None:
    with sessions_lock:
        return sessions.get(session_id, {}).copy() if session_id in sessions else None


def cleanup_old_sessions(max_age: float = 600.0):
    """Remove sessions older than max_age seconds."""
    now = time.time()
    with sessions_lock:
        expired = [
            sid for sid, s in sessions.items()
            if now - s.get("created_at", 0) > max_age
        ]
        for sid in expired:
            del sessions[sid]


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

model_cache = {}
model_lock = threading.Lock()
model_ready = threading.Event()


def aggressive_cleanup():
    for _ in range(3):
        gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    """Load Foundation-1 checkpoint and keep warm on GPU."""
    with model_lock:
        if "model" in model_cache:
            return model_cache["model"], model_cache["config"], model_cache["device"]

        print("Loading Foundation-1 model...")
        print(f"  Config : {FOUNDATION_CONFIG_PATH}")
        print(f"  Ckpt   : {FOUNDATION_CKPT_PATH}")

        with open(FOUNDATION_CONFIG_PATH, "r") as f:
            model_config = json.load(f)

        model = create_model_from_config(model_config)
        state_dict = load_ckpt_state_dict(FOUNDATION_CKPT_PATH)
        model.load_state_dict(state_dict, strict=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Use fp16 on CUDA if the checkpoint is fp16
        sample_tensor = next(iter(state_dict.values()))
        if sample_tensor.dtype == torch.float16 and device == "cuda":
            model = model.half()
            print("  Using fp16")

        model.eval()
        model.requires_grad_(False)

        model_cache["model"] = model
        model_cache["config"] = model_config
        model_cache["device"] = device

        sr = model_config.get("sample_rate", 44100)
        ss = model_config.get("sample_size", 2097152)
        print(f"  Model loaded on {device}")
        print(f"  Sample rate: {sr}  Sample size: {ss}")
        print(f"  Diffusion objective: {getattr(model, 'diffusion_objective', 'unknown')}")

        model_ready.set()
        return model, model_config, device


# ---------------------------------------------------------------------------
# BPM / duration helpers
# ---------------------------------------------------------------------------

def nearest_foundation_bpm(host_bpm: float) -> int:
    return min(SUPPORTED_BPMS, key=lambda b: abs(b - host_bpm))


def derive_duration(bars: int, bpm: float) -> float:
    return bars * 4 * 60.0 / bpm


def time_stretch_ratio(host_bpm: float, foundation_bpm: int) -> float:
    return host_bpm / foundation_bpm


# ---------------------------------------------------------------------------
# Time-stretch
# ---------------------------------------------------------------------------

def apply_time_stretch(audio_tensor: torch.Tensor, ratio: float, sample_rate: int) -> torch.Tensor:
    if abs(ratio - 1.0) < 0.001:
        return audio_tensor

    np_audio = audio_tensor.cpu().numpy()

    try:
        import pyrubberband as pyrb
        np_audio_t = np_audio.T
        stretched = pyrb.time_stretch(np_audio_t, sample_rate, ratio)
        return torch.from_numpy(stretched.T.copy()).to(audio_tensor.dtype)
    except ImportError:
        pass

    virtual_sr = int(sample_rate * ratio)
    resampler_down = torchaudio.transforms.Resample(sample_rate, virtual_sr)
    resampler_up = torchaudio.transforms.Resample(virtual_sr, sample_rate)
    stretched = resampler_up(resampler_down(audio_tensor.cpu()))
    return stretched.to(audio_tensor.dtype)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(data: dict) -> str:
    parts = []

    family = (data.get("family") or "").strip()
    subfamily = (data.get("subfamily") or "").strip()
    if family:
        parts.append(family)
    if subfamily:
        parts.append(subfamily)

    for knob in ["descriptor_knob_a", "descriptor_knob_b", "descriptor_knob_c"]:
        val = (data.get(knob) or "").strip()
        if val:
            parts.append(val)

    for extra in (data.get("descriptors_extra") or []):
        val = (extra if isinstance(extra, str) else "").strip()
        if val:
            parts.append(val)

    for tag_key in ["spatial_tags", "band_tags", "wave_tech_tags", "style_tags"]:
        for tag in (data.get(tag_key) or []):
            val = (tag if isinstance(tag, str) else "").strip()
            if val:
                parts.append(val)

    behavior = data.get("behavior_tags") or []
    if isinstance(behavior, str):
        behavior = [b.strip() for b in behavior.split(",") if b.strip()]
    parts.extend(behavior)

    if data.get("reverb_enabled") and data.get("reverb_amount"):
        parts.append(data["reverb_amount"])
    if data.get("delay_enabled") and data.get("delay_type"):
        parts.append(data["delay_type"])
    if data.get("distortion_enabled") and data.get("distortion_amount"):
        parts.append(data["distortion_amount"])
    if data.get("phaser_enabled") and data.get("phaser_amount"):
        parts.append(data["phaser_amount"])
    if data.get("bitcrush_enabled") and data.get("bitcrush_amount"):
        parts.append(data["bitcrush_amount"])

    bars = data.get("bars", 4)
    foundation_bpm = data.get("_foundation_bpm", 120)
    key_root = (data.get("key_root") or "C").strip()
    key_mode = (data.get("key_mode") or "minor").strip().lower()

    parts.append(f"{bars} Bars")
    parts.append(f"{foundation_bpm} BPM")
    parts.append(f"{key_root} {key_mode}")

    seen = set()
    deduped = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)

    prompt = ", ".join(deduped)
    return prompt if prompt else FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_request(data: dict) -> list:
    errors = []
    bars = data.get("bars")
    if bars is not None and bars not in SUPPORTED_BARS:
        errors.append(f"bars must be one of {SUPPORTED_BARS}")

    key_root = data.get("key_root")
    if key_root and key_root not in VALID_KEY_ROOTS:
        errors.append(f"key_root must be one of {VALID_KEY_ROOTS}")

    key_mode = data.get("key_mode")
    if key_mode and key_mode.lower() not in VALID_KEY_MODES:
        errors.append(f"key_mode must be one of {VALID_KEY_MODES}")

    host_bpm = data.get("host_bpm")
    if host_bpm is not None and (host_bpm < 40 or host_bpm > 300):
        errors.append("host_bpm must be between 40 and 300")

    return errors


# ---------------------------------------------------------------------------
# Background generation worker
# ---------------------------------------------------------------------------

def generation_worker(session_id: str, data: dict):
    """Runs in a background thread. Updates session progress as it goes."""
    t_start = time.time()

    # Wait for GPU availability
    acquired = generation_semaphore.acquire(timeout=30)
    if not acquired:
        update_session(session_id,
                       status="failed",
                       generation_in_progress=False,
                       error="GPU busy — another generation is in progress")
        return

    try:
        seed = data["_seed"]
        bars = data["_bars"]
        host_bpm = data["_host_bpm"]
        foundation_bpm = data["_foundation_bpm"]
        gen_duration = data["_gen_duration"]
        prompt = data["_prompt"]
        guidance_scale = data["_guidance_scale"]
        steps = data["_steps"]
        stretch_ratio_val = data["_stretch_ratio"]
        key_root = data.get("key_root", "C")
        key_mode = data.get("key_mode", "minor")

        update_session(session_id, status="generating", progress=0)

        print(f"[{session_id}] Generate request:")
        print(f"  seed={seed} bars={bars} host_bpm={host_bpm}")
        print(f"  foundation_bpm={foundation_bpm} duration={gen_duration:.2f}s")
        print(f"  stretch_ratio={stretch_ratio_val:.4f}")
        print(f"  steps={steps} guidance={guidance_scale}")
        print(f"  prompt: {prompt}")

        model, config, device = load_model()
        sample_rate = int(config.get("sample_rate", 44100))
        sample_size = int(config.get("sample_size", 2097152))

        gen_samples = min(int(gen_duration * sample_rate), sample_size)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": gen_duration,
        }]

        # Progress callback — fired each diffusion step
        def on_step(callback_info):
            step = callback_info.get("i", 0) + 1
            pct = int(step / steps * 90)  # reserve last 10% for post-processing
            update_session(session_id, step=step, progress=pct)

        # Inference
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=guidance_scale,
                conditioning=conditioning,
                sample_size=gen_samples,
                sample_rate=sample_rate,
                seed=seed,
                device=device,
                batch_size=1,
                callback=on_step,
            )

        update_session(session_id, progress=92, status="stretching" if abs(stretch_ratio_val - 1.0) >= 0.001 else "encoding")

        audio = output[0].cpu().float()

        # Trim to exact duration
        expected_samples = int(gen_duration * sample_rate)
        if audio.shape[-1] > expected_samples:
            audio = audio[:, :expected_samples]

        # Time-stretch to host BPM if needed
        if abs(stretch_ratio_val - 1.0) >= 0.001:
            update_session(session_id, transform_in_progress=True, progress=94)
            audio = apply_time_stretch(audio, stretch_ratio_val, sample_rate)
            host_duration = derive_duration(bars, host_bpm)
            host_samples = int(host_duration * sample_rate)
            if audio.shape[-1] > host_samples:
                audio = audio[:, :host_samples]
            update_session(session_id, transform_in_progress=False, progress=96)

        audio = audio.clamp(-1.0, 1.0)
        final_duration = audio.shape[-1] / sample_rate

        # Encode to base64 WAV
        update_session(session_id, progress=97)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, audio, sample_rate, format="wav")
        audio_bytes = audio_buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Also save to disk
        filename = f"foundation_{session_id}_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        gen_time = time.time() - t_start
        print(f"[{session_id}] Done in {gen_time:.2f}s -> {output_path}")

        update_session(
            session_id,
            status="completed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=100,
            audio_data=audio_b64,
            meta={
                **data.get("_original_request", {}),
                "session_id": session_id,
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "gen_duration": round(gen_duration, 4),
                "stretch_ratio": round(stretch_ratio_val, 4),
                "final_duration": round(final_duration, 4),
                "key": f"{key_root} {key_mode}",
                "prompt": prompt,
                "generation_time": round(gen_time, 2),
                "output_path": output_path,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{session_id}] Error: {e}")
        update_session(
            session_id,
            status="failed",
            generation_in_progress=False,
            transform_in_progress=False,
            error=str(e),
        )
    finally:
        generation_semaphore.release()
        aggressive_cleanup()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if model_ready.is_set():
        model, config, device = load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "sample_rate": config.get("sample_rate"),
            "sample_size": config.get("sample_size"),
        })
    return jsonify({"status": "starting", "model_loaded": False}), 503


@app.route("/ready", methods=["GET"])
def ready():
    if model_ready.is_set():
        return jsonify({"ready": True}), 200
    return jsonify({"ready": False}), 503


@app.route("/randomize", methods=["POST"])
def randomize():
    """
    Generate a randomized preset using RC's weighted prompt engine.
    Returns decomposed values so the JUCE plugin can populate its knobs.

    Optional JSON body:
      seed (int):       -1 or omit for random, otherwise deterministic
      mode (str):       "standard" or "mix" (default "standard")
      variant (str):    "auto", "M1", or "T1" (default "auto")
      family_hint (str): lock to a specific family (e.g. "Synth"), or omit
    """
    import random as stdlib_random

    try:
        rc = _load_rc_prompt()
    except Exception as e:
        return jsonify({"success": False, "error": f"RC prompt engine unavailable: {e}"}), 500

    data = request.get_json(silent=True) or {}

    seed = int(data.get("seed", -1))
    if seed == -1:
        seed = stdlib_random.randint(0, 2**31 - 1)

    mode = (data.get("mode") or "standard").strip().lower()
    variant = (data.get("variant") or "auto").strip()
    family_hint = (data.get("family_hint") or "").strip() or None

    # --- Run RC's engine to get the structured anchor ---
    vt = rc.choose_variant_type(mode=mode, variant=variant)
    profile = rc.normalize_mode_to_profile(mode)
    if vt == "T1":
        profile = "mix"

    base_rng = stdlib_random.Random(seed)
    anchor = rc.build_anchor(base_rng, profile=profile, family_hint=family_hint)

    family = str(anchor["family"])
    subfamily = str(anchor["sub"])
    tags = list(anchor["tags"])
    fx_tokens = list(anchor["fx"])
    melody_str = str(anchor["melody"])
    wet = bool(anchor["wet"])

    # --- Decompose FX tokens into reverb / delay / distortion knob values ---
    reverb_value = ""
    delay_value = ""
    distortion_value = ""
    phaser_value = ""
    bitcrush_value = ""

    reverb_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["reverb"])}
    delay_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["delay"])}
    distortion_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["distortion"])}
    phaser_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["phaser"])}
    bitcrush_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["bitcrush"])}

    for tok in fx_tokens:
        tok_lower = tok.lower()
        if tok_lower in reverb_items:
            reverb_value = tok
        elif tok_lower in delay_items:
            delay_value = tok
        elif tok_lower in distortion_items:
            distortion_value = tok
        elif tok_lower in phaser_items:
            phaser_value = tok
        elif tok_lower in bitcrush_items:
            bitcrush_value = tok

    # --- Decompose tags into descriptor buckets ---
    timbre_set = {t.lower() for t in rc.TIMBRE_TAGS}
    spatial_set = {t.lower() for t in rc.SPATIAL_TAGS}
    band_set = {t.lower() for t in rc.BAND_TAGS}
    wave_set = {t.lower() for t in rc.WAVE_TECH_TAGS}
    style_set = {t.lower() for t in rc.STYLE_TAGS}

    descriptors = []
    spatial = []
    band = []
    wave_tech = []
    style = []

    for t in tags:
        tl = t.lower()
        if tl in timbre_set:
            descriptors.append(t)
        elif tl in spatial_set:
            spatial.append(t)
        elif tl in band_set:
            band.append(t)
        elif tl in wave_set:
            wave_tech.append(t)
        elif tl in style_set:
            style.append(t)
        else:
            descriptors.append(t)

    # --- Decompose melody string into behavior components ---
    melody_parts = [m.strip() for m in melody_str.split(",") if m.strip()]

    speed_set = {s.lower() for s in rc.SPEED}
    rhythm_set = {r.lower() for r in rc.RHYTHM}
    contour_set = {c.lower() for c in rc.CONTOUR}
    density_set = {d.lower() for d in rc.DENSITY}
    structure_set = {s.lower() for s in rc.STRUCTURE_GENERIC + ["bassline"]}

    speed_val = ""
    structure_val = ""
    rhythm_vals = []
    contour_vals = []
    density_vals = []

    for part in melody_parts:
        pl = part.lower()
        if pl in speed_set:
            speed_val = part
        elif pl in structure_set:
            structure_val = part
        elif pl in rhythm_set:
            rhythm_vals.append(part)
        elif pl in contour_set:
            contour_vals.append(part)
        elif pl in density_set:
            density_vals.append(part)

    # --- Build the full prompt string (same as generate would use) ---
    full_prompt = rc.prompt_generator_variants(
        seed=seed, mode=mode, variant=variant,
        allow_timbre_mix=True, family_hint=family_hint,
    )

    return jsonify({
        "success": True,
        "seed": seed,
        "mode": mode,
        "variant": vt,

        # Main instrument controls
        "family": family,
        "subfamily": subfamily,

        # Descriptor knobs (first 3 mapped to knob_a/b/c, rest in extras)
        "descriptor_knob_a": descriptors[0] if len(descriptors) > 0 else "",
        "descriptor_knob_b": descriptors[1] if len(descriptors) > 1 else "",
        "descriptor_knob_c": descriptors[2] if len(descriptors) > 2 else "",
        "descriptors_extra": descriptors[3:],

        # FX knobs
        "reverb_enabled": reverb_value != "",
        "reverb_amount": reverb_value,
        "delay_enabled": delay_value != "",
        "delay_type": delay_value,
        "distortion_enabled": distortion_value != "",
        "distortion_amount": distortion_value,
        "phaser_enabled": phaser_value != "",
        "phaser_amount": phaser_value,
        "bitcrush_enabled": bitcrush_value != "",
        "bitcrush_amount": bitcrush_value,

        # Behavior / melody
        "behavior_tags": melody_parts,
        "speed": speed_val,
        "structure": structure_val,
        "rhythm": rhythm_vals,
        "contour": contour_vals,
        "density": density_vals,

        # Additional tag detail (for plugins that want finer control)
        "spatial_tags": spatial,
        "band_tags": band,
        "wave_tech_tags": wave_tech,
        "style_tags": style,
        "all_tags": tags,

        # The full assembled prompt (for preview / debug)
        "prompt": full_prompt,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Accept a generation request, return a session_id immediately.
    Generation runs in background — poll /poll_status/<session_id>.
    """
    cleanup_old_sessions()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        errors = validate_request(data)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        # Check model readiness
        if not model_ready.is_set():
            return jsonify({"success": False, "error": "loading model — warming up"}), 503

        # Parse and resolve params
        seed = int(data.get("seed", -1))
        bars = int(data.get("bars", 4))
        host_bpm = float(data.get("host_bpm", 120.0))
        guidance_scale = float(data.get("guidance_scale", 7.0))
        steps = int(data.get("steps", 100))
        custom_override = (data.get("custom_prompt_override") or "").strip()
        key_root = data.get("key_root", "C")
        key_mode = data.get("key_mode", "minor")

        foundation_bpm = nearest_foundation_bpm(host_bpm)
        data["_foundation_bpm"] = foundation_bpm
        gen_duration = derive_duration(bars, foundation_bpm)

        if custom_override:
            prompt = custom_override
            if "Bars" not in prompt:
                prompt += f", {bars} Bars"
            if "BPM" not in prompt:
                prompt += f", {foundation_bpm} BPM"
            if key_root.lower() not in prompt.lower():
                prompt += f", {key_root} {key_mode}"
        else:
            prompt = build_prompt(data)

        if seed == -1:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        stretch_ratio_val = time_stretch_ratio(host_bpm, foundation_bpm)

        # Create session
        session_id = str(uuid.uuid4())[:12]

        # Stash resolved values for the worker
        data["_seed"] = seed
        data["_bars"] = bars
        data["_host_bpm"] = host_bpm
        data["_foundation_bpm"] = foundation_bpm
        data["_gen_duration"] = gen_duration
        data["_prompt"] = prompt
        data["_guidance_scale"] = guidance_scale
        data["_steps"] = steps
        data["_stretch_ratio"] = stretch_ratio_val
        data["_original_request"] = {
            k: v for k, v in data.items() if not k.startswith("_")
        }

        create_session(session_id, {
            "steps": steps,
            "prompt": prompt,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
        })

        # Launch background worker
        thread = threading.Thread(
            target=generation_worker,
            args=(session_id, data),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
            "gen_duration": round(gen_duration, 4),
            "stretch_ratio": round(stretch_ratio_val, 4),
            "prompt": prompt,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/audio2audio", methods=["POST"])
def audio2audio():
    """
    Audio-to-audio timbre transfer using RC's init_audio variation path.
    Accepts input audio (base64 WAV), a text prompt describing the target
    timbre, and an init_noise_level controlling how much of the original
    audio to preserve.

    The input audio is time-stretched to the nearest Foundation BPM before
    inference, and the output is stretched back to host_bpm afterward
    (same round-trip our /generate endpoint does).

    JSON body:
      audio_data (str):         base64-encoded WAV of the input audio (required)
      prompt (str):             target timbre description (required)
      host_bpm (float):         BPM of the input audio (required)
      bars (int):               4 or 8 (default 8)
      init_noise_level (float): 0.01–1.0; low = preserve input, high = more generation (default 0.25)
      seed (int):               -1 for random (default -1)
      steps (int):              diffusion steps (default 75)
      guidance_scale (float):   CFG scale (default 7.0)
      key_root (str):           e.g. "A#" (default "C")
      key_mode (str):           "major" or "minor" (default "minor")
    """
    cleanup_old_sessions()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        audio_b64 = (data.get("audio_data") or "").strip()
        if not audio_b64:
            return jsonify({"success": False, "error": "audio_data (base64 WAV) is required"}), 400

        host_bpm = data.get("host_bpm")
        if host_bpm is None:
            return jsonify({"success": False, "error": "host_bpm is required"}), 400
        host_bpm = float(host_bpm)

        if not model_ready.is_set():
            return jsonify({"success": False, "error": "loading model — warming up"}), 503

        bars = int(data.get("bars", 8))
        init_noise_level = float(data.get("init_noise_level", 0.25))
        init_noise_level = max(0.01, min(1.0, init_noise_level))
        seed = int(data.get("seed", -1))
        steps = int(data.get("steps", 75))
        guidance_scale = float(data.get("guidance_scale", 7.0))
        key_root = data.get("key_root", "C")
        key_mode = data.get("key_mode", "minor")

        foundation_bpm = nearest_foundation_bpm(host_bpm)
        stretch_ratio_val = time_stretch_ratio(host_bpm, foundation_bpm)
        gen_duration = derive_duration(bars, foundation_bpm)

        # Build prompt the same way /generate does — supports both
        # custom_prompt_override (raw string) and knob-based reconstruction.
        custom_override = (data.get("custom_prompt_override") or data.get("prompt") or "").strip()
        data["_foundation_bpm"] = foundation_bpm
        if custom_override:
            prompt_text = custom_override
            if "Bars" not in prompt_text:
                prompt_text += f", {bars} Bars"
            if "BPM" not in prompt_text:
                prompt_text += f", {foundation_bpm} BPM"
            if key_root.lower() not in prompt_text.lower():
                prompt_text += f", {key_root} {key_mode}"
        else:
            prompt_text = build_prompt(data)

        # Decode the input audio from base64
        audio_bytes = base64.b64decode(audio_b64)
        audio_buf = io.BytesIO(audio_bytes)
        input_waveform, input_sr = torchaudio.load(audio_buf)

        # Pre-stretch: host_bpm -> foundation_bpm so the model hears it at
        # a tempo it was trained on.  The ratio we need is the inverse of
        # the output stretch: we want to speed the audio UP to foundation_bpm.
        pre_stretch_ratio = foundation_bpm / host_bpm
        if abs(pre_stretch_ratio - 1.0) >= 0.001:
            input_waveform = apply_time_stretch(input_waveform, pre_stretch_ratio, input_sr)

        if seed == -1:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        session_id = str(uuid.uuid4())[:12]

        create_session(session_id, {
            "steps": steps,
            "prompt": prompt_text,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
            "mode": "audio2audio",
        })

        thread = threading.Thread(
            target=audio2audio_worker,
            args=(session_id,),
            kwargs={
                "input_waveform": input_waveform,
                "input_sr": int(input_sr),
                "prompt": prompt_text,
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "gen_duration": gen_duration,
                "stretch_ratio": stretch_ratio_val,
                "init_noise_level": init_noise_level,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "key_root": key_root,
                "key_mode": key_mode,
            },
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "seed": seed,
            "foundation_bpm": foundation_bpm,
            "init_noise_level": init_noise_level,
            "prompt": prompt_text,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def audio2audio_worker(
    session_id: str,
    *,
    input_waveform: torch.Tensor,
    input_sr: int,
    prompt: str,
    seed: int,
    bars: int,
    host_bpm: float,
    foundation_bpm: int,
    gen_duration: float,
    stretch_ratio: float,
    init_noise_level: float,
    steps: int,
    guidance_scale: float,
    key_root: str,
    key_mode: str,
):
    """Background worker for audio-to-audio generation."""
    t_start = time.time()

    acquired = generation_semaphore.acquire(timeout=30)
    if not acquired:
        update_session(session_id,
                       status="failed",
                       generation_in_progress=False,
                       error="GPU busy — another generation is in progress")
        return

    try:
        update_session(session_id, status="generating", progress=0)

        print(f"[{session_id}] Audio2Audio request:")
        print(f"  seed={seed} bars={bars} host_bpm={host_bpm}")
        print(f"  foundation_bpm={foundation_bpm} duration={gen_duration:.2f}s")
        print(f"  init_noise_level={init_noise_level} steps={steps} guidance={guidance_scale}")
        print(f"  prompt: {prompt}")

        model, config, device = load_model()
        sample_rate = int(config.get("sample_rate", 44100))
        sample_size = int(config.get("sample_size", 2097152))

        gen_samples = min(int(gen_duration * sample_rate), sample_size)

        # Resample input to model sample rate if needed
        if input_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(input_sr, sample_rate)
            input_waveform = resampler(input_waveform)

        # Pad or crop to match generation length
        if input_waveform.shape[-1] < gen_samples:
            pad = gen_samples - input_waveform.shape[-1]
            input_waveform = torch.nn.functional.pad(input_waveform, (0, pad))
        elif input_waveform.shape[-1] > gen_samples:
            input_waveform = input_waveform[:, :gen_samples]

        # RC's generate_diffusion_cond expects init_audio as (sample_rate, tensor)
        init_audio = (sample_rate, input_waveform)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": gen_duration,
        }]

        def on_step(callback_info):
            step = callback_info.get("i", 0) + 1
            pct = int(step / steps * 90)
            update_session(session_id, step=step, progress=pct)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=guidance_scale,
                conditioning=conditioning,
                sample_size=gen_samples,
                sample_rate=sample_rate,
                seed=seed,
                device=device,
                batch_size=1,
                init_audio=init_audio,
                init_noise_level=init_noise_level,
                callback=on_step,
            )

        update_session(session_id, progress=92,
                       status="stretching" if abs(stretch_ratio - 1.0) >= 0.001 else "encoding")

        audio = output[0].cpu().float()

        # Trim to exact generation duration
        expected_samples = int(gen_duration * sample_rate)
        if audio.shape[-1] > expected_samples:
            audio = audio[:, :expected_samples]

        # Post-stretch: foundation_bpm -> host_bpm
        if abs(stretch_ratio - 1.0) >= 0.001:
            update_session(session_id, transform_in_progress=True, progress=94)
            audio = apply_time_stretch(audio, stretch_ratio, sample_rate)
            host_duration = derive_duration(bars, host_bpm)
            host_samples = int(host_duration * sample_rate)
            if audio.shape[-1] > host_samples:
                audio = audio[:, :host_samples]
            update_session(session_id, transform_in_progress=False, progress=96)

        audio = audio.clamp(-1.0, 1.0)
        final_duration = audio.shape[-1] / sample_rate

        # Encode to base64 WAV
        update_session(session_id, progress=97)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, audio, sample_rate, format="wav")
        audio_bytes = audio_buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Save to disk
        filename = f"foundation_a2a_{session_id}_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        gen_time = time.time() - t_start
        print(f"[{session_id}] Audio2Audio done in {gen_time:.2f}s -> {output_path}")

        update_session(
            session_id,
            status="completed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=100,
            audio_data=audio_b64,
            meta={
                "session_id": session_id,
                "mode": "audio2audio",
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "init_noise_level": init_noise_level,
                "gen_duration": round(gen_duration, 4),
                "stretch_ratio": round(stretch_ratio, 4),
                "final_duration": round(final_duration, 4),
                "key": f"{key_root} {key_mode}",
                "prompt": prompt,
                "generation_time": round(gen_time, 2),
                "output_path": output_path,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{session_id}] Audio2Audio error: {e}")
        update_session(
            session_id,
            status="failed",
            generation_in_progress=False,
            transform_in_progress=False,
            error=str(e),
        )
    finally:
        generation_semaphore.release()
        aggressive_cleanup()


@app.route("/poll_status/<session_id>", methods=["GET"])
def poll_status(session_id: str):
    """
    Returns the exact JSON shape the gary4juce poller expects:
      success, generation_in_progress, transform_in_progress,
      progress (0-100), status, audio_data, error, queue_status
    """
    session = get_session(session_id)
    if session is None:
        return jsonify({
            "success": False,
            "error": f"unknown session: {session_id}",
        }), 404

    status = session["status"]
    gen_in_progress = session["generation_in_progress"]
    xform_in_progress = session["transform_in_progress"]
    progress = session["progress"]

    # Build queue_status for queued state
    queue_status = {}
    if status == "queued":
        queue_status = {
            "status": "queued",
            "position": 1,
            "message": "waiting for GPU",
            "estimated_time": "~5s",
            "estimated_seconds": 5,
        }
    elif status in ("generating", "stretching", "encoding"):
        queue_status = {"status": "ready"}

    response = {
        "success": True,
        "generation_in_progress": gen_in_progress,
        "transform_in_progress": xform_in_progress,
        "progress": progress,
        "status": status,
        "queue_status": queue_status,
    }

    if status == "completed":
        response["audio_data"] = session.get("audio_data", "")
        response["meta"] = session.get("meta", {})
    elif status == "failed":
        response["success"] = False
        response["error"] = session.get("error", "unknown error")

    return jsonify(response)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def warmup():
    try:
        print("Warming up Foundation-1...")
        load_model()
        print("Foundation-1 ready.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    warmup_thread = threading.Thread(target=warmup, daemon=True)
    warmup_thread.start()

    port = int(os.environ.get("PORT", 8015))
    print(f"Starting Foundation-1 API on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)
