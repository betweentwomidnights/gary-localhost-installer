"""Named and explicitly overridable inference settings for Foundation-1."""

from numbers import Real
import math


INFERENCE_PROFILES = {
    "gary_fallback": {
        "sampler_type": "dpmpp-2m-sde",
        "sigma_min": 0.5,
        "sigma_max": 50.0,
        "rho": 1.0,
        "steps": 100,
        "guidance_scale": 7.0,
    },
    "royalcities": {
        "sampler_type": "dpmpp-3m-sde",
        "sigma_min": 0.01,
        "sigma_max": 100.0,
        "rho": 1.0,
        "steps": 100,
        "guidance_scale": 7.0,
    },
}

SUPPORTED_SAMPLERS = {
    "k-heun",
    "k-lms",
    "k-dpmpp-2s-ancestral",
    "k-dpm-2",
    "k-dpm-fast",
    "k-dpm-adaptive",
    "dpmpp-2m-sde",
    "dpmpp-3m-sde",
}


def _number(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    if not isinstance(value, Real):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def resolve_inference_settings(data: dict) -> dict:
    """Resolve a named profile, then apply request-level overrides."""
    profile_name = data.get("inference_profile", "gary_fallback")
    if not isinstance(profile_name, str):
        raise ValueError("inference_profile must be a string")
    if profile_name not in INFERENCE_PROFILES:
        choices = ", ".join(sorted(INFERENCE_PROFILES))
        raise ValueError(f"inference_profile must be one of: {choices}")

    settings = dict(INFERENCE_PROFILES[profile_name])

    sampler = data.get("sampler_type", data.get("sampler", settings["sampler_type"]))
    if not isinstance(sampler, str):
        raise ValueError("sampler_type must be a string")
    if "sampler" in data and "sampler_type" in data and data["sampler"] != data["sampler_type"]:
        raise ValueError("sampler and sampler_type must match when both are provided")
    if sampler not in SUPPORTED_SAMPLERS:
        choices = ", ".join(sorted(SUPPORTED_SAMPLERS))
        raise ValueError(f"sampler_type must be one of: {choices}")

    settings.update({
        "inference_profile": profile_name,
        "sampler_type": sampler,
        "sigma_min": _number(data, "sigma_min", settings["sigma_min"]),
        "sigma_max": _number(data, "sigma_max", settings["sigma_max"]),
        "rho": _number(data, "rho", settings["rho"]),
        "guidance_scale": _number(data, "guidance_scale", settings["guidance_scale"]),
    })

    raw_steps = data.get("steps", settings["steps"])
    if isinstance(raw_steps, bool):
        raise ValueError("steps must be an integer")
    try:
        steps = int(raw_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be an integer") from exc
    if isinstance(raw_steps, Real) and float(raw_steps) != steps:
        raise ValueError("steps must be an integer")
    settings["steps"] = steps

    if settings["steps"] < 1:
        raise ValueError("steps must be at least 1")
    if settings["sigma_min"] <= 0:
        raise ValueError("sigma_min must be greater than 0")
    if settings["sigma_max"] <= settings["sigma_min"]:
        raise ValueError("sigma_max must be greater than sigma_min")
    if settings["rho"] <= 0:
        raise ValueError("rho must be greater than 0")
    if settings["guidance_scale"] < 0:
        raise ValueError("guidance_scale must be at least 0")

    return settings


def resolve_audio2audio_inference_settings(data: dict, init_noise_level: float) -> dict:
    """Resolve settings and report the variation schedule's effective maximum."""
    settings = resolve_inference_settings(data)
    return {
        **settings,
        "configured_sigma_max": settings["sigma_max"],
        "sigma_max": float(init_noise_level),
    }
