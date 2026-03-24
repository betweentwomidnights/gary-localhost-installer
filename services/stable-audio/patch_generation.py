# /app/patch_generation.py

from pathlib import Path
import importlib
import sys


def ensure_sample_rf_guided():
    # 1. Import the installed sampling module so we get the real path
    try:
        sampling_mod = importlib.import_module(
            "stable_audio_tools.inference.sampling"
        )
    except ImportError as e:
        print("ERROR: Could not import stable_audio_tools.inference.sampling:", e)
        sys.exit(1)

    sampling_path = Path(sampling_mod.__file__)
    print(f"Patching sampling.py at: {sampling_path}")

    if not sampling_path.exists():
        print("ERROR: sampling.py does not exist at that path")
        sys.exit(1)

    text = sampling_path.read_text()

    # 2. Idempotency: if we already patched it, bail out
    sentinel = "# === BEGIN collabage patch: guided rectified-flow sampling ==="
    if sentinel in text:
        print("Patch already present, skipping")
        return

    patch_code = """
# === BEGIN collabage patch: guided rectified-flow sampling ===

@torch.no_grad()
def sample_rf_guided(
    model_fn,
    noise,
    init_data=None,
    steps=100,
    sampler_type="euler",
    sigma_max=1,
    device="cuda",
    callback=None,
    guidance=None,
    **extra_args,
):
    # 1. Same sigma_max + init_data handling as sample_rf
    if sigma_max > 1:
        sigma_max = 1

    if init_data is not None:
        x = init_data * (1 - sigma_max) + noise * sigma_max
    else:
        x = noise

    # 2. Same logSNR → t schedule as sample_rf
    logsnr_max = math.log(((1 - sigma_max) / sigma_max) + 1e-6) if sigma_max < 1 else -6
    logsnr = torch.linspace(logsnr_max, 2, steps + 1, device=x.device, dtype=x.dtype)
    t = torch.sigmoid(-logsnr)
    t[0] = sigma_max
    t[-1] = 0

    # 3. Dispatch per sampler_type to *guided* variants
    if sampler_type == "euler":
        return sample_discrete_euler_guided(
            model_fn, x, sigmas=t, sigma_max=sigma_max, guidance=guidance,
            callback=callback, **extra_args
        )
    elif sampler_type == "dpmpp":
        # later: sample_flow_dpmpp_guided(...)
        return sample_flow_dpmpp(model_fn, x, sigmas=t, sigma_max=sigma_max,
                                 callback=callback, **extra_args)
    elif sampler_type == "pingpong":
        return sample_flow_pingpong_guided(
            model_fn, x, sigmas=t, sigma_max=sigma_max,
            guidance=guidance, callback=callback, **extra_args
        )
    else:
        return sample_rf(  # fallback to unguided
            model_fn, noise, init_data=init_data, steps=steps,
            sampler_type=sampler_type, sigma_max=sigma_max,
            device=device, callback=callback, **extra_args
        )


@torch.no_grad()
def sample_discrete_euler_guided(model, x, sigmas, sigma_max=1, guidance=None,
                                 callback=None, dist_shift=None, disable_tqdm=False, **extra_args):
    ts = x.new_ones([x.shape[0]])

    print(f"[EulerGuided] guidance is None? {guidance is None}")

    t = sigmas  # same as sample_discrete_euler

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr

        v = model(x, t_curr_tensor, **extra_args)

        # ---- Hawley-style guidance hook ----
        if guidance is not None:
            v = apply_latent_inpaint_guidance(
                x=x,
                v=v,
                t=float(t_curr),
                guidance=guidance,
            )

        x = x + dt * v

        if callback is not None:
            denoised = x - t_prev * v
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised})

    return x


@torch.no_grad()
def sample_flow_pingpong_guided(model, x, steps=None, sigma_max=1, sigmas=None,
                                callback=None, dist_shift=None, guidance=None, **extra_args):
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:
        t = torch.linspace(sigma_max, 0, steps + 1, device=x.device, dtype=x.dtype)
        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    else:
        t = sigmas

    for i in trange(len(t) - 1, disable=False):
        t_curr = t[i]
        denoised = x - t_curr * model(x, t_curr * ts, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr,
                      'sigma_hat': t_curr, 'denoised': denoised})

        # In pingpong, denoised ≈ x_1; we can use that for latent inpainting too.
        if guidance is not None:
            # treat denoised as "final latent" proxy
            denoised = apply_latent_inpaint_guidance_pingpong(
                x=x,
                denoised=denoised,
                t=float(t_curr),
                guidance=guidance,
            )

        t_next = t[i + 1]
        x = (1 - t_next) * denoised + t_next * torch.randn_like(x)

    return x


def apply_latent_inpaint_guidance(x, v, t, guidance):
    # guidance: {
    #   "mode": "latent_inpaint",
    #   "M_sq": tensor,   # [B, 1 or C, T_latent]
    #   "z_y": tensor,    # encoded guitar latents
    #   "strength": float,
    #   "t_min": float,
    #   "t_max": float,
    # }

    if guidance.get("mode") != "latent_inpaint":
        return v

    if t < guidance.get("t_min", 0.0) or t > guidance.get("t_max", 1.0):
        return v

    M_sq = guidance["M_sq"]
    z_y = guidance["z_y"]
    strength = guidance.get("strength", 1.0)

    t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)

    # Hawley latent-only inpainting:
    # z1_hat = z_t + (1 - t) * v_t
    z1_hat = x + (1.0 - t_tensor) * v

    # Analytic gradient of L = || M ⊙ (z1_hat - z_y) ||^2
    grad = M_sq * (z1_hat - z_y)

    # Time scaling ~(1 - t) / t
    time_scale = (1.0 - t_tensor) / max(t, 1e-4)
    dv = -strength * time_scale * grad

    return v + dv


def apply_latent_inpaint_guidance_pingpong(x, denoised, t, guidance):
    # Same idea as apply_latent_inpaint_guidance, but acting directly
    # on 'denoised' (our proxy for the final latent z_1).

    if guidance.get("mode") != "latent_inpaint":
        return denoised

    if t < guidance.get("t_min", 0.0) or t > guidance.get("t_max", 1.0):
        return denoised

    M_sq = guidance["M_sq"]
    z_y = guidance["z_y"]
    strength = guidance.get("strength", 1.0)

    t_tensor = torch.tensor(t, device=denoised.device, dtype=denoised.dtype)

    # Here z1_hat is just the current denoised estimate
    z1_hat = denoised
    grad = M_sq * (z1_hat - z_y)

    time_scale = (1.0 - t_tensor) / max(t, 1e-4)
    delta = -strength * time_scale * grad

    return denoised + delta

# === END collabage patch ===
"""

    sampling_path.write_text(text + "\n\n" + patch_code)
    print("Added sample_rf_guided to sampling.py")

if __name__ == "__main__":
    ensure_sample_rf_guided()
