# sa3 localhost backend

Local Stable Audio 3 service for gary4local.

LoRA registration uses a Carey-style flow in the control center: the user points
at an SA3 `.ckpt` or `.safetensors` checkpoint and, optionally, a dataset folder
containing training `.txt` sidecars. The registry is written to
`%APPDATA%/Gary4JUCE/sa3/lora_registry.json`, and prompt dice pools live under
`%APPDATA%/Gary4JUCE/sa3/prompts`.

## Hugging Face access

SA3 uses the same saved Hugging Face token as the existing stable-audio service.
The token alone is not enough: the same Hugging Face account must also accept
the model terms for:

- https://huggingface.co/stabilityai/stable-audio-3-medium
- https://huggingface.co/google/t5gemma-b-b-ul2

The service health endpoint does not load or download the model. First load
happens when `/load` or a generation endpoint is called.

## LoRA endpoints

- `GET /loras` returns configured LoRAs.
- `POST /reload` rebuilds the loaded model and preloads the current registry.
- `GET /prompts?lora=name` returns default prompt pools merged with any selected
  LoRA prompt JSONs.

## LoRA training

The control center includes a first Gary-native SA3 LoRA training flow powered
by vendored pieces of [dada-bots' underfit project](https://github.com/dada-bots/underfit).
It does not embed the underfit dashboard. Instead, Tauri launches
`train_lora_job.py`, which stages the SA3 medium base checkpoint from Hugging
Face, pre-encodes an audio folder, runs underfit's raw PyTorch trainer, then
copies the final `.safetensors` adapter to `%APPDATA%/Gary4JUCE/sa3/loras` and
adds it to the existing SA3 LoRA catalog.

Training uses the saved Gary4local Hugging Face token and needs access to
`stabilityai/stable-audio-3-medium-base`. The SA3 generation service should be
stopped before training so the GPU has enough VRAM.

Plain `.txt` sidecars are literal prompts: the complete trimmed file becomes
the clip's prompt. Labels such as `Title:` have no special parsing in a text
sidecar. Structured JSON and embedded audio metadata are composed into labelled
prompt parts instead. The dataset editor links to both the
[Underfit metadata guide](https://github.com/dada-bots/underfit#2-optional-add-metadata-for-prompts)
and the official
[SA3 prompting guide](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md).
LoRA dice pools copy text-sidecar prompts while omitting a trailing numeric BPM
tag because Gary supplies tempo separately.

Training status and logs persist when the control center closes. The training
window can also cancel preprocessing or training and clean up the active child
processes.

An optional experimental loudness fix can normalize every track to a target
encoded latent RMS during pre-encoding. The recommended `0.90` target matches
the base model's latent loudness. Because the VAE is nonlinear, normalization
may encode each track several times and therefore makes pre-encoding slower.
Random training crops are unchanged and continue to re-roll from the stored
full-track latents.

## Runtime hardening

All sampler and manual decode boundaries align latent tensors to the
pretransform decoder's device and dtype. This includes classifier-free guidance
paths that may return float32 latents while a half-precision decoder is loaded.

When `model_half` is enabled, model weights are converted to float16 before the
model moves to CUDA. The resulting loaded model is unchanged, but the transfer
avoids a transient full-precision GPU allocation.

## Continuation modes

`POST /continue` supports `continuation_mode` values `inpaint` and
`latent_prefix`. `latent_prefix` pins the encoded source as a fixed latent prefix
and forces the `pingpong` sampler.

## Output shaping

SA3 applies local loudness defaults from the service environment and echoes the
applied values in `meta.loudness`. Gary4local exposes the main knobs as an
advanced "sa3 output shaping" panel:

- `peak_normalize_db` / `SA3_PEAK_NORMALIZE_DB`, default `2.0`
- `limiter_ceiling_db` / `SA3_LIMITER_CEILING_DB`, default `-0.3`
- `latent_rescale` / `SA3_LATENT_RESCALE`, default `1.0`
- `latent_shift` / `SA3_LATENT_SHIFT`, default `0.0`
- `latent_target_std` / `SA3_LATENT_TARGET_STD`, default off
- `continuation_tail_pad` / `SA3_CONTINUE_TAIL_PAD`, default `6`

Use `off` for dB fields to disable that stage. A positive peak-normalize target
is intended to be paired with the limiter.
