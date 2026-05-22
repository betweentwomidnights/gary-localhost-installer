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
