# changelog

this is where we're keeping the version history that used to live at the top
of the main README. the README should stay focused on what gary4local is now;
this file gets to remember how we got here.

## v0.1.16

this is a small hotfix for carey environment rebuilds on clean machines.

- pin `trove-classifiers==2026.5.22.10` for Carey's isolated Hatchling build
  environment because the newer `2026.6.1.19` wheel is missing required
  package-name metadata
- treat omitted and explicit `null` sampling controls the same way in gary
- initialize Stable Audio's model duration before validating explicit loop
  bars
- clarify that carey's completion duration means final total duration, not
  seconds appended

compatible with gary4juce v4.0.2.

## v0.1.15

we've got integrated ACE-Step LoRA training now.

- caption and prepare ACE-Step datasets with the 0.6B, 1.7B, or 4B captioner
- edit captions, genres, BPM, key, lyrics, and other sidecar metadata before
  training
- train LoRA or DoRA adapters against regular base or XL-base
- use Min-SNR loss weighting, best-checkpoint tracking, and the experimental
  balanced attention + MLP profile
- offload frozen model components and run a conservative VRAM preflight before
  the first batch
- automatically repair safe missing captioning/training dependencies
- keep standard and XL LoRAs and caption pools isolated automatically
- register completed adapters and their prompt pools with carey

the carey service also handles model offloading more cleanly now, which makes
it much easier to swap between base, turbo, SFT, and XL models while using
gary4juce.

see the [ACE-Step LoRA training guide](docs/ace-step-lora-training.md) for the
honest version of what we've tested and what remains experimental.

fair warning... this trainer has only been tested on a 5070 laptop GPU with
training runs using `ace-step-v15-base`. plz let me know if you have any issues
with `xl-base`.

## v0.1.14

v0.1.14 makes Hugging Face permission failures explicit when downloading gated
Stable Audio 3 models.

- preserves the underlying Hugging Face error when a generic cache error wraps
  a `401` or `403` response
- explains when a fine-grained token needs public gated-repository read access
- labels a stored token as saved rather than implying its permissions have
  already been validated
- places the gated-token permission guide directly on the sa3 model screen
- repairs older sa3 environments by installing missing LoRA training
  dependencies before preprocessing begins
- cancels sa3 LoRA training without flashing PowerShell or taskkill windows

## v0.1.13

v0.1.13 hardens Hugging Face onboarding and model downloads, especially for
users who are new to gated repositories and fine-grained access tokens.

- adds an in-app visual guide for enabling public gated-repository access on a
  fine-grained Hugging Face token
- shows actionable model-download errors directly in the model list
- uses Hugging Face's official snapshot downloader for resumable downloads and
  reliable cache layout on Windows
- detects incomplete sa3 snapshots instead of presenting them as ready
- loads complete Stable Audio 3 Medium and bundled T5Gemma files directly from
  the local cache, avoiding unnecessary Hub checks during inference

## v0.1.12

v0.1.12 adds Stable Audio 3 LoRA training directly to the Windows control
center. the trainer is a focused integration of
[dada-bots' underfit project](https://github.com/dada-bots/underfit), adapted
to use gary4local's existing sa3 environment, saved Hugging Face token, model
storage, and LoRA registry.

- choose an audio dataset, edit optional text-sidecar prompts, and start
  training with practical defaults for consumer NVIDIA GPUs
- follow selectable, auto-scrolling logs and persisted job progress, or cancel
  preprocessing and training from the same window
- copy completed `.safetensors` adapters into gary4local's sa3 LoRA folder and
  register them for generation automatically
- optionally normalize each track's encoded latent RMS to the base-model target
  before training
- align sampler latents with the decoder's device and precision before decode
- convert half-precision models before moving them to the GPU to reduce peak
  loading memory
