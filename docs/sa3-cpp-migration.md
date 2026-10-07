# SA3 native migration

Started October 7, 2026 on `feature/sa3-cpp-migration`, from gary4local
`61c8e51` (v0.4.0). The first service is SA3; Jerry (`stable-audio`) and
Foundation remain on Python until their own API/model validation is complete.

## Runtime baseline and ownership

The latest published sa3.cpp release inspected is
[v0.1.1](https://github.com/betweentwomidnights/sa3.cpp/releases/tag/v0.1.1),
commit `3602b87393f439773ea022138530a53a164b333e`. Its Windows core contains
`sa3-server`, `sat-server`, and the native training tools. Install core plus
one GPU backend and reuse gary4local's existing `cudart-12.8` pack. Do not use
the standalone archive or copy CUDA DLLs into each service.

gary4juce-specific HTTP request/response compatibility belongs in gary4local.
Keep sa3.cpp's unified `/generate` API: `/transform` and `/continue` are client
contracts to translate, not missing generation operations. Reusable pipeline
capabilities and HTTP access to their controls belong in sa3.cpp. Audit/test
those on `feature/gary4local-api-compat` in
`C:\dev\sa3-cpp-gary4local-compat`, based on published v0.1.1. The user's
existing `feature/lora-envelope` checkout is independent and must not be reset.
A v0.1.2 release is a candidate once compatibility is verified; it has not been
published or pinned by this work.

gary4local owns migration consent, storage inventory, download verification,
backend preferences, app settings, LoRA catalog integration and training UI.
Audio algorithms should remain in sa3.cpp's pipeline, with endpoint handlers
translating the established request/response contract.

## First slice implemented

Storage settings now has an explicit **preview SA3 cleanup** scan. It runs
off the UI thread and inventories only the active profile's SA3 `env`/`.venv`
and the two known PyTorch model repositories. It uses the same effective
Hugging Face cache path as the model manager. Legacy storage honors external
HF cache overrides; custom storage follows the active runtime's managed cache.
No folders are created, no files are removed, and the manifest remains Python.

The view explains what must survive migration: LoRAs, prompts, training jobs,
checkpoints, decoder adapter source, shared runtimes, native model weights,
settings, tokens and datasets. It excludes the shared UV cache from the space
estimate. Other storage profiles remain under the existing storage maintenance
flow. Canonical path containment checks omit unsafe redirected candidates.
Estimates are logical file sizes, not a guarantee of recovered disk space.

SA3 service code is retained in this slice: prompt building and audio metadata
analysis also use its Python environment. Auditing/replacing these helpers is
required before retiring that environment and before deleting bundled code.
Training model copies under `sa3/training/models` are also retained until their
ownership and conversion needs have been audited.

## Compatibility audit against the published API

| Area | Current Python API | Published v0.1.1 | Required work |
|---|---|---|---|
| Generate and loop | `/generate`, `/generate/loop`, `/poll_status` | Present | Compare validation, padding, defaults, status and result metadata; route existence is not parity |
| Transform | `/transform`, base64 `audio_data`, `strength` | Audio-to-audio primitives exposed through `/generate` + `init_path` | Native request adapter with decoded/resampled audio, correct output length and metadata |
| Continue | `/continue`, `continuation_seconds`, continuation modes, splice controls | Inpainting/splice primitives exposed through `/generate`; request-level splice controls/returned splice metadata missing | Translate source duration to `inpaint_start` without subtracting overlap twice; expose reusable controls/metadata upstream; separately resolve `latent_prefix` parity |
| Lifecycle | `/ready`, `/load`, `/reload`, `/unload` | Only `/unload`; health uses `loaded` | Add explicit readiness/loading errors and idle-only lifecycle operations; maintain `loaded` compatibility |
| LoRA discovery | Catalog/registry of safetensors with indexes | Scans GGUF adapters/source exports | Convert originals to separate GGUFs, retain originals, reconcile names/indexes and default selection |
| Training | Tauri launches Python jobs and reads status/checkpoints | Native CLI + `sa3-train-web` API | Adapt launch/status/cancel/checkpoint registration and verify user-facing options |
| Model precision | Medium PyTorch inference/base | GGUF model families and independent component tiers | Download shared encoder/tokenizer once; choose DiT tier independently from AE/text precision |

The first native upstream changes expose `mask_overlap`, `splice_source`,
`splice_xfade`, and `splice_gain_match` on `/generate`, validate them before
queueing, and return pipeline measurements under completed `meta.splice`.
These are generic controls, not additional Gary-specific route aliases. The
native API continues to use `init_path`; the host adapter must stage decoded
base64 WAV uploads under managed storage and retire them after the server has
read them, including failed/cancelled requests.

The current gary4juce client (`Source/PluginEditor.Jerry.cpp`) sends `shift`
where the native server expects `dist_shift`. A host adapter must map and
normalize its value; silently leaving the native default would ignore the UI.
It also sends `continuation_tail_pad`, `continuation_mode`, and per-LoRA
intervals. Full-range intervals currently used by this client are compatible;
non-default intervals and the selectable `latent_prefix` mode require a real
capability audit, rather than being dropped by the wrapper. The inspected
v0.1.1 `GenParams` contains inpainting but no `latent_prefix` implementation.

Native source splicing already exists at the pipeline layer. With splicing
enabled, the host must pass the source duration as `inpaint_start`, because the
pipeline pulls the mask back by `mask_overlap` itself. Output `target_samples`
must equal the source plus requested extension, independently of the padded
generation canvas. Transform similarly uses the source length as its target,
maps `strength` to `init_noise_level`, and leaves inpaint disabled.

The native server's `/health` does not prove models are loaded. A successful
runtime `--props` probe likewise proves backend availability, not inference or
training parity. Do not treat either as a migration validation marker.

## Shared install design for three services

The current native installer uses `services/<id>/native`. Before adding Jerry
or Foundation as native services, introduce a bundle identity (e.g. `sa3`) so
one version/backend installation supplies separate `sa3-server` and `sat-server`
processes. Track consumers independently. Serialize bundle install/update/removal
and block replacement while any consumer or training process uses its binaries.
Retain service-specific ports, settings and model directories. Share the CUDA
pack across all GGML projects; keep Yuey's ggml DLLs in its own bundle because
its ABI/version need not match SA3's.

For SA3 initial defaults, use an unquantized DiT on supported hardware, guided
by measurements on the RTX 5070 Laptop. Do not interpret "full precision" as
forcing F32 for every component: v0.1.1 resolves text encoding and autoencoder
encoding independently, preferring F16 text and F32 AE. Provide quantized DiT
selection for smaller hardware and validate training with the chosen base.

## Migration transaction to implement after parity

1. Detect legacy SA3 assets without marking new installs as needing cleanup.
   Block a storage-root change until restart. Show the concrete active root,
   effective HF cache, retained artifacts, removable paths and estimated sizes.
2. Install and verify the pinned C++ bundle/backend and shared CUDA dependencies
   using the Yuey download/staging/probe path. Preserve the old Python runtime
   until the native path is functional. On low-disk hosts, explain the space
   needed before offering explicit early removal of reproducible weights/envs.
3. Download verified GGUF components; preserve tokens/gating behavior and model
   licenses. Convert imported/trained/decoder LoRAs to separate files. Require
   successful validation of the converted adapters before modifying the catalog.
4. Wire native inference and training into gary4local, then validate on the
   laptop together. Preserve datasets/sidecars, prompts, checkpoint history and
   external references. Never delete original adapters as a cleanup side effect.
5. Offer **clean up and migrate to sa3.cpp** with an exact removable allowlist.
   Stop SA3 inference/training/helper jobs, reject active builds/downloads,
   re-resolve every path immediately before removal, and preserve other services.
   Use existing managed deletion primitives rather than raw broad directory
   deletion. Retire source files only after every helper dependency is replaced.
6. Persist per-profile migration progress atomically; distinguish native installed,
   switched, cleanup pending and cleanup complete. A failed download/conversion/
   deletion must leave a truthful, retryable state. App upgrades must not reseed
   retired Python source. UV cache cleanup remains explicit and optional because
   other Python services still consume it.

## First host validation dataset

The user selected `~/Downloads/koan_dset`, resolved on this laptop to
`C:\Users\thegr\Downloads\koan_dset`. The folder exists and contains 42 WAV
files and 42 matching caption TXT files. Preserve the source files; put training
outputs, logs and any generated cache under managed runtime storage. The first
host training validation should launch a short run from gary4local before a
longer quality run, then register/select its native checkpoint and generate
through the client with that adapter.

## Validation before enabling cleanup

First-slice checks completed: four Rust inventory tests passed, `npm run check`
reported zero errors/warnings, and the production frontend build passed. The
upstream CPU `sa3-server` build, runtime contract test, and new HTTP request
validation test passed. The HTTP suite exercises invalid splice settings and
missing-model job failure without downloading weights. A separate tiny real-model
CPU continuation smoke check passed using existing small-music GGUFs: native
48 kHz resampling, exact output length, measured splice metadata, and both normal
and consume polling. It is a transport check, not audio-quality validation.
No GPU inference/training,
native switchover, destructive cleanup, release publication or release pinning
has been performed in this slice.

- Fresh/default/legacy/custom roots; pending restart; HF overrides; redirected
  paths; low disk; interrupted downloads; retry; partial cleanup failures.
- Preserve Jerry/Foundation/Yuey, all adapters/checkpoints/prompts, external
  datasets, settings/tokens, shared CUDA and unrelated model repositories.
- Published-package install with no developer override; backend/precision
  selection on the RTX 5070 Laptop; generate, loop, transform, continuation,
  seed recall, loudness controls and LoRA blending through gary4juce.
- Native training launched from gary4local with a real dataset; progress/logs,
  cancel/restart, checkpoint cadence and selection, imported adapter conversion,
  active adapter reload and inference with the resulting checkpoint.
- Run the same API contract suite against the published native release, not
  only the local branch. Broader CUDA/Vulkan/Metal trainer validation upstream
  supports this work but does not replace host integration validation.
