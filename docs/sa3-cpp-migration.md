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

The native installer supports `services/<bundle>/native` alongside the existing
service-specific path. The SA3 bundle supplies separate `sa3-server` and
`sat-server` processes with one version/backend installation. Consumer state
and bundle reservations are tracked independently; Jerry and Foundation still
need their own native manifest and API/model integration. Retain service-specific
ports, settings and model directories. Share the CUDA pack across all GGML
projects; keep Yuey's ggml DLLs in its own bundle because its ABI/version need
not match SA3's.

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

## Shared runtime preparation implemented

The manifest now pins the published v0.1.1 core, CUDA and Vulkan archives once
in `nativeBundles.sa3`. A service references this bundle and supplies its own
executable, arguments and native environment overrides. SA3 remains selected as
a Python service while its native candidate can be prepared beside it. Jerry
and Foundation are not wired to the bundle yet; tests exercise their future
shared paths and reservations without changing their active manifests.

Storage settings provides **prepare SA3 runtime** with build/download progress,
failure details and the installed version/backend. Pending storage restart
blocks preparation. Prepared native runtimes appear separately from Python
environments in storage maintenance, while a shared bundle is counted once.
Its stamp keeps the shared CUDA pack protected even before native switchover.

Native installs now probe the staging candidate before swapping it into place.
A failed probe preserves the installed runtime. Install/removal operations
serialize shared package mutations; a bundle cannot be installed or removed
while a native consumer, registered trainer/converter, or consumer build uses
it. The workload reservations still need connecting to the native training and
conversion process lifecycle when those host commands are implemented.

The production installer was exercised headlessly with the published archives,
without a package/build override, in an isolated validation root:
`artifacts/sa3-migration/runtime`. It verified all hashes, installed the shared
`cudart-12.8` pack, selected CUDA automatically, and probed the RTX 5070 Laptop
GPU (7.9 GiB) successfully. The bundle includes SA3/SAT servers, trainer and
adapter converter. The active Python installation was not changed. This
release's `--props` does not include a precision recommendation; the candidate
explicitly selects F16 DiT, F16 text and F32 AE pending model measurements.

Reproduce the explicit package validation from `control-center/src-tauri`:

```powershell
$env:GARY4LOCAL_SA3_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/runtime'
cargo test --lib installs_published_sa3_bundle -- --ignored --nocapture
```

The regular Rust suite passed 111 tests (two explicit network/GPU checks
ignored). The published install test separately passed. Three offline release
pin tests ensure updating a bundle leaves consumers and unrelated package pins
unchanged and refuses an incomplete checksum set without writing. Frontend
checks/build passed. Native model setup, client shim, training integration and
destructive migration remain outstanding.

## Client adapter and upstream capability work

`sa3_adapter.rs` now translates the public generation routes into native
`/generate`: text, loop, transform and continuation. It maps `shift`, validates
WAV uploads, preserves source rate/channel metadata, stages uploads under the
active runtime and separates final sample count from the padded generation
canvas. Native parsing reads the uploaded WAV before acknowledging the job,
so acknowledged uploads are removed immediately. Ambiguous transport failures
retain their source until storage maintenance can safely reclaim it after the
native process has stopped; automatic reclamation still needs implementing.

The adapter preserves native seeds, status/audio and actual splice measurements
while adding Gary's request metadata. Poll/consume behavior is covered through
HTTP. Saved loudness/splice/tail defaults seed the adapter; explicit client
fields override them. Native selection will bind the public service port before
launching its child, serve the C++ API privately on 18006, and own the adapter
task with the process entry. The manifest still selects Python.
LoRA/prompt catalog routes, default/decoder adapter selection and native training
remain to be wired before offering switchover.

The upstream compatibility branch now implements real fixed latent-prefix
sampling in addition to local inpainting conditioning. It also accepts an
explicit conditioning duration independent of the output crop, hidden inpaint
canvas padding, mono WAV input and non-narrowed HTTP integer seeds. Results
report actual conditioned seconds/schedule frames, prefix tokens and full
latent canvas size. `/health.capabilities` advertises these controls. The
adapter checks them before uploading/queuing; published v0.1.1 is refused for
this path because it would silently ignore required fields. A compatible
v0.1.2 release is still pending validation and publication.

Seven adapter unit/HTTP tests passed, including rejection of the old capability
contract before writing uploads. A separate real native CPU integration check
passed for all four public routes against the local compatibility build and
existing small-music models. It checks mono 48 kHz transform input, source
resampling, exact returned lengths, independent conditioning length, large
recalled seed, prefix/splice measurements, identical normal/consume metadata
and removal of acknowledged uploads. One-step synthetic output tests transport
and geometry, not audio quality. Artifacts are under
`artifacts/sa3-migration/adapter-smoke`.

```powershell
# From control-center/src-tauri; use the isolated local compatibility build.
$env:GARY4LOCAL_SA3_SMOKE_BINARY = 'C:/dev/sa3-cpp-gary4local-compat/build-gary-compat/bin/Release/sa3-server.exe'
$env:GARY4LOCAL_SA3_SMOKE_MODELS = 'C:/dev/sa3.cpp/models'
$env:GARY4LOCAL_SA3_ADAPTER_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/adapter-smoke'
cargo test --lib real_native_adapter_smoke -- --ignored --nocapture
```

Non-default LoRA intervals/layer filters and non-pingpong samplers are still
explicitly rejected rather than silently changed. Existing gary4juce requests
use full intervals; broader Python feature parity remains an audit requirement.
GPU model generation, host UI/client validation, converted/trained adapters and
the cleanup transaction are still outstanding.

### Native lifecycle integration

The compatibility server now provides generic `/load`, `/reload`, `/ready` and
idle-only `/unload` routes, advertised as `capabilities.model_lifecycle`.
Health reports initialization, loading, the last load error/duration and the
number of admitted generations. An admission reservation protects lifecycle
changes from new submissions; queued as well as running jobs block reload,
unload and model switching without making the request wait behind generation.
Failure to load is a 503 and remains visible in health/readiness. Unload clears
it. Load is idempotent once the pipeline is initialized.

The public adapter forwards these routes, preserves their HTTP statuses and
errors, and aliases health fields to `model_loaded`, `model_loading` and
`model_error`. An old release lacking the capability is refused. Load/reload
have a longer request timeout; local native connections are not pooled. The
real adapter test exposed a stale connection after a longer load, and using
fresh loopback connections fixed the subsequent health request.

Native initialization reads/validates weights by phase and frees them; readiness
does not promise persistent GPU residency. Frugal generation retains the
initialized pipeline, so readiness remains true after it frees weight tensors.
Full unload makes readiness false. This preserves sa3.cpp's memory behavior.

The real CPU server test passed successful/idempotent load, reload, readiness,
frugal completion and unload. With two admitted jobs, reload/unload/model
selection returned 409 in under one second. Missing-model load failures and
error recovery passed without GPU/models. The host adapter's real integration
test then passed load/readiness/reload/unload alongside all four generation
routes. Native training and LoRA catalogs/conversion are still outstanding.

## First host validation dataset

The user selected `~/Downloads/koan_dset`, resolved on this laptop to
`C:\Users\thegr\Downloads\koan_dset`. The folder exists and contains 42 WAV
files and 42 matching caption TXT files. Preserve the source files; put training
outputs, logs and any generated cache under managed runtime storage. The first
host training validation should launch a short run from gary4local before a
longer quality run, then register/select its native checkpoint and generate
through the client with that adapter.

## Native model preparation

The migration/storage screen and SA3 model panel now offer a preparation step
for medium GGUF models while the Python service remains selected. Inference
DiT choices are F16 (default), Q8_0, Q5_K_M and Q4_K_M. The optional training
base has its own F16 (default) or Q4_K_M selection. These controls choose what
to prepare; active native precision selection still needs wiring during the
service migration slice.

`control-center/src-tauri/sa3-models.json` pins each file's HF repository,
immutable revision, exact size and SHA-256. The files come from
`thepatch/stable-audio-3-medium-GGUF`,
`thepatch/stable-audio-3-medium-base-GGUF` and
`thepatch/t5gemma-b-b-ul2-GGUF`. Native weights are plain files under
`<active-root>/models/sa3`; the existing Python HF cache stays separate.
Every DiT tier and training base shares one F32 SAME-L decoder, one F32
conditioner, one F16 text encoder and one tokenizer. Components own disjoint
file sets so storage accounting and removal do not double-count shared files.

Preparation hashes existing files before reuse, resumes partial downloads and
discards downloads with wrong hashes. Pending storage restart blocks mutations.
The selected runtime root is resolved before creating model subfolders; a
subfolder redirected outside it is refused. Model preparation claims a service
reservation before queuing: native launches, trainers/converters, overlapping
preparation and removal cannot race the downloads. The Python start stays
available. Removal holds both service/model locks through checked file deletion
and removes the component's partial files as well. Present-status checks use
exact file sizes and do not trust a stale successful-download state; preparation
performs the full hash verification.

The live catalog check verified every pinned filename, size and LFS hash at its
exact HF revision. It also downloaded the 793 KB conditioner, reused it without
a network request, and repaired a corrupted local copy through the production
transfer path. Artifacts are in `artifacts/sa3-migration/model-smoke`. Full model
downloads and real desktop UI interaction still need validation.

```powershell
# From control-center/src-tauri.
$env:GARY4LOCAL_SA3_MODEL_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/model-smoke'
cargo test --lib downloads_published_sa3_conditioner_and_verifies_catalog -- --ignored --nocapture
```

At this checkpoint, 127 regular Rust tests pass (four explicit integration
checks ignored); the new live model check passed separately. Frontend checks
report zero errors/warnings and the production build passes.
At that checkpoint native training and final cleanup remained outstanding. The
following slice adds exported-adapter conversion and client catalog mapping.

## Native LoRA preparation and client mapping

Storage preflight and the LoRA manager now offer **prepare / verify adapters**.
The native converter creates separate GGUF files in
`<active-root>/sa3/native-loras`; originals and the Python catalog stay intact.
The native catalog records source/configuration/converter/output SHA-256 values.
Verified conversions are reused. Changed inputs produce separate revisions;
older files remain recoverable. Failed conversions report an error per adapter
and remove unpublished staging files. A damaged cached conversion is repaired
only while the shared native bundle is idle. Runtime replacement/removal is
blocked while conversion runs, and pending storage restart blocks preparation.

Exported `.safetensors` can carry embedded `lora_config`, as Gary's trainer
already emits, or use a matching JSON sidecar. Legacy `.ckpt` sources now use
a one-time CPU export through the existing SA3 Python environment, before that
environment is retired. The host bundles the audited upstream exporter rather
than importing Gary's model code. The helper uses restricted tensor-only loading,
requires a JSON-compatible LoRA config, embeds it in safetensors, retains integer
tensor types, and refuses to replace existing outputs. Unsupported pickles report
an error; there is no unrestricted loading fallback.

Exports live in immutable folders under `sa3/native-loras/legacy-exports`, with
original-checkpoint, helper, safetensors and configuration hashes. Preparation
exports all registered CKPT training history, including older checkpoints whose
currently selected adapter is safetensors. The original files and legacy catalog
are unchanged. Switching back to an older checkpoint can rediscover its verified
cache without Python. A missing/corrupt native GGUF can also be repaired using
that cache after environment removal. A changed checkpoint or damaged exported
source needs a new export while Python still exists; it blocks migration if the
environment is unavailable. Activation verifies every legacy historical export
and automatically discovered configuration sidecars as well as selected adapters.

The host adapter exposes the legacy `/loras` and `/prompts` endpoints. It maps
registered logical names to their exact converted revision and keeps native
filenames out of client selection. The native server's model-dimension checks
filter the creative LoRA menu. Unknown, unprepared or incompatible adapters fail
before upload staging or queue admission. `lora=none` selects no creative
adapter; `lora=default` resolves the configured name or first compatible entry.
Blended requests preserve their individual strengths. Non-default intervals
and layer filters remain explicit compatibility gaps; the optional decoder
correction also still needs native preparation/selection integration.

Native prompt lookup uses the existing managed prompt directory and preserves
repeated `lora` query arguments and the established prompt-pool schema. The
wrapper validates prompt names before forwarding. Bundled default prompt
initialization and Python-free prompt building still need integration.

The actual published converter successfully prepared
`~/Downloads/koan_small_step3000.safetensors`, including its DoRA metadata.
The test verified original preservation, reuse without rewriting, repair of a
damaged GGUF, and an independently recorded legacy checkpoint error. Its isolated
artifacts are in `artifacts/sa3-migration/lora-smoke-1`.

The compatible upstream CPU server then passed generate, loop, transform and
continuation through the host adapter with that LoRA applied. It also passed
prompt-pool merging, rejection before staging, lifecycle controls and medium
model incompatibility filtering. Native logs confirm one adapter queued on each
generation. These one-step, short-audio tests prove transport/application paths;
they do not validate quality, CUDA memory use or training. Artifacts are in
`artifacts/sa3-migration/adapter-lora-smoke-2`.

```powershell
# From control-center/src-tauri; converter preparation needs a new isolated root.
$env:GARY4LOCAL_SA3_LORA_CONVERTER = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/runtime/services/sa3/native/sa3-lora-convert.exe'
$env:GARY4LOCAL_SA3_LORA_SOURCE = 'C:/Users/thegr/Downloads/koan_small_step3000.safetensors'
$env:GARY4LOCAL_SA3_LORA_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/lora-smoke-new'
cargo test --lib real_native_lora_preparation -- --ignored --nocapture

# Add these to the existing real_native_adapter_smoke command above.
$env:GARY4LOCAL_SA3_LORA_REGISTRY = $env:GARY4LOCAL_SA3_LORA_SMOKE_ROOT
$env:GARY4LOCAL_SA3_ADAPTER_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/adapter-lora-smoke-new'
cargo test --lib real_native_adapter_smoke -- --ignored --nocapture
```

Regular Rust checks at this slice pass 131 tests, with five explicit integration
checks ignored. The two live LoRA checks passed separately. Frontend type checks
report zero errors/warnings and the production build passes. SA3 remains on
Python; no cleanup, training or runtime switch has occurred.

The user selected `~/Downloads/koan_dset` for native training validation. The
read-only inventory found 42 WAV/caption pairs, all stereo 44.1 kHz PCM24, with
track lengths from 45.176 to 288 seconds. The inventory is saved under
`artifacts/sa3-migration/koan-dataset-inventory.json`; the dataset was not changed.
The
native CLI already has cooperative cancellation hooks at sample boundaries,
but the published CLI does not expose them and the Windows training web server
uses `TerminateProcess`. Expose generic CLI cancellation/progress upstream, then
connect Gary's existing training controls, checkpoint registration and resume
handling to that contract. Keep latent caches under managed storage rather than
writing them into the user's dataset.

## Native training launch, cancellation and resume

The upstream CLI now exposes `--control-info`, `--progress-file` and
`--cancel-file`. Progress/result JSON is replaced atomically on Windows as well
as POSIX. Cancellation uses the existing native sample-boundary hook, including
during pre-encode; it saves immutable adapter/optimizer checkpoint pairs after
updates. Cancelling before the first optimizer update now returns cancellation
without fabricating an adapter. The control parser/writer tests and real CLI
missing-model/pre-load-cancel smoke test pass without a GPU. These generic
changes belong in the candidate sa3.cpp release, rather than Gary's adapter.

Gary's training modal offers an explicit C++ validation path while the service
default remains Python. Native launch requires compatible control capabilities,
prepared shared text/decoder components and the chosen F16 or Q4_K_M base. It
uses explicit base/component paths so a quantized training base does not require
the matching inference tier to be installed. It carries the existing rank,
batch size, checkpoint cadence, crop length, learning rate, fixed prompt,
target RMS and layer-scope choices into the native job. Native training needs no
Python environment or HF token once its pinned models are prepared.

Native jobs share Gary's current-job status contract, logs and managed process
ownership. Cancellation requests a native save rather than terminating the
process. The shared runtime stays reserved through process exit and adapter
registration; SA3 cannot launch another Python/native generation process during
native training. Terminal GGUF adapters are copied into the native catalog,
leaving run/checkpoint files intact. Resume uses the original named job and its
immutable checkpoint pair, with steps as the new total target. Fixed-prompt
resume reuses the original prompt-config file because native compatibility
fingerprints include its path and modification time.

Latent caches live under managed `sa3/training/native-latents`, partitioned by
decoder hash/precision and target RMS. The dataset is read-only. The native CLI
validates its own checkpoint compatibility rather than treating every GGUF as
an interchangeable resume source.

The real host job round trip passed on the RTX 5070 Laptop CUDA backend using
the user's 42-track Koan dataset, F16 medium base/text weights, F32 decoder,
rank-16 DoRA core scope and 47-second crops. All five component files matched
the published SHA-256 pins. It pre-encoded 42 tracks, requested cancellation
after update 1, stopped and registered a checkpoint at update 2, then resumed
to update 4 in a new run. The second run reused all 42 latent caches. SHA-256
checks proved the resume source checkpoint and every original dataset file were
unchanged; no cache directory was created in the dataset. Both run logs,
progress JSON, checkpoint pairs and catalog revisions are retained in
`artifacts/sa3-migration/training-smoke-1`.

This was the production Rust native job machinery exercised headlessly, using
a locally built compatible CLI plus the verified published CUDA backend and
shared CUDA pack. It is not a desktop UI/client validation or a published
0.1.2-package test. Full-track/default-length training, quantized-base host
training, inference with the newly trained adapter, native checkpoint-history
selection, prompt-pool registration and orphan recovery/resume UX still need
validation or integration before cleanup can be enabled.

At this slice, 133 regular Rust tests pass (six explicit integration checks
ignored); the live CUDA round trip passed separately. Frontend checks report
zero errors/warnings and the production build passes. The service default,
Python installation and models have not been removed or switched.

```powershell
# From control-center/src-tauri; use a new isolated root for each invocation.
$env:GARY4LOCAL_SA3_TRAIN_SMOKE_ROOT = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/training-smoke-new'
$env:GARY4LOCAL_SA3_TRAIN_MODELS = 'C:/dev/sa3.cpp/models'
$env:GARY4LOCAL_SA3_TRAIN_BINARY = 'C:/dev/sa3-cpp-gary4local-compat/build-gary-compat/bin/Release/sa3-train.exe'
$env:GARY4LOCAL_SA3_TRAIN_DATASET = 'C:/Users/thegr/Downloads/koan_dset'
$env:GGML_BACKEND_PATH = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/runtime/services/sa3/native/ggml-cuda.dll'
$env:PATH = 'C:/dev/gary-localhost-installer/artifacts/sa3-migration/runtime/native-runtimes/cudart-12.8;' + $env:PATH
cargo test --lib real_native_training_cancel_resume -- --ignored --nocapture
```

## Native checkpoint history and caption prompts

Native LoRA entries now retain caption-folder metadata and expose the paired
checkpoints from every managed training run with the same logical name. The
checkpoint selector distinguishes resumed branches by job ID, including branches
that produced the same step number. Only adapter/optimizer pairs within their
managed run are selectable. Selecting an adapter copies it into the immutable
native adapter cache and preserves its original training files and strength.

The training modal offers that history for resume and restores the original
dataset, duration, base precision, fixed prompt and optimizer settings. The step
target remains editable as the new total. Failed jobs with a valid checkpoint
pair remain resumable even before they have a registered final adapter.

Caption prompt generation now runs in Rust for both legacy and native LoRAs.
It preserves the legacy BPM/key-tail stripping, relative-path sort, UTF-8 BOM
handling and case-insensitive prompt deduplication. Automatic native registration
preserves an existing curated pool; the existing manual rebuild action explicitly
rebuilds pools. Dataset links outside the source folder are rejected and directory
cycles are skipped. Prompt output uses the selected managed storage root.

The real history test passed against the CUDA-trained adapter under
`artifacts/sa3-migration/training-smoke-1`: both cancelled and resumed checkpoint
branches remained available, an earlier checkpoint could be selected and the
final adapter restored, original GGUF hashes stayed unchanged, and a native
prompt pool was registered from all 42 Koan captions. This uses production Rust
helpers with real artifacts; desktop interaction remains untested.

At this slice, 136 regular Rust tests pass, with seven explicit integration
checks excluded from the regular suite. Frontend checks and build pass.

## Full-duration CUDA training and trained-adapter inference

The host training check also passed at the UI's default 285.35-second window,
rank 16, batch 1, core DoRA scope, RMS correction off, and fixed prompt
`glitch hop, neurofunk, electronica`. Both bases reused the 42 cached Koan
tracks, cancelled cooperatively, resumed from the checkpoint pair, registered
the final adapter, and preserved the dataset and original checkpoint hashes.

| Base | Recorded full-step times | Cancel / resumed target | Artifact root |
| --- | --- | --- | --- |
| F16 | 291–297 seconds | step 2 / step 4 | `artifacts/sa3-migration/training-fulltrack-smoke-1` |
| Q4_K_M | 8.22–10.04 seconds | step 1 / step 3 | `artifacts/sa3-migration/training-q4-fulltrack-smoke-1` |

These are trainer log timings on the RTX 5070 Laptop with 8 GB VRAM, including
the full update, rather than the narrower progress-JSON timing. F16 completed
correctly but was approximately 30 times slower at this duration. A memory
sample during Q4 training showed 7,426 MiB used; it is not a peak measurement.
The Q4 source GGUF's SHA-256 matched the pinned model catalog. Inference still
defaults to F16. Model preparation and full-track training recommend Q4 on
CUDA GPUs with at most 9 GiB reported VRAM; explicit choices and restored
resume settings take precedence. Smaller GPUs may also need shorter crops.

CUDA inference through the production Rust wrapper passed with the F16-trained
medium adapter, F16 inference DiT/T5, and F32 decoder. The four public client
routes, seed recall, normal/consume polling, prompt pools, incompatible-adapter
filtering, load/readiness/reload/unload, and rejected legacy requests passed.
The same CUDA suite also passed with the Q4-trained adapter loaded into the F16
inference model, under `artifacts/sa3-migration/adapter-cuda-q4-trained-smoke-1`.
Artifacts are under `artifacts/sa3-migration/adapter-cuda-lora-smoke-1`, including
`trained-adapter-listening.wav`: a 12-second, 20-step, seed-42 generation using
the registered adapter at strength 0.8. The tiny route checks validate transport;
three or four training updates do not establish fine-tune quality. These tests
use local compatibility tools with the published CUDA backend/runtime pack.
Desktop/gary4juce interaction and an unmodified compatible release installation
remain required before cleanup.

## Profile runtime activation

Storage now offers **verify and select C++** for a prepared inference precision.
Activation reserves the bundle and blocks SA3 launch, rebuild, removal and
training launch while it checks the native trainer control schema, hashes the
selected inference/text/decoder files against the pinned model catalog, and
rechecks registered adapter sources, configuration sidecars and native copies.
It then launches a private native server, requires the four compatibility
capabilities, loads the model, checks readiness, and verifies a tiny real
generation's audio geometry. Progress identifies each file and validation phase.
Errors preserve the prior runtime selection and all Python files.

Successful verification atomically stores `sa3/native-runtime.json` within the
active storage profile, then updates the service manager to use native launch.
The selected DiT precision overrides the manifest's initial F16 argument while
T5/decoder precision stays independent. Other service runtime choices and other
storage profiles remain unchanged. A restart restores that choice. Corrupt,
inaccessible or redirected selection files block SA3 instead of silently
restoring Python. Changing precision preserves previous cleanup status.

An enabled decoder squeak-fix setting currently blocks activation until its
native conversion/selection is integrated. Unprepared legacy adapters also
block activation. Published v0.1.1 lacks the required trainer/server controls,
so the button reports that a compatible release is needed; this slice does not
claim v0.1.1 is a migration-ready package.

The read-only cleanup inventory now also excludes canonical paths overlapping
the native bundle, native models/shared packs, other services or HF cache
repositories, and original adapters/configurations/checkpoints referenced by
the catalogs. Tests include a real Windows junction that redirects the old
environment into the native bundle and an original adapter inside a candidate
PyTorch repository. Both are preserved.

Two live CUDA checks passed under
`artifacts/sa3-migration/profile-activation-smoke-1`: full model verification,
private generation and persisted activation left the original Python marker
unchanged; a later service-manager launch used the saved native choice and
passed public adapter load/readiness before stopping. They use the local
compatibility build, a developer override and isolated storage, rather than
a published compatible package or the user's live desktop profile.

Legacy checkpoint validation passed with the published native converter and
two CKPT fixtures reconstructed from the real Koan safetensors adapter (579
adapter tensors); these are format fixtures, not original user CKPTs. Under
`artifacts/sa3-migration/legacy-export-smoke-2`, both selected and historical
checkpoints were exported, the fixture's Python junction was temporarily renamed,
and cache reuse, GGUF repair and old-export rediscovery passed without Python.
Original checkpoint hashes and legacy catalog bytes stayed unchanged. Tampering
with a historical exported configuration blocked migration. The junction was
restored; the actual Python environment was never changed. Three upstream tests
also passed for tensor/config preservation, output collisions, invalid inputs and
rejected pickle execution.

143 regular Rust tests pass, with ten explicit integration tests excluded.
Frontend checks/build pass. The cleanup transaction, native decoder adapter
integration, Python helper retirement, published-package validation and joint
desktop/client validation remain outstanding.

## Validation before enabling cleanup

First-slice checks completed: four Rust inventory tests passed, `npm run check`
reported zero errors/warnings, and the production frontend build passed. The
upstream CPU `sa3-server` build, runtime contract test, and new HTTP request
validation test passed. The HTTP suite exercises invalid splice settings and
missing-model job failure without downloading weights. A separate tiny real-model
CPU continuation smoke check passed using existing small-music GGUFs: native
48 kHz resampling, exact output length, measured splice metadata, and both normal
and consume polling. It is a transport check, not audio-quality validation.
The subsequent preparation slice pins and probes the published v0.1.1 bundle
as described above. The later CUDA training round trip is recorded in the
native-training section. GPU inference/client validation, native switchover,
destructive cleanup and release publication remain outstanding.

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
