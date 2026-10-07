# changelog

this is where we're keeping the version history that used to live at the top
of the main README. the README should stay focused on what gary4local is now;
this file gets to remember how we got here.

## v0.4.0-rc.4

yuey now installs [yuey.cpp v0.2.2](https://github.com/betweentwomidnights/yuey.cpp/releases/tag/v0.2.2),
fixing the short-score token-budget failure found with two-bar MIDI. after
updating the app, press `update runtime` on yuey's row; replacing the app alone
doesn't replace an already installed native runtime.

SA3's default request ceiling and Carey's completion limit are now 380 seconds,
matching gary4juce's localhost controls. SA3 continuation counts the source and
added audio against that total. existing explicit SA3_MAX_DURATION overrides
still apply. this raises the accepted duration, not a guarantee of memory use
or generation speed on every GPU.

compatible with gary4juce v5.0.0-rc.4.

## v0.4.0-rc.2

yuey now installs [yuey.cpp v0.2.1](https://github.com/betweentwomidnights/yuey.cpp/releases/tag/v0.2.1),
with faster continuations and a timing summary for each job. existing installs
need to press `update runtime` on yuey's row to replace 0.2.0.

the default natural-length ceiling is now 96 seconds, matching the remote
backend. a natural continuation is limited by the new bars it adds. old preview
settings carrying the former default of 180 move to 96 once; other chosen
limits stay. the log pane also has a `copy log` button for reporting slow jobs.

compatible with gary4juce v5.0.0-rc.3.

## v0.4.0

### yuey, the first native service

yuey runs [YuE2](https://github.com/multimodal-art-projection/YuE) through
[yuey.cpp](https://github.com/betweentwomidnights/yuey.cpp) on port 8007. it's
the first service with no Python environment: "install runtime" replaces
"build env". gary4local detects the GPU, downloads yuey's core package plus one
GGML backend (CUDA on NVIDIA, Vulkan on AMD and Intel), checks both against the
SHA-256s pinned in the manifest, unpacks them, and runs `yue2-server --props`
to confirm the backend actually came up before it calls the runtime installed.
if the automatic choice doesn't come up, it tries the next backend and says so
in the panel. there's no CPU runtime: a song takes minutes on a GPU, so on a
CPU yuey would only look broken.

the CUDA runtime (cudart and cuBLAS, about 550 MB) is shared by every native
service, so it installs once under `native-runtimes/`, from its own release on
this repo, `runtime-cudart-12.8.1`. how all of this is packaged, published and
pinned is in [native runtime packages](docs/native-runtime-packages.md), which
sa3.cpp, acestep.cpp and audiocraft.cpp will follow as they come over.

### the yuey panel

the panel shows the runtime's backend and version, a backend override (auto,
cuda or vulkan), and the model tier, picked from the GPU's memory unless you
choose one. a runtime install finishes by downloading the shared models and
the recommended tier, and yuey won't start until they're present.

a generation section sets three server defaults for every client, gary4juce
included: the instrumental method ("official YuE" moves the melody onto an
instrument, "our original" takes it out), whether the instrumental LoRA is
used, and how long a song yuey may write for itself (30-600 seconds, 96 by
default, the same as the remote backend; a continuation is held to it by what
it adds). saving restarts yuey. a request that sets one of these itself still
wins. pre-release installs that saved settings under the old default of 180
move to 96 once; a ceiling anyone actually chose stays.

yuey is pinned at [v0.2.1](https://github.com/betweentwomidnights/yuey.cpp/releases/tag/v0.2.1),
which makes continuation faster. an 8-bar continuation of a 25-second clip went
from 46s to 31s on an RTX 5070 Laptop over CUDA, and from 37s to 32s over Vulkan.
each job also writes a one-line timing summary to yuey's log, and the log pane
has a "copy log" button, so a slow job can be reported by pasting it.

### storage

the storage window lists native runtimes as runtimes, removed with "remove
runtime" and brought back with "install runtime", and gives each shared runtime
a row of its own. the CUDA runtime can only be removed once nothing installed
uses it: after the last CUDA runtime is removed, or after a switch to Vulkan.

compatible with gary4juce v5.0.0.

## v0.3.2

### foundation-1 inference profiles

Foundation-1 generation and audio-to-audio routes now resolve named sampler
profiles before applying explicit request overrides. `gary_fallback` preserves
the sampler defaults used by existing local generations, while `royalcities`
matches the sampler and sigma range passed by RoyalCities' Gradio UI.

Both routes report the resolved profile, sampler, sigma range, rho, steps, and
guidance in their accepted and completed metadata. Audio-to-audio also keeps
the configured sigma maximum visible while reporting the effective variation
maximum actually passed to the sampler.

### sa3 offline startup

SA3 no longer calls Hugging Face's login helper during service startup. A saved
token is normalized into the environment without a validation request, so a
fully cached model can load and generate while the machine is offline. Missing
model files still use the normal authenticated download path when networking is
available.

compatible with gary4juce v4.0.14.

## v0.3.1

### sa3 decoder squeak fix

SA3 can download and enable the SAME-L decoder LoRA as an optional squeak fix.
It is kept separate from user style LoRAs, then merged into the autoencoder
decoder when the model loads so style-LoRA selection remains unchanged and the
render path has no live LoRA parametrization overhead.

Commit-pinned Hugging Face snapshots are resolved directly, so the downloaded
decoder checkpoint remains loadable offline even when the cache has no
`refs/main` entry. Existing legacy-storage users continue resolving it from
their normal Hugging Face cache, and the old-storage cleanup inventory includes
that managed decoder repository after they move to custom storage.

### sa3 continuation splice

SA3 continuations now regenerate the final 0.2 seconds of the source by default,
then restore the original source over the earlier kept head and use a short
equal-power crossfade into the newly decoded audio. The splice happens before
peak normalization and limiting so chained continuations keep a seamless
boundary without bypassing output shaping. Advanced controls expose the mask
overlap, crossfade, RMS gain matching, and an off switch for source restoration.
Overlap is defensively clamped to retain at least 50 ms of source, and a silent
generated head no longer causes RMS matching to attenuate the restored source.

Runtime service stamps now include a build-time content hash as well as the app
version. Corrected QA installers refresh bundled backend code even when their
semantic version has not changed, while preserving environments and models.

SA3's Underfit model registries and training templates are now tracked instead
of being supplied by an ignored local dashboard folder, making clean Git
checkouts and release-builder checkouts produce the same training bundle.
Generated Python `*.egg-info` directories are also excluded from runtime
staging so local editable installs cannot leak build metadata into a release.

compatible with gary4juce v4.0.14.

## v0.3.0

### custom runtime storage

gary4local no longer assumes all of its mutable data belongs in
`%APPDATA%\Gary4JUCE`. fresh installs put service environments, models, caches,
logs, and Gary-trained LoRAs in `gary4local-data` beside the installed app. if
you choose an install folder on another drive, the large runtime follows it.

existing installs keep using `%APPDATA%\Gary4JUCE` when Gary data is already
there. legacy mode also leaves the Hugging Face cache variables alone, so an
existing `~/.cache/huggingface` remains visible and models don't need to be
downloaded again just because gary4local updated.

the storage panel can choose a different runtime for the next restart. the
saved Hugging Face token and app settings follow automatically. Gary-trained
SA3 and Carey LoRAs can be copied after the move; external LoRAs are left alone
and can be registered again from their original files. copying is deliberately
not moving: the source LoRAs stay where they were until you decide what to do
with them.

old-storage cleanup only unlocks after the app has restarted somewhere else.
it lists exactly which old environments, managed models, and rebuildable caches
it found before anything is removed. it doesn't quietly delete the LoRAs left
behind in the old profile.

### storage maintenance

the storage panel can clear uv's package cache and remove individual service
environments. every model panel can remove models it owns, including
MelodyFlow's model, Carey components, Jerry finetunes, and Foundation-1. model
and environment sizes are shown before cleanup so the large number in Explorer
isn't just a mystery.

Hugging Face snapshots often point at the same blobs, and older Gary installs
could leave both a snapshot copy and its blob behind. size reporting avoids
counting links twice, while cleanup can reclaim a duplicate blob without
removing the surviving model file. the larger scans run away from the UI thread
so opening the storage panel doesn't make the whole app look frozen.

### gary (MusicGen)

gary's generation routes accept a seed now, matching the advanced controls in
gary4juce v4.0.13. the optional fast-generation path can be turned off from the
control center, and its xFormers fallback now reaches generation on machines
where xFormers isn't installed.

startup imports are lighter, audio no longer gets routed through TorchCodec,
and `hf_xet` is installed for model repositories that use Hugging Face's Xet
storage. `pesq`, which gary doesn't use for inference, is no longer part of the
environment.

compatible with gary4juce v4.0.13.

## v0.2.1

### carey (ACE-Step) LoRA trainer

the base/xl-base selector was sending the short aliases `base` and `xl-base`,
which no longer resolve to anything: `MODEL_MAP` is keyed by the real folder
names (`acestep-v15-base`, `acestep-v15-xl-base`). the selector has also moved
into its own **preparation + training model** section, because it decides how
the dataset gets preprocessed as well as what gets trained.

the worse half was `load_silence_latent`. its third search step scanned every
known variant subdirectory and took the first `silence_latent.pt` it found, so
selecting xl-base without its assets downloaded silently trained against base's
latent instead of failing. it now resolves the selected model or raises. if you
had every carey model downloaded, step 2 always matched and you never reached
the fallback — this only shows up on a partial install.

the trainer also gets LoRA catalog controls, refuses to reuse a name that is
already registered, and cleans up checkpoints more carefully when a run ends or
is cancelled.

### sa3

LoRA layer scope is selectable, and the default drops from the full reference
set to `transformer-core`: the seven attention and feed-forward projections in
each of the 24 transformer blocks, 168 adapters, matching the efficient MLX
scope. the full 229-target reference scope and a 228-target variant without the
duration conditioner are both still available for exact recipe parity. 168
trains faster and produces a normal SA3 LoRA checkpoint.

auto-labelling can now pick its captioner, including the 4B ACE-Step model if
you have the VRAM for it. the prompt pool falls back to the bundled dice prompts
when it can't be reached, and LoRAs trained by gary rather than sa3 are cleaned
out instead of sitting in the list.

### terry (MelodyFlow)

**use seed** submits a fixed seed and the box fills in with whatever seed the
last transform ran. worth being straight about why it exists: seed matters far
less for melodyflow transformations than it does for sa3 LoRA blending. it is
here so the same transform can be run on two machines and compared, which is
what testing new hardware needs. every transform draws noise twice — once
sampling the VAE posterior of the encoded prompt in `flow.generate`, and again
per solver step while regularizing — and both come from the global RNG, so one
`torch.manual_seed` before `edit()` covers the whole run. it is not
bit-identical: the VAE encoder's convolutions pick nondeterministic cuDNN
algorithms, so two same-seed euler transforms correlate at 1.000000 without
being byte equal.

terry no longer resamples to 32kHz before editing. that path round-tripped
correctly — input resampled down, output written back down, results matching the
source in pitch and tempo — which is why it survived this long. running at the
model's own 48kHz means it sees more detail in the input. the tradeoff is
length: the 750-latent window at the VAE's 25Hz frame rate is 30 seconds of
48kHz audio, where the 32kHz stream stretched the same window to 45.

finding how much audio fits used to be a binary search with a GPU encode per
iteration, and it caught every exception as "too long", so a failing encoder
became a silent one-second result rather than an error. the latent count is
exactly linear in sample count, so it is one calculation and one encode now.

foundation returns a real message when a host reports an impossible tempo.
savihost sent 3159345 BPM through gary4juce and the reply was a bare 400 with
nothing in the service log; the range check was working, but the response only
carried a plural `errors` key that clients don't read. it now names the value
and the accepted range in both the response and the log.

the `[OK] xformers memory efficient attention available` line prints once per
process instead of roughly 48 times per generation — it runs from
`StreamingMultiheadAttention`'s constructor, and terry rebuilds its model
between requests.

### elsewhere

the model panel shows how much disk each downloaded model is using, and every
LoRA picker remembers the folder you were last in.

compatible with gary4juce v4.0.12.

## v0.2.0

v0.2.0 swaps the SA3 LoRA trainer over to stable-audio-3's own Lightning
trainer instead of the hand-written training loop we were carrying. it's the
same code the inference service already uses, so the training step matches the
reference exactly instead of slowly drifting from it. `pytorch-lightning`
installs itself the first time you train, so you shouldn't need **rebuild env**
for it.

the shared trigger word finally does what you'd expect. it gets prepended to
every caption, so you end up training on `my-trigger, some caption` and the word
comes to mean the style. before this it was *replacing* the caption around 60%
of the time, which meant a lot of steps trained on a bare trigger with nothing
describing the audio. if you trained a LoRA with a trigger word on an older
version, it's worth retraining it. the "prepend shared phrase" toggle in the
prompt editor is gone too — sidecars are just captions now, and the trigger is
handled for you.

two LoRA annoyances are fixed:

- a freshly trained LoRA didn't show up in gary4juce until you opened the
  **add loras** popup. the trainer wrote the catalog, but nothing rebuilt the
  registry until that window happened to open.
- switching a LoRA's training checkpoint while the SA3 service was running
  silently did nothing, because adapters get baked in when the model loads. it
  now reloads the service so the switch actually takes effect.

training should also sit better on 8 GB cards. the VAE and the T5Gemma text
encoder get freed or pushed to CPU once they aren't needed, which is roughly
2 GB back, and the logs now show the caption for each clip plus the trigger word
being applied so you can see what you're actually training on.

compatible with gary4juce v4.0.7.

## v0.1.19

v0.1.19 adds full auto-labeling to the SA3 trainer. it uses the ACE-Step
captioner to pull genre keywords for SA3 prompt sidecars, alongside the local
BPM and key helpers. use **rebuild env** before using the new SA3 trainer
features.

both integrated trainers are more resilient now: caption/preprocessing and
checkpoint checks are stricter, interrupted launchers and child workers recover
cleanly, and Windows now owns managed inference and training processes as a
group so an unexpected app exit cannot leave GPU workers running. closing to
the tray still leaves work running as intended.

compatible with gary4juce v4.0.6.

## v0.1.18

v0.1.18 fixes Carey seed reporting so the seed shown after a random generation
is the seed that actually produced the audio.

ACE-Step LoRA sidecars now treat lyrics as BYOL metadata. the captioner still
helps with captions, genres, BPM, and key metadata, but it no longer writes
LM-hallucinated lyrics into vocal sidecars. the sidecar editor shows a grey
lyrics template instead, so users can paste or write the real words themselves.

compatible with gary4juce v4.0.4.

## v0.1.17

v0.1.17 adds carey seed support for lego, complete, and cover, plus an optional
[ScragVAE](https://huggingface.co/scragnog/Ace-Step-1.5-ScragVAE) decoder
toggle for ACE-Step.

it also keeps the local ACE-Step LoRA trainer moving forward. early testing says
regular `acestep-v15-base` is still the safer lego model when no LoRA is loaded,
while `xl-base` gets much more exciting once you have a matching xl-base LoRA,
especially for vocals and backing vocals.

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
