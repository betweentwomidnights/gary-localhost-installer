# changelog

this is where we're keeping the version history that used to live at the top
of the main README. the README should stay focused on what gary4local is now;
this file gets to remember how we got here.

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
