# changelog

this is where we're keeping the version history that used to live at the top
of the main README. the README should stay focused on what gary4local is now;
this file gets to remember how we got here.

## v0.2.1

terry gets a seed. tick **use seed** and the box below it fills in with whatever
seed the last transform actually ran, so you can send the same one again and get
that take back. it works the way sa3 and carey already do. one honest caveat: the
same seed isn't bit-identical run to run, because the gpu picks its own
convolution algorithms and a 75 step solve amplifies the difference. it is very,
very close — close enough to compare two machines, but don't expect byte
equality.

**terry now runs at its native 48kHz, and transforms cap at 30 seconds instead
of 45.** we were resampling input to 32kHz and writing the output back at 32kHz,
and those two mistakes cancelled out for listening, which is why nobody caught
it. the model was hearing everything a fifth high the whole time. the extra 15
seconds was a side effect of the same mismatch. output files are 48kHz now.

foundation tells you what went wrong when your host reports a nonsense tempo.
savihost sent 3159345 BPM through gary4juce and all you got back was a bare 400
with nothing in the log. it now says which value it rejected and what range it
takes, in both the response and the service log.

terry also stops re-deriving something it already knew: finding how much audio
fits used to binary search with a gpu encode per step, and it swallowed every
error while doing it, so a failing encoder turned into a silent one second
result instead of an error. and the `[OK] xformers memory efficient attention
available` line now appears once instead of roughly 48 times per generation.

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
