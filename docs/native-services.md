# native services

every model gary4local runs now runs on a native GGML runtime: a prebuilt
server per model family, built in its own repo, downloaded and checked by
gary4local, and run on your GPU with CUDA or Vulkan. no Python, no torch, no
build step on your machine.

this doc is about what those runtimes are, where they come from, how they
differ from the projects they're based on, and the gary4local-specific behavior
that doesn't belong on the front page. how a runtime is packaged, pinned and
installed is in [native runtime packages](native-runtime-packages.md). the old
Python services are still in the repo, and [python services](python-services.md)
explains why.

| service | port | runtime | based on |
|---|---|---|---|
| `gary` (MusicGen) | 8000 | [audiocraft.cpp](https://github.com/betweentwomidnights/audiocraft.cpp) `musicgen-server` | Meta's [audiocraft](https://github.com/facebookresearch/audiocraft) |
| `terry` (MelodyFlow) | 8002 | audiocraft.cpp `melodyflow-server` | Meta's audiocraft |
| `carey` (ACE-Step 1.5) | 8003 | [acestep.cpp](https://github.com/betweentwomidnights/acestep.cpp) `ace-server` | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), via ServeurpersoCom's acestep.cpp |
| `jerry` (Stable Audio Open Small) | 8005 | [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp) `sat-server` | Stability's stable-audio-tools |
| `sa3` (Stable Audio 3) | 8006 | sa3.cpp `sa3-server`, behind a small adapter | Stability's [stable-audio-3](https://github.com/stability-ai/stable-audio-3) |
| `foundation-1` | 8015 | sa3.cpp `sat-server` | [Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) by RoyalCities |
| `yuey` (YuE2) | 8007 | [yuey.cpp](https://github.com/betweentwomidnights/yuey.cpp) `yue2-server` | [YuE2](https://github.com/multimodal-art-projection/YuE) by HKUST and M-A-P |

all four runtimes build on the same ggml fork,
[betweentwomidnights/ggml](https://github.com/betweentwomidnights/ggml), pinned
at upstream GGML 0.29 plus the training patches sa3.cpp and acestep.cpp need.

## backends

on Windows, every runtime offers CUDA and Vulkan and tries CUDA first on
NVIDIA. AMD and Intel get Vulkan. CUDA's runtime DLLs install once under
`native-runtimes/` in runtime storage and every CUDA runtime shares them.

on Linux (amd64 and arm64), every runtime is Vulkan only, on NVIDIA as well as
AMD and Intel. there's no Linux CUDA build, and there's no CPU runtime
anywhere: a machine without an NVIDIA, AMD or Intel GPU is told so when it
tries to install one.

each install runs the server's `--props` check to confirm the backend actually
came up. on auto, a backend that fails falls through to the next one, and the
panel says so rather than quietly running somewhere else. a backend you pick
yourself is the only one tried, so it fails loudly instead. changing it takes a
runtime reinstall.

## moving from Python

if you had a service set up on Python before its runtime existed, it stays on
Python until you press `migrate runtime`. migration installs the runtime beside
the Python env, prepares the models, converts any LoRAs (as copies; originals
stay where they are), then starts the native server on a private port and runs
real jobs through it before it switches anything. cleanup is a separate step
you start and review, and it never lists your LoRAs, training runs, prompts or
outputs. a fresh install skips all of this and sets up native from the start.
the switch is recorded per storage profile, so another profile keeps its own
state.

## gary and terry: audiocraft.cpp

audiocraft.cpp was written in our repo, not forked from another .cpp project.
it ports MusicGen's `generate_continuation`, MelodyFlow's `edit`, the T5-base
conditioner, EnCodec 32kHz and MelodyFlow's VAE, and it matches PyTorch closely
(6000 of 6000 tokens for MusicGen, cosine 0.9999 for terry). it deliberately
leaves out melody and style conditioning, `top_p`, extending past 30 seconds,
two-step CFG and torch's seeds.

**gary**

- `musicgen-server` takes port 8000 itself, no adapter. it serves
  `/api/models`, `process_audio`, `continue_music`, `retry_music` and
  `poll_status`.
- `/api/models` lists the GGUFs actually on disk instead of Python's hardcoded
  list. gary4juce picks the model per request.
- converted so far: `vanya_ai_dnb_0.1` (small, and gary4juce's fallback),
  `hoenn_lofi`, `keygen-gary-v2-large-12`, and Meta's small, medium and large.
  the rest of the `thepatch/*` finetunes (bleeps, keygen-gary medium and the
  other v2 sizes, gary_orchestra_2) aren't converted yet. migration cleanup
  removes their Python downloads and names them first; they'll show up in
  gary4juce again once they're converted.
- precision is F16, Q8_0 or Q4_K_M. the tier you choose decides which vanya
  file gets downloaded; the server runs the most precise copy of a model it
  finds on disk. Q8_0 and Q4_K_M are only checked by round trip so far, so
  listening is the remaining test.
- since audiocraft.cpp v0.1.4, `continue_music` and `retry_music` return your
  input with the continuation appended, at the input's sample rate, in stereo.
- a request with no description generates unguided, which is faster and
  wanders, same as audiocraft. a continuation can open with a near-silent beat;
  that comes from audiocraft's own codes, not the port.
- short input isn't looped or padded the way the Python service did. it's
  only cropped.
- models are released after every request, so many finetunes fit on a small
  GPU. F16 large needs about 6.5 GB.

**terry**

- `melodyflow-server` on port 8002 serves `/variations`, `/transform`,
  `transform_audio`, `undo_transform` and `poll_status`. `/variations` works,
  where the Python service returned a 500.
- it runs at 48kHz stereo, and transforms crop to the first 30 seconds.
- one model: `thepatch/melodyflow-t24-30secs-GGUF`, F16.
- 33 presets. a custom prompt replaces the preset's. a missing seed gets a
  random one, and the seed comes back with the result, but seeds don't
  reproduce the Python service's output.
- about 30 seconds for a 30 second transform on an RTX 5070 Laptop.

the FA2, fp16 and static KV switches from the Python services don't exist here,
and the panels hide them once a service is native.

## carey: acestep.cpp

acestep.cpp is our fork of [ServeurpersoCom's acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)
(through twilwa's Sawhee), which ports ACE-Step 1.5 by ACE Studio and StepFun.
on top of upstream, the fork adds:

- `ace-train`, native LoRA and DoRA training with resume, a preprocessing
  cache, progress and cancel files.
- functional adapters by default, so a LoRA runs on a quantized base without
  merging and requantizing it. DoRA works on quantized bases too.
- `/caption` and `/analyze`, with BPM and key helpers ported from sa3.cpp, and
  genres in the understand pass, limited to ACE-Step's own vocabulary.
- `adapter_info` in `/props`, job progress and failure reasons for clients,
  and `ACE_*` environment variables for configuration.
- VAE tiles sized to the free GPU memory, and training on songs longer than
  600 seconds.

we don't use upstream's web UI, `/lm` generation from gary4juce, merge-mode
adapters or single-file adapters.

**in gary4local**

- `ace-server` takes port 8003 itself. there's no `/lego` or `/cover` route:
  gary4juce sends every task to `/synth` as a `task_type`:

  | gary4juce | `task_type` | model |
  |---|---|---|
  | lego | `lego` | base only |
  | continue | `repaint` | base, sft or turbo (turbo fixed at 8 steps, CFG 1.0) |
  | complete | `complete` | base only |
  | cover | `cover-nofsq` | turbo or base (turbo fixed at 8 steps, CFG 1.0) |

- 2B versus XL is chosen in gary4juce now, by model name, not with a toggle in
  gary4local. models load per request and are freed afterwards.
- for lego mode without a LoRA, regular base is still the recommended path.
  plain xl-base has been rough in my testing. with a matching xl-base LoRA,
  though, lego vocals and backing vocals can be excellent. xl-base vocal LoRAs
  can occasionally bleed a little instrumentation into the output. the funny
  part is that the bleed usually fits the source audio anyway. (that was on the
  Python service; it's the same model.)
- models come from `Serveurperso/ACE-Step-1.5-GGUF`: base, sft, turbo and their
  XL versions, each in BF16, Q8_0, Q6_K, Q5_K_M and Q4_K_M, plus the shared
  embedding model and VAE. the recommendation is the biggest tier that fits
  with 2 GiB to spare. on an 8 GB card that's 2B at BF16 and XL at Q8_0, so
  XL doesn't need 16 GB anymore unless you want it at BF16.
- with more than one precision of a model downloaded, gary4juce renders the
  most precise one present.
- LoRAs are listed in `carey/lora_catalog.json` in runtime storage and
  published to ace-server as PEFT folders under `carey/adapters`. gary4juce
  only shows LoRAs whose model family matches the size you picked. a new or
  changed LoRA shows up after carey restarts.
- the dice pool for a LoRA is built from each song's caption, its genre line,
  and combinations of its tags. auto-labeling takes a BPM from the filename
  first, then a confident local estimate, then the LM, and never writes LM
  lyrics.
- the trainer trains on base or xl-base, sends DoRA as row-wise DoRA, uses
  batch size 1, and takes windows of 1 to 600 seconds (45 to 60 by default).
  windows over 240 seconds are untested and need far more VRAM. the
  [ACE-Step training guide](ace-step-lora-training.md) has the rest.
- local C++ carey needs gary4juce v5.0.2 or later.

## sa3, jerry and foundation-1: sa3.cpp

sa3.cpp was written in our repo, not forked. it ports Stable Audio 3 and the
classic stable-audio-tools pipeline behind Stable Audio Open Small, Stable
Audio Open 1.0, Foundation-1 and Keybeds, checked against PyTorch as it went.
Stability's stable-audio-3 and stable-audio-tools are its reference;
RoyalCities' tools and model card are the reference for Foundation; underfit
informed its LoRA and DoRA training. on top of those it adds native training,
LoRA automation lanes, a C ABI and a web studio, and a C++ port of
Foundation's prompt engine that doesn't reproduce Python's RNG byte for byte.

one runtime bundle in `services/sa3/native` serves all three services.

**sa3**

- `sa3-server` runs on a private port (18006). a small adapter in gary4local
  keeps the Python-era API on 8006: generate, loop, transform, continue (both
  `inpaint` and `latent_prefix`), prompt dice, LoRAs, load and unload, output
  shaping, and LoRA automation lanes.
- requests cap at 380 seconds and only take the `pingpong` sampler.
- models: the T5Gemma encoder plus the SAME-L decoder and conditioner, then one
  DiT tier: F16, Q8_0, Q5_K_M or Q4_K_M. training uses a separate medium-base
  DiT, F16 or Q4_K_M (Q4_K_M is suggested on CUDA cards with 9 GB or less).
- LoRAs are GGUF. migration converts your Python LoRAs into
  `sa3/native-loras` and leaves the originals where they are. a `.ckpt` LoRA
  needs the old Python env once to export it.
- the trainer is `sa3-train`, row-wise DoRA, rank 16 by default, over 168
  layers (core), all 229 (full), or 228 (full without the `seconds_total`
  conditioner). `train on full tracks` is a 285-second
  window: a track that fits is used whole, a longer one gives a random
  285-second stretch each step. since v0.2.4, tracks longer than 600 seconds
  are encoded whole instead of being cut at 600.
- the optional decoder correction is a small LoRA on the decoder
  (`squeakfix_v3`), converted natively.

**jerry**

- `sat-server --model saos` takes port 8005 itself. one process serves the
  Stable Audio Open Small base and its kickbass and jerry grunge finetunes,
  picked per request.
- up to 11 seconds. 8 steps and CFG 1 for the base, 50 and 4 for finetunes.
- the native jerry grunge really is the finetune. the Python service's loader
  quietly ran the base model in its place, so the two sound different.

**foundation-1**

- `sat-server --model foundation-1` takes port 8015 itself: generate, loop,
  randomize, audio2audio.
- by default it samples like the old gary profile: DPM++ 2M SDE, sigma 0.5 to
  50, 100 steps, CFG 7. RoyalCities' own settings (DPM++ 3M SDE, sigma 0.01 to
  100) are used when the request asks for `inference_profile: "royalcities"`,
  which is the toggle in gary4juce, off by default.
- `host_bpm` stretches without changing pitch, and big ratios can degrade the
  audio.

## yuey: yuey.cpp

yuey.cpp ports YuE2 (the AR stage, the flow-matching stage and its 48kHz
Oobleck VAE), plus SheetSage2 and MERT for transcription. the instrumental and
continuation adapters are third-party, from Mothersuperior. official YuE is the
reference for the ABC and lyrics dialect. yuey.cpp adds:

- the two-voice ABC read the way YuE's own tools read it, with typed BPM, key
  and meter, and `/score/transform`.
- remix and continue through transcription. continue also extracts a
  real-audio prefix and needs the continuation adapter.
- frugal loading: models load per job and the generator is freed before the VAE
  decodes, in 512-frame windows. that's what stopped long songs running 8 GB
  cards out of memory at the very end.
- the decode step replayed as a CUDA graph, which roughly doubled decode speed.
- FLAC on the wire, and a fix for transcription on Vulkan that shipped before
  gary4local ever installed a runtime.

**in gary4local**

- `yue2-server` on 8007. on Windows it's CUDA first on NVIDIA; on an RTX 5070
  Laptop CUDA is the faster backend (a 4000-token instrumental took 96 seconds
  on CUDA against 154 on Vulkan). yuey's own docs call CUDA the fully validated
  backend and Vulkan a work in progress.
- installing the runtime doesn't download models. `download recommended models`
  fetches the shared set (VAE, tokenizer, SheetSage2, the semantic tokenizer,
  and the instrumental and continuation adapters) and one generation tier, and
  yuey won't start until they're there.
- models come from [thepatch/YuE2-3B-GGUF](https://huggingface.co/thepatch/YuE2-3B-GGUF)
  into `models/yuey`. `Q4_K_M` is recommended for 8 GB GPUs, `Q8_0` for 12 GB
  and `BF16` for 16 GB. Q5_K_M isn't published yet, so a card the server would
  give Q5_K_M gets Q4_K_M.
- gary4local always names a tier at launch: the one you picked, else the
  recommended one, else the smallest downloaded. `yue2-server` on its own would
  take the most precise it can find, which on a small card means running out
  of memory. changing the tier restarts a running yuey.
- the yuey generation panel sets three server defaults, passed as env at
  launch and applied to every client, gary4juce included. saving restarts a
  running yuey, and a request that sets one of these itself still wins:
  - `YUE2_INSTRUMENTAL_METHOD`: what an instrumental job does with the sung or
    planned melody. "official YuE" (`transfer`) moves it onto an instrument and
    is the default; "our original" (`rest`) takes it out and leaves the backing
    part.
  - `YUE2_USE_INSTRUMENTAL_ADAPTER`: whether the instrumental adapter is used.
    on by default.
  - `YUE2_NATURAL_MAX_SECONDS`: the natural-length ceiling, 30 to 600 seconds.
    gary4local defaults to 96 like the remote backend (the server's own default
    is 180). a continuation is held to it by the bars it adds.

## stems

[stems.cpp](https://github.com/betweentwomidnights/stems.cpp) is a sibling
runtime too, but gary4juce runs it directly. gary4local doesn't install it.
