# custom model backends

gary4local wraps several music models behind local services that can be shared
by gary4juce, the sa3 Ableton extension, and whatever else we build next. this
doc keeps track of the custom behavior and localhost-specific optimizations
that don't really belong on the front page.

## gary localhost optimizations

the local `gary` service applies several MusicGen inference optimizations that
are specific to the localhost deployment:

- `musicgen_fast.py` converts remaining float32 parameters and buffers to fp16
  to avoid extra dtype conversion overhead during generation.
- self-attention layers are patched to use a pre-allocated static KV cache
  instead of repeatedly growing tensors with `torch.cat`.
- if available in the local gary environment, Flash Attention 2 is patched in
  directly for MusicGen self-attention.
- the service performs a small first-load kernel warmup pass per model/device
  so later generations start faster.

for localhost we intentionally **don't** enable `torch.compile` by default.
gary unloads models after generation so we can support many finetunes on
smaller GPUs without keeping large model instances resident, and that model
lifecycle usually makes compile overhead a poor tradeoff.

## terry localhost optimizations

the local `terry` service supports an optional Flash Attention 2 path for
MelodyFlow on CUDA:

- terry runs MelodyFlow on `torch 2.7.1` with the same Windows FA2 wheel family
  used by the other CUDA services.
- `melodyflow_fast.py` patches the DiT self-attention blocks at runtime to call
  FA2 directly when the wheel is installed and the tensors are in a supported
  CUDA dtype.
- cross-attention stays on the existing AudioCraft attention path because
  those blocks can carry masks, so the FA2 patch stays focused on the large
  self-attention passes over audio latents.
- the optimization is disabled by default and can be enabled with
  `MELODYFLOW_USE_FLASH_ATTN=1` or from the `gary4local` terry panel when that
  feature is included in the build.

## carey localhost notes

the local `carey` service tracks more of the custom
[ace-lego](https://github.com/betweentwomidnights/ace-lego) behavior while still
being practical on smaller Windows GPUs.

see the [ACE-Step LoRA training guide](ace-step-lora-training.md) for the
integrated trainer workflow, hardware caveats, captioning notes, and current
experimental recommendations.

- `lego`, `extract`, `complete`, and `cover` are all exposed through the
  localhost wrapper.
- `lego` uses the active base checkpoint family only. when the XL toggle is
  off it routes to `acestep-v15-base`; when the XL toggle is on it routes to
  `acestep-v15-xl-base`. it does not expose turbo or SFT choices.
- for lego mode without a LoRA, regular `acestep-v15-base` is still the
  recommended path. plain `acestep-v15-xl-base` has been rough in testing.
  with a matching xl-base LoRA, though, lego vocals and backing vocals can be
  excellent.
- some lego targets with xl-base LoRAs are still lightly tested, and xl-base
  vocal LoRAs can occasionally bleed a little instrumentation into the output.
  the funny part is that the bleed usually fits the source audio anyway.
- `cover` always routes to the turbo checkpoint and stays fixed at 8 steps /
  CFG 1.0.
- `complete` accepts `base`, `turbo`, or `sft` from localhost clients. `turbo`
  stays fixed at 8 steps / CFG 1.0, while `base` and `sft` keep editable steps
  and CFG.
- the `gary4local` carey UI includes an XL toggle. when enabled, those same
  localhost model choices map to `acestep-v15-xl-base`, `acestep-v15-xl-sft`,
  and `acestep-v15-xl-turbo` under the hood instead of the regular checkpoints.
- the carey panel has a separate `add lora` flow instead of mixing user
  adapters into the checkpoint download UI.
- each local LoRA entry stores a name, `model family` (`standard` or `xl`), a
  checkpoint folder, and an optional separate captions/source folder. this
  supports the common Side-Step workflow where the exported adapter and the
  training sidecars don't live together.
- if a checkpoint folder includes `metadata.json`, gary4local reads `scale`,
  `backends`, and `model_family` from it. accepted backend tags are `base`
  and `turbo`; lego uses the `base` tag. if metadata is missing, the app defaults to
  `scale: 1.0`, `backends: ["base", "turbo"]`, and infers the family from the
  folder name or current XL mode.
- saving entries writes `%APPDATA%\Gary4JUCE\carey\lora_registry.json`.
  building captions writes `%APPDATA%\Gary4JUCE\carey\captions.json` by
  scanning `.txt` sidecars from the chosen captions/source folder or, if
  omitted, from the checkpoint folder itself.
- the bundled `default` caption pool is seeded from
  `services/carey/default_captions.json`. per-LoRA pools are built from the
  sidecar `caption:` and `genre:` lines. LoRAs without sidecars still work for
  generation, but the gary4juce dice button falls back to the default pool.
- gary4juce only sees LoRAs whose `model family` matches the current carey XL
  toggle. `standard` LoRAs are hidden when XL is on, and `xl` LoRAs are hidden
  when XL is off.
- we currently recommend 16 GB of VRAM for the XL toggle even though some
  slower first-use generations may still succeed below that on certain cards.
- model startup is intentionally lazy. switching the XL toggle updates the
  routing config, but carey waits until the first request to download or
  initialize the required checkpoint.
- during decode, localhost carey can temporarily offload the DiT model so the
  VAE decode step has more VRAM available.
- the decode path falls back through progressively safer modes, including
  tiled decode, CPU-offloaded decode, and full CPU decode, instead of
  hard-failing immediately on lower-memory GPUs.

this keeps ACE-Step usable on consumer cards where generation may fit in VRAM
but decode is the step that would otherwise tip the process into an
out-of-memory failure.

## sa3 localhost notes

the local `sa3` service mirrors the remote gary4juce contract for generate,
loop, transform, continue, prompt dice, LoRA listing, and poll status. it was
built from the same backend contract as
[sa3-api](https://github.com/betweentwomidnights/sa3-api), which remains the
clean reference for the remote SA3 API shape.

- sa3 runs on `http://localhost:8006`.
- LoRA entries are managed in the control center and written to
  `%APPDATA%\Gary4JUCE\sa3\lora_registry.json`.
- the integrated LoRA trainer uses a stripped-down version of
  [underfit](https://github.com/dada-bots/underfit) and automatically registers
  completed adapters.
- the optional dataset prompt editor creates and edits same-name `.txt`
  sidecars using the literal prompt format consumed during training.
- training jobs expose live logs, persist their status when the control center
  closes, and can be cancelled gracefully.
- prompt dice pools live under `%APPDATA%\Gary4JUCE\sa3\prompts`.
- `continue` supports both `inpaint` and `latent_prefix` continuation modes.
- output shaping controls expose the first pass at local loudness management
  for hot LoRA outputs.
- manual and preview decode paths normalize latents to the decoder's device and
  dtype, including CFG paths that may return float32 latents.
- half-precision sa3 models are converted before transfer to CUDA, reducing
  transient GPU memory pressure during load and reload.
