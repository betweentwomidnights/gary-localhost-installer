# ACE-Step LoRA training in gary4local

This guide covers the ACE-Step trainer built into gary4local. It is not meant
to be the final word on training ACE-Step LoRAs. This is still experimental,
and part of the fun is figuring out what works for your own music.

## Fair warning: what has actually been tested

This trainer has only been tested on an NVIDIA RTX 5070 Laptop GPU with 8 GB of
VRAM. Regular `acestep-v15-base` training works there. I have not yet been able
to validate training against `acestep-v15-xl-base`, although the pipeline is
designed to support it.

You will probably want 16–24 GB of VRAM for XL-base training. The trainer
offloads frozen model components and performs a VRAM preflight before the first
batch, but that check cannot make an oversized model fit on a small GPU.

I have gotten decent instrumental results from regular base. In my testing,
though, Stable Audio 3 has generally been the better option for purely
instrumental datasets. ACE-Step seems especially interesting when the dataset
contains vocalists.

## The basic workflow

1. Choose a folder of audio files and give the LoRA a name and trigger word.
2. Run **caption / prepare**. This fills missing sidecars with ACE-Step's
   captioner and performs the optional BPM/key sanity check.
3. Open **edit prompts / sidecars** and review the generated metadata. Human
   ears still win.
4. Choose the model, instrumental/vocal mode, adapter type, epochs, learning
   rate, and maximum track length.
5. Leave the advanced settings alone for a first run unless you have a reason
   to change them.
6. Start training. The trainer saves periodic checkpoints, keeps a smoothed
   best-loss checkpoint, and registers the selected adapter with Carey when the
   run finishes.

## A note about the captioner

The captioner's results will not be perfect. We use ACE-Step's own captioner
because the base model was trained around this style of metadata. Even when a
caption or genre does not perfectly describe your music, it still resembles
the conditioning language ACE-Step expects.

Use the 4B captioner when you have enough GPU memory; it should produce the best
metadata. The 1.7B model is a practical good-quality choice. The 0.6B model is
available for constrained hardware, but it is not recommended when either of
the larger models fits.

Do not be afraid to edit the results. BPM, key, time signature, lyrics, genre,
and captions remain editable in the sidecar UI. The local BPM/key check is also
deliberately conservative: it is there to catch obvious captioner mistakes,
not to overrule a musician.

At generation time, Gary4JUCE's dice button draws from the caption and genre
pool associated with the LoRA. In practice, that means the exact wording of
every training caption may matter less than you expect. I personally hate
prompting, so I often let RNGesus keep rolling until I hear something I like.

## Trigger words and varied prompts

The trigger word is prepended during preprocessing like this:

```text
your-trigger-word, original caption
```

This lets you try the learned distribution against prompts beyond the exact
captions in the dataset. The trigger should be distinctive and easy to type.
How literally you use it—or whether you mostly rely on the dice button—is up to
you.

## Base, Turbo, and SFT

One of the useful parts of ACE-Step LoRAs is that an adapter trained against
the standard base checkpoint can also be applied to the standard Turbo and SFT
checkpoints. Early results here have favored applying a regular-base LoRA to
the regular SFT model.

You do not have to keep the model families straight by hand. Gary4local records
whether you trained against regular base or XL-base, then keeps that LoRA with
the matching standard or XL family automatically.

XL-SFT remains more uncertain. There is an
[ACE-Step issue describing rhythm, tempo, and key instability in XL-SFT](https://github.com/ace-step/ACE-Step-1.5/issues/1203).
The issue was closed automatically for inactivity rather than confirmed fixed,
so treat it as an unresolved caveat—not proof that every XL-SFT generation is
broken.

## Understanding the training controls

### Adapter type

DoRA is the recommended default. Plain LoRA is a somewhat lighter and simpler
alternative. If memory is extremely tight, LoRA may be worth trying.

### Epochs

One epoch is one complete pass over every track in the dataset. More epochs
mean more opportunities to learn the material, but training for longer is not
automatically better. Listen to saved checkpoints when possible instead of
assuming the last epoch must be the best one.

### Learning rate

Learning rate controls how aggressively each optimizer step changes the
adapter. `1e-4` is a lighter touch; `3e-4` trains harder and is the current
default. If a LoRA becomes harsh, unstable, or too literal, a lower learning
rate is one of the first things to try.

### Rank

Rank controls the capacity of the adapter: roughly, how much room it has to
learn changes to the base model without modifying the full model weights.
Ranks 64 and 128 currently appear to be the most useful range for ACE-Step.

Rank 128 consumes substantially more VRAM, especially with balanced attention
and MLP coverage. If rank 64 is already near your GPU's limit, do not assume
128 will fit. The VRAM preflight will block configurations that are clearly
unsafe, but leaving additional headroom is still wise on a display GPU.

### Balanced attention + MLP

The **balanced attention + MLP** profile is experimental. It is informed by
ideas explored in [koda-dernet's Side-Step](https://github.com/koda-dernet/Side-Step),
but Gary does not perform Side-Step's adaptive Fisher analysis. Instead, it
uses a fixed, architecture-level distribution of rank across self-attention,
cross-attention, and feed-forward projections.

That is intentionally simpler and less dataset-specific. So far it has made my
test LoRAs sound cleaner, but that is an early result—not a guarantee.

If you want adaptive analysis and a more advanced training setup, use
[Side-Step](https://github.com/koda-dernet/Side-Step).

### Batch size and gradient accumulation

Batch size and gradient accumulation should usually remain at 1 for a small
dataset such as a single album. That gives each track its own optimizer update
instead of averaging several tracks into one update.

Larger datasets may benefit from accumulating gradients across multiple
examples. The tradeoff is fewer, smoother optimizer updates and additional
memory or time. Change these settings because the dataset calls for it, not
because larger numbers look more powerful.

### Min-SNR loss weighting

ACE-Step training samples different flow timesteps, which correspond to
different mixtures of clean audio and noise. With a flat mean-squared-error
loss, high-signal timesteps can produce disproportionately strong gradients and
dominate the update.

Min-SNR estimates the signal-to-noise ratio at each sampled timestep and caps
the influence of those high-SNR examples. In plain language, it keeps the easy,
cleaner parts of the denoising problem from shouting over everything else. The
goal is a more balanced learning signal across noise levels, not a stronger
effect or a replacement for good data.

The default gamma of 5 is a sensible starting point. Leave it there unless you
are deliberately comparing loss-weighting behavior. Gary's implementation is
adapted to ACE-Step's flow interpolation and follows the method described by
[Hang et al., “Efficient Diffusion Training via Min-SNR Weighting Strategy”](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html).

## What is still on the wish list

Gary4JUCE's Lego mode currently keeps LoRAs disabled. Now that regular
v1.5-base training works locally, I want to open that path and test whether a
vocal LoRA can influence Lego-mode vocals.

Training a LoRA on a specific guitar style and using it in Lego mode also
sounds incredibly based. I do not think many people have tried LoRAs with Lego
mode yet, so I would like to confirm whether it is cool or stupid before
presenting it as a feature.

Until then: save checkpoints, change one variable at a time, and trust your
ears more than the loss graph.
