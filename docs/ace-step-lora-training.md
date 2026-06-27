# ACE-Step LoRA training in gary4local

this isn't meant to be the definitive guide to training ACE-Step LoRAs. it's
mostly a collection of what i've tested so far, what seems to be working for
me, and enough explanation to help you start your own experiments.

## fair warning

this trainer started on an NVIDIA RTX 5070 Laptop GPU with 8 GB of VRAM, which
is enough for regular `acestep-v15-base` training with the current offload
setup. `acestep-v15-xl-base` training has now worked in local testing too, but
i'd still treat it as a bigger-card workflow. you'll probably want 16-24 GB of
VRAM before spending much time there.

the trainer does offload frozen model components and run a VRAM safety check
before the first batch. that's made regular `acestep-v15-base` training work
on my 8 GB card, but it obviously can't make an XL model fit where it doesn't
fit.

i was able to get some decent instrumental results with regular base, but i'm
finding that sa3 seems to be the better option for purely instrumental data.
ACE-Step seems better suited to data with vocalists in it. that's just where my
own testing has landed so far; your data may disagree.

## the basic workflow

1. pick a folder of audio files and choose a name and trigger word for the
   LoRA.
2. press **caption / prepare** to fill in missing sidecars and run the optional
   BPM/key helper.
3. open **edit prompts / sidecars** and fix anything obviously wrong. add your
   own lyrics here if the dataset has vocals.
4. choose your model, adapter type, epochs, learning rate, maximum track
   length, and any advanced training-schedule settings you actually mean to
   test.
5. press **train LoRA** and let the VRAM preflight decide whether the selected
   settings are safe enough to begin.

the trainer saves checkpoints along the way and keeps a smoothed best-loss
checkpoint separately from the final epoch. it registers the selected adapter
with carey when training finishes.

## fun fact: base, turbo, and SFT

one of the cool parts about training ACE-Step LoRAs is that you can apply a
LoRA trained on regular base to regular turbo and SFT too. my early results
seem to favor applying a LoRA trained on regular base to the SFT model.

gary4local handles the standard/XL family separation for you. if you train on
regular base, the LoRA is registered as standard. if you train on XL-base, it's
registered as XL. carey only loads LoRAs and caption pools belonging to the
currently selected family.

i'm not sure XL LoRAs will be as successful on XL-SFT. there's an
[ACE-Step issue reporting tempo, rhythm, and key instability with XL-SFT](https://github.com/ace-step/ACE-Step-1.5/issues/1203).
that issue was closed automatically due to inactivity rather than confirmed
fixed, so i'm treating it as an unresolved question rather than proof that
XL-SFT is always broken.

## a note about the captioner

the results aren't going to be perfect. the reason we use ACE-Step's captioner
is because, in my mind, the base model was originally trained on a pipeline
using this kind of captioner model. even when the captions and genres don't
perfectly represent your music, they do represent the style of caption this
model seems to prefer.

use the 4B captioner if you have the GPU memory. it should give you the best
results. the 1.7B model is a good practical option. the 0.6B model is there for
low-memory situations, but i don't recommend it if one of the larger models
fits.

everything remains editable. the captioner can get the caption, genre, BPM, or
key wrong. the BPM/key helper is intentionally conservative, and a musician
should trust their ears over either model.

lyrics are BYOL for now. ACE-Step has a separate transcriber model we may wire
up later, but understand_music lyrics are hallucinated rather than transcribed,
so gary4local does not write them into vocal sidecars. the lyrics editor shows
a grey structure template when the field is empty; that template is only a
guide and is not saved unless you type it yourself.

at the end of the day, your prompt is probably going to get populated by the
dice button in gary4juce, so it doesn't matter all that much what every caption
says, in my opinion. the sidecars still matter because their captions and
genres become the dice pool associated with your LoRA.

the trigger word gets applied like this:

```text
your-trigger-word, original caption
```

that lets you try applying your LoRA's fine-tuned distribution to more varied
captions instead of only repeating the training text. like everything else,
this is going to be experimental for you, and you'll have to figure out how
you prefer to work. i personally hate prompting, so i mostly let rnjesus decide
through the dice button until i hear what i like.

## a few training controls worth explaining

### epochs

one epoch is one pass over every track. more epochs means more chances for the
LoRA to learn your data, but more isn't automatically better. the best
checkpoint may happen before the final epoch, which is why the trainer saves
them separately.

### learning rate

learning rate is basically how hard you want it to train. `1e-4` is a lighter
touch; `3e-4` trains harder and is our current default. if the result becomes
harsh, unstable, or far too literal, lowering the learning rate is a sensible
first experiment.

### rank

rank is basically how much room you give the adapter to adjust the model. it
isn't rewriting the original base-model weights directly, but a higher rank
does let the LoRA train more adapter parameters.

the best results with ACE-Step seem to come from rank 64 or possibly 128. be
careful with 128 if your GPU is already close to its limit at 64, especially
with balanced attention + MLP enabled. the preflight will reject configurations
that are clearly unsafe, but i still prefer leaving some breathing room on a
GPU that's also driving the display.

### balanced attention + MLP

the **balanced attention + MLP** option in the advanced settings is pretty
experimental. it's based on some of the ideas explored in
[koda-dernet's Side-Step](https://github.com/koda-dernet/Side-Step).

rather than add adaptive Fisher analysis to our training pipeline, we opted to
do something simpler that should hopefully achieve good results. gary uses a
fixed distribution of rank across self-attention, cross-attention, and MLP
projections. so far, in testing, it's made my LoRA sound cleaner. that's an
early personal result, not a promise.

if you want Fisher analysis or a more advanced setup, use
[Side-Step](https://github.com/koda-dernet/Side-Step).

### batch size and gradient accumulation

batch size and gradient accumulation should probably remain at 1 unless you're
using a lot of data. if you're training on a single album, i'd rather let each
track produce its own optimizer update than average a bunch of tracks together
before updating.

larger datasets may benefit from larger batches or gradient accumulation, but
they trade more frequent, noisy updates for fewer, smoother updates. i'd only
change these because your dataset gives you a reason to.

### Min-SNR loss weighting (the codexplanation)

ACE-Step trains at many different flow timesteps. you can think of those as
different mixtures of clean audio and noise. with ordinary flat loss
weighting, the cleaner, high-signal timesteps can produce much stronger
gradients and start shouting over the rest of the training problem.

Min-SNR estimates the signal-to-noise ratio at each sampled timestep and caps
the influence of those high-SNR examples. in plainer language: it asks the
easy, cleaner examples to stop hogging the conversation so the LoRA can learn
from a better-balanced range of noise levels.

it doesn't rescue bad data, improve the captions, or simply make the LoRA
stronger. it only changes how the loss is balanced. gamma 5 is a sensible
starting point, and i'd leave it there unless you're specifically testing loss
weighting. gary's implementation adapts the method from
[Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy"](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)
to ACE-Step's flow interpolation.

## lego mode

gary4juce can use carey LoRAs in lego mode too. lego still only exposes the
base route: regular base when XL mode is off, and XL-base when XL mode is on.
the registry family tag decides which LoRAs are visible, so standard adapters
stay with regular base and XL adapters stay with XL-base.

right now, i would use regular `acestep-v15-base` for lego mode if you don't
have a LoRA loaded. plain `acestep-v15-xl-base` vocals have been pretty awful
for me so far.

the story changes once you have an xl-base LoRA trained. with a matching LoRA,
lego vocals and backing vocals can be incredible. some non-vocal lego tasks are
still lightly tested, and i've heard occasional instrument bleed from xl-base
vocal LoRAs. the bleed tends to fit the instrumental anyway, so it may
still be useful in a mix.

for now: save checkpoints, change one thing at a time, and trust your ears more
than the loss graph.
