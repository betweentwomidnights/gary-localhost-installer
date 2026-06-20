# ACE-Step 1.5

ACE-Step music generation service for Gary4JUCE.

## LoRA training quality note

The local trainer is optimized for `acestep-v15-base` on consumer GPUs. Its
balanced attention/MLP profile, Min-SNR weighting, and best-checkpoint selection
substantially improve the local workflow, but regular-base adapters should not
be expected to match LoRAs trained against `acestep-v15-xl-base`.

XL-base training remains available when the checkpoint and sufficient VRAM are
present, but it does not fit reliably on an 8 GB GPU. For the highest adapter
quality, use an XL-capable GPU or remote training backend and apply the resulting
adapter to the matching XL model family.

Local runs offload ACE-Step's frozen encoder/tokenizer components after setup,
then measure the remaining CUDA headroom. Unsafe configurations stop before the
first training batch with a specific VRAM budget message, reducing the risk of a
driver-level out-of-memory hang.
