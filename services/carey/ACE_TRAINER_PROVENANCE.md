# ACE-Step Trainer Provenance

Gary's ACE-Step training implementation is informed by
[koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step).

## Pinned Reference

The permitted reference baseline is Side-Step commit
`0a80d14f62d2482ef970955ae4a039ba99d83bf9`, released as
`0.9.0-beta` on February 19, 2026. That snapshot includes the two-pass
preprocessor and Preprocessing++/Fisher work and was distributed under the
MIT License. A copy of that license is included in `SIDESTEP_MIT_LICENSE`.

Side-Step changed its repository license to CC BY-NC-SA 4.0 in commit
`db0a4294c96fca365d49bb7ed49ae9b89d83702f` on March 5, 2026. Code and
documentation first published under that later license must not be copied
into Gary without separate permission from the Side-Step maintainers.

## Development Rule

- Keep the exact MIT commit available as the provenance and comparison source.
- Preserve the Side-Step copyright and MIT notice with derived trainer code.
- Implement behavior introduced after the license change independently from
  public interfaces, observed behavior, and Gary's own requirements.
- Do not vendor the current Side-Step tree or its CC BY-NC-SA documentation.
- Use full-precision adapter training with standard AdamW. Gary does not depend
  on bitsandbytes or model quantization for ACE-Step training because output
  quality inside the DAW takes priority over that memory tradeoff.

The local development reference worktree is intentionally detached at the
pinned commit. It is not part of Gary's shipped files.

## Min-SNR

Gary's Min-SNR loss weighting is independently expressed from the method in
Hang et al., [Efficient Diffusion Training via Min-SNR Weighting
Strategy](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)
(ICCV 2023). The signal-to-noise ratio is derived from ACE-Step's own linear
flow interpolation rather than copied from the later CC-licensed Side-Step
implementation.

## Best Checkpoint Tracking

MA5 training-loss checkpoint selection is restored from the pinned MIT
baseline. Gary preserves `best/` and `final/` as separate adapters and registers
`best/` when it is available so the last epoch remains available for comparison.

## Static Balanced Adapter Profile

Gary does not ship or recreate Side-Step's Fisher analysis. The static balanced
profile is an independent aggregate of projection-family ranks observed in two
user-owned PEFT adapter configurations (one instrumental and one vocal). It
intentionally discards layer-specific rankings and retains only a fixed
attention/MLP family allocation that applies equally to base and XL decoders.
