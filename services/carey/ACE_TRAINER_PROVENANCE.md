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
