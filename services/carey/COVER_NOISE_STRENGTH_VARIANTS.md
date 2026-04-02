# Cover Noise Strength Variants

This project currently defaults to the same `cover_noise_strength` behavior used on the remote spark backend.

## Default Variant: Spark Parity

File:
- `checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py`

Behavior:
- Computes `effective_noise_level = 1.0 - cover_noise_strength`
- Chooses the nearest valid turbo timestep to that noise level
- Initializes `xt` from `renoise(src_latents, nearest_t, noise)`
- Truncates the turbo schedule to begin at that exact timestep
- Switches from cover conditioning to non-cover conditioning once, when `step_idx >= cover_steps`

Why this is the default:
- It matches the running remote backend more closely
- It makes local and remote settings easier to reason about

## Alternate Variant: Step-Skip Approximation

This was an experimental local-only implementation used during debugging.

Behavior:
- Converts `cover_noise_strength` into an integer count of skipped early turbo steps
- Starts from the first remaining timestep after those skipped steps

Observed behavior:
- It allows higher `cover_noise_strength` values like `0.5+` without immediately collapsing into "almost just source audio"
- It tends to stay more creative than the spark-parity implementation at the same settings
- It may be attractive as an optional compile-time variant or future UI toggle

## Listening Notes

Observed locally during testing:
- Spark-parity behavior stays closer to the source at the same settings
- The alternate step-skip variant leaves more room for reinterpretation
- Both can sound good, but they serve slightly different artistic goals

## Future Direction

If we want to expose both paths cleanly, the best next step is a named mode, for example:
- `spark_parity`
- `creative_step_skip`

That could begin as a compile-time option and later become a user-facing toggle once we are confident about the UX.
