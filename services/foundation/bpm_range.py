"""Host tempo validation for Foundation-1.

`host_bpm` never reaches the model. It only picks the nearest Foundation
preset BPM and then becomes a Rubber Band time-stretch ratio —
`host_bpm / foundation_bpm` on the way out, and its inverse on the way in for
/audio2audio. With the presets pinned to 100-150 BPM, the range below keeps
that ratio inside roughly 0.2x-6.7x, which Rubber Band handles fine.

So the bounds are DAW tempo limits rather than musical ones. Ableton Live's
tempo range is exactly 20-999 BPM and Logic, FL, and Reaper all fit inside
that, so legitimate tempo automation is never rejected. Hosts reporting a
bogus tempo are — Savihost has been seen sending 3159345 BPM through
gary4juce, which asks Rubber Band for a 21062x stretch.
"""

import math

MIN_HOST_BPM = 20.0
MAX_HOST_BPM = 999.0


class HostBpmError(ValueError):
    """Raised when host_bpm can't be turned into a usable stretch ratio."""


def _fmt(bpm: float) -> str:
    """Readable tempo — no trailing zeros, and no scientific notation for the
    absurd values this module exists to catch."""
    return f"{bpm:.2f}".rstrip("0").rstrip(".")


def resolve_host_bpm(value, default: float | None = None) -> float:
    """Coerce an incoming host_bpm to a float inside the supported range.

    Pass a default to make the field optional; without one, a missing value
    is an error.
    """
    if value is None:
        if default is None:
            raise HostBpmError("host_bpm is required")
        return float(default)

    try:
        bpm = float(value)
    except (TypeError, ValueError):
        raise HostBpmError(f"host_bpm must be a number, got {value!r}") from None

    if not math.isfinite(bpm):
        raise HostBpmError(f"host_bpm must be a finite number, got {value!r}")

    if not MIN_HOST_BPM <= bpm <= MAX_HOST_BPM:
        raise HostBpmError(
            f"host_bpm {_fmt(bpm)} is outside the supported range "
            f"{_fmt(MIN_HOST_BPM)}-{_fmt(MAX_HOST_BPM)} BPM — audio can't be "
            f"time-stretched that far. Check the tempo your host is reporting "
            f"to the plugin."
        )

    return bpm
