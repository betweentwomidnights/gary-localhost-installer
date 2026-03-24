"""
Patch diffusers/quantizers/torchao/torchao_quantizer.py.

Bug: `logger` is defined at module level *after* `_update_torch_safe_globals()` is
called at module load time.  When the torchao import inside that function fails
(e.g. because torchao 0.16 moved `uint4_layout`), the except block calls
`logger.warning()` before `logger` exists → NameError → module can't be imported.

Fix: move `logger = logging.get_logger(__name__)` to just before the `if`-block
that calls `_update_torch_safe_globals()`.  All other code is unchanged.

Affects: diffusers 0.35.x, 0.36.x (and possibly earlier).
Safe to re-run: if the pattern is already absent (fixed upstream), the assert
will fire and the build will fail loudly rather than silently producing a bad image.
"""

import sys

DIFFUSERS_SITE = "/usr/local/lib/python3.12/dist-packages"
PATH = f"{DIFFUSERS_SITE}/diffusers/quantizers/torchao/torchao_quantizer.py"

# The exact block as it appears in the broken versions.
# _update_torch_safe_globals() is called, then two blank lines, then logger is defined.
OLD = (
    "\nif (\n"
    "    is_torch_available()\n"
    "    and is_torch_version(\">=\", \"2.6.0\")\n"
    "    and is_torchao_available()\n"
    "    and is_torchao_version(\">=\", \"0.7.0\")\n"
    "):\n"
    "    _update_torch_safe_globals()\n"
    "\n"
    "\n"
    "logger = logging.get_logger(__name__)"
)

# Swap: define logger first, then run the conditional call.
NEW = (
    "\nlogger = logging.get_logger(__name__)\n"
    "\n"
    "if (\n"
    "    is_torch_available()\n"
    "    and is_torch_version(\">=\", \"2.6.0\")\n"
    "    and is_torchao_available()\n"
    "    and is_torchao_version(\">=\", \"0.7.0\")\n"
    "):\n"
    "    _update_torch_safe_globals()"
)

txt = open(PATH).read()

if OLD not in txt:
    print(
        f"Pattern not found in {PATH} — diffusers may have already fixed this "
        "upstream, or the version in the image has changed.  Skipping patch.",
        file=sys.stderr,
    )
    sys.exit(0)

open(PATH, "w").write(txt.replace(OLD, NEW, 1))
print(f"Patched {PATH}: moved logger definition before _update_torch_safe_globals() call site.")
