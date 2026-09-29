# native runtime packages

this is the contract between the GGML service repos and gary4local: what a
service release has to contain, what its executable has to do, and how
gary4local finds, checks and installs it. yuey ([yuey.cpp](https://github.com/betweentwomidnights/yuey.cpp))
is the first service built this way, and it's the reference implementation.
sa3.cpp, audiocraft.cpp and acestep.cpp follow the same pattern as they come
over. when a service needs something this doesn't cover, change this document
first, then the repos.

for now yuey is the only native service in gary4local. sa3, foundation-1,
jerry, gary and carey stay on their pytorch environments until yuey is fully
validated.

## who owns what

- **each service repo** builds and publishes its own packages from a GitHub
  release: `ci/package-windows.ps1` does the building and
  `.github/workflows/release.yml` runs it. yuey's side, with its release
  checklist, is [packaging](https://github.com/betweentwomidnights/yuey.cpp/blob/main/docs/packaging.md).
- **gary-localhost-installer** hosts the shared CUDA runtime as its own release
  and pins every package by URL and SHA-256 in `services/manifests/services.json`.
- **gary4local** reads the manifest, picks a backend for the machine's GPU,
  downloads and verifies the packages, unpacks them, and checks the result runs
  before it calls a runtime installed.

nothing is ever compiled on a user's machine, and nothing is downloaded without
a hash to check it against.

## what a service release contains

one GitHub release per version, tagged `vX.Y.Z`, with these assets:

| asset | contents |
|---|---|
| `<service>-vX.Y.Z-windows-x64-core.zip` | the server executable (and any CLI tools), the service's own DLLs, `ggml.dll`, `ggml-base.dll`, every `ggml-cpu-*.dll` variant, `LICENSE`, `LICENSE-ggml.txt`, `THIRD_PARTY_NOTICES.md`, `BUILD-INFO.json` |
| `<service>-vX.Y.Z-windows-x64-cuda.zip` | `ggml-cuda.dll`, nothing else |
| `<service>-vX.Y.Z-windows-x64-vulkan.zip` | `ggml-vulkan.dll`, nothing else |
| `<service>-vX.Y.Z-windows-x64-standalone.zip` | core, both backends and the CUDA runtime in one folder, for people running the service on its own. gary4local never downloads it. |
| `SHA256SUMS` | `<sha256>  <file name>` per zip, LF line endings, no BOM, so `sha256sum -c` works on it |

`<service>` is the name the service goes by in gary4local (`yuey`), not
necessarily its repo's name.

the split is the whole trick. ggml loads backend DLLs from the executable's
folder, so unpacking core plus one backend zip into the same folder is all the
backend selection there is. an NVIDIA machine and an AMD machine run the same
executable with a different DLL beside it.

the zips are flat, with no folder inside. core goes down first and the backend
unpacks over it. a backend DLL must never be in core (the package script fails
the build if one leaks), because it would load on every machine.

`BUILD-INFO.json` says where a package came from, without needing the release
page:

```json
{
  "service": "yuey",
  "version": "v0.2.0",
  "commit": "<the service repo's commit>",
  "ggml_commit": "<the ggml submodule's commit>",
  "dirty": false,
  "platform": "windows-x64",
  "backends": ["cuda", "vulkan"],
  "cuda": "12.8.1",
  "vulkan_sdk": "1.4.350.0",
  "built_utc": "2026-09-29T21:18:02Z"
}
```

`dirty: true` means someone packaged a checkout with local changes by hand. CI
never produces one, and one should never be pinned.

## how the packages are built

every service builds the same way, so one gary4local install path works for
all of them:

- `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`. nothing
  is tuned to the machine that built it, and the CPU backend picks its own
  instruction set at load time.
- no CUDA architecture list, so ggml's portable default covers Maxwell through
  Blackwell.
- CUDA 12.8.1 and Vulkan SDK 1.4.350.0, on GitHub's `windows-2022` runner with
  Visual Studio 2022. the CUDA toolkit's MSBuild integration targets 2022, so
  the runner is pinned rather than `windows-latest`.
- the Vulkan SDK comes from LunarG's own installer, not a trimmed install.
  ggml's Vulkan backend asks for SPIRV-Headers' CMake config, which trimmed
  installs leave out. that's what failed yuey's first release dry run.
- the same ggml fork everywhere (`betweentwomidnights/ggml`), pinned as a
  submodule. each service still ships its own copy of the ggml DLLs in its own
  folder. nothing is shared between services except the CUDA runtime.

the package script runs the tests with the CPU backend loaded dynamically,
exactly as it'll be on a user's machine, before it packages anything. the same
script runs in CI and on a developer machine, so a package built by hand is the
package CI would have built.

## what the executable has to do

gary4local treats the server as a black box with a few promises:

- **`--version`** prints the bare version (`0.2.0`) and exits.
- **`--props`** prints a JSON document to stdout and exits 0, without binding a
  port or loading a model. gary4local runs it right after unpacking, to confirm
  the backend it installed actually came up. the fields it reads:
  - `devices`, a list of `{backend, description, type, memory_total_bytes}`.
    `backend` is ggml's registry name (`CUDA`, `Vulkan`, `Metal`, `CPU`), and
    a GPU backend counts as working only if one of these names it.
  - `hardware.recommended_encoding`, optionally, for services with
    quantization tiers.
- **it's configured by env vars**, so the manifest can stay declarative. at
  minimum: its port, its models folder, and the device, which gary4local fills
  from `${NATIVE_BACKEND}` (`cuda` or `vulkan`).
- **`GET /health`** answers once it's running. that's how gary4local knows the
  service is up.
- **it needs nothing outside its own folder** except the GPU driver and, for
  CUDA, the shared runtime on `PATH`.

the version is written once, in the service's build (`project(VERSION)` in
CMake), and the executable reports that. the package script refuses to
package a release whose tag doesn't match what the built server reports.

## the shared CUDA runtime

`ggml-cuda.dll` needs `cudart64_12.dll`, `cublas64_12.dll` and
`cublasLt64_12.dll`. that's about 550 MB, and it's identical for every service,
so gary4local installs it once, under `native-runtimes/` in runtime storage,
and puts it on `PATH` for any service whose CUDA package asks for it.

it's published from this repo, not from any service:

- `control-center/src-tauri/scripts/package_cuda_runtime.ps1` packages the
  three DLLs, NVIDIA's EULA (as `NVIDIA-CUDA-EULA.txt`) and a
  `RUNTIME-INFO.json`.
- `.github/workflows/cuda-runtime.yml` builds it in CI, dispatched by hand.
  without `publish` it's a dry run that keeps the zip as a workflow artifact.
  with it, the pack becomes release `runtime-cudart-<x.y.z>`, which is never
  marked latest, so gary4local's own release stays the one the releases page
  leads with.

the zip and the tag are named for the exact toolkit (`cudart-12.8.1-windows-x64.zip`
on `runtime-cudart-12.8.1`), so a published pack never changes. the manifest
names it by the compatibility class services ask for (`cudart-12.8`), and a
CUDA package lists that class in `requires`. the workflow refuses to publish
over an existing runtime release.

a new pack is only needed when the services move to a new CUDA toolkit. when
that happens, publish the new pack, pin it, and move the services over in the
same gary4local release.

## the manifest

each native service's entry in `services/manifests/services.json` has
`"runtime": "native"` and a `native` block:

```json
"native": {
  "executable": "yue2-server.exe",
  "version": "v0.2.0",
  "platforms": {
    "windows-x64": {
      "core": {"url": ".../yuey-v0.2.0-windows-x64-core.zip", "sha256": "..."},
      "backends": {
        "cuda": {"url": ".../yuey-v0.2.0-windows-x64-cuda.zip", "sha256": "...", "requires": ["cudart-12.8"]},
        "vulkan": {"url": ".../yuey-v0.2.0-windows-x64-vulkan.zip", "sha256": "..."}
      },
      "prefer": ["cuda", "vulkan"]
    }
  }
}
```

and the shared runtimes sit once at the top level:

```json
"nativeRuntimes": {
  "cudart-12.8": {"url": ".../runtime-cudart-12.8.1/cudart-12.8.1-windows-x64.zip", "sha256": "..."}
}
```

a `sha256` of `"unpublished"` means nothing has been pinned yet. gary4local
refuses to install it and says why, rather than downloading something it can't
check.

`prefer` is the service's own order for automatic backend choice. yuey prefers
CUDA on NVIDIA even though Vulkan also runs there, because gary4juce lives
inside a DAW and Vulkan has stalled beside one. each service gets measured for
its own order.

pins are written by a script, never by hand:

```bash
node control-center/src-tauri/scripts/pin_native_release.mjs --service yuey --repo betweentwomidnights/yuey.cpp --tag v0.2.0
node control-center/src-tauri/scripts/pin_native_release.mjs --runtime cudart-12.8 --repo betweentwomidnights/gary-localhost-installer --tag runtime-cudart-12.8.1
```

both read the release's `SHA256SUMS` and edit the manifest in place, so the
diff is only the lines that changed. a service pin also sets the service's
`version`.

## how gary4local installs a runtime

"install runtime" does this:

1. **detects the GPU** and lists the backends the machine can run, in the
   service's `prefer` order. a backend picked in the service's panel is the
   only candidate, so if it fails, the install fails loudly. there's no CPU
   runtime: these models take minutes per song on a GPU, so on a CPU they'd
   only look broken, and a machine without a supported GPU is told so.
2. **downloads core**, resuming a partial download and keeping it only if its
   SHA-256 matches.
3. **for each candidate backend**: downloads its zip and any shared runtime it
   `requires` (installed once; a runtime already there with the pinned hash is
   kept), unpacks core plus the backend into a staging folder, and runs
   `--props` with the runtime on `PATH`. the first backend that reports its
   devices wins. if the automatic choice doesn't come up, the next one is tried,
   and the panel says so instead of quietly running somewhere else.
4. **swaps the staging folder into `services/<id>/native`**, keeping the old
   install until the new one is in place, and writes `gary-native.json` (the
   version, the backend it ended up on, the one that was asked for, and any
   fallback reason) and the `--props` output beside the executable.

an installed runtime whose `gary-native.json` version differs from the
manifest's shows "update runtime". the executable and its DLLs are locked while
the service runs, so a rebuild refuses to start until it's stopped.

the binaries aren't code-signed. downloads made by gary4local itself carry no
Mark of the Web, so SmartScreen doesn't apply, but Smart App Control can still
block the executable. when it does, gary4local's error explains how to get past
it.

## testing before a release exists

two overrides, both read from gary4local's environment at launch:

- **`GARY4LOCAL_NATIVE_PACKAGE_DIR`** points at a folder of packages with its
  own `SHA256SUMS`, such as a service repo's `dist\`. the whole install flow
  runs against it: same selection, same checks, hashes from that folder. a
  runtime pack built locally is named for the local toolkit
  (`cudart-12.8.0-...`), so for a runtime, the one pack in the folder named for
  its class (`cudart-12.8`) stands in for the pinned file. build one with a
  service's `package-windows.ps1 -CudaRuntime`, or with this repo's
  `package_cuda_runtime.ps1`.
- **`GARY4LOCAL_NATIVE_DIR_<ID>`** (for yuey, `GARY4LOCAL_NATIVE_DIR_YUEY`)
  points straight at a build folder, such as `C:\dev\yue2.cpp\build-cuda\bin\Release`.
  nothing is installed; gary4local just runs that executable.
  `GARY4LOCAL_NATIVE_BACKEND_<ID>` picks the device, and defaults to auto.

## releasing, end to end

1. **the service.** cut and publish its release by its own checklist (yuey's is
   in [packaging](https://github.com/betweentwomidnights/yuey.cpp/blob/main/docs/packaging.md)),
   including a CI dry run before anything is public.
2. **the CUDA runtime**, only when it changes: dispatch `cuda-runtime.yml` as a
   dry run, then again with `publish`.
3. **pin** the new release (and runtime, if it changed) with
   `pin_native_release.mjs`.
4. **test the published path** with no override set, on CUDA and on Vulkan:
   install, start, and run the service from gary4juce. this is the first time
   gary4local downloads from the real URLs, so don't skip it.
5. **ship gary4local** with the pin, by [the release guide](releasing/PHASE2_RELEASE.md).

once gary4local pins a release, its assets don't change. a fix is a new patch
version, never a new file under an old tag. the one exception is rerunning a
service's workflow to attach assets that are missing, before anything has
pinned the release.

## bringing another service over

for sa3.cpp, audiocraft.cpp and acestep.cpp:

1. copy yuey's `ci/package-windows.ps1` and `.github/workflows/release.yml`,
   rename the service and its executables, and keep everything else, including
   the checks. differences should be deliberate and written down here.
2. make the server keep the promises in [what the executable has to do](#what-the-executable-has-to-do).
3. add a `THIRD_PARTY_NOTICES.md` for whatever else the service vendors.
4. add its manifest entry with `"sha256": "unpublished"`, test it through
   `GARY4LOCAL_NATIVE_PACKAGE_DIR`, then release and pin as above.

## not yet

- **macOS and Linux.** the platform key (`windows-x64`) leaves room for
  `macos-arm64` with a Metal backend and `linux-x64`, but nothing is built for
  them yet.
- **code signing.** see above. it would take a certificate, and the Smart App
  Control workaround is the price of not having one.
