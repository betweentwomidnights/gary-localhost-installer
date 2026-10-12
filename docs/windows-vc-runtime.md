# Shared Windows Visual C++ prerequisite

Native Yuey, SA3, ACE-Step and other MSVC builds need the Microsoft Visual C++
x64 runtime even when they use Vulkan or CPU. Developer tools are not required:
many machines already have the Redistributable from another application. That
explains why Yuey can work on a machine without Visual Studio, but it does not
prove that a clean Windows installation has the dependency.

gary4local owns one check and repair path for every native service:

- Before a service probe or launch, check the versions of `msvcp140.dll`,
  `vcruntime140.dll`, `vcruntime140_1.dll` and `vcomp140.dll` in Windows System32.
- The **Install runtime** action repairs a missing/outdated prerequisite before
  staging a service. Its existing shared installer lock serializes this work.
- Download the exact pinned Microsoft x64 installer into the shared download
  cache. Verify SHA-256, a valid Microsoft Corporation Authenticode signature,
  and the expected installer version before execution.
- Windows requests administrator consent when repair is needed. Ordinary service
  startup never downloads, elevates or installs anything. Already current systems
  skip the repair entirely, including the download.
- Cancellation/failure does not install or replace a service. A reboot-required
  result asks the user to restart and retry. Success requires rechecking the DLLs.

This uses Microsoft's central installation rather than an app-local DLL pack.
Microsoft recommends central deployment because Windows can service the runtime
independently. See [deployment methods](https://learn.microsoft.com/en-us/cpp/windows/choosing-a-deployment-method)
and [supported runtime downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
The prerequisite is independent of the per-service ggml libraries and shared
CUDA PATH pack. Uninstalling a service must not uninstall this system runtime,
which other applications can also depend on.

`control-center/src-tauri/windows/vc-runtime.json` pins the official installer
14.51.36247.0 and SHA-256
`843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c`.
The minimum is 14.44.35211.0 for the currently packaged VS2022/v143 builds.
When a service moves to newer build tools, review that minimum against its
runtime imports and toolchain; do not change only the downloadable installer.
Update the URL, hash and installer version together after checking Microsoft's
signature. Do not pin a mutable `aka.ms` redirect without its exact hash.

## Validation and remaining acceptance

On the development PC, all five focused tests pass, including the real Win32
file-version reader and the explicit read-only test proving that a current
system does not create a cache, download or elevate. The downloaded pinned EXE
passed its SHA-256 and the actual PowerShell publisher/version check without
being executed.

Still exercise Windows Sandbox/a VM without developer tools or an existing VC
runtime: install each native service through the UI, approve repair once, launch
the next service without another repair, cancel UAC and retry, and test an older
runtime, network/hash/signature failure, MSI contention and reboot-required
results. Unit tests cover exit-code decisions; they do not replace those system
installation scenarios. Never remove the host's shared runtime to simulate them.
