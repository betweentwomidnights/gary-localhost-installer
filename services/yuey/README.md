# yuey

yuey is gary4local's first native service. It has no Python environment and
no source in this folder: gary4local downloads a prebuilt
[yuey.cpp](https://github.com/betweentwomidnights/yuey.cpp) `yue2-server`
package for the machine's GPU and runs it on port 8007.

- `native/` holds the downloaded runtime. The service's "install runtime"
  button fills it: the core package plus one GPU backend, checked against the
  SHA-256 pinned in `services/manifests/services.json`. yuey uses CUDA on
  NVIDIA and Vulkan on AMD and Intel; Vulkan stays available on NVIDIA from
  the yuey panel, but stalls on long renders there.
- The GGUF models live under the runtime storage's `models/yuey/` and come
  from [thepatch/YuE2-3B-GGUF](https://huggingface.co/thepatch/YuE2-3B-GGUF).
  The model weights are CC BY-NC 4.0.

This folder is tracked so that app updates, which replace every bundled
service folder, keep `native/` in place instead of deleting yuey outright.

For development against a local build instead of a published package:

```powershell
$env:GARY4LOCAL_NATIVE_DIR_YUEY = "C:\dev\yue2.cpp\build-cuda\bin\Release"
# optional; defaults to auto
$env:GARY4LOCAL_NATIVE_BACKEND_YUEY = "cuda"
```

Or, to exercise the full install flow before a release exists, point
gary4local at the output of yuey.cpp's `ci\package-windows.ps1`:

```powershell
$env:GARY4LOCAL_NATIVE_PACKAGE_DIR = "C:\dev\yue2.cpp\dist"
```
