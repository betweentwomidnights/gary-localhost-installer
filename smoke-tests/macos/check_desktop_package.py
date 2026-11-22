"""Validate a macOS preview; GPU and GUI behavior still require the M4 gate."""
import hashlib
import json
import os
import pathlib
import plistlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAURI = ROOT / "control-center/src-tauri"


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    version = json.loads((TAURI / "tauri.conf.json").read_text())["version"]
    bundle = TAURI / "target/aarch64-apple-darwin/release/bundle"
    app = bundle / "macos/gary4local.app"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "com.betweentwomidnights.gary4local", info
    assert info["CFBundleShortVersionString"] == version, info
    binary = app / "Contents/MacOS" / info["CFBundleExecutable"]
    assert run("lipo", "-archs", str(binary)).split() == ["arm64"]
    dependencies = run("otool", "-L", str(binary))
    assert "/opt/homebrew/" not in dependencies and "/usr/local/" not in dependencies, dependencies
    manifests = list(app.rglob("services/manifests/services.json"))
    assert len(manifests) == 1, manifests
    assert manifests[0].read_bytes() == (ROOT / "services/manifests/services.json").read_bytes()
    for file in app.rglob("*"):
        if file.is_file():
            assert file.suffix not in {".gguf", ".safetensors", ".p12", ".p8"}, file
    run("codesign", "--verify", "--deep", "--strict", str(app))
    signed = os.environ.get("SIGNED") == "true"
    signature = subprocess.run(["codesign", "-dvv", str(app)], text=True, capture_output=True, check=True).stderr
    if signed:
        assert "Authority=Developer ID Application:" in signature, signature
        run("xcrun", "stapler", "validate", str(app))
        run("spctl", "--assess", "--type", "execute", "--verbose", str(app))
    images = list((bundle / "dmg").glob("*.dmg"))
    assert len(images) == 1, images
    image = images[0]
    if signed:
        # The app is already notarized/stapled by Tauri before DMG creation.
        # Give the downloadable image its own ticket for offline distribution.
        run("codesign", "--force", "--timestamp", "--sign", os.environ["APPLE_SIGNING_IDENTITY"], str(image))
        result = json.loads(run("xcrun", "notarytool", "submit", str(image),
            "--key", os.environ["APPLE_API_KEY_PATH"], "--key-id", os.environ["APPLE_API_KEY"],
            "--issuer", os.environ["APPLE_API_ISSUER"], "--wait", "--output-format", "json"))
        assert result.get("status") == "Accepted", result
        run("xcrun", "stapler", "staple", str(image))
        run("xcrun", "stapler", "validate", str(image))
    output = TAURI / "target/desktop-macos"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"gary4local_{version}_macos-arm64.app.zip"
    run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), str(archive))
    import shutil
    disk_image = output / f"gary4local_{version}_aarch64.dmg"
    shutil.copy2(image, disk_image)
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in [archive, disk_image]}
    (output / "SHA256SUMS-macos-arm64").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    metadata = {"service": "gary4local-desktop", "version": version,
        "commit": os.environ.get("GITHUB_SHA", run("git", "-C", str(ROOT), "rev-parse", "HEAD")),
        "platform": "macos-arm64", "signed": signed, "notarized": signed,
        "updater_enabled": False, "m4_validation": "required before sharing",
        "dependencies": dependencies, "sha256": hashes}
    (output / "BUILD-INFO.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
