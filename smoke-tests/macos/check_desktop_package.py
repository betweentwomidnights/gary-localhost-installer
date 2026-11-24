"""Validate a macOS preview; GPU and GUI behavior still require the M4 gate."""
import hashlib
import json
import os
import pathlib
import plistlib
import shutil
import subprocess
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAURI = ROOT / "control-center/src-tauri"


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def updater_archive(app, archive, signed):
    """Pack the stapled app the way tauri-plugin-updater unpacks it.

    The updater drops each entry's first path component and replaces the
    installed bundle with the rest, so every entry must sit under gary4local.app/.
    """
    env = dict(os.environ, COPYFILE_DISABLE="1")  # no AppleDouble ._ files
    subprocess.run(["tar", "--no-xattrs", "-czf", str(archive), "-C", str(app.parent), app.name],
                   check=True, env=env)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert names and all(n == app.name or n.startswith(app.name + "/") for n in names), names[:5]
        assert not any("/._" in n for n in names), "AppleDouble files in the updater archive"
    # Unpack with the system tar and check the result is still the signed,
    # stapled app a user would get from the disk image.
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(["tar", "-xzf", str(archive), "-C", scratch], check=True)
        unpacked = pathlib.Path(scratch) / app.name
        run("codesign", "--verify", "--deep", "--strict", str(unpacked))
        if signed:
            run("xcrun", "stapler", "validate", str(unpacked))
            run("spctl", "--assess", "--type", "execute", "--verbose", str(unpacked))


def main():
    updater_enabled = os.environ.get("VITE_ENABLE_APP_UPDATER", "1") != "0"
    update_channel = os.environ.get("GARY4LOCAL_UPDATE_CHANNEL", "stable")
    assert updater_enabled and update_channel == "preview", (updater_enabled, update_channel)
    version = json.loads((TAURI / "tauri.conf.json").read_text())["version"]
    bundle = TAURI / "target/aarch64-apple-darwin/release/bundle"
    app = bundle / "macos/gary4local.app"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "com.betweentwomidnights.gary4local", info
    assert info["CFBundleShortVersionString"] == version, info
    icon = app / "Contents/Resources" / info["CFBundleIconFile"]
    expected_icon = TAURI / "icons/icon.icns"
    assert icon.read_bytes() == expected_icon.read_bytes(), f"Unexpected Dock icon: {icon}"
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
    update_archive = output / f"gary4local_{version}_aarch64.app.tar.gz"
    updater_archive(app, update_archive, signed)
    disk_image = output / f"gary4local_{version}_aarch64.dmg"
    shutil.copy2(image, disk_image)
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
              for file in [archive, update_archive, disk_image]}
    (output / "SHA256SUMS-macos-arm64").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    metadata = {"service": "gary4local-desktop", "version": version,
        "commit": os.environ.get("GITHUB_SHA", run("git", "-C", str(ROOT), "rev-parse", "HEAD")),
        "platform": "macos-arm64", "signed": signed, "notarized": signed,
        "updater_enabled": updater_enabled, "update_channel": update_channel,
        "updater_archive": update_archive.name, "updater_signed": False, "m4_validation": "required before sharing",
        "dependencies": dependencies, "sha256": hashes}
    (output / "BUILD-INFO.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
