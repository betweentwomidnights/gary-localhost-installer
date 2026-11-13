"""Check a native Linux desktop DEB before retaining it as a preview artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TAURI = ROOT / 'control-center/src-tauri'


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['linux-x64', 'linux-arm64'], required=True)
    parser.add_argument('--arch', choices=['amd64', 'arm64'], required=True)
    args = parser.parse_args()
    assert {'linux-x64': 'amd64', 'linux-arm64': 'arm64'}[args.platform] == args.arch
    assert os.environ['VITE_ENABLE_APP_UPDATER'] == '0'
    assert os.environ['GARY4LOCAL_LINUX_PREVIEW'] == '1'
    packages = list((TAURI / 'target/release/bundle/deb').glob('*.deb'))
    assert len(packages) == 1, packages
    package = packages[0]
    assert run('dpkg-deb', '-f', str(package), 'Architecture') == args.arch
    assert run('dpkg-deb', '-f', str(package), 'Package') == 'gary4local-preview'
    depends = run('dpkg-deb', '-f', str(package), 'Depends')
    assert all(lib in depends for lib in ['libwebkit2gtk-4.1-0', 'libayatana-appindicator3-1', 'libvulkan1'])
    with tempfile.TemporaryDirectory(prefix='gary-desktop-check-') as directory:
        unpacked = Path(directory)
        subprocess.run(['dpkg-deb', '-x', str(package), str(unpacked)], check=True)
        binary = unpacked / 'usr/bin/gary4local-preview'
        launcher = unpacked / 'usr/bin/gary4local-preview-launch'
        assert binary.is_file() and os.access(binary, os.X_OK)
        assert launcher.is_file() and os.access(launcher, os.X_OK)
        elf = run('readelf', '-h', str(binary))
        assert ('AArch64' if args.arch == 'arm64' else 'Advanced Micro Devices X86-64') in elf
        dependencies = run('ldd', str(binary))
        assert 'not found' not in dependencies, dependencies
        desktop = list((unpacked / 'usr/share/applications').glob('*.desktop'))
        assert len(desktop) == 1
        subprocess.run(['desktop-file-validate', str(desktop[0])], check=True)
        assert 'Exec=gary4local-preview-launch' in desktop[0].read_text()
        manifests = list(unpacked.rglob('services/manifests/services.json'))
        assert len(manifests) == 1, manifests
        assert manifests[0].read_bytes() == (ROOT / 'services/manifests/services.json').read_bytes()
        for path in unpacked.rglob('*'):
            if path.is_file():
                assert path.suffix not in {'.gguf', '.safetensors', '.ckpt', '.pt', '.pth'}, path
                assert not path.name.startswith('libggml'), path
        # Exercise the installed menu launcher with a harmless environment dumper.
        home = unpacked / 'fake-home'
        env = {'HOME': str(home), 'PATH': '/usr/bin:/bin', 'GARY4LOCAL_PREVIEW_BINARY': '/usr/bin/env'}
        output = subprocess.check_output([str(launcher)], env=env, text=True)
        values = dict(line.split('=', 1) for line in output.splitlines())
        preview = home / 'gary4local-packaging/desktop-gui'
        assert values['GARY4LOCAL_PREVIEW_HOME'] == str(preview)
        assert 'GARY4LOCAL_RUNTIME_ROOT' not in values
        assert values['XDG_CONFIG_HOME'] == str(preview / 'config')
        assert values['XDG_DATA_HOME'] == str(preview / 'share')
        assert values['XDG_CACHE_HOME'] == str(preview / 'cache')
        explicit = subprocess.check_output([str(launcher)], env=dict(env, GARY4LOCAL_RUNTIME_ROOT='/explicit/test-data'), text=True)
        assert 'GARY4LOCAL_RUNTIME_ROOT=/explicit/test-data' in explicit.splitlines()
        invalid = subprocess.run([str(launcher)], env=dict(env, GARY4LOCAL_PREVIEW_HOME='relative'), capture_output=True)
        assert invalid.returncode != 0
    assert (ROOT / 'control-center/dist/index.html').is_file()
    output = TAURI / 'target/desktop-preview'
    output.mkdir(exist_ok=True)
    version = run('dpkg-deb', '-f', str(package), 'Version')
    # Tauri uses the display name (including spaces) for its internal filename.
    # Keep the distributed asset name stable and convenient for shell commands.
    destination = output / f'gary4local-preview_{version}_{args.arch}.deb'
    shutil.copy2(package, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (output / ('SHA256SUMS-' + args.platform)).write_text(f'{digest}  {destination.name}\n')
    info = {'service': 'gary4local-desktop', 'version': version,
            'commit': os.environ.get('GITHUB_SHA', run('git', '-C', str(ROOT), 'rev-parse', 'HEAD')),
            'platform': args.platform, 'architecture': args.arch, 'experimental': True,
            'updater_enabled': False, 'package': destination.name, 'sha256': digest,
            'package_checks': 'passed', 'elf_dependencies': dependencies,
            'build_os': Path('/etc/os-release').read_text(),
            'rust': run('rustc', '--version'), 'node': run('node', '--version')}
    (output / 'BUILD-INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    shutil.copy2(ROOT / 'docs/linux-desktop-preview.md', output / 'README.md')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
