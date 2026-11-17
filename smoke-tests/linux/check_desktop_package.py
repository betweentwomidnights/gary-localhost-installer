"""Check a native Linux desktop DEB before retaining it as a CI artifact."""
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
    packages = list((TAURI / 'target/release/bundle/deb').glob('*.deb'))
    assert len(packages) == 1, packages
    package = packages[0]
    assert run('dpkg-deb', '-f', str(package), 'Architecture') == args.arch
    assert run('dpkg-deb', '-f', str(package), 'Package') == 'gary4local'
    # One identity: the package takes over an installed gary4local-preview.
    for field in ('Conflicts', 'Replaces'):
        assert 'gary4local-preview' in run('dpkg-deb', '-f', str(package), field), field
    depends = run('dpkg-deb', '-f', str(package), 'Depends')
    assert all(lib in depends for lib in ['libwebkit2gtk-4.1-0', 'libayatana-appindicator3-1', 'libvulkan1'])
    with tempfile.TemporaryDirectory(prefix='gary-desktop-check-') as directory:
        unpacked = Path(directory)
        subprocess.run(['dpkg-deb', '-x', str(package), str(unpacked)], check=True)
        binary = unpacked / 'usr/bin/gary4local'
        launcher = unpacked / 'usr/bin/gary4local-launch'
        assert binary.is_file() and os.access(binary, os.X_OK)
        assert launcher.is_file() and os.access(launcher, os.X_OK)
        elf = run('readelf', '-h', str(binary))
        assert ('AArch64' if args.arch == 'arm64' else 'Advanced Micro Devices X86-64') in elf
        dependencies = run('ldd', str(binary))
        assert 'not found' not in dependencies, dependencies
        desktop = list((unpacked / 'usr/share/applications').glob('*.desktop'))
        assert len(desktop) == 1
        assert desktop[0].name == 'gary4local.desktop'
        subprocess.run(['desktop-file-validate', str(desktop[0])], check=True)
        assert 'Exec=gary4local-launch' in desktop[0].read_text()
        assert 'Icon=gary4local' in desktop[0].read_text()
        assert 'StartupWMClass=gary4local' in desktop[0].read_text()
        for size in [32, 128, 256]:
            icon = unpacked / f'usr/share/icons/hicolor/{size}x{size}/apps/gary4local.png'
            source = TAURI / f'icons/linux/{size}x{size}.png'
            assert icon.read_bytes() == source.read_bytes(), icon
        manifests = list(unpacked.rglob('services/manifests/services.json'))
        assert len(manifests) == 1, manifests
        assert manifests[0].read_bytes() == (ROOT / 'services/manifests/services.json').read_bytes()
        for path in unpacked.rglob('*'):
            if path.is_file():
                assert path.suffix not in {'.gguf', '.safetensors', '.ckpt', '.pt', '.pth'}, path
                assert not path.name.startswith('libggml'), path
        # Exercise the installed menu launcher with a harmless environment dumper.
        home = unpacked / 'fake-home'
        env = {'HOME': str(home), 'PATH': '/usr/bin:/bin', 'GARY4LOCAL_BINARY': '/usr/bin/env'}
        def launch(**extra):
            output = subprocess.check_output([str(launcher)], env=dict(env, **extra), text=True)
            return dict(line.split('=', 1) for line in output.splitlines())
        # By default it is an ordinary install: the standard XDG folders, untouched.
        values = launch()
        assert not any(name in values for name in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'GARY4LOCAL_HOME'))
        assert 'GARY4LOCAL_RUNTIME_ROOT' not in values
        assert not home.exists(), 'the default launch created folders'
        # GARY4LOCAL_HOME opts into one self-contained folder.
        sandbox = home / 'gary4local-sandbox'
        values = launch(GARY4LOCAL_HOME=str(sandbox))
        assert values['XDG_CONFIG_HOME'] == str(sandbox / 'config')
        assert values['XDG_DATA_HOME'] == str(sandbox / 'share')
        assert values['XDG_CACHE_HOME'] == str(sandbox / 'cache')
        assert launch(GARY4LOCAL_RUNTIME_ROOT='/explicit/test-data')['GARY4LOCAL_RUNTIME_ROOT'] == '/explicit/test-data'
        invalid = subprocess.run([str(launcher)], env=dict(env, GARY4LOCAL_HOME='relative'), capture_output=True)
        assert invalid.returncode != 0
    assert (ROOT / 'control-center/dist/index.html').is_file()
    output = TAURI / 'target/desktop-linux'
    output.mkdir(exist_ok=True)
    version = run('dpkg-deb', '-f', str(package), 'Version')
    # Tauri uses the display name (including spaces) for its internal filename.
    # Keep the distributed asset name stable and convenient for shell commands.
    destination = output / f'gary4local_{version}_{args.arch}.deb'
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
    shutil.copy2(ROOT / 'docs/linux-desktop.md', output / 'README.md')
    shutil.copy2(TAURI / 'linux/install.sh', output / 'INSTALL.sh')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
