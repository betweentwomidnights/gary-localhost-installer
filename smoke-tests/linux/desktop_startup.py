"""Start the installed preview under a virtual display, without starting engines."""
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import time

root = Path(__file__).resolve().parents[2] / 'control-center/src-tauri/target/desktop-preview'
profile = root / 'startup-data'
assert not profile.exists()
evidence = root / 'STARTUP.log'
env = dict(os.environ, GARY4LOCAL_PREVIEW_HOME=str(profile), LIBGL_ALWAYS_SOFTWARE='1')
with evidence.open('w') as log:
    process = subprocess.Popen(['/usr/bin/gary4local-preview-launch'], env=env, cwd='/',
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        manifest = profile / 'data/services/manifests/services.json'
        while not manifest.is_file():
            assert process.poll() is None, evidence.read_text()
            assert time.monotonic() < deadline, evidence.read_text()
            time.sleep(.2)
        time.sleep(5)
        assert process.poll() is None, evidence.read_text()
        # A live process and seeded manifest can still hide a blank WebKit view.
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
        candidates = re.findall(r'^\s+(0x[0-9a-fA-F]+).*"gary4local', tree, re.M)
        windows = []
        for candidate in candidates:
            geometry = subprocess.check_output(['xwininfo', '-id', candidate], text=True)
            width = re.search(r'Width:\s+(\d+)', geometry)
            height = re.search(r'Height:\s+(\d+)', geometry)
            if (width and height and int(width[1]) >= 900 and int(height[1]) >= 550
                    and 'Map State: IsViewable' in geometry):
                windows.append(candidate)
        assert len(windows) == 1, tree
        assert windows, tree
        screenshot = root / 'STARTUP.png'
        subprocess.run(['import', '-window', windows[0], str(screenshot)], check=True, timeout=15)
        colors = int(subprocess.check_output(['identify', '-format', '%k', str(screenshot)], text=True))
        assert colors > 32, f'Blank or unrendered desktop: only {colors} colors'
        data = json.loads(manifest.read_text())
        assert len(data['services']) == 7
        result = {'installed_package_started': True, 'services_seeded': 7,
                  'rendered_ui_screenshot': 'STARTUP.png', 'rendered_colors': colors,
                  'runtime_root': str(profile / 'data'), 'engine_started': False,
                  'display': 'Xvfb software rendering; hardware GUI validation remains manual'}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    result['supervisor_stopped'] = True
    (root / 'STARTUP.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
