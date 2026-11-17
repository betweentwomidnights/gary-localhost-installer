"""Real Spark PTY/GPU lifecycle test for the human Gary console.

python3 gary_console.py /absolute/path/to/gary UNIQUE_CASE
Reads existing weights. Keeps evidence, removes only its fresh stopped runtime.
"""
import array
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import wave

ROOT = Path('/home/kev/gary4local-packaging').resolve(strict=True)
binary = Path(sys.argv[1]).resolve(strict=True)
case = sys.argv[2]
assert case and all(c.isalnum() or c == '-' for c in case)
models = Path('/home/kev/sa3.cpp/models')
before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
assert before
assert int(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()) <= 10
work = ROOT / 'headless' / ('gary-console-' + case)
work.mkdir(exist_ok=False)
evidence = work / 'evidence'; evidence.mkdir()
config = work / 'sa3.json'
runtime = work / 'sa3.data'
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); public = s.getsockname()[1]
env = dict(os.environ, GARY4LOCAL_HEADLESS_CONFIG=str(config))
external = None
terminal = None
results = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}


class Terminal:
    def __init__(self, name):
        self.name, self.text, self.consumed = name, '', 0
        self.master, slave = pty.openpty()
        self.process = subprocess.Popen([str(binary)], stdin=slave, stdout=slave, stderr=slave, env=env, cwd=work)
        os.close(slave)

    def send(self, text):
        os.write(self.master, text.encode())

    def expect(self, needle, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            offset = self.text.find(needle, self.consumed)
            if offset >= 0:
                self.consumed = offset + len(needle)
                return
            assert time.monotonic() < deadline, (needle, self.text)
            if select.select([self.master], [], [], .1)[0]:
                try:
                    chunk = os.read(self.master, 16384)
                except OSError as error:
                    raise AssertionError((needle, self.text)) from error
                assert chunk, (needle, self.text)
                self.text += chunk.decode(errors='replace')

    def exited(self):
        self.process.wait(timeout=15)
        assert self.process.returncode == 0, self.text
        (evidence / (self.name + '.txt')).write_text(self.text)
        os.close(self.master)

    def cleanup(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        (evidence / (self.name + '.txt')).write_text(self.text)


def command(action, extra=()):
    p = subprocess.run([str(binary), action, '--json'] + list(extra), env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, (action, p.stdout, p.stderr)
    result = json.loads(p.stdout)
    assert result['schema_version'] == 1 and result['ok'], result
    return result['data']


def http(route, body=None):
    request = urllib.request.Request(f'http://127.0.0.1:{public}' + route,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def no_native_children():
    deadline = time.monotonic() + 10
    while True:
        owned = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if (proc / 'cwd').resolve(strict=True).is_relative_to(runtime) and b'sa3-server' in (proc / 'cmdline').read_bytes():
                    owned.append(int(proc.name))
            except (OSError, ValueError):
                pass
        if not owned:
            break
        assert time.monotonic() < deadline, owned
        time.sleep(.1)
    assert not command('status')['running']
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', public)) != 0


try:
    terminal = Terminal('setup-and-lifecycle')
    terminal.expect('not configured'); terminal.expect('gary> ')
    assert not config.exists() and not runtime.exists()
    terminal.send('/setup\n'); terminal.expect('SA3 models: ')
    terminal.send('/cancel\n'); terminal.expect('no config was saved'); terminal.expect('gary> ')
    assert not config.exists()
    terminal.send('/setup\n'); terminal.expect('SA3 models: ')
    terminal.send(str(models) + '\n'); terminal.expect('Client port')
    terminal.send(str(public) + '\n'); terminal.expect('GPU device')
    terminal.send('\n'); terminal.expect('Generation precision')
    terminal.send('\n'); terminal.expect('Save [y/N]: ')
    terminal.send('y\n'); terminal.expect('Created config'); terminal.expect('gary> ')
    assert config.exists() and not runtime.exists()
    results['guided_setup_cancel_and_save_without_runtime_creation'] = True
    terminal.send('/doctor\n'); terminal.expect('preflight: blocked'); terminal.expect('gary> ')
    assert not runtime.exists()
    terminal.send('/install\n'); terminal.expect('SA3 runtime installed'); terminal.expect('gary> ')
    terminal.send('/start\n/stop\n'); terminal.expect('Starting SA3'); terminal.expect('SA3 stopped.'); terminal.expect('gary> ')
    no_native_children()
    results['startup_cancellation_releases_owned_children_and_lock'] = True
    terminal.send('/start sa3\n'); terminal.expect('SA3 ready at '); terminal.expect('gary> ')
    state = command('status')
    assert state['running'] and state['health']
    first_session = state['state']['session']
    terminal.send('/status\n'); terminal.expect('running'); terminal.expect('gary> ')
    submitted = http('/generate', {'prompt': 'a short warm synth chord', 'duration': 2, 'tail_pad': 0,
                                 'steps': 2, 'seed': 42, 'cfg_scale': 2,
                                 'peak_normalize_db': None, 'limiter_ceiling_db': None})
    deadline = time.monotonic() + 180
    while True:
        completed = http('/poll_status/' + submitted['session_id'])
        assert completed['status'] != 'failed', completed
        if completed['status'] == 'completed':
            break
        assert time.monotonic() < deadline, completed
        time.sleep(.2)
    audio = base64.b64decode(completed['audio_data'])
    with wave.open(io.BytesIO(audio)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 2, 44100, 88200)
        assert any(array.array('h', wav.readframes(wav.getnframes())))
    (evidence / 'generation.wav').write_bytes(audio)
    shutil.copy2(runtime / 'logs/sa3-native.log', evidence / 'generation-native.log')
    results['interactive_owned_service_generation'] = {'audio_sha256': hashlib.sha256(audio).hexdigest(), 'frames': 88200}
    terminal.send('/stop sa3\n'); terminal.expect('SA3 stopped.'); terminal.expect('gary> ')
    no_native_children()
    terminal.send('/start\n'); terminal.expect('SA3 ready at '); terminal.expect('gary> ')
    assert command('status')['state']['session'] != first_session
    terminal.send('/quit\n'); terminal.expect('Goodbye.'); terminal.exited(); terminal = None
    no_native_children()
    results['stop_restart_and_quit_stop_only_owned_service'] = True
    # A console must not acquire or stop a host that was already running.
    with (evidence / 'external.ndjson').open('w') as out, (evidence / 'external.stderr').open('w') as err:
        external = subprocess.Popen([str(binary.parent / 'gary4local-server'), 'serve', '--config', str(config), '--json'],
                                    stdout=out, stderr=err, cwd=work)
    deadline = time.monotonic() + 180
    while True:
        assert external.poll() is None, (evidence / 'external.stderr').read_text()
        try:
            lines = (evidence / 'external.ndjson').read_text().splitlines()
            if lines and json.loads(lines[0])['event'] == 'ready':
                external_session = json.loads(lines[0])['data']['session']; break
        except json.JSONDecodeError:
            pass
        assert time.monotonic() < deadline
        time.sleep(.1)
    terminal = Terminal('external-owner-preserved')
    terminal.expect('Running in another session.'); terminal.expect('gary> ')
    terminal.send('/quit\n'); terminal.expect('Goodbye.'); terminal.exited(); terminal = None
    assert external.poll() is None and command('status')['running']
    assert command('stop', ['--session', external_session])['stopped']
    external.wait(timeout=15); assert external.returncode == 0
    no_native_children()
    results['quitting_console_preserves_external_service'] = True
    terminal = Terminal('ctrl-c')
    terminal.expect('gary> '); terminal.process.send_signal(signal.SIGINT)
    terminal.expect('Goodbye.'); terminal.exited(); terminal = None
    no_native_children()
    results['ctrl_c_without_enter'] = True
    assert before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
    results['existing_models_unchanged'] = True
    results['runtime_bytes_before_cleanup'] = sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())
    assert runtime.resolve().parent == work.resolve() and work.resolve().parent == ROOT / 'headless'
    assert (runtime / '.gary4local-headless').is_file() and not any(p.is_symlink() for p in runtime.rglob('*'))
    shutil.rmtree(runtime)
    results['fresh_owned_runtime_removed'] = True
    (evidence / 'RESULT.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
finally:
    if terminal:
        terminal.cleanup()
    if external and external.poll() is None:
        external.terminate()
        try:
            external.wait(timeout=15)
        except subprocess.TimeoutExpired:
            external.kill(); external.wait(timeout=5)
