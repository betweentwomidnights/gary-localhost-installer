"""Spark agent-CLI smoke. Reads existing weights, keeps evidence, removes own runtime.

python3 headless_agent_cli.py /absolute/path/gary4local-server UNIQUE_CASE
No sudo, model downloads/copies, desktop profile changes or service registration.
"""
import array
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
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
models = Path('/home/kev/sa3.cpp/models').resolve(strict=True)
before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
assert before
utilization = int(subprocess.check_output([
    'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'
], text=True).strip())
assert utilization <= 10, f'GPU busy: {utilization}%'
work = ROOT / 'headless' / ('agent-cli-' + case)
work.mkdir(exist_ok=False)
evidence = work / 'evidence'
evidence.mkdir()
config = work / 'sa3.json'
runtime = work / 'sa3.data'
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    public = s.getsockname()[1]
process = None
native = None
private = None
session = None
results = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}


def envelope(value, command=None):
    assert value['schema_version'] == 1 and value['kind'] in ('event', 'result'), value
    assert set(('command', 'event', 'ok', 'exit_code', 'data', 'error')) <= value.keys()
    assert value['ok'] == (value['exit_code'] == 0), value
    if command:
        assert value['command'] == command, value
    return value


def command(action, extra=(), expected=0, configured=True):
    args = [str(binary), action, '--json']
    if configured:
        args += ['--config', str(config)]
    p = subprocess.run(args + list(extra), capture_output=True, text=True, timeout=180)
    assert p.returncode == expected, (action, p.returncode, p.stdout, p.stderr)
    lines = p.stdout.splitlines()
    assert len(lines) == 1, (action, lines)
    result = envelope(json.loads(lines[0]), action)
    assert result['kind'] == 'result' and result['exit_code'] == expected, result
    (evidence / (action + '-last.json')).write_text(p.stdout)
    if p.stderr:
        (evidence / (action + '-stderr.txt')).write_text(p.stderr)
    return result


def http(route, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{public}' + route, data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def closed():
    for port in (public, private):
        if port is not None:
            with socket.socket() as s:
                assert s.connect_ex(('127.0.0.1', port)) != 0, port


def wait_ready():
    stream = evidence / 'serve.ndjson'
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert process.poll() is None, (stream.read_text(), (evidence / 'serve.stderr').read_text())
        lines = stream.read_text().splitlines()
        if lines:
            ready = envelope(json.loads(lines[0]), 'serve')
            assert ready['kind'] == 'event' and ready['event'] == 'ready' and ready['ok'], ready
            return ready['data']
        time.sleep(.1)
    raise AssertionError('Startup timed out')


try:
    discovery = command('capabilities', configured=False)['data']
    assert discovery['features']['training'] is False
    assert discovery['features']['model_downloads'] is False
    assert any(p['name'] == 'vulkan' and p['install_supported'] for p in discovery['published_backends'])
    command('schema', ['--target', 'config'], configured=False)
    command('schema', ['--target', 'response'], configured=False)
    extra = ['--models-dir', str(models), '--port', str(public)]
    assert command('init', extra)['data']['created']
    assert not command('init', extra)['data']['created']
    assert not runtime.exists()
    command('validate')
    blocked = command('doctor', expected=4)
    assert blocked['error']['code'] == 'not_ready' and blocked['data']['read_only']
    assert not runtime.exists(), 'Read-only preflight created storage'
    results['init_validate_doctor_without_mutation'] = True
    installed = command('install')['data']
    assert command('install')['data'] == installed
    preflight = command('doctor')['data']
    assert preflight['can_attempt_serve'] and preflight['read_only']
    results['explicit_manifest_install_and_reinstall'] = installed
    with (evidence / 'serve.ndjson').open('w') as out, (evidence / 'serve.stderr').open('w') as err:
        process = subprocess.Popen([str(binary), 'serve', '--config', str(config), '--json'],
                                   stdout=out, stderr=err, cwd=work)
    ready = wait_ready()
    session = ready['session']
    assert ready['public_url'] == f'http://127.0.0.1:{public}'
    assert Path(ready['log_path']) == runtime / 'logs' / 'sa3-native.log'
    state = command('status')['data']
    assert state['running'] and state['health'] and state['state']['session'] == session
    native, private = state['state']['native_pid'], state['state']['native_port']
    mismatch = command('stop', ['--session', 'stale-' + session], expected=1)
    assert mismatch['error']['code'] == 'session_mismatch'
    assert not (runtime / 'stop.request').exists()
    assert command('status')['data']['running'] and process.poll() is None
    results['mismatched_session_preserves_running_service'] = True
    results['ready'] = ready
    generation = http('/generate', {
        'prompt': 'a short warm synth chord', 'duration': 2, 'tail_pad': 0,
        'steps': 2, 'seed': 42, 'cfg_scale': 2,
        'peak_normalize_db': None, 'limiter_ceiling_db': None,
    })
    deadline = time.monotonic() + 180
    while True:
        completed = http('/poll_status/' + generation['session_id'])
        assert completed['status'] != 'failed', completed
        if completed['status'] == 'completed':
            break
        assert time.monotonic() < deadline, completed
        time.sleep(.2)
    audio = base64.b64decode(completed.pop('audio_data'))
    (evidence / 'generation.wav').write_bytes(audio)
    with wave.open(io.BytesIO(audio)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 2, 44100, 88200)
        samples = array.array('h', wav.readframes(wav.getnframes()))
        assert any(samples)
    results['generation'] = {'sha256': hashlib.sha256(audio).hexdigest(), 'frames': 88200,
                             'sample_rate': 44100, 'peak_pcm16': max(map(abs, samples))}
    assert command('stop', ['--session', session])['data']['stopped']
    process.wait(timeout=15)
    assert process.returncode == 0
    deadline = time.monotonic() + 5
    while alive(native) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not alive(native)
    closed()
    messages = [envelope(json.loads(line), 'serve') for line in (evidence / 'serve.ndjson').read_text().splitlines()]
    assert len(messages) == 2
    assert messages[-1]['kind'] == 'result' and messages[-1]['ok']
    assert messages[-1]['data']['session'] == session and messages[-1]['data']['stopped']
    assert command('stop', ['--session', session])['data']['stopped']
    assert not command('status')['data']['running']
    legacy = subprocess.run([str(binary), 'status', '--config', str(config)],
                            capture_output=True, text=True, check=True)
    assert 'schema_version' not in json.loads(legacy.stdout)
    results['ndjson_terminal_result_fenced_stop_repeat_and_legacy_status'] = True
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
    assert before == after
    results['model_sources_unchanged'] = True
    results['owned_child_exited_and_ports_closed'] = True
    shutil.copy2(runtime / 'logs' / 'sa3-native.log', evidence / 'sa3-native.log')
    results['runtime_bytes_before_cleanup'] = sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())
    # This exact root was freshly created by this test, and its owner is stopped.
    assert runtime.resolve().parent == work.resolve() and work.resolve().parent == ROOT / 'headless'
    assert (runtime / '.gary4local-headless').is_file() and not any(p.is_symlink() for p in runtime.rglob('*'))
    shutil.rmtree(runtime)
    results['fresh_runtime_removed_after_validation'] = not runtime.exists()
    (evidence / 'RESULT.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
finally:
    if process and process.poll() is None:
        # Ask only the session captured from this process to stop. Never signal
        # any PID read from stale state and never affect other GPU workloads.
        if session:
            command('stop', ['--session', session])
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
