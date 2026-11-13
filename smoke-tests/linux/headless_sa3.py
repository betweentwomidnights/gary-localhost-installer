"""Exercise the GUI-free SA3 host on Spark using existing, read-only models.

Usage: python3 headless_sa3.py /absolute/path/to/gary4local-server cuda|vulkan [case] [--manifest]
Artifacts and all owned processes stay under gary4local-packaging/headless.
"""
import array
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

ROOT = Path('/home/kev/gary4local-packaging')
binary = Path(sys.argv[1]).resolve(strict=True)
backend = sys.argv[2]
assert backend in {'cuda', 'vulkan'}
use_manifest = '--manifest' in sys.argv[3:]
cases = [arg for arg in sys.argv[3:] if arg != '--manifest']
assert len(cases) <= 1
assert not use_manifest or backend == 'vulkan'
suffix='-'+cases[0] if cases else ''
assert all(c.isalnum() or c=='-' for c in suffix)
work = ROOT/'headless'/('sa3-'+backend+suffix)
work.mkdir(exist_ok=False)
evidence = work/'evidence'; evidence.mkdir()
runtime_root = work/'data'
models = Path('/home/kev/sa3.cpp/models')
before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
assert before
utilization = int(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
assert utilization <= 10, f'GPU busy: {utilization}%'
dist = ROOT/'sa3.cpp/dist-linux' if backend=='cuda' else ROOT/'ci-exercise/43297964568/artifacts'
sums = {name: h for h, name in (line.split() for line in (dist/'SHA256SUMS-linux-arm64').read_text().splitlines())}
def package(part):
    path = dist/f'sa3-v0.2.3-linux-arm64-{part}.zip'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sums[path.name]
    return {'path': str(path), 'sha256': sums[path.name]}
def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]
public, private = port(), None
config = {'runtime_root': str(runtime_root), 'models_dir': str(models),
          'core': package('core'), 'backend': package(backend),
          'device': 'CUDA0' if backend=='cuda' else 'Vulkan0',
          'public_port': public,
          'model': 'medium', 'encoding': 'f16', 't5_encoding': 'f16', 'ae_encoding': 'f32', 'threads': 4,
          'defaults': {}}
if use_manifest:
    del config['core']; del config['backend']
config_path = work/'config.json'
config_path.write_text(json.dumps(config, indent=2)+'\n')
def command(action, check=True, path=config_path):
    p = subprocess.run([str(binary), action, '--config', str(path)], capture_output=True, text=True, timeout=180)
    if check: assert p.returncode==0, (action, p.stderr)
    return p
def http(route, body=None, timeout=15):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{public}'+route, data=data,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response: return json.load(response)
def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split()[2]!='Z'
    except FileNotFoundError: return False
def closed():
    deadline=time.monotonic()+5
    while True:
        listening=[]
        for p in (p for p in (public, private) if p is not None):
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',p))==0: listening.append(p)
        if not listening: return
        assert time.monotonic()<deadline, f'Ports still listening: {listening}'
        time.sleep(.05)
process=None
native_pid=None
def start(label):
    global process, native_pid, private
    log=(evidence/(label+'-supervisor.log')).open('w')
    process=subprocess.Popen([str(binary),'serve','--config',str(config_path)], stdout=log, stderr=subprocess.STDOUT, cwd=work)
    log.close()
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        assert process.poll() is None, (label, (evidence/(label+'-supervisor.log')).read_text())
        try:
            state=json.loads((runtime_root/'state.json').read_text())
            if state['pid']==process.pid and state['status']=='running':
                native_pid=state['native_pid']
                private=state['native_port']
                return state
        except (FileNotFoundError, json.JSONDecodeError): pass
        time.sleep(.1)
    raise AssertionError('Startup timeout')
def stopped(expected=0):
    process.wait(timeout=12)
    assert process.returncode==expected, process.returncode
    deadline=time.monotonic()+5
    while alive(native_pid) and time.monotonic()<deadline: time.sleep(.05)
    assert not alive(native_pid), 'Owned native child survived'
    closed()

results={}
try:
    # A bad checksum must fail before producing an installed runtime.
    bad=dict(config, core=dict(package('core'),sha256='0'*64), backend=package(backend))
    bad_path=work/'bad-checksum.json'; bad_path.write_text(json.dumps(bad))
    rejected=command('install',check=False,path=bad_path)
    assert rejected.returncode!=0 and 'SHA256 mismatch' in rejected.stderr
    results['bad_checksum_rejected']=True
    results['install']=json.loads(command('install').stdout)
    results['idempotent_install']=json.loads(command('install').stdout)
    if use_manifest:
        for part in ('core', 'vulkan'):
            expected=package(part)['sha256']
            cached=runtime_root/'downloads'/(expected+'.zip')
            assert hashlib.sha256(cached.read_bytes()).hexdigest()==expected
        results['published_manifest_downloads_verified']=True
        # A cached install succeeds with a valid HTTPS URL that cannot resolve.
        manifest=json.loads((ROOT/'headless/distribution-source/services/manifests/services.json').read_text())
        platforms=manifest['nativeBundles']['sa3']['platforms']['linux-arm64']
        platforms['core']['url']='https://cached.invalid/core.zip'
        platforms['backends']['vulkan']['url']='https://cached.invalid/vulkan.zip'
        override=work/'cached-manifest.json'; override.write_text(json.dumps(manifest))
        custom=dict(config,manifest=str(override))
        custom_path=work/'override.json';custom_path.write_text(json.dumps(custom))
        assert json.loads(command('install',path=custom_path).stdout)==results['install']
        results['verified_cache_needs_no_network']=True
        # A bad published checksum fails without replacing the installed runtime.
        platforms['core']['url']=json.loads((ROOT/'headless/distribution-source/services/manifests/services.json').read_text())['nativeBundles']['sa3']['platforms']['linux-arm64']['core']['url']
        platforms['core']['sha256']='0'*64
        override.write_text(json.dumps(manifest))
        rejected=command('install',check=False,path=custom_path)
        assert rejected.returncode!=0 and 'download was discarded' in rejected.stderr,rejected.stderr
        assert not (runtime_root/'downloads'/('0'*64+'.zip.partial')).exists()
        platforms['core']['sha256']=package('core')['sha256']
        manifest['nativeBundles']['sa3']['version']='v0.0.0-invalid'
        override.write_text(json.dumps(manifest))
        rejected=command('install',check=False,path=custom_path)
        assert rejected.returncode!=0 and 'metadata' in rejected.stderr,rejected.stderr
        assert json.loads(command('install').stdout)==results['install']
        assert len(list((runtime_root/'runtimes').iterdir()))==1
        results['bad_manifest_candidates_preserve_installed_runtime']=True
    bad_model=dict(config,ae_encoding='f16')
    bad_model_path=work/'missing-component.json'; bad_model_path.write_text(json.dumps(bad_model))
    rejected=command('serve',check=False,path=bad_model_path)
    assert rejected.returncode!=0 and 'SA3 model preparation failed' in rejected.stderr
    closed(); results['missing_model_component_rejected_before_ready']=True
    # Occupied public ports must not create an orphan native server.
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',public)); occupied.listen()
        rejected=command('serve',check=False)
        assert rejected.returncode!=0 and 'Cannot bind SA3 client adapter' in rejected.stderr
    closed(); results['occupied_port_rejected']=True
    first=start('generation')
    results['running_status']=json.loads(command('status').stdout)
    assert results['running_status']['running'] and results['running_status']['health']
    # No second daemon or package install may mutate an active root.
    for action in ('serve','install'):
        rejected=command(action,check=False)
        assert rejected.returncode!=0 and 'active operation' in rejected.stderr
    results['concurrent_operations_rejected']=True
    assert http('/loras')['loras']==[]
    started=time.monotonic()
    submitted=http('/generate', {'prompt':'a short warm synth chord', 'duration':2, 'tail_pad':0,
                                'steps':2, 'seed':42, 'cfg_scale':2,
                                'peak_normalize_db':None, 'limiter_ceiling_db':None})
    session=submitted['session_id']
    deadline=time.monotonic()+180
    while True:
        completed=http('/poll_status/'+session)
        assert completed['status']!='failed', completed
        if completed['status']=='completed': break
        assert time.monotonic()<deadline, completed
        time.sleep(.2)
    audio=base64.b64decode(completed.pop('audio_data'))
    (evidence/'generation.wav').write_bytes(audio)
    (evidence/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    with wave.open(io.BytesIO(audio)) as wav:
        assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())==(2,2,44100,88200)
        samples=array.array('h',wav.readframes(wav.getnframes())); assert any(samples)
    generation_log=(runtime_root/'logs/sa3-native.log').read_text()
    backend_lines=[line for line in generation_log.splitlines() if '[sa3] backend:' in line]
    assert backend_lines and all(config['device'].lower() in line.lower() and 'NVIDIA GB10' in line for line in backend_lines), backend_lines
    shutil.copy2(runtime_root/'logs/sa3-native.log', evidence/'generation-native.log')
    results['generation']={'session':session,'elapsed_seconds':time.monotonic()-started,
                           'audio_sha256':hashlib.sha256(audio).hexdigest(),'frames':88200,'sample_rate':44100,
                           'channels':2,'peak_pcm16':max(map(abs,samples)),'backend_lines':backend_lines}
    results['stop']=json.loads(command('stop').stdout); stopped()
    assert not json.loads(command('status').stdout)['running']
    # Restarting must ignore the previous session's stop request.
    second=start('sigterm'); assert second['session']!=first['session']
    time.sleep(.4); assert process.poll() is None
    process.send_signal(signal.SIGTERM); stopped()
    results['restart_and_sigterm']=True
    start('native-crash')
    os.kill(native_pid,signal.SIGKILL); stopped(expected=1)
    results['native_crash_closes_adapter']=True
    start('owner-crash')
    process.kill(); stopped(expected=-signal.SIGKILL)
    results['owner_crash_kills_native']=True
    assert not json.loads(command('status').stdout)['running']
    # An OS-released lock must allow recovery after an ungraceful exit.
    start('recovery'); command('stop'); stopped()
    results['recovery_after_crash']=True
    after={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in models.glob('*.gguf')}
    assert before==after
    results['existing_models_unchanged']=True
    results['all_test_ports_closed']=True
    results['backend']=backend
    (evidence/'RESULT.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
finally:
    if process and process.poll() is None:
        command('stop',check=False)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
    if native_pid and alive(native_pid):
        os.killpg(native_pid,signal.SIGKILL)
