import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const script = fileURLToPath(new URL('./pin_native_release.mjs', import.meta.url));
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gary-pin-'));
  const manifest = path.join(root, 'services.json');
  const sums = path.join(root, 'SHA256SUMS');
  const packageDef = { version: 'v0.1.1', platforms: { 'windows-x64': {
    core: { url: 'https://github.com/owner/sa3.cpp/releases/download/v0.1.1/sa3-v0.1.1-core.zip', sha256: 'a'.repeat(64) },
    backends: { cuda: { url: 'https://github.com/owner/sa3.cpp/releases/download/v0.1.1/sa3-v0.1.1-cuda.zip', sha256: 'b'.repeat(64), requires: ['cudart-12.8'] } },
  } } };
  const source = { nativeBundles: { sa3: packageDef }, services: [
    { id: 'sa3', native: { bundle: 'sa3', executable: 'sa3-server.exe' } },
    { id: 'other', native: packageDef },
  ], nativeRuntimes: { 'cudart-12.8': { url: 'https://example.com/cudart.zip', sha256: 'c'.repeat(64) } } };
  fs.writeFileSync(manifest, JSON.stringify(source, null, 2));
  fs.writeFileSync(sums, `${'d'.repeat(64)}  sa3-v0.1.2-core.zip\n${'e'.repeat(64)}  sa3-v0.1.2-cuda.zip\n`);
  return { root, manifest, sums, source, run: (...target) => spawnSync(process.execPath, [script,
    ...target, '--repo', 'owner/sa3.cpp', '--tag', 'v0.1.2', '--manifest', manifest, '--sums', sums], { encoding: 'utf8' }) };
}

test('bundle pin changes only its packages, preserving consumers and other pins', () => {
  const f = fixture();
  try {
    const result = f.run('--bundle', 'sa3');
    assert.equal(result.status, 0, result.stderr);
    const updated = JSON.parse(fs.readFileSync(f.manifest, 'utf8'));
    assert.equal(updated.nativeBundles.sa3.version, 'v0.1.2');
    const platform = updated.nativeBundles.sa3.platforms['windows-x64'];
    assert.equal(platform.core.sha256, 'd'.repeat(64));
    assert.match(platform.core.url, /v0\.1\.2\/sa3-v0\.1\.2-core\.zip$/);
    assert.equal(platform.backends.cuda.sha256, 'e'.repeat(64));
    assert.deepEqual(platform.backends.cuda.requires, ['cudart-12.8']);
    assert.deepEqual(updated.services, f.source.services);
    assert.deepEqual(updated.nativeRuntimes, f.source.nativeRuntimes);
  } finally { fs.rmSync(f.root, { recursive: true }); }
});

test('consumer service pin directs the caller to the bundle without writing', () => {
  const f = fixture();
  try {
    const before = fs.readFileSync(f.manifest, 'utf8');
    const result = f.run('--service', 'sa3');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /--bundle sa3/);
    assert.equal(fs.readFileSync(f.manifest, 'utf8'), before);
  } finally { fs.rmSync(f.root, { recursive: true }); }
});

test('incomplete release hashes leave the manifest intact', () => {
  const f = fixture();
  try {
    fs.writeFileSync(f.sums, `${'d'.repeat(64)}  sa3-v0.1.2-core.zip\n`);
    const before = fs.readFileSync(f.manifest, 'utf8');
    const result = f.run('--bundle', 'sa3');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /no sa3-v0.1.2-cuda.zip/);
    assert.equal(fs.readFileSync(f.manifest, 'utf8'), before);
  } finally { fs.rmSync(f.root, { recursive: true }); }
});
