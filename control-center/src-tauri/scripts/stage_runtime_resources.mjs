import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const tauriDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(tauriDir, '..', '..');
const resourcesDir = path.join(tauriDir, 'resources');
const stagedServicesDir = path.join(resourcesDir, 'services');

const serviceNames = ['gary', 'melodyflow', 'stable-audio', 'carey', 'foundation'];

const excludedPathSegments = new Set([
  '.git',
  '.venv',
  'env',
  '.claude',
  '.cache',
  '.pytest_cache',
  'checkpoints',
  '__pycache__',
  'smoke-tests',
]);

const excludedBasenames = new Set([
  '.DS_Store',
  'Thumbs.db',
  'smoke.wav',
  'smoke.mp3',
]);

const excludedSuffixes = ['.pyc', '.log'];

const careyExcludedPrefixes = [
  'acestep/docs',
  'acestep/examples',
  'acestep/assets',
  'acestep/.github',
  'acestep/docker-patches',
  'acestep/gradio_outputs',
];

function log(message) {
  console.log(`[stage-runtime] ${message}`);
}

function normalizeRelativePath(relativePath) {
  return relativePath.split(path.sep).join('/');
}

function shouldSkipPath(serviceName, relativePath) {
  if (!relativePath) {
    return false;
  }

  const normalized = normalizeRelativePath(relativePath);
  const parts = normalized.split('/').filter(Boolean);
  const basename = parts[parts.length - 1] ?? '';

  if (parts.some((part) => excludedPathSegments.has(part))) {
    return true;
  }

  if (excludedBasenames.has(basename)) {
    return true;
  }

  if (excludedSuffixes.some((suffix) => basename.endsWith(suffix))) {
    return true;
  }

  if (
    serviceName === 'carey' &&
    careyExcludedPrefixes.some(
      (prefix) => normalized === prefix || normalized.startsWith(`${prefix}/`),
    )
  ) {
    return true;
  }

  return false;
}

function copyTree(serviceName, srcRoot, dstRoot, relativePath = '') {
  const currentSourceDir = relativePath ? path.join(srcRoot, relativePath) : srcRoot;
  const entries = fs.readdirSync(currentSourceDir, { withFileTypes: true });

  for (const entry of entries) {
    const nextRelativePath = relativePath
      ? path.join(relativePath, entry.name)
      : entry.name;

    if (shouldSkipPath(serviceName, nextRelativePath)) {
      continue;
    }

    const srcPath = path.join(srcRoot, nextRelativePath);
    const dstPath = path.join(dstRoot, nextRelativePath);

    if (entry.isDirectory()) {
      fs.mkdirSync(dstPath, { recursive: true });
      copyTree(serviceName, srcRoot, dstRoot, nextRelativePath);
      continue;
    }

    if (entry.isFile()) {
      fs.mkdirSync(path.dirname(dstPath), { recursive: true });
      fs.copyFileSync(srcPath, dstPath);
      continue;
    }

    if (entry.isSymbolicLink()) {
      const target = fs.realpathSync(srcPath);
      const stat = fs.statSync(target);
      if (stat.isDirectory()) {
        fs.mkdirSync(dstPath, { recursive: true });
        copyTree(serviceName, target, dstPath);
      } else {
        fs.mkdirSync(path.dirname(dstPath), { recursive: true });
        fs.copyFileSync(target, dstPath);
      }
    }
  }
}

function stageService(name) {
  const srcDir = path.join(repoRoot, 'services', name);
  const dstDir = path.join(stagedServicesDir, name);

  if (!fs.existsSync(srcDir) || !fs.statSync(srcDir).isDirectory()) {
    log(`WARNING: missing source directory: ${srcDir}`);
    return;
  }

  log(`staging ${name}...`);
  fs.mkdirSync(dstDir, { recursive: true });
  copyTree(name, srcDir, dstDir);
}

function copyFileIfPresent(srcPath, dstPath) {
  if (!fs.existsSync(srcPath) || !fs.statSync(srcPath).isFile()) {
    return false;
  }

  fs.mkdirSync(path.dirname(dstPath), { recursive: true });
  fs.copyFileSync(srcPath, dstPath);
  return true;
}

function main() {
  log(`repo root: ${repoRoot}`);
  log(`staging into: ${stagedServicesDir}`);

  fs.mkdirSync(resourcesDir, { recursive: true });
  fs.rmSync(stagedServicesDir, { recursive: true, force: true });
  fs.mkdirSync(stagedServicesDir, { recursive: true });

  for (const serviceName of serviceNames) {
    stageService(serviceName);
  }

  const manifestSrc = path.join(repoRoot, 'services', 'manifests', 'services.json');
  const manifestDst = path.join(stagedServicesDir, 'manifests', 'services.json');
  if (!copyFileIfPresent(manifestSrc, manifestDst)) {
    throw new Error(`Required manifest is missing: ${manifestSrc}`);
  }

  const sessionStoreSrc = path.join(repoRoot, 'services', 'local_session_store.py');
  const sessionStoreDst = path.join(stagedServicesDir, 'local_session_store.py');
  copyFileIfPresent(sessionStoreSrc, sessionStoreDst);

  const repoIcon = path.join(repoRoot, 'icon.png');
  const stagedIcon = path.join(resourcesDir, 'icon.png');
  if (!copyFileIfPresent(repoIcon, stagedIcon)) {
    fs.rmSync(stagedIcon, { force: true });
  }

  log('done');
}

main();
