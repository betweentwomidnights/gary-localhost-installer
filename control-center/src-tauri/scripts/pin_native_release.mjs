// Pin a native service, or a shared runtime, to a published release: rewrite
// its URLs and SHA-256s in services/manifests/services.json from the release's
// SHA256SUMS. See docs/native-runtime-packages.md.
//
// Usage:
//   node control-center/src-tauri/scripts/pin_native_release.mjs \
//     --service yuey --repo betweentwomidnights/yuey.cpp --tag v0.2.1 [--sums path/to/SHA256SUMS]
//   node control-center/src-tauri/scripts/pin_native_release.mjs \
//     --runtime cudart-12.8 --repo betweentwomidnights/gary-localhost-installer --tag runtime-cudart-12.8.1
//
// A service pin touches only URLs under the service's currently pinned
// release, so a shared runtime hosted elsewhere is left alone. A runtime pin
// rewrites that one nativeRuntimes entry to the single zip its release carries.
// The file is edited in place rather than re-serialised, so the diff is only
// the lines that changed.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.resolve(__dirname, '..', '..', '..', 'services', 'manifests', 'services.json');

function fail(message) {
  console.error(`[pin-native] ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key?.startsWith('--') || argv[i + 1] === undefined) fail(`bad arguments near ${key}`);
    args[key.slice(2)] = argv[i + 1];
  }
  if (Boolean(args.service) === Boolean(args.runtime)) fail('give --service or --runtime, not both');
  for (const required of ['repo', 'tag']) {
    if (!args[required]) fail(`--${required} is required`);
  }
  return args;
}

function parseSums(text) {
  const sums = new Map();
  for (const line of text.split(/\r?\n/)) {
    const match = line.trim().match(/^([0-9a-fA-F]{64})\s+\*?(.+)$/);
    if (match) sums.set(match[2].trim(), match[1].toLowerCase());
  }
  return sums;
}

const args = parseArgs(process.argv.slice(2));
const text = fs.readFileSync(manifestPath, 'utf8');

let sumsText;
if (args.sums) {
  sumsText = fs.readFileSync(args.sums, 'utf8');
} else {
  const url = `https://github.com/${args.repo}/releases/download/${args.tag}/SHA256SUMS`;
  const response = await fetch(url);
  if (!response.ok) fail(`could not fetch ${url}: HTTP ${response.status}`);
  sumsText = await response.text();
}
const sums = parseSums(sumsText);
if (sums.size === 0) fail('SHA256SUMS has no entries');

if (args.runtime) {
  // A runtime release carries exactly one zip, named for the exact toolkit it
  // came from; the manifest names it by the compatibility class.
  if (!JSON.parse(text).nativeRuntimes?.[args.runtime]) fail(`${args.runtime} is not in nativeRuntimes`);
  if (sums.size !== 1) fail(`a runtime release should carry one zip; ${args.tag} has ${sums.size}`);
  const [[file, sha]] = [...sums.entries()];
  const name = args.runtime.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const entry = new RegExp(`("${name}":\\s*\\{\\s*"url":\\s*")[^"]*(",\\s*"sha256":\\s*")[^"]*(")`);
  if (!entry.test(text)) fail(`could not find ${args.runtime}'s url and sha256`);
  const url = `https://github.com/${args.repo}/releases/download/${args.tag}/${file}`;
  const updated = text.replace(entry, `$1${url}$2${sha}$3`);
  JSON.parse(updated);
  fs.writeFileSync(manifestPath, updated);
  console.log(`[pin-native] ${args.runtime} -> ${url}  ${sha}`);
  process.exit(0);
}

const service = JSON.parse(text).services.find((entry) => entry.id === args.service);
if (!service?.native) fail(`${args.service} is not a native service in ${manifestPath}`);
const oldTag = service.native.version;

const releasePrefix = `https://github.com/${args.repo}/releases/download/`;
let pinned = 0;
let updated = text.replace(
  /("url":\s*")([^"]+)(",\s*"sha256":\s*")([^"]*)(")/g,
  (whole, open, url, middle, _oldSha, close) => {
    if (!url.startsWith(releasePrefix)) return whole;
    const [urlTag, ...rest] = url.slice(releasePrefix.length).split('/');
    if (urlTag !== oldTag) return whole;
    const oldFile = rest.join('/');
    const file = oldFile.split(oldTag).join(args.tag);
    const sha = sums.get(file);
    if (!sha) {
      if (oldFile.includes(oldTag)) fail(`the ${args.tag} SHA256SUMS has no ${file}`);
      console.log(`[pin-native] ${file} is not in ${args.tag}; keeping its ${oldTag} pin`);
      return whole;
    }
    pinned += 1;
    console.log(`[pin-native] ${file}  ${sha}`);
    return `${open}${releasePrefix}${args.tag}/${file}${middle}${sha}${close}`;
  },
);
if (pinned === 0) fail(`nothing under ${releasePrefix}${oldTag} was found to pin`);

// The service's own version, which is what marks an installed runtime outdated.
const serviceStart = updated.indexOf(`"id": "${args.service}"`);
const versionField = `"version": "${oldTag}"`;
const versionAt = updated.indexOf(versionField, serviceStart);
if (serviceStart < 0 || versionAt < 0) fail(`could not find ${args.service}'s version field`);
updated =
  updated.slice(0, versionAt) +
  `"version": "${args.tag}"` +
  updated.slice(versionAt + versionField.length);

JSON.parse(updated);
fs.writeFileSync(manifestPath, updated);
console.log(`[pin-native] ${args.service} ${oldTag} -> ${args.tag}: ${pinned} packages pinned`);
