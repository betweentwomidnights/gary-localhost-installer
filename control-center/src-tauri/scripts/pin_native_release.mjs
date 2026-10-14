// Pin a native service, shared bundle, or shared runtime to a published release: rewrite
// its URLs and SHA-256s in services/manifests/services.json from the release's
// SHA256SUMS. See docs/native-runtime-packages.md.
//
// Usage:
//   node control-center/src-tauri/scripts/pin_native_release.mjs \
//     --service yuey --repo betweentwomidnights/yuey.cpp --tag v0.2.1 [--sums path/to/SHA256SUMS]
//   node control-center/src-tauri/scripts/pin_native_release.mjs \
//     --runtime cudart-12.8 --repo betweentwomidnights/gary-localhost-installer --tag runtime-cudart-12.8.1
//   node control-center/src-tauri/scripts/pin_native_release.mjs \
//     --bundle sa3 --repo betweentwomidnights/sa3.cpp --tag v0.1.1
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
  if (['service', 'runtime', 'bundle'].filter((key) => args[key]).length !== 1) fail('give exactly one of --service, --bundle or --runtime');
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
const manifestPath = args.manifest ?? path.resolve(__dirname, '..', '..', '..', 'services', 'manifests', 'services.json');
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

const manifest = JSON.parse(text);
const target = args.bundle
  ? manifest.nativeBundles?.[args.bundle]
  : manifest.services.find((entry) => entry.id === args.service)?.native;
if (!target) fail(`${args.bundle ?? args.service} has no native package definition in ${manifestPath}`);
if (target.bundle) fail(`pin the shared package with --bundle ${target.bundle}`);
const oldTag = target.version;

// Work within the selected definition. Another service may pin a different
// version of the same repository; it must not change as a side effect.
function objectRange(source, at) {
  const start = source.indexOf('{', at);
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let i = start; i < source.length; i++) {
    const character = source[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') quoted = false;
    } else if (character === '"') quoted = true;
    else if (character === '{') depth++;
    else if (character === '}' && --depth === 0) return [start, i + 1];
  }
  fail('could not locate the selected package definition');
}

const escapedName = (name) => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function namedObject(source, name, from = 0) {
  const match = new RegExp(`"${escapedName(name)}"\\s*:\\s*\\{`, 'g');
  match.lastIndex = from;
  const found = match.exec(source);
  if (!found) fail(`could not find ${name}'s package definition`);
  return objectRange(source, found.index);
}
let range;
if (args.bundle) {
  const [start, end] = namedObject(text, 'nativeBundles');
  const [bundleStart, bundleEnd] = namedObject(text.slice(start, end), args.bundle);
  range = [start + bundleStart, start + bundleEnd];
} else {
  const id = new RegExp(`"id"\\s*:\\s*"${escapedName(args.service)}"`).exec(text);
  if (!id) fail(`could not find service ${args.service}`);
  range = namedObject(text, 'native', id.index);
}
const definition = text.slice(...range);

const releasePrefix = `https://github.com/${args.repo}/releases/download/`;
let pinned = 0;
let updated = definition.replace(
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
const versionField = new RegExp(`("version"\\s*:\\s*)"${escapedName(oldTag)}"`);
if (!versionField.test(updated)) fail(`could not find ${args.bundle ?? args.service}'s version field`);
updated = updated.replace(versionField, (_whole, prefix) => `${prefix}${JSON.stringify(args.tag)}`);
updated = text.slice(0, range[0]) + updated + text.slice(range[1]);

JSON.parse(updated);
fs.writeFileSync(manifestPath, updated);
console.log(`[pin-native] ${args.bundle ?? args.service} ${oldTag} -> ${args.tag}: ${pinned} packages pinned`);
