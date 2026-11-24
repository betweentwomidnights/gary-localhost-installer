#!/bin/sh
# Install this extracted archive as a managed Gary version, plus a small
# launcher that runs whichever version is current. `gary update` installs
# newer versions beside it later and switches `current`, so the extracted
# folder isn't needed once this finishes.
set -eu
bin_dir="$HOME/.local/bin"
case "${XDG_DATA_HOME:-}" in
  /*) app_dir="$XDG_DATA_HOME/gary4local/headless-app" ;;
  *) app_dir="$HOME/.local/share/gary4local/headless-app" ;;
esac
config=''
replace=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bin-dir|--config|--app-dir)
      [ "$#" -ge 2 ] || { echo "$1 needs an absolute path" >&2; exit 2; }
      case "$2" in /*) ;; *) echo "$1 needs an absolute path" >&2; exit 2 ;; esac
      case "$1" in
        --bin-dir) bin_dir=$2 ;;
        --config) config=$2 ;;
        *) app_dir=$2 ;;
      esac
      shift 2 ;;
    --replace) replace=1; shift ;;
    --help)
      echo 'Usage: sh install-gary.sh [--bin-dir DIR] [--config FILE] [--app-dir DIR] [--replace]'
      echo '--replace moves an existing gary command aside to gary.previous instead of refusing.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
package_dir=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
for required in gary gary4local-server BUILD-INFO.json; do
  [ -f "$package_dir/$required" ] || { echo "Missing $required in $package_dir" >&2; exit 1; }
done
[ -x "$package_dir/gary" ] || { echo "Missing executable: $package_dir/gary" >&2; exit 1; }
version=$("$package_dir/gary" --version) || { echo "$package_dir/gary doesn't run on this machine." >&2; exit 1; }
case "$version" in
  ''|*[!0-9.]*) echo "Unexpected version from $package_dir/gary: $version" >&2; exit 1 ;;
esac
versions="$app_dir/versions"
dest="$versions/$version"
launcher="$bin_dir/gary"
umask 077
mkdir -p "$bin_dir" "$versions"
temporary="$bin_dir/.gary-install-$$"
staging="$versions/.staging-$$"
trap 'rm -f "$temporary"; rm -rf "$staging"' EXIT HUP INT TERM
# Shell-quote paths as data; never evaluate a user-supplied path.
quote() { printf "'"; printf '%s' "$1" | sed "s/'/'\\\\''/g"; printf "'"; }
(
  set -C
  {
    printf '#!/bin/sh\n'
    if [ -n "$config" ]; then
      printf 'export GARY4LOCAL_HEADLESS_CONFIG='; quote "$config"; printf '\n'
    fi
    printf 'exec '; quote "$app_dir/current/gary"; printf ' "$@"\n'
  } > "$temporary"
)
chmod 755 "$temporary"

# Decide about the launcher before changing anything.
place_launcher=1
if [ -e "$launcher" ] || [ -L "$launcher" ]; then
  if [ ! -L "$launcher" ] && cmp -s "$temporary" "$launcher"; then
    place_launcher=0
  elif [ "$replace" = 1 ]; then
    if [ -e "$launcher.previous" ] || [ -L "$launcher.previous" ]; then
      echo "Refusing to replace $launcher: $launcher.previous already exists. Move it away first." >&2
      exit 1
    fi
  else
    echo "Refusing to replace existing $launcher. Inspect it, then rerun with --replace to move it aside to $launcher.previous, or choose another --bin-dir." >&2
    exit 1
  fi
fi

# Copy only the archive's own files, never whatever else shares its folder.
if [ -e "$dest" ]; then
  echo "Gary $version is already installed in $dest"
else
  mkdir "$staging"
  for name in gary gary4local-server install-gary.sh BUILD-INFO.json README.md AGENT-WORKFLOW.md \
      headless.example.json headless.local.example.json gary4local-server.service.example \
      dependencies.txt TOOLCHAIN.txt ELF-DEPENDENCIES.txt; do
    if [ -f "$package_dir/$name" ]; then cp -p "$package_dir/$name" "$staging/$name"; fi
  done
  mv "$staging" "$dest"
fi

# Point a link at a version by renaming a new link over it, as gary update does.
point() {
  link_temporary="$app_dir/.$1-$$"
  rm -f "$link_temporary"
  ln -s "$2" "$link_temporary"
  mv -T "$link_temporary" "$app_dir/$1" 2>/dev/null || { rm -f "$link_temporary"; ln -sfn "$2" "$app_dir/$1"; }
}
old=$(readlink "$app_dir/current" 2>/dev/null || true)
if [ "$old" != "versions/$version" ]; then
  if [ -n "$old" ]; then point previous "$old"; fi
  point current "versions/$version"
fi
echo "Current Gary version: $version ($app_dir)"

if [ "$place_launcher" = 0 ]; then
  echo "Gary command already installed: $launcher"
else
  if [ -e "$launcher" ] || [ -L "$launcher" ]; then
    mv "$launcher" "$launcher.previous"
    echo "Moved the old command aside: $launcher.previous"
  fi
  # Hard-link creation is atomic and refuses a racing existing destination.
  ln "$temporary" "$launcher"
  echo "Installed Gary command: $launcher"
fi
if [ "$package_dir" != "$(CDPATH='' cd -P "$dest" && pwd)" ]; then
  echo "This extracted folder is no longer needed: $package_dir"
fi
case ":$PATH:" in
  *":$bin_dir:"*) echo 'Run: gary' ;;
  *) printf 'For this shell, run: export PATH='; quote "$bin_dir"; printf ':"$PATH"\nThen run: gary\n' ;;
esac
echo 'Later versions install with: gary update'
echo 'No shell profile, service or model files were changed.'
