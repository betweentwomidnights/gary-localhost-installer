#!/bin/sh
# Install a small launcher; keep the extracted preview binaries in place.
set -eu
bin_dir="$HOME/.local/bin"
config=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bin-dir|--config)
      [ "$#" -ge 2 ] || { echo "$1 needs an absolute path" >&2; exit 2; }
      case "$2" in /*) ;; *) echo "$1 needs an absolute path" >&2; exit 2 ;; esac
      if [ "$1" = --bin-dir ]; then bin_dir=$2; else config=$2; fi
      shift 2 ;;
    --help) echo 'Usage: sh install-gary.sh [--bin-dir DIR] [--config FILE]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
package_dir=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
binary="$package_dir/gary"
[ -x "$binary" ] || { echo "Missing executable: $binary" >&2; exit 1; }
mkdir -p "$bin_dir"
launcher="$bin_dir/gary"
temporary="$bin_dir/.gary-install-$$"
umask 077
# Shell-quote paths as data; never evaluate a user-supplied path.
quote() { printf "'"; printf '%s' "$1" | sed "s/'/'\\\\''/g"; printf "'"; }
trap 'rm -f "$temporary"' EXIT HUP INT TERM
(
  set -C
  {
    printf '#!/bin/sh\n'
    if [ -n "$config" ]; then
      printf 'export GARY4LOCAL_HEADLESS_CONFIG='; quote "$config"; printf '\n'
    fi
    printf 'exec '; quote "$binary"; printf ' "$@"\n'
  } > "$temporary"
)
chmod 755 "$temporary"
if [ -e "$launcher" ] || [ -L "$launcher" ]; then
  if [ ! -L "$launcher" ] && cmp -s "$temporary" "$launcher"; then
    echo "Gary command already installed: $launcher"
  else
    echo "Refusing to replace existing $launcher. Choose another --bin-dir or inspect it first." >&2
    exit 1
  fi
else
  # Hard-link creation is atomic and refuses a racing existing destination.
  ln "$temporary" "$launcher"
  echo "Installed Gary command: $launcher"
fi
echo "Keep this extracted preview in place: $package_dir"
case ":$PATH:" in
  *":$bin_dir:"*) echo 'Run: gary' ;;
  *) printf 'For this shell, run: export PATH='; quote "$bin_dir"; printf ':"$PATH"\nThen run: gary\n' ;;
esac
echo 'No shell profile, service or model files were changed.'
