#!/bin/sh
set -eu
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  printf '%s\n' 'Usage: sh INSTALL.sh /path/to/gary4local-preview_VERSION_ARCH.deb' >&2
  exit 1
fi
if [ "$(dpkg-deb -f "$1" Package)" != gary4local-preview ]; then
  printf '%s\n' 'This installer only accepts the gary4local-preview package.' >&2
  exit 1
fi
# APT's _apt user cannot traverse private home directories. Stage only this
# public package in a readable temporary directory, without changing HOME modes.
staging=$(mktemp -d /var/tmp/gary4local-preview-install.XXXXXX)
trap 'rm -f "$staging/package.deb"; rmdir "$staging"' EXIT
chmod 755 "$staging"
install -m 644 -- "$1" "$staging/package.deb"
sudo apt install --reinstall "$staging/package.deb"
