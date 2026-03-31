#!/bin/bash
set -euo pipefail

# Stage runtime resources for Tauri bundling.
# Copies service source code into src-tauri/resources/services/ so Tauri's
# resource bundler can include them in the installer.
#
# Usage: called by npm "build" script before `tauri build`, or manually.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAURI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TAURI_DIR}/../.." && pwd)"

RESOURCES_DIR="${TAURI_DIR}/resources"
SERVICES_DST="${RESOURCES_DIR}/services"

echo "[stage-runtime] repo root: ${REPO_ROOT}"
echo "[stage-runtime] staging into: ${SERVICES_DST}"

# Clean previous staging
rm -rf "${SERVICES_DST}"
mkdir -p "${SERVICES_DST}"

copy_service() {
  local name="$1"
  local src="${REPO_ROOT}/services/${name}"
  local dst="${SERVICES_DST}/${name}"

  if [[ ! -d "${src}" ]]; then
    echo "[stage-runtime] WARNING: missing source directory: ${src}"
    return
  fi

  echo "[stage-runtime] staging ${name}..."
  mkdir -p "${dst}"

  rsync -a --delete \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude "env/" \
    --exclude ".claude/" \
    --exclude ".cache/" \
    --exclude ".pytest_cache/" \
    --exclude "checkpoints/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.log" \
    --exclude ".DS_Store" \
    --exclude "Thumbs.db" \
    --exclude "smoke-tests/" \
    --exclude "smoke.wav" \
    --exclude "smoke.mp3" \
    "${src}/" "${dst}/"
}

# Stage each service
copy_service "gary"
copy_service "melodyflow"
copy_service "stable-audio"
copy_service "carey"
copy_service "foundation"

# Stage the manifest
mkdir -p "${SERVICES_DST}/manifests"
cp "${REPO_ROOT}/services/manifests/services.json" "${SERVICES_DST}/manifests/services.json"

# Stage shared top-level service modules used by multiple services.
if [[ -f "${REPO_ROOT}/services/local_session_store.py" ]]; then
  cp "${REPO_ROOT}/services/local_session_store.py" "${SERVICES_DST}/local_session_store.py"
fi

# Remove heavyweight non-runtime content from carey/acestep
rm -rf "${SERVICES_DST}/carey/acestep/docs" 2>/dev/null || true
rm -rf "${SERVICES_DST}/carey/acestep/examples" 2>/dev/null || true
rm -rf "${SERVICES_DST}/carey/acestep/assets" 2>/dev/null || true
rm -rf "${SERVICES_DST}/carey/acestep/.github" 2>/dev/null || true
rm -rf "${SERVICES_DST}/carey/acestep/docker-patches" 2>/dev/null || true
rm -rf "${SERVICES_DST}/carey/acestep/gradio_outputs" 2>/dev/null || true

# Copy icon.png to resources root (used for tray icon)
if [[ -f "${REPO_ROOT}/icon.png" ]]; then
  cp "${REPO_ROOT}/icon.png" "${RESOURCES_DIR}/icon.png"
fi

echo "[stage-runtime] done"
