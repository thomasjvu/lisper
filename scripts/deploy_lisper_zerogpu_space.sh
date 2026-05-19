#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_DIR="${SPACE_DIR:-$ROOT_DIR/spaces/lisper-zerogpu}"
SPACE_ID="${SPACE_ID:-thomasjvu/lisper-zerogpu}"
SPACE_PRIVATE="${SPACE_PRIVATE:-0}"

exclude_generated=(--exclude "__pycache__/**" --exclude "*.pyc" --delete "__pycache__/**" --delete "*.pyc")

if command -v hf >/dev/null 2>&1; then
  hf auth whoami
  if [[ "$SPACE_PRIVATE" == "1" ]]; then
    hf upload "$SPACE_ID" "$SPACE_DIR" --type space --private "${exclude_generated[@]}" --commit-message "Deploy Lisper ZeroGPU Space"
  else
    hf upload "$SPACE_ID" "$SPACE_DIR" --type space "${exclude_generated[@]}" --commit-message "Deploy Lisper ZeroGPU Space"
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli whoami
  if [[ "$SPACE_PRIVATE" == "1" ]]; then
    huggingface-cli upload "$SPACE_ID" "$SPACE_DIR" --repo-type space --private "${exclude_generated[@]}" --commit-message "Deploy Lisper ZeroGPU Space"
  else
    huggingface-cli upload "$SPACE_ID" "$SPACE_DIR" --repo-type space "${exclude_generated[@]}" --commit-message "Deploy Lisper ZeroGPU Space"
  fi
else
  echo "Install the Hugging Face CLI first: pip install -U huggingface_hub[cli]" >&2
  exit 1
fi
