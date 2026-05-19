#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${ALLOW_EXPERIMENTAL_E4B_SPACE:-0}" != "1" ]]; then
  cat <<'EOF' >&2
Refusing to deploy the private E4B experiment by default.

Hackathon submission should use only:
  SPACE_ID=thomasjvu/lisper-zerogpu

The E4B Space is not submission-ready: it has no merged full checkpoint and no
passing held-out eval/publish verdict. If you intentionally want to update that
private experiment, rerun with:
  ALLOW_EXPERIMENTAL_E4B_SPACE=1 scripts/deploy_lisper_zerogpu_e4b_space.sh
EOF
  exit 1
fi

export SPACE_DIR="${SPACE_DIR:-$ROOT_DIR/spaces/lisper-zerogpu}"
export SPACE_ID="${SPACE_ID:-thomasjvu/lisper-zerogpu-e4b}"
export SPACE_PRIVATE="${SPACE_PRIVATE:-1}"

"$ROOT_DIR/scripts/deploy_lisper_zerogpu_space.sh"

cat <<'EOF'

E4B Space source uploaded. After the E4B full model repo is populated, set:

LISPER_ZERO_GPU_MODEL_ID=google/gemma-4-E4B-it
LISPER_ZERO_GPU_ADAPTER_ID=thomasjvu/lisper-gemma4-e4b-audio-lora
LISPER_ZERO_GPU_DTYPE=float16
LISPER_ZERO_GPU_LOAD_IN_4BIT=1
LISPER_ZERO_GPU_MAX_SEQ_LENGTH=2048
LISPER_ZERO_GPU_SIZE=large
LISPER_ZERO_GPU_MAX_NEW_TOKENS=96

Keep LISPER_ZERO_GPU_EAGER_LOAD=0 until the Space variables and private/gated model access are configured, then switch it back to 1 for smoke testing.
EOF
