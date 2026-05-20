#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

HF_SOURCE_PATH=""
MEGATRON_PATH=""
RAW_EXPORT_PATH=""
HF_EXPORT_PATH=""
PROMPT="The capital of France is"
SKIP_SETUP=0
TRUST_REMOTE_CODE=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") --hf-source-path PATH --megatron-path PATH --raw-export-path PATH --hf-export-path PATH [options]

Options:
  --hf-source-path PATH     Original HuggingFace Gemma-3 checkpoint
  --megatron-path PATH      Megatron checkpoint directory
  --raw-export-path PATH    Temporary raw HF text export directory
  --hf-export-path PATH     Final merged HF export directory
  --prompt TEXT             Verification prompt
  --skip-setup              Skip environment sync step
  --trust-remote-code       Enable trust_remote_code in Python scripts
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-source-path) HF_SOURCE_PATH="$2"; shift 2 ;;
    --megatron-path) MEGATRON_PATH="$2"; shift 2 ;;
    --raw-export-path) RAW_EXPORT_PATH="$2"; shift 2 ;;
    --hf-export-path) HF_EXPORT_PATH="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$HF_SOURCE_PATH" || -z "$MEGATRON_PATH" || -z "$RAW_EXPORT_PATH" || -z "$HF_EXPORT_PATH" ]]; then
  usage
  exit 1
fi

if [[ $SKIP_SETUP -eq 0 ]]; then
  bash "${SCRIPT_DIR}/setup_bridge_env.sh"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

find_free_port() {
  "$PYTHON_BIN" - <<'PY'
import socket
import sys

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sys.stdout.write(f"{sock.getsockname()[1]}\n")
sock.close()
PY
}

run_with_bridge_env() {
  local port
  port=$(find_free_port)
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="$port"
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0
  export PYTHONPATH="${REPO_ROOT}/src"
  "$@"
}

TRUST_ARGS=()
if [[ $TRUST_REMOTE_CODE -eq 1 ]]; then
  TRUST_ARGS+=(--trust-remote-code)
fi

cd "${REPO_ROOT}"
run_with_bridge_env "$PYTHON_BIN" "${SCRIPT_DIR}/convert_gemma3.py" \
  --hf-model-path "$HF_SOURCE_PATH" \
  --megatron-path "$MEGATRON_PATH" \
  "${TRUST_ARGS[@]}"

run_with_bridge_env "$PYTHON_BIN" "${SCRIPT_DIR}/export_gemma3.py" \
  --hf-source-path "$HF_SOURCE_PATH" \
  --megatron-path "$MEGATRON_PATH" \
  --raw-export-path "$RAW_EXPORT_PATH" \
  --hf-export-path "$HF_EXPORT_PATH" \
  "${TRUST_ARGS[@]}"

"$PYTHON_BIN" "${SCRIPT_DIR}/verify_roundtrip.py" \
  --original-path "$HF_SOURCE_PATH" \
  --roundtrip-path "$HF_EXPORT_PATH" \
  --prompt "$PROMPT" \
  "${TRUST_ARGS[@]}"
