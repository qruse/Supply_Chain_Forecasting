#!/usr/bin/env bash
set -Eeuo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${COMMON_SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/test_logs"
ARTIFACT_DIR="${ROOT_DIR}/artifacts"
TMP_DIR="${ROOT_DIR}/tmps"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-12}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-32}"
BASELINE_HIDDEN_SIZE="${BASELINE_HIDDEN_SIZE:-48}"
BASELINE_NUM_LAYERS="${BASELINE_NUM_LAYERS:-1}"
BASELINE_DROPOUT="${BASELINE_DROPOUT:-0.10}"
BASELINE_PATIENCE="${BASELINE_PATIENCE:-3}"
BASELINE_LR="${BASELINE_LR:-0.001}"
TSFM_BATCH_SIZE="${TSFM_BATCH_SIZE:-16}"
TSFM_STEPS="${TSFM_STEPS:-100}"
TSFM_SPLIT="${TSFM_SPLIT:-both}"
TSFM_MODEL="${TSFM_MODEL:-all}"
TSFM_DATASET="${TSFM_DATASET:-all}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}" "${TMP_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

run_logged() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}_$(timestamp).log"
  echo "[run] ${name}"
  echo "[log] ${log_path}"
  "$@" > "${log_path}" 2>&1
  tail -40 "${log_path}"
}

run_logged_shell() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}_$(timestamp).log"
  echo "[run] ${name}"
  echo "[log] ${log_path}"
  bash -lc "$*" > "${log_path}" 2>&1
  tail -40 "${log_path}"
}

all_exist() {
  local path
  for path in "$@"; do
    [[ -s "${path}" ]] || return 1
  done
  return 0
}

skip_or_run() {
  local name="$1"
  shift
  local outputs=()
  while [[ "$#" -gt 0 && "$1" != "--" ]]; do
    outputs+=("$1")
    shift
  done
  shift
  if [[ "${FORCE_REBUILD}" != "1" ]] && all_exist "${outputs[@]}"; then
    echo "[skip] ${name}: cached outputs exist"
    return 0
  fi
  run_logged "${name}" "$@"
}

skip_or_run_shell() {
  local name="$1"
  shift
  local outputs=()
  while [[ "$#" -gt 0 && "$1" != "--" ]]; do
    outputs+=("$1")
    shift
  done
  shift
  if [[ "${FORCE_REBUILD}" != "1" ]] && all_exist "${outputs[@]}"; then
    echo "[skip] ${name}: cached outputs exist"
    return 0
  fi
  run_logged_shell "${name}" "$@"
}
