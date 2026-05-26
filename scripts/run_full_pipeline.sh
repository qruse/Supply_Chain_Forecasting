#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

RUN_DATA="${RUN_DATA:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_TSFMS="${RUN_TSFMS:-1}"

echo "root=${ROOT_DIR}"
echo "python=${PYTHON_BIN}"
echo "device=${DEVICE}"
echo "run_data=${RUN_DATA} run_baselines=${RUN_BASELINES} run_tsfms=${RUN_TSFMS}"
echo "tsfm_dataset=${TSFM_DATASET} tsfm_model=${TSFM_MODEL} tsfm_split=${TSFM_SPLIT} tsfm_steps=${TSFM_STEPS}"
echo "force_rebuild=${FORCE_REBUILD}"

if [[ "${RUN_DATA}" == "1" ]]; then
  "${SCRIPT_DIR}/steps/01_prepare_demand_data.sh"
  "${SCRIPT_DIR}/steps/02_prepare_inventory_data.sh"
  "${SCRIPT_DIR}/steps/03_prepare_leadtime_data.sh"
  "${SCRIPT_DIR}/steps/04_validate_data_quality.sh"
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  "${SCRIPT_DIR}/steps/10_train_baselines.sh"
  "${SCRIPT_DIR}/steps/11_test_baselines.sh"
fi

if [[ "${RUN_TSFMS}" == "1" ]]; then
  "${SCRIPT_DIR}/steps/20_run_tsfm.sh"
fi

echo "full pipeline completed"
