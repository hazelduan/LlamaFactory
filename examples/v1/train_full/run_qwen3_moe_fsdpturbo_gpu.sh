#!/usr/bin/env bash
set -euo pipefail

# Single node:
#   EP_DISPATCHER=eager NPROC_PER_NODE=8 bash examples/v1/train_full/run_qwen3_moe_fsdpturbo_gpu.sh
# Multi-node: run on every node with the same MASTER_ADDR/MASTER_PORT and NNODES,
# changing only NODE_RANK. EP_SIZE defaults to the total world size.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/v1/train_full/train_full_qwen3_moe_fsdpturbo_gpu.yaml}"
MODEL="${MODEL:-${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}}"
TRAIN_DATASET="${TRAIN_DATASET:-${ROOT_DIR}/data/v1_sft_demo.yaml}"
EP_DISPATCHER="${EP_DISPATCHER:-eager}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-10}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
ENABLE_ACTIVATION_CHECKPOINTING="${ENABLE_ACTIVATION_CHECKPOINTING:-true}"

if [[ "${EP_DISPATCHER}" != "eager" && "${EP_DISPATCHER}" != "fused" ]]; then
  echo "EP_DISPATCHER must be eager or fused, got: ${EP_DISPATCHER}" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file does not exist: ${CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${TRAIN_DATASET}" ]]; then
  echo "Dataset config does not exist: ${TRAIN_DATASET}" >&2
  exit 2
fi

export USE_V1=1
export FORCE_TORCHRUN=1
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(python - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
fi
export NPROC_PER_NODE

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
EP_SIZE="${EP_SIZE:-${WORLD_SIZE}}"

if ((NPROC_PER_NODE < 1)); then
  echo "No visible CUDA device was found." >&2
  exit 2
fi

if ((EP_SIZE < 1 || WORLD_SIZE % EP_SIZE != 0)); then
  echo "EP_SIZE must divide total world size: ${WORLD_SIZE} % ${EP_SIZE} != 0." >&2
  exit 2
fi

python - "${EP_DISPATCHER}" <<'PY'
import sys

import torch
import torch.nn.functional as F
import transformers
import fsdp_turbo

dispatcher = sys.argv[1]
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the active Python environment.")
if dispatcher == "fused" and not (hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")):
    raise RuntimeError("GPU fused EP requires a PyTorch build that provides a grouped_mm implementation.")

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"FSDPTurbo: {fsdp_turbo.__file__}")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
PY

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/Qwen3.5-35B-A3B/full/fsdpturbo_gpu_${EP_DISPATCHER}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/fsdpturbo_gpu}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen35_a3b_${EP_DISPATCHER}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

overrides=(
  "model=${MODEL}"
  "train_dataset=${TRAIN_DATASET}"
  "output_dir=${OUTPUT_DIR}"
  "max_steps=${MAX_STEPS}"
  "cutoff_len=${CUTOFF_LEN}"
  "enable_activation_checkpointing=${ENABLE_ACTIVATION_CHECKPOINTING}"
  "dist_config.ep_size=${EP_SIZE}"
  "dist_config.ep_dispatcher=${EP_DISPATCHER}"
)

if [[ -n "${GLOBAL_BATCH_SIZE:-}" ]]; then
  overrides+=("global_batch_size=${GLOBAL_BATCH_SIZE}")
fi

cd "${ROOT_DIR}"
echo "Launching Qwen3.5-35B-A3B with dispatcher=${EP_DISPATCHER}, world_size=${WORLD_SIZE}, ep_size=${EP_SIZE}."
echo "Log: ${LOG_FILE}"

llamafactory-cli sft "${CONFIG}" "${overrides[@]}" 2>&1 | tee "${LOG_FILE}"
