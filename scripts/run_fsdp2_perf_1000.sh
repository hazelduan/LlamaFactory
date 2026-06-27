#!/usr/bin/env bash
set -euo pipefail
cd /home/dxq/LlamaFactory
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export USE_V1=1
export PATH=/home/dxq/envs/dxq_swift_megatron/bin:$PATH
export PYTHONPATH=/home/dxq/LlamaFactory/src:/home/dxq/MindSpeed:${PYTHONPATH:-}
export NPROC_PER_NODE=16
export FORCE_TORCHRUN=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
exec /home/dxq/envs/dxq_swift_megatron/bin/llamafactory-cli sft examples/v1/train_full/train_full_fsdp2_perf_1000.yaml
