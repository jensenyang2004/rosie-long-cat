#!/usr/bin/env bash
# Run a script sharded over 2 GPUs (data parallel), as in mice.
#
#   ./run_2gpu.sh infer_baseline.py --exp_name turbo_mice [args...]
#   GPU0=2 GPU1=3 ./run_2gpu.sh infer_baseline.py ...
set -euo pipefail

SCRIPT="$1"; shift
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${DIR}/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU0}" python "${DIR}/${SCRIPT}" --num_shards 2 --shard_id 0 "$@" > "${LOG_DIR}/shard0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="${GPU1}" python "${DIR}/${SCRIPT}" --num_shards 2 --shard_id 1 "$@" > "${LOG_DIR}/shard1.log" 2>&1 &
PID1=$!
echo "shard 0 (GPU ${GPU0}) PID ${PID0} -> ${LOG_DIR}/shard0.log"
echo "shard 1 (GPU ${GPU1}) PID ${PID1} -> ${LOG_DIR}/shard1.log"
wait "${PID0}" "${PID1}"
echo "Both shards finished."
