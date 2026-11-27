#!/usr/bin/env bash
# Run a script sharded over 2 GPUs (data parallel), as in mice.
#
#   ./run_2gpu.sh infer_baseline.py --exp_name turbo_mice [args...]
#   GPU0=2 GPU1=3 ./run_2gpu.sh infer_baseline.py ...
#   LOG_TAG=bind_mice_ ./run_2gpu.sh infer_bind.py ...     (logs/bind_mice_shard{0,1}.log; default logs/shard{0,1}.log)
#
# Exits non-zero if either shard fails.
set -euo pipefail

SCRIPT="$1"; shift
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_TAG="${LOG_TAG:-}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG0="${LOG_DIR}/${LOG_TAG}shard0.log"
LOG1="${LOG_DIR}/${LOG_TAG}shard1.log"

CUDA_VISIBLE_DEVICES="${GPU0}" python "${DIR}/${SCRIPT}" --num_shards 2 --shard_id 0 "$@" > "${LOG0}" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="${GPU1}" python "${DIR}/${SCRIPT}" --num_shards 2 --shard_id 1 "$@" > "${LOG1}" 2>&1 &
PID1=$!
echo "shard 0 (GPU ${GPU0}) PID ${PID0} -> ${LOG0}"
echo "shard 1 (GPU ${GPU1}) PID ${PID1} -> ${LOG1}"
failed=0
wait "${PID0}" || { echo "shard 0 failed: tail ${LOG0}"; failed=1; }
wait "${PID1}" || { echo "shard 1 failed: tail ${LOG1}"; failed=1; }
[ "${failed}" = 0 ] || exit 1
echo "Both shards finished."
