#!/usr/bin/env bash
# End to end (the official edit, one instruction), binding only, and binding + steering, in that order, on one or
# both benches. Steering defaults are the tuned values (2026-11-27): ctc 0.5, src 0.6, bg 0.35, ins 0.025 with the
# centred instruction anchor (minus the mean update over every prompt slot, padding included). ONLY=single runs the
# same steering without binding, on the official one-instruction edit (infer_single.py).
#
#   ./run_all.sh mice                            all three on MICE-Bench, one GPU
#   ./run_all.sh both                            MICE-Bench, then LoMOE-Bench
#   ./run_all.sh mice --sample_ids 06 09 276     smoke test: extra args go to all three scripts (run-control flags
#                                                only, e.g. --sample_ids, --overwrite)
#   GPU=1 ./run_all.sh lomoe                     on GPU 1 (CUDA_VISIBLE_DEVICES; default: as already set)
#   ONLY="steer" INS=0.05 ./run_all.sh mice      only the steered run, another instruction strength
#   ONLY=single ./run_all.sh mice                steering without binding (one instruction, masks for steering only)
#   DRY=1 ./run_all.sh both                      print the commands without running them
#
# Environment (defaults):
#   GPU (unset = the visible GPU)   OFFLOAD 1 (--cpu_offload; 0 = off)
#   ONLY "baseline bind steer" (or any of them and single)   CTC 0.5   SRC 0.6   BG 0.35   INS 0.025   INS_CENTRE prompts (raw = the raw anchor)
#   TAG (appended to every exp name)
#
# Results: results/longcat_turbo_<bench>, results/longcatbind_bind_<bench>,
#          results/longcatrot_rot_<bench>_c<CTC>_s<SRC>_b<BG>_i<INS>_<INS_CENTRE>,
#          results/longcatsingle_single_<bench>_c<CTC>_s<SRC>_b<BG>_i<INS>_<INS_CENTRE>; logs in logs/<exp name>.log.
# A finished sample is skipped, so rerunning the same command resumes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BENCH="${1:?usage: ./run_all.sh mice|lomoe|both [extra args for all three scripts]}"; shift
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="${GPU}"
ONLY="${ONLY:-baseline bind steer}"
CTC="${CTC:-0.5}"; SRC="${SRC:-0.6}"; BG="${BG:-0.35}"; INS="${INS:-0.025}"
INS_CENTRE="${INS_CENTRE:-prompts}"
TAG="${TAG:+_${TAG}}"
COMMON=()
[ "${OFFLOAD:-1}" = 1 ] && COMMON+=(--cpu_offload)

case "${BENCH}" in
  mice|lomoe) BENCHES=("${BENCH}") ;;
  both) BENCHES=(mice lomoe) ;;
  *) echo "bench must be mice, lomoe or both, not ${BENCH}" >&2; exit 2 ;;
esac
case "${INS_CENTRE}" in
  raw) CENTRE_ARG=() ;;
  prompts|text) CENTRE_ARG=(--ins_centre "${INS_CENTRE}") ;;
  *) echo "INS_CENTRE must be prompts, text or raw" >&2; exit 2 ;;
esac
mkdir -p logs

run() {  # run <log name> <script> <args...>
  local log="$1" script="$2"; shift 2
  echo "== $(date '+%F %T') ${script} $*"
  [ "${DRY:-0}" = 1 ] && return 0
  python "${script}" "$@" 2>&1 | tee "logs/${log}.log"
}

count() {  # count <run folder>
  local f=("$1"/*_result.png)
  [ -e "${f[0]}" ] || f=()
  echo "   ${#f[@]} results in $1"
}

for b in "${BENCHES[@]}"; do
  root="../3D-bench/image/${b}_bench"
  base="turbo_${b}${TAG}"
  bind="bind_${b}${TAG}"
  rot="rot_${b}_c${CTC}_s${SRC}_b${BG}_i${INS}_${INS_CENTRE}${TAG}"
  single="single_${b}_c${CTC}_s${SRC}_b${BG}_i${INS}_${INS_CENTRE}${TAG}"
  if [[ " ${ONLY} " == *" baseline "* ]]; then
    run "${base}" infer_baseline.py --dataset_root "${root}" --exp_name "${base}" ${COMMON[@]+"${COMMON[@]}"} "$@"
    count "results/longcat_${base}"
  fi
  if [[ " ${ONLY} " == *" bind "* ]]; then
    run "${bind}" infer_bind.py --dataset_root "${root}" --exp_name "${bind}" ${COMMON[@]+"${COMMON[@]}"} "$@"
    count "results/longcatbind_${bind}"
  fi
  if [[ " ${ONLY} " == *" steer "* ]]; then
    run "${rot}" infer_steer.py --dataset_root "${root}" --exp_name "${rot}" \
      --rot_ctc "${CTC}" --rot_src "${SRC}" --rot_bg "${BG}" --rot_ins "${INS}" ${CENTRE_ARG[@]+"${CENTRE_ARG[@]}"} \
      ${COMMON[@]+"${COMMON[@]}"} "$@"
    count "results/longcatrot_${rot}"
  fi
  if [[ " ${ONLY} " == *" single "* ]]; then
    run "${single}" infer_single.py --dataset_root "${root}" --exp_name "${single}" \
      --rot_ctc "${CTC}" --rot_src "${SRC}" --rot_bg "${BG}" --rot_ins "${INS}" ${CENTRE_ARG[@]+"${CENTRE_ARG[@]}"} \
      ${COMMON[@]+"${COMMON[@]}"} "$@"
    count "results/longcatsingle_${single}"
  fi
done
echo "== $(date '+%F %T') done"
