#!/usr/bin/env bash
# Overnight orchestrator: get a good acrobatic policy by morning.
#
# Cartwheel is already a good policy at iter 3000 (mean_reward ~22, ee_term ~1.3,
# body_pos_err ~9 cm). All night budget goes into double_kong, which plateaued
# under default tracking termination thresholds because of the inversion phase.
#
# Stages:
#   S1: DoubleKong-relaxed env, FRESH, full motion, 7000 iters  (~3.3 h)
#   S2: DoubleKong-relaxed env, FRESH, TRIMMED motion (jump-only), 5000 iters  (~1.4 h)
#
# Each stage writes its own log under logs/orchestrator/. The final run dirs
# land in logs/rsl_rl/g1_double_kong/. Stages run sequentially.

set -uo pipefail

PROJECT_ROOT="/home/ev/repos/otus_rl_project"
LOG_DIR="${PROJECT_ROOT}/logs/orchestrator"
mkdir -p "${LOG_DIR}"

ORCH_LOG="${LOG_DIR}/night_acro_orchestrator.log"
ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] [orch] $*" | tee -a "${ORCH_LOG}"; }

log "==== night orchestrator START ===="
log "host pid=$$  user=$(id -un)"
log "project root=${PROJECT_ROOT}"

cd "${PROJECT_ROOT}" || { log "FATAL cd"; exit 1; }

run_stage() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/night_${name}.log"
  log "---- stage ${name} BEGIN -> ${logfile} ----"
  log "cmd: $*"
  : > "${logfile}"
  "$@" >> "${logfile}" 2>&1
  local rc=$?
  log "---- stage ${name} END rc=${rc} ----"
  return ${rc}
}

# ----------------------------- Stage 1: full kong, relaxed env --------------
log ">> Stage 1: fresh Mjlab-DoubleKong-Unitree-G1, FULL motion, 7000 iters"
run_stage "s1_kong_relaxed_full" \
  make exec CMD="otus-train Mjlab-DoubleKong-Unitree-G1 \
    --motion-file=/workspace/otus_rl_project/src/assets/motions/g1/double_kong.npz \
    --env.scene.num-envs=4096 \
    --agent.max-iterations=7000 \
    --agent.run-name=kong-relaxed-7k \
    --agent.logger=tensorboard \
    --agent.upload-model=False"

log ">> Stage 1 finished. Checkpoints in g1_double_kong/*kong-relaxed-7k*"

# ----------------------------- Stage 2: trimmed kong ------------------------
TRIMMED_NPZ="/workspace/otus_rl_project/src/assets/motions/g1/double_kong_trim.npz"
log ">> Build trimmed motion (frames 120..260) -> ${TRIMMED_NPZ}"
make exec CMD="python /workspace/otus_rl_project/scripts/trim_motion.py \
  --input  /workspace/otus_rl_project/src/assets/motions/g1/double_kong.npz \
  --output ${TRIMMED_NPZ} \
  --start-frame 120 --end-frame 260" >> "${ORCH_LOG}" 2>&1 || log "WARN: trim failed"

log ">> Stage 2: fresh Mjlab-DoubleKong-Unitree-G1, TRIMMED motion, 5000 iters"
run_stage "s2_kong_trim_5k" \
  make exec CMD="otus-train Mjlab-DoubleKong-Unitree-G1 \
    --motion-file=${TRIMMED_NPZ} \
    --env.scene.num-envs=4096 \
    --agent.max-iterations=5000 \
    --agent.run-name=kong-trim-5k \
    --agent.logger=tensorboard \
    --agent.upload-model=False"

log ">> Stage 2 finished."
log "==== night orchestrator DONE ===="
