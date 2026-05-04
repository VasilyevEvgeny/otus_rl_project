#!/usr/bin/env bash
# Acrobatics orchestrator: train two MimicKit-G1 motions sequentially.
#   1. cartwheel    (g1_cartwheel.npz)        — sideways inversion (roll 180°)
#   2. double_kong  (g1_double_kong.npz)     — running dive vault + recovery
# Both: 4000 iter, 4096 envs, ETA ~2h each. Total ~4h.
set -uo pipefail

ROOT="/home/ev/repos/otus_rl_project"
LOG_DIR="$ROOT/logs/orchestrator"
mkdir -p "$LOG_DIR"
cd "$ROOT"

ts()  { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/acrobatics_orchestrator.log"; }

log "acrobatics-orchestrator started"

run_train() {
    local NPZ_NAME="$1"
    local RUN_NAME="$2"
    local LOG_FILE="$LOG_DIR/train_${RUN_NAME}.log"
    local NPZ="/workspace/otus_rl_project/src/assets/motions/g1/${NPZ_NAME}.npz"
    log "starting training: motion=$NPZ_NAME run=$RUN_NAME"
    log "  log file: $LOG_FILE"
    make track-train NPZ="$NPZ" ITERS=4000 NUM_ENVS=4096 RUN="$RUN_NAME" \
        > "$LOG_FILE" 2>&1
    local RC=$?
    log "  exit code: $RC"
    local RUN_DIR
    RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*"${RUN_NAME}"* 2>/dev/null | head -1)
    log "  run dir: $RUN_DIR"
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/policy.onnx" ]; then
        log "  onnx OK ($(stat -c %s "$RUN_DIR/policy.onnx") bytes)"
    fi
    return $RC
}

run_train cartwheel   acro-cartwheel   || log "acro-cartwheel train FAILED"
run_train double_kong acro-double-kong || log "acro-double-kong train FAILED"

log "acrobatics-orchestrator finished"
