#!/usr/bin/env bash
# Daytime orchestrator: train two more trackers sequentially
#   1. fallAndGetUp1_subject1  (rolls / recoveries)
#   2. fight1_subject2         (martial arts kicks/spins)
# Both: 4000 iter, 4096 envs. ETA ~1.5h each.
set -uo pipefail

ROOT="/home/ev/repos/otus_rl_project"
LOG_DIR="$ROOT/runs/night"
mkdir -p "$LOG_DIR"
cd "$ROOT"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/day_orchestrator.log"; }

log "day-orchestrator started"

run_train() {
    local NPZ_NAME="$1"
    local RUN_NAME="$2"
    local LOG_FILE="$LOG_DIR/train_${RUN_NAME}.log"
    local NPZ="/workspace/otus_rl_project/src/assets/motions/g1/${NPZ_NAME}.npz"
    log "starting training: motion=$NPZ_NAME run=$RUN_NAME"
    log "  log file: $LOG_FILE"
    make track-train \
        NPZ="$NPZ" \
        ITERS=4000 NUM_ENVS=4096 RUN="$RUN_NAME" \
        > "$LOG_FILE" 2>&1
    local RC=$?
    log "  exit code: $RC"
    local RUN_DIR
    RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*"${RUN_NAME}"* 2>/dev/null | head -1)
    log "  run dir: $RUN_DIR"
    if [ -n "$RUN_DIR" ]; then
        local CKPT
        CKPT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
        log "  ckpt:    $CKPT"
        if [ -f "$RUN_DIR/policy.onnx" ]; then
            log "  onnx OK ($(stat -c %s "$RUN_DIR/policy.onnx") bytes)"
        else
            log "  WARN: policy.onnx missing"
        fi
    fi
    return $RC
}

run_train fallAndGetUp1_subject1 fall-day || log "fall-day train FAILED"
run_train fight1_subject2          fight-day || log "fight-day train FAILED"

# Append updated artifact list to the night report
REPORT="$LOG_DIR/NIGHT-RUN-REPORT.md"
log "appending day-run artifacts to $REPORT"
{
    echo
    echo "---"
    echo
    echo "# Day Run (Phase 2) — Acrobatic-ish motions"
    echo
    echo "_Appended: $(ts)_"
    echo
    echo "Two more motion trackers trained sequentially on the same task."
    echo
    echo "## Artifacts"
    echo
    echo '```'
    for RUN_NAME in fall-day fight-day; do
        RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*"${RUN_NAME}"* 2>/dev/null | head -1)
        CKPT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
        printf "%-12s %s\n" "$RUN_NAME run:" "$RUN_DIR"
        printf "%-12s %s\n" "$RUN_NAME ckpt:" "$CKPT"
    done
    echo '```'
    echo
    echo "## How to view"
    echo
    for RUN_NAME in fall-day fight-day; do
        RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*"${RUN_NAME}"* 2>/dev/null | head -1)
        CKPT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
        case "$RUN_NAME" in
            fall-day) NPZ="fallAndGetUp1_subject1.npz" ;;
            fight-day) NPZ="fight1_subject2.npz" ;;
        esac
        echo "### $RUN_NAME"
        echo '```bash'
        echo "make track-play-viser \\"
        echo "    NPZ=/workspace/otus_rl_project/src/assets/motions/g1/$NPZ \\"
        echo "    CKPT=$CKPT"
        echo '```'
        echo
    done
    echo "## Final-iteration metrics"
    echo
    for RUN_NAME in fall-day fight-day; do
        echo "### $RUN_NAME"
        echo '```'
        grep -E "(Learning iteration 39[89][0-9]|Mean reward|error_body_pos|error_joint_pos|error_anchor_pos)" \
            "$LOG_DIR/train_${RUN_NAME}.log" 2>/dev/null | tail -25
        echo '```'
        echo
    done
} >> "$REPORT"

log "day-orchestrator finished"
