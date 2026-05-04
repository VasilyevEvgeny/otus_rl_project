#!/usr/bin/env bash
# Overnight pipeline orchestrator.
#   1. Wait for the running dance tracker training (PID passed as $1) to finish
#   2. Launch jumps tracker training (LAFAN1 jumps1_subject1, 4000 iters)
#   3. Export both checkpoints to ONNX with metadata
#   4. Write a brief report to runs/night/NIGHT-RUN-REPORT.md
set -uo pipefail

DANCE_PID="${1:?usage: orchestrator.sh <dance_pid>}"
ROOT="/home/ev/repos/otus_rl_project"
LOG_DIR="$ROOT/runs/night"
REPORT="$LOG_DIR/NIGHT-RUN-REPORT.md"
mkdir -p "$LOG_DIR"
cd "$ROOT"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/orchestrator.log"; }

log "orchestrator started; waiting for dance training PID=$DANCE_PID to finish"

# 1. Wait for dance to finish
while ps -p "$DANCE_PID" > /dev/null 2>&1; do
    sleep 30
done
log "dance training process exited"

# Find the dance run directory
DANCE_RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*dance-night* 2>/dev/null | head -1)
if [ -n "$DANCE_RUN_DIR" ]; then
    DANCE_CKPT=$(ls -t "$DANCE_RUN_DIR"/model_*.pt 2>/dev/null | head -1)
    log "dance run dir: $DANCE_RUN_DIR"
    log "dance ckpt:    $DANCE_CKPT"
else
    log "WARN: dance run dir not found"
    DANCE_CKPT=""
fi

# 2. Launch jumps training (sequential, full GPU)
log "launching jumps tracker training, 4000 iters, 4096 envs"
make track-train \
    NPZ=/workspace/otus_rl_project/src/assets/motions/g1/jumps1_subject1.npz \
    ITERS=4000 NUM_ENVS=4096 RUN=jumps-night \
    > "$LOG_DIR/train_jumps.log" 2>&1
JUMPS_RC=$?
log "jumps training exit code: $JUMPS_RC"

JUMPS_RUN_DIR=$(ls -td logs/rsl_rl/g1_tracking/*jumps-night* 2>/dev/null | head -1)
if [ -n "$JUMPS_RUN_DIR" ]; then
    JUMPS_CKPT=$(ls -t "$JUMPS_RUN_DIR"/model_*.pt 2>/dev/null | head -1)
    log "jumps run dir: $JUMPS_RUN_DIR"
    log "jumps ckpt:    $JUMPS_CKPT"
else
    log "WARN: jumps run dir not found"
    JUMPS_CKPT=""
fi

# 3. Export both .pt -> .onnx (the training already exports policy.onnx automatically,
#    but we also re-export with our own pipeline to embed metadata for keyboard sim2sim).
#    For tracking tasks, ONNX is automatically exported by mjlab during training; just verify.
log "checking auto-exported ONNX files"
for d in "$DANCE_RUN_DIR" "$JUMPS_RUN_DIR"; do
    [ -z "$d" ] && continue
    if [ -f "$d/policy.onnx" ]; then
        log "  $d/policy.onnx OK ($(stat -c %s "$d/policy.onnx") bytes)"
    else
        log "  $d/policy.onnx MISSING — attempting re-export"
        if ls "$d"/model_*.pt > /dev/null 2>&1; then
            CKPT=$(ls -t "$d"/model_*.pt | head -1)
            make exec CMD="otus-export --task Unitree-G1-Tracking-No-State-Estimation --checkpoint $CKPT --out $d/policy.onnx" \
                >> "$LOG_DIR/orchestrator.log" 2>&1 || log "    WARN: re-export failed"
        fi
    fi
done

# 4. Write report
log "writing $REPORT"
{
    echo "# Overnight Run Report"
    echo
    echo "_Generated: $(ts)_"
    echo
    echo "## What ran"
    echo
    echo "Two motion-tracking trainings of \`Unitree-G1-Tracking-No-State-Estimation\`:"
    echo
    echo "1. **dance1_subject2** (LAFAN1, 130 s of dancing, ~3000 iter)"
    echo "2. **jumps1_subject1** (LAFAN1, 244 s of jumps, ~4000 iter)"
    echo
    echo "## Artifacts"
    echo
    echo '```'
    echo "dance run:   $DANCE_RUN_DIR"
    echo "dance ckpt:  $DANCE_CKPT"
    echo "jumps run:   $JUMPS_RUN_DIR"
    echo "jumps ckpt:  $JUMPS_CKPT"
    echo '```'
    echo
    echo "## How to view"
    echo
    echo 'On laptop, open the SSH tunnel: `ssh -L 8080:localhost:8080 ev@<server>`'
    echo
    echo 'On server, in the project root:'
    echo
    echo '```bash'
    echo '# Dance tracker'
    echo "make track-play-viser \\"
    echo "    NPZ=/workspace/otus_rl_project/src/assets/motions/g1/dance1_subject2.npz \\"
    echo "    CKPT=$DANCE_CKPT"
    echo
    echo '# Jumps tracker'
    echo "make track-play-viser \\"
    echo "    NPZ=/workspace/otus_rl_project/src/assets/motions/g1/jumps1_subject1.npz \\"
    echo "    CKPT=$JUMPS_CKPT"
    echo '```'
    echo
    echo "Then point your browser at <http://localhost:8080>."
    echo
    echo "## Logs"
    echo
    echo '- Dance training:   `runs/night/train_dance.log`'
    echo '- Jumps training:   `runs/night/train_jumps.log`'
    echo '- Orchestrator:     `runs/night/orchestrator.log`'
    echo
    echo "## Final-iteration metrics"
    echo
    echo "### Dance"
    echo '```'
    grep -E "(Learning iteration|Mean reward|error_body_pos|error_joint_pos|error_anchor_pos)" \
        "$LOG_DIR/train_dance.log" 2>/dev/null | tail -20
    echo '```'
    echo
    echo "### Jumps"
    echo '```'
    grep -E "(Learning iteration|Mean reward|error_body_pos|error_joint_pos|error_anchor_pos)" \
        "$LOG_DIR/train_jumps.log" 2>/dev/null | tail -20
    echo '```'
} > "$REPORT"

log "orchestrator finished"
