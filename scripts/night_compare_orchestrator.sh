#!/usr/bin/env bash
# Phase-4 prep: train PPO baseline, PPO+AMP, and ARS on the frozen comparison task,
# sequentially on a single GPU, with Eval/* logging enabled. Writes a status report
# to runs/compare/COMPARE-RUN-REPORT.md and prints the run dirs that the
# `otus-extract-tb` extractor should be pointed at.
#
# Usage:
#   bash scripts/night_compare_orchestrator.sh                   # full run
#   bash scripts/night_compare_orchestrator.sh --smoke           # 2 iters each, tiny envs
#   PPO_ITERS=3000 AMP_ITERS=3000 ARS_ITERS=500 NUM_ENVS=4096 \
#     bash scripts/night_compare_orchestrator.sh
#
# The script expects:
#   - the otus_rl_project Docker container to be running (`make up`)
#   - GPU 0 to be free for the duration (sequential single-GPU runs)
#
# Skip individual algorithms by setting the corresponding ITERS=0:
#   PPO_ITERS=0 bash scripts/night_compare_orchestrator.sh   # skip PPO baseline
set -uo pipefail

# -----------------------------------------------------------------------------
# Defaults — overridable via environment variables
# -----------------------------------------------------------------------------
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
fi

if (( SMOKE == 1 )); then
    : "${PPO_ITERS:=2}"
    : "${AMP_ITERS:=2}"
    : "${ARS_ITERS:=2}"
    : "${SAC_ITERS:=2}"
    : "${NUM_ENVS:=64}"
    : "${EVAL_INTERVAL:=1}"
    : "${EVAL_NUM_ENVS:=16}"
    : "${EVAL_NUM_STEPS:=8}"
    : "${RUN_TAG:=smoke}"
    # ARS requires num_envs == 2 * num_directions * envs_per_direction; pick small valid combo.
    : "${ARS_NUM_DIRECTIONS:=16}"
    : "${ARS_TOP_DIRECTIONS:=8}"
    : "${ARS_ROLLOUT_STEPS:=16}"
    # SAC: tiny rollout + few updates so smoke completes in seconds.
    : "${SAC_ROLLOUT_STEPS:=2}"
    : "${SAC_NUM_RANDOM_STEPS:=1}"
    : "${SAC_NUM_UPDATES_PER_ITER:=2}"
    : "${SAC_BATCH_SIZE:=64}"
    : "${SAC_REPLAY_BUFFER_SIZE:=4096}"
else
    : "${PPO_ITERS:=3000}"
    : "${AMP_ITERS:=3000}"
    : "${ARS_ITERS:=500}"
    : "${SAC_ITERS:=1000}"
    : "${NUM_ENVS:=4096}"
    : "${EVAL_INTERVAL:=50}"
    : "${EVAL_NUM_ENVS:=256}"
    : "${EVAL_NUM_STEPS:=96}"
    : "${RUN_TAG:=phase4}"
    : "${ARS_NUM_DIRECTIONS:=64}"
    : "${ARS_TOP_DIRECTIONS:=32}"
    : "${ARS_ROLLOUT_STEPS:=200}"
    : "${SAC_ROLLOUT_STEPS:=8}"
    : "${SAC_NUM_RANDOM_STEPS:=4}"
    : "${SAC_NUM_UPDATES_PER_ITER:=64}"
    : "${SAC_BATCH_SIZE:=4096}"
    : "${SAC_REPLAY_BUFFER_SIZE:=500000}"
fi
: "${SEED:=42}"
: "${AMP_EXPERT_CSV:=src/assets/motions/g1/dance1_subject2.csv}"

ROOT="/home/ev/repos/otus_rl_project"
COMPARE_DIR="$ROOT/runs/compare"
REPORT="$COMPARE_DIR/COMPARE-RUN-REPORT.md"
mkdir -p "$COMPARE_DIR"
cd "$ROOT"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(ts)] $*" | tee -a "$COMPARE_DIR/orchestrator.log"; }

CONTAINER="otus_rl_project"
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || {
    log "ERROR: container '$CONTAINER' is not running. Start it with: make up"
    exit 1
}

run_in_container() {
    docker exec "$CONTAINER" bash -lc "cd /workspace/otus_rl_project && $*"
}

latest_run_dir() {
    # $1 = experiment subdir under logs/rsl_rl/, $2 = run-name suffix to match
    local subdir="$1" suffix="$2"
    ls -td "logs/rsl_rl/${subdir}/"*"${suffix}"* 2>/dev/null | head -1
}

log "===================================================="
log "compare-orchestrator started (smoke=$SMOKE, tag=$RUN_TAG)"
log "PPO_ITERS=$PPO_ITERS AMP_ITERS=$AMP_ITERS ARS_ITERS=$ARS_ITERS SAC_ITERS=$SAC_ITERS"
log "NUM_ENVS=$NUM_ENVS EVAL_INTERVAL=$EVAL_INTERVAL"
log "===================================================="

PPO_RUN_DIR=""
AMP_RUN_DIR=""
ARS_RUN_DIR=""
SAC_RUN_DIR=""

# -----------------------------------------------------------------------------
# 1. PPO baseline on Otus-G1-Walk-Compare (stage labels intentionally kept as N/3 for backward compatibility with prior reports — SAC is appended as stage 4).
# -----------------------------------------------------------------------------
if (( PPO_ITERS > 0 )); then
    PPO_RUN_NAME="ppo_${RUN_TAG}"
    PPO_LOG="$COMPARE_DIR/train_ppo_${RUN_TAG}.log"
    log "[1/3] PPO baseline: iters=$PPO_ITERS run=$PPO_RUN_NAME log=$PPO_LOG"
    run_in_container "otus-train Otus-G1-Walk-Compare \
        --env.scene.num-envs $NUM_ENVS \
        --agent.max-iterations $PPO_ITERS \
        --agent.seed $SEED \
        --agent.run-name $PPO_RUN_NAME \
        --agent.logger tensorboard \
        --agent.upload-model False \
        --agent.locomotion-eval-interval $EVAL_INTERVAL \
        --agent.locomotion-eval-num-envs $EVAL_NUM_ENVS \
        --agent.locomotion-eval-num-steps $EVAL_NUM_STEPS" \
        > "$PPO_LOG" 2>&1
    PPO_RC=$?
    log "  exit code: $PPO_RC"
    PPO_RUN_DIR=$(latest_run_dir "g1_walk_compare" "$PPO_RUN_NAME")
    log "  run dir: ${PPO_RUN_DIR:-<not found>}"
else
    log "[1/3] PPO baseline SKIPPED (PPO_ITERS=0)"
fi

# -----------------------------------------------------------------------------
# 2. PPO + AMP on Otus-G1-Walk-AMP
# -----------------------------------------------------------------------------
if (( AMP_ITERS > 0 )); then
    AMP_RUN_NAME="amp_${RUN_TAG}"
    AMP_LOG="$COMPARE_DIR/train_amp_${RUN_TAG}.log"
    log "[2/3] PPO+AMP: iters=$AMP_ITERS run=$AMP_RUN_NAME log=$AMP_LOG"
    run_in_container "otus-train-amp \
        --agent.amp.expert-csv-paths \"('${AMP_EXPERT_CSV}',)\" \
        --env.scene.num-envs $NUM_ENVS \
        --agent.max-iterations $AMP_ITERS \
        --agent.seed $SEED \
        --agent.run-name $AMP_RUN_NAME \
        --agent.logger tensorboard \
        --agent.upload-model False \
        --agent.locomotion-eval-interval $EVAL_INTERVAL \
        --agent.locomotion-eval-num-envs $EVAL_NUM_ENVS \
        --agent.locomotion-eval-num-steps $EVAL_NUM_STEPS" \
        > "$AMP_LOG" 2>&1
    AMP_RC=$?
    log "  exit code: $AMP_RC"
    AMP_RUN_DIR=$(latest_run_dir "g1_walk_amp" "$AMP_RUN_NAME")
    log "  run dir: ${AMP_RUN_DIR:-<not found>}"
else
    log "[2/3] PPO+AMP SKIPPED (AMP_ITERS=0)"
fi

# -----------------------------------------------------------------------------
# 3. ARS on Otus-G1-Walk-Compare
# -----------------------------------------------------------------------------
if (( ARS_ITERS > 0 )); then
    ARS_RUN_NAME="ars_${RUN_TAG}"
    ARS_LOG="$COMPARE_DIR/train_ars_${RUN_TAG}.log"
    log "[3/3] ARS: iters=$ARS_ITERS run=$ARS_RUN_NAME log=$ARS_LOG"
    run_in_container "otus-train-ars \
        --num-iterations $ARS_ITERS \
        --num-envs $NUM_ENVS \
        --num-directions $ARS_NUM_DIRECTIONS \
        --top-directions $ARS_TOP_DIRECTIONS \
        --rollout-steps $ARS_ROLLOUT_STEPS \
        --seed $SEED \
        --run-name $ARS_RUN_NAME \
        --eval-interval $EVAL_INTERVAL \
        --eval-num-envs $EVAL_NUM_ENVS \
        --eval-num-steps $EVAL_NUM_STEPS" \
        > "$ARS_LOG" 2>&1
    ARS_RC=$?
    log "  exit code: $ARS_RC"
    ARS_RUN_DIR=$(latest_run_dir "g1_walk_compare_ars" "$ARS_RUN_NAME")
    log "  run dir: ${ARS_RUN_DIR:-<not found>}"
else
    log "[3/3] ARS SKIPPED (ARS_ITERS=0)"
fi

# -----------------------------------------------------------------------------
# 4. SAC on Otus-G1-Walk-Compare
# -----------------------------------------------------------------------------
if (( SAC_ITERS > 0 )); then
    SAC_RUN_NAME="sac_${RUN_TAG}"
    SAC_LOG="$COMPARE_DIR/train_sac_${RUN_TAG}.log"
    log "[4/4] SAC: iters=$SAC_ITERS run=$SAC_RUN_NAME log=$SAC_LOG"
    run_in_container "otus-train-sac \
        --num-iterations $SAC_ITERS \
        --num-envs $NUM_ENVS \
        --rollout-steps $SAC_ROLLOUT_STEPS \
        --num-random-steps $SAC_NUM_RANDOM_STEPS \
        --num-updates-per-iter $SAC_NUM_UPDATES_PER_ITER \
        --batch-size $SAC_BATCH_SIZE \
        --replay-buffer-size $SAC_REPLAY_BUFFER_SIZE \
        --seed $SEED \
        --run-name $SAC_RUN_NAME \
        --eval-interval $EVAL_INTERVAL \
        --eval-num-envs $EVAL_NUM_ENVS \
        --eval-num-steps $EVAL_NUM_STEPS" \
        > "$SAC_LOG" 2>&1
    SAC_RC=$?
    log "  exit code: $SAC_RC"
    SAC_RUN_DIR=$(latest_run_dir "g1_walk_compare_sac" "$SAC_RUN_NAME")
    log "  run dir: ${SAC_RUN_DIR:-<not found>}"
else
    log "[4/4] SAC SKIPPED (SAC_ITERS=0)"
fi

# -----------------------------------------------------------------------------
# Status report
# -----------------------------------------------------------------------------
log "writing $REPORT"
{
    echo "# Comparison-Run Report (Phase 4 prep)"
    echo
    echo "_Generated: $(ts) — tag=\`$RUN_TAG\`, smoke=\`$SMOKE\`_"
    echo
    echo "Three RL algorithms trained sequentially on a single GPU against the"
    echo "frozen \`Otus-G1-Walk-Compare\` task (PPO+AMP uses \`Otus-G1-Walk-AMP\`,"
    echo "which shares the same env config). All runs log identical \`Eval/*\`"
    echo "scalars via \`LocomotionCompareVelocityRunner\`."
    echo
    echo "## Settings"
    echo
    echo '| Parameter | Value |'
    echo '|---|---|'
    echo "| PPO iters | \`$PPO_ITERS\` |"
    echo "| AMP iters | \`$AMP_ITERS\` |"
    echo "| ARS iters | \`$ARS_ITERS\` |"
    echo "| SAC iters | \`$SAC_ITERS\` |"
    echo "| num_envs | \`$NUM_ENVS\` |"
    echo "| eval_interval | \`$EVAL_INTERVAL\` |"
    echo "| eval_num_envs | \`$EVAL_NUM_ENVS\` |"
    echo "| eval_num_steps | \`$EVAL_NUM_STEPS\` |"
    echo "| seed | \`$SEED\` |"
    echo "| amp_expert_csv | \`$AMP_EXPERT_CSV\` |"
    echo
    echo "## Run directories"
    echo
    echo '```'
    echo "ppo: ${PPO_RUN_DIR:-<skipped/not-found>}"
    echo "amp: ${AMP_RUN_DIR:-<skipped/not-found>}"
    echo "ars: ${ARS_RUN_DIR:-<skipped/not-found>}"
    echo "sac: ${SAC_RUN_DIR:-<skipped/not-found>}"
    echo '```'
    echo
    echo "## Extract metrics + figures"
    echo
    echo "Run inside the container:"
    echo
    echo '```bash'
    pairs=()
    [[ -n "$PPO_RUN_DIR" ]] && pairs+=("ppo=$PPO_RUN_DIR")
    [[ -n "$AMP_RUN_DIR" ]] && pairs+=("amp=$AMP_RUN_DIR")
    [[ -n "$ARS_RUN_DIR" ]] && pairs+=("ars=$ARS_RUN_DIR")
    [[ -n "$SAC_RUN_DIR" ]] && pairs+=("sac=$SAC_RUN_DIR")
    if (( ${#pairs[@]} > 0 )); then
        echo "otus-extract-tb \\"
        echo "    --run ${pairs[*]} \\"
        echo "    --out-dir docs/results"
    else
        echo "# (no successful runs to extract)"
    fi
    echo '```'
    echo
    echo "## Logs"
    echo
    [[ "$PPO_ITERS" -gt 0 ]] && echo '- PPO  : `'"$PPO_LOG"'`'
    [[ "$AMP_ITERS" -gt 0 ]] && echo '- AMP  : `'"$AMP_LOG"'`'
    [[ "$ARS_ITERS" -gt 0 ]] && echo '- ARS  : `'"$ARS_LOG"'`'
    [[ "$SAC_ITERS" -gt 0 ]] && echo '- SAC  : `'"$SAC_LOG"'`'
    echo '- orchestrator: `'"$COMPARE_DIR/orchestrator.log"'`'
    echo
    echo "## Final-iteration metrics (raw tail)"
    echo
    if [[ -n "$PPO_RUN_DIR" ]]; then
        echo "### PPO"
        echo '```'
        grep -E "(Learning iteration|Mean reward|Eval/velocity_tracking)" \
            "$PPO_LOG" 2>/dev/null | tail -20
        echo '```'
        echo
    fi
    if [[ -n "$AMP_RUN_DIR" ]]; then
        echo "### PPO+AMP"
        echo '```'
        grep -E "(Learning iteration|Mean reward|AMP/disc_loss|Eval/velocity_tracking)" \
            "$AMP_LOG" 2>/dev/null | tail -20
        echo '```'
        echo
    fi
    if [[ -n "$ARS_RUN_DIR" ]]; then
        echo "### ARS"
        echo '```'
        grep -E "(iter|mean_R|Eval/velocity_tracking)" \
            "$ARS_LOG" 2>/dev/null | tail -20
        echo '```'
        echo
    fi
    if [[ -n "$SAC_RUN_DIR" ]]; then
        echo "### SAC"
        echo '```'
        grep -E "(it=|mean_step_R|alpha=|critic_loss|Eval/velocity_tracking)" \
            "$SAC_LOG" 2>/dev/null | tail -20
        echo '```'
    fi
} > "$REPORT"

log "compare-orchestrator finished"
log "report: $REPORT"
