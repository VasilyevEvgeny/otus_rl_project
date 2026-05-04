#!/usr/bin/env bash
# Watcher: tail the cartwheel training log; when iter 3000 appears,
# kill the orchestrator (so double_kong does NOT start) and the
# training process itself. After that, the next checkpoint flush
# (model_3000.pt) is already on disk because saves happen at multiples
# of 500.
set -uo pipefail

LOG="/home/ev/repos/otus_rl_project/logs/orchestrator/train_acro-cartwheel.log"
ORCH_LOG="/home/ev/repos/otus_rl_project/logs/orchestrator/acrobatics_orchestrator.log"
ORCH_PID_PAT="acrobatics_orchestrator.sh"
TRAIN_PID_PAT="train\.py.*acro-cartwheel"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log_w() { echo "[$(ts)] [stop3k] $*" | tee -a "$ORCH_LOG"; }

log_w "watcher started; waiting for iteration 3000 in $LOG"

# Stream the log; bail out as soon as we see iter 3000.
( tail -F -n 0 "$LOG" 2>/dev/null & echo $! >/tmp/stop3k_tail.pid ) | \
  while IFS= read -r line; do
    # Strip ANSI to be safe.
    clean=$(printf '%s' "$line" | sed -E 's/\x1b\[[0-9;]*m//g')
    if [[ "$clean" =~ Learning\ iteration\ 3000/ ]]; then
      log_w "iter 3000 reached, stopping training and orchestrator"
      # 1) kill orchestrator first so the next step (double_kong) does not start
      pkill -KILL -f "$ORCH_PID_PAT" 2>/dev/null || true
      # 2) kill training process(es)
      pkill -KILL -f "$TRAIN_PID_PAT" 2>/dev/null || true
      # 3) belt-and-braces: kill any lingering python that mentions cartwheel npz
      pkill -KILL -f "cartwheel\.npz" 2>/dev/null || true
      log_w "kill signals sent; waiting 5s for teardown"
      sleep 5
      log_w "remaining processes:"
      pgrep -af "train\.py|acrobatics_orchestrator|cartwheel" | tee -a "$ORCH_LOG" || true
      log_w "watcher exiting"
      # kill the tail
      kill "$(cat /tmp/stop3k_tail.pid 2>/dev/null)" 2>/dev/null || true
      exit 0
    fi
  done
