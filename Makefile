# Convenience wrappers around `docker compose` for rootless Docker.
# All targets assume DOCKER_HOST points to the rootless socket (done in ~/.bashrc).

SHELL := /usr/bin/env bash

COMPOSE   := docker compose -f docker/compose.yaml
CONTAINER := otus_rl_project
UID       := $(shell id -u)
GID       := $(shell id -g)

# Pass host UID/GID to build, so files on bind-mounted /home volumes match host.
export UID
export GID

.PHONY: help
help:                           ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: check
check:                          ## Verify rootless Docker + GPU CDI are live
	@echo "DOCKER_HOST=$$DOCKER_HOST"
	@docker info 2>&1 | grep -E "Server Version|Root Dir|rootless|CDI" || true
	@echo "---"
	@docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L

.PHONY: assets
assets:                         ## Download G1 MJCF assets from MuJoCo Menagerie
	bash scripts/setup_assets.sh

.PHONY: build
build:                          ## Build the main image (otus_rl_project:latest)
	DOCKER_BUILDKIT=1 $(COMPOSE) build --progress=plain

# --- Persistent dev-container workflow --------------------------------------

.PHONY: up
up:                             ## Start the persistent dev container in background
	$(COMPOSE) up -d shell
	@echo "container '$(CONTAINER)' is up. Attach with: make attach"

.PHONY: down
down:                           ## Stop and remove persistent container(s)
	$(COMPOSE) down

.PHONY: stop
stop:                           ## Stop the container without removing
	$(COMPOSE) stop shell

.PHONY: start
start:                          ## Start previously-stopped container
	$(COMPOSE) start shell

.PHONY: restart
restart:                        ## Restart the container
	$(COMPOSE) restart shell

.PHONY: attach
attach:                         ## Attach a new bash session to the running container
	@docker ps --format '{{.Names}}' | grep -qx $(CONTAINER) || { \
		echo ">>> container '$(CONTAINER)' is not running; starting it..."; \
		$(MAKE) up; \
	}
	docker exec -it $(CONTAINER) bash

.PHONY: exec
exec:                           ## Run an arbitrary command inside container: make exec CMD="..."
	@flags="-i"; [ -t 0 ] && flags="-it"; \
	docker exec $$flags $(CONTAINER) bash -lc "$${CMD:-echo 'pass CMD=...'}"

.PHONY: ps
ps:                             ## Show container status
	$(COMPOSE) ps

.PHONY: logs
logs:                           ## Tail container logs
	$(COMPOSE) logs -f --tail=100 shell

# --- One-shot ephemeral workflow --------------------------------------------

.PHONY: shell
shell:                          ## Open an ephemeral bash (dies on exit; use `attach` for persistent)
	$(COMPOSE) run --rm shell bash

.PHONY: list-tasks
list-tasks:                     ## List all registered training tasks
	@$(MAKE) --no-print-directory exec CMD="otus-list-tasks"

.PHONY: train
train:                          ## Start training in the persistent container. Pass CMD="..." to override
	@$(MAKE) --no-print-directory exec CMD="$${CMD:-otus-train --help}"

.PHONY: train-ars
train-ars:                      ## ARS-V2 training on Otus-G1-Walk-Compare. Vars: ITERS, NUM_ENVS, RUN, EXTRA="--noise-std 0.04 ..."
	@ITERS="$${ITERS:-500}"; NUM_ENVS="$${NUM_ENVS:-4096}"; RUN="$${RUN:-ars}"; EXTRA="$${EXTRA:-}"; \
	echo ">>> ARS  iters=$$ITERS  num_envs=$$NUM_ENVS  run=$$RUN  extra=[$$EXTRA]"; \
	$(MAKE) --no-print-directory exec CMD="otus-train-ars --num-iterations $$ITERS --num-envs $$NUM_ENVS --run-name $$RUN $$EXTRA"

.PHONY: play
play:                           ## Run sim2sim keyboard deploy (requires X11). Pass CMD="otus-play --policy ..." to customize
	xhost +SI:localuser:$$USER >/dev/null 2>&1 || true
	@$(MAKE) --no-print-directory exec CMD="$${CMD:-otus-play --help}"

.PHONY: export
export:                         ## Re-export a .pt checkpoint to ONNX. Pass CMD="otus-export --task ... --checkpoint ..." to customize
	@$(MAKE) --no-print-directory exec CMD="$${CMD:-otus-export --help}"

.PHONY: play-viser
play-viser:                     ## Headless web viewer (Viser) on http://localhost:8080. Latest checkpoint by default.
	@CKPT="$${CKPT:-$$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1)}"; \
	if [ -z "$$CKPT" ]; then echo ">>> no checkpoint found under logs/rsl_rl"; exit 1; fi; \
	TASK="$${TASK:-Unitree-G1-Flat}"; \
	EXTRA="$${EXTRA:-}"; \
	echo ">>> task=$$TASK  ckpt=$$CKPT"; \
	echo ">>> open http://localhost:8080 on your laptop (after ssh -L 8080:localhost:8080)"; \
	$(MAKE) --no-print-directory exec CMD="otus-play-mjlab $$TASK --viewer viser --num-envs 1 --checkpoint-file $$CKPT $$EXTRA"

# --- Motion imitation (tracking) workflow ---------------------------------

.PHONY: track-prep
track-prep:                     ## Convert a CSV motion to NPZ. Vars: CSV (path), NAME (output base), FPS_IN (30), FPS_OUT (50), LINE_RANGE (e.g. 2191,2431 to slice frames)
	@CSV="$${CSV:?set CSV=path/to/motion.csv}"; \
	NAME="$${NAME:-$$(basename $$CSV .csv)}"; \
	FPS_IN="$${FPS_IN:-30}"; FPS_OUT="$${FPS_OUT:-50}"; \
	LR_FLAG=""; if [ -n "$${LINE_RANGE:-}" ]; then LR_FLAG="--line-range $$LINE_RANGE"; fi; \
	echo ">>> CSV=$$CSV  NAME=$$NAME  fps=$$FPS_IN->$$FPS_OUT  range=$${LINE_RANGE:-(full file)}"; \
	$(MAKE) --no-print-directory exec CMD="cd /workspace/otus_rl_project && python /opt/third_party/unitree_rl_mjlab/scripts/csv_to_npz.py --robot g1 --input-file $$CSV --output-name $$NAME --input-fps $$FPS_IN --output-fps $$FPS_OUT $$LR_FLAG"

.PHONY: track-train
track-train:                    ## Train motion-tracking policy. Vars: NPZ (path), ITERS (default 3000), NUM_ENVS (default 4096), RUN (run-name)
	@NPZ="$${NPZ:?set NPZ=path/to/motion.npz}"; \
	ITERS="$${ITERS:-3000}"; NUM_ENVS="$${NUM_ENVS:-4096}"; \
	RUN="$${RUN:-tracker}"; \
	TASK="$${TASK:-Unitree-G1-Tracking-No-State-Estimation}"; \
	echo ">>> task=$$TASK npz=$$NPZ iters=$$ITERS envs=$$NUM_ENVS run=$$RUN"; \
	$(MAKE) --no-print-directory exec CMD="otus-train $$TASK --motion-file=$$NPZ --env.scene.num-envs=$$NUM_ENVS --agent.max-iterations=$$ITERS --agent.run-name=$$RUN --agent.logger=tensorboard --agent.upload-model=False"

.PHONY: track-play-viser
track-play-viser:               ## Replay a trained tracker in Viser web viewer. Vars: CKPT (default = latest), NPZ (motion file)
	@CKPT="$${CKPT:-$$(ls -t logs/rsl_rl/g1_tracking/*/model_*.pt 2>/dev/null | head -1)}"; \
	if [ -z "$$CKPT" ]; then echo ">>> no tracking checkpoint under logs/rsl_rl/g1_tracking"; exit 1; fi; \
	NPZ="$${NPZ:?set NPZ=path/to/motion.npz}"; \
	TASK="$${TASK:-Unitree-G1-Tracking-No-State-Estimation}"; \
	echo ">>> task=$$TASK ckpt=$$CKPT motion=$$NPZ"; \
	echo ">>> open http://localhost:8080 on your laptop"; \
	$(MAKE) --no-print-directory exec CMD="otus-play-mjlab $$TASK --viewer viser --num-envs 1 --motion-file=$$NPZ --checkpoint-file=$$CKPT"

.PHONY: track-replay-ref
track-replay-ref:               ## Replay raw motion reference (no trained policy). Vars: NPZ (motion file)
	@NPZ="$${NPZ:?set NPZ=path/to/motion.npz}"; \
	TASK="$${TASK:-Unitree-G1-Tracking-No-State-Estimation}"; \
	echo ">>> task=$$TASK motion=$$NPZ  agent=zero (no policy, just replay the human reference)"; \
	echo ">>> open http://localhost:8080 on your laptop"; \
	$(MAKE) --no-print-directory exec CMD="otus-play-mjlab $$TASK --viewer viser --num-envs 1 --motion-file=$$NPZ --agent zero --no-terminations"

.PHONY: tb
tb:                             ## TensorBoard on http://localhost:6006 (run inside the persistent container)
	@$(MAKE) --no-print-directory exec CMD="tensorboard --logdir=/workspace/otus_rl_project/logs --host=0.0.0.0 --port=6006"

# --- Phase 4 (cross-algorithm comparison report) --------------------------

.PHONY: compare-night
compare-night:                  ## Run PPO/AMP/ARS overnight on the comparison task. Vars: PPO_ITERS, AMP_ITERS, ARS_ITERS, NUM_ENVS, RUN_TAG, SEED
	bash scripts/night_compare_orchestrator.sh

.PHONY: compare-smoke
compare-smoke:                  ## ~30-second smoke test of the orchestrator pipeline (2 iters each, 64 envs)
	bash scripts/night_compare_orchestrator.sh --smoke

.PHONY: compare-extract
compare-extract:                ## Extract Eval/AMP scalars to docs/results/. Vars: PPO_RUN, AMP_RUN, ARS_RUN (default: latest of each)
	@PPO_RUN="$${PPO_RUN:-$$(ls -td logs/rsl_rl/g1_walk_compare/*ppo* 2>/dev/null | head -1)}"; \
	AMP_RUN="$${AMP_RUN:-$$(ls -td logs/rsl_rl/g1_walk_amp/*amp* 2>/dev/null | head -1)}"; \
	ARS_RUN="$${ARS_RUN:-$$(ls -td logs/rsl_rl/g1_walk_compare_ars/*ars* 2>/dev/null | head -1)}"; \
	PAIRS=""; \
	[ -n "$$PPO_RUN" ] && PAIRS="$$PAIRS ppo=$$PPO_RUN"; \
	[ -n "$$AMP_RUN" ] && PAIRS="$$PAIRS amp=$$AMP_RUN"; \
	[ -n "$$ARS_RUN" ] && PAIRS="$$PAIRS ars=$$ARS_RUN"; \
	if [ -z "$$PAIRS" ]; then echo ">>> no run dirs found under logs/rsl_rl/g1_walk_*; pass them via PPO_RUN=, AMP_RUN=, ARS_RUN="; exit 1; fi; \
	echo ">>> extracting --run$$PAIRS --out-dir docs/results"; \
	$(MAKE) --no-print-directory exec CMD="otus-extract-tb --run$$PAIRS --out-dir docs/results"

.PHONY: prune
prune:                          ## Remove dangling images + build cache (rootless only)
	docker image prune -f
	docker builder prune -f

.PHONY: size
size:                           ## Show disk usage of rootless Docker data root
	du -sh $$HOME/.local/share/docker 2>/dev/null || true
	docker system df
