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
	echo ">>> task=$$TASK  ckpt=$$CKPT"; \
	echo ">>> open http://localhost:8080 on your laptop (after ssh -L 8080:localhost:8080)"; \
	$(MAKE) --no-print-directory exec CMD="otus-play-mjlab $$TASK --viewer viser --num-envs 1 --checkpoint-file $$CKPT"

.PHONY: tb
tb:                             ## TensorBoard on http://localhost:6006 (run inside the persistent container)
	@$(MAKE) --no-print-directory exec CMD="tensorboard --logdir=/workspace/otus_rl_project/logs --host=0.0.0.0 --port=6006"

.PHONY: prune
prune:                          ## Remove dangling images + build cache (rootless only)
	docker image prune -f
	docker builder prune -f

.PHONY: size
size:                           ## Show disk usage of rootless Docker data root
	du -sh $$HOME/.local/share/docker 2>/dev/null || true
	docker system df
