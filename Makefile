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

.PHONY: train
train:                          ## Start training. Pass TASK=... CMD="..." to override
	$(COMPOSE) run --rm train bash -lc "$${CMD:-otus-train --help}"

.PHONY: play
play:                           ## Launch sim2sim deploy (requires X11)
	xhost +SI:localuser:$$USER >/dev/null 2>&1 || true
	$(COMPOSE) run --rm play

.PHONY: export
export:                         ## Export latest checkpoint to ONNX
	$(COMPOSE) run --rm export

.PHONY: tb
tb:                             ## Run tensorboard on http://localhost:6006
	$(COMPOSE) run --rm --service-ports shell tensorboard --logdir=/workspace/otus_rl_project/runs --host=0.0.0.0

.PHONY: prune
prune:                          ## Remove dangling images + build cache (rootless only)
	docker image prune -f
	docker builder prune -f

.PHONY: size
size:                           ## Show disk usage of rootless Docker data root
	du -sh $$HOME/.local/share/docker 2>/dev/null || true
	docker system df
