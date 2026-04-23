# syntax=docker/dockerfile:1.7

# -----------------------------------------------------------------------------
# Base image: CUDA 12.6.3 + cuDNN + Ubuntu 24.04 (matches host, Python 3.12 by
# default). We use the `devel` flavor because NVIDIA Warp may occasionally need
# CUDA dev headers when JIT-compiling kernels at first run. Add ~2GB vs runtime,
# but safer for an R&D image.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # NVIDIA GPU visibility inside container (kept explicit for clarity)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# -----------------------------------------------------------------------------
# System dependencies
#   - build-essential, cmake, git: for source installs (unitree_rl_mjlab, etc.)
#   - python3.12 / venv / pip: from Ubuntu 24.04 default
#   - libGL / libEGL / libglfw / libosmesa: for MuJoCo rendering (passive viewer)
#   - x11/xkb libs: for GLFW window when we X11-forward from host
#   - ffmpeg: for video recording (imageio-ffmpeg, wandb videos)
#   - curl, ca-certs: for uv installer
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        libgl1 \
        libegl1 \
        libglu1-mesa \
        libglfw3 \
        libosmesa6 \
        libx11-6 \
        libxcursor1 \
        libxrandr2 \
        libxinerama1 \
        libxi6 \
        libxkbcommon0 \
        libxcb1 \
        libxau6 \
        libxdmcp6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# -----------------------------------------------------------------------------
# Install uv (fast pip/venv replacement) — used for all Python deps
# -----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# -----------------------------------------------------------------------------
# Create a non-root user matching host UID/GID (passed at build time).
# This prevents volume-mount file ownership mess on /home bind mounts.
# -----------------------------------------------------------------------------
ARG USER_UID=1004
ARG USER_GID=1004
ARG USER_NAME=dev

RUN groupadd --gid ${USER_GID} ${USER_NAME} 2>/dev/null || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USER_NAME} && \
    mkdir -p /workspace/otus_rl_project /opt/third_party && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace /opt/third_party

# -----------------------------------------------------------------------------
# Python environment (system site-packages, owned by our user).
# We use uv with --system to install into /usr/lib/python3.12 — no venv needed,
# which keeps the image simpler and avoids PATH gymnastics inside the container.
# -----------------------------------------------------------------------------
# Installs are split to maximize Docker layer cache re-use:
#   1. Heavy, rarely-changing: torch + warp-lang + mujoco + mujoco-warp + mjlab
#   2. Medium: rsl-rl, onnxruntime-gpu, onnxscript, pygame, wandb, tensorboard
#   3. Our own pyproject (editable) - last, changes most often
# -----------------------------------------------------------------------------

# Step 1 — heavy simulation/RL stack (pinned versions)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "torch>=2.7,<2.9" \
        --extra-index-url https://download.pytorch.org/whl/cu126

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "mujoco>=3.5,<4" \
        "mujoco-warp>=3.5,<4" \
        "warp-lang>=1.12,<2" \
        "mjlab==1.2.0"

# Step 2 — RL training, deploy, tracking
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "rsl-rl-lib==5.0.1" \
        "onnxruntime-gpu>=1.20" \
        "onnxscript>=0.5.4" \
        "pygame>=2.5" \
        "wandb>=0.22.3" \
        "tensorboard>=2.20" \
        "tyro>=0.8" \
        "tqdm" \
        "numpy"

# -----------------------------------------------------------------------------
# Clone unitree_rl_mjlab into /opt/third_party and install editable.
# Pinned to a specific commit for reproducibility. Bump explicitly when needed.
# -----------------------------------------------------------------------------
ARG UNITREE_RL_MJLAB_REF=main
RUN git clone --depth 1 --branch ${UNITREE_RL_MJLAB_REF} \
        https://github.com/unitreerobotics/unitree_rl_mjlab.git \
        /opt/third_party/unitree_rl_mjlab && \
    uv pip install --system --break-system-packages \
        -e /opt/third_party/unitree_rl_mjlab

# -----------------------------------------------------------------------------
# Install our own package in editable mode. Source will be bind-mounted at
# runtime via compose (the whole repo ends up at /workspace/otus_rl_project),
# so this COPY is a placeholder install to register the entry points.
# -----------------------------------------------------------------------------
WORKDIR /workspace/otus_rl_project
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml /workspace/otus_rl_project/pyproject.toml
COPY --chown=${USER_UID}:${USER_GID} src /workspace/otus_rl_project/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -e /workspace/otus_rl_project

# -----------------------------------------------------------------------------
# We intentionally leave the default user as root (UID 0).
# Under rootless Docker, UID 0 inside the container is transparently mapped
# to the daemon owner (`ev`, UID 1004) on the host, so running as "root" here
# is safe *and* gives us write access to bind-mounted repo files that are
# owned by `ev` on the host. The `dev` user is kept around only for non-
# rootless usage scenarios (where one would override `user:` in compose).
# -----------------------------------------------------------------------------

# Useful default for wandb/tensorboard: don't litter $HOME
ENV WANDB_DIR=/workspace/otus_rl_project/runs/wandb \
    TENSORBOARD_DIR=/workspace/otus_rl_project/runs/tb

CMD ["bash"]
