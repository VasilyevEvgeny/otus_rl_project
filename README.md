# otus_rl_project

Reinforcement-learning pipeline for the **Unitree G1** humanoid:
train complex skills (locomotion first, then acrobatics — `spin kick`,
`cartwheel`, `back-flip`), export the policy to **ONNX**, and run
**sim2sim** in MuJoCo with keyboard control.

Everything runs inside **rootless Docker**. The daemon, images and data
live in `$HOME/.local/share/docker`, so this project does not interfere
with the system-wide Docker installation or with other users on the
machine.

---

## Stack

| Layer | Choice |
|---|---|
| Simulator | [MuJoCo](https://mujoco.org) + [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) (GPU-accelerated MuJoCo on NVIDIA Warp) |
| RL framework | [`mjlab`](https://github.com/mujocolab/mjlab) (Isaac-Lab-style API on top of MuJoCo-Warp) |
| G1 tasks / configs | [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) |
| Algorithm | PPO via [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) (v5.0.1) |
| Deploy | ONNX (`onnxruntime-gpu`) + MuJoCo passive viewer + pygame keyboard |
| Experiment tracking | TensorBoard + Weights & Biases |
| Runtime | rootless Docker + `docker compose` + `Makefile` |

Target GPU: **NVIDIA RTX 4090 (24 GB)**, CUDA 12.6, driver 590.x.

---

## Layout

```
otus_rl_project/
├── docker/
│   ├── Dockerfile          # CUDA 12.6 + cuDNN + Python 3.12 + uv / mjlab / unitree_rl_mjlab
│   └── compose.yaml        # services: shell, train, play, export
├── .devcontainer/
│   └── devcontainer.json   # "Reopen in Container" for Cursor / VS Code
├── src/otus_rl_project/    # our Python package (editable install)
│   ├── train/main.py       # otus-train  — RL training entry point
│   ├── deploy/keyboard_play.py  # otus-play   — sim2sim with keyboard
│   └── deploy/export_onnx.py    # otus-export — .pt -> .onnx
├── scripts/
│   └── setup_assets.sh     # pulls unitree_g1 from MuJoCo Menagerie
├── assets/g1/              # MJCF + meshes (not tracked, populated by script)
├── runs/                   # checkpoints, logs, ONNX (not tracked)
├── motions/                # mocap data for motion tracking (not tracked)
├── third_party/            # Menagerie clone and friends (not tracked)
├── pyproject.toml          # otus-rl-project package + dependencies
├── Makefile                # thin wrappers around docker compose
└── README.md
```

---

## Quick start

**Prerequisite:** rootless Docker is already installed on the host for
user `ev` (see `## Rootless Docker setup` below if you need to reproduce
this on another machine).

```bash
# 0. Verify rootless Docker + GPU passthrough work
make check

# 1. Download the G1 MJCF model (~30 MB)
make assets

# 2. Build the main image (~8–10 GB, takes a while on first build)
make build

# 3. Bring up the persistent dev container in the background
make up

# 4. Attach an interactive shell (re-runnable, no restart needed)
make attach

# 5. Kick off training (override the command via CMD=...)
make train CMD="otus-train --task Unitree-G1-Flat --num-envs 4096 --iterations 2000"

# 6. Export the latest checkpoint to ONNX
make export

# 7. Sim2sim deploy with keyboard (requires X11 forwarding)
make play
```

To open the project *inside* the container from Cursor / VS Code:

1. Install the **Dev Containers** extension
   (`ms-vscode-remote.remote-containers`, by Microsoft).
2. In the repo root run: command palette →
   **`Dev Containers: Reopen in Container`**.
   The IDE will read `.devcontainer/devcontainer.json` and open
   `/workspace/otus_rl_project` directly.

If you prefer **`Attach to Running Container`** instead, it also works:
the container sets `HOME=/workspace/otus_rl_project`, so the
"Open Folder" dialog lands in the project directory by default.

---

## Roadmap

- [x] **Rootless Docker**: per-user daemon, data-root in `/home`, CDI GPU passthrough.
- [x] **Project skeleton**: directory layout, pyproject, Dockerfile, compose, Makefile.
- [x] **Image build**: `make build` completes, `make check` sees the GPU.
- [x] **mjlab smoke test**: `python -c "import mjlab; import mujoco_warp"` + demo env.
- [ ] **MVP locomotion**: train `Unitree-G1-Flat` (joystick locomotion) to stable walking.
- [ ] **Sim2sim deploy**: keyboard -> command vector -> ONNX -> MuJoCo viewer.
- [ ] **Motion imitation**: pick one acrobatic motion, train a tracker.
- [ ] **Final demo**: drive the acrobatic motion from the keyboard in MuJoCo.
- [ ] *(optional)* Sim2real on a physical G1 via `unitree_sdk2`.
- [ ] *(optional)* Ablation PPO vs SAC / TD3 via `skrl` / `rl_games`.

---

## Rootless Docker setup (reference)

What has already been done on this machine:

```bash
# system dependencies
sudo apt install -y uidmap fuse-overlayfs dbus-user-session slirp4netns docker-ce-rootless-extras

# allow unprivileged user namespaces (Ubuntu 24.04)
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | \
    sudo tee /etc/sysctl.d/60-apparmor-userns.conf
sudo sysctl --system

# per-user rootless docker daemon
dockerd-rootless-setuptool.sh install

# keep the daemon alive across logouts
sudo loginctl enable-linger $USER

# in ~/.bashrc
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

After this, `docker info` reports `rootless` and
`Docker Root Dir: ~/.local/share/docker`.

GPU access is provided via CDI (Container Device Interface):

```bash
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.6.3-base nvidia-smi
```

Neither edits to `/etc/nvidia-container-runtime/config.toml` nor
membership in the `docker` group are required for rootless mode.

---

## Disk hygiene

- Image: ~10 GB (under `/home/ev/.local/share/docker`).
- Build cache: up to ~15 GB transiently during `make build`. Clean with `make prune`.
- `runs/` grows over time (checkpoints, tensorboard, rollout videos). Prune
  manually or archive to `wandb artifact`.
- `assets/g1` is ~30 MB, `third_party/mujoco_menagerie` ~100 MB (sparse checkout).

`df -h $HOME` should stay above 100 GB free during normal work.
