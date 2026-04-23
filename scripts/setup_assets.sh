#!/usr/bin/env bash
# Download Unitree G1 MJCF model from MuJoCo Menagerie and place it in assets/g1.
# Menagerie is ~1 GB total; we only need ~30 MB for G1, so we use sparse-checkout.
#
# Usage:   bash scripts/setup_assets.sh
# Idempotent: re-running is a no-op.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
ASSETS="${ROOT}/assets/g1"

MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF="${MENAGERIE_REF:-main}"

mkdir -p "${THIRD_PARTY}" "${ASSETS}"

if [[ ! -d "${THIRD_PARTY}/mujoco_menagerie/.git" ]]; then
    echo "[assets] cloning MuJoCo Menagerie (sparse, only unitree_g1) ..."
    git -C "${THIRD_PARTY}" clone --depth 1 --branch "${MENAGERIE_REF}" \
        --filter=blob:none --no-checkout "${MENAGERIE_URL}"
    git -C "${THIRD_PARTY}/mujoco_menagerie" sparse-checkout init --cone
    git -C "${THIRD_PARTY}/mujoco_menagerie" sparse-checkout set unitree_g1
    git -C "${THIRD_PARTY}/mujoco_menagerie" checkout "${MENAGERIE_REF}"
else
    echo "[assets] Menagerie already cloned, pulling latest ..."
    git -C "${THIRD_PARTY}/mujoco_menagerie" fetch --depth 1 origin "${MENAGERIE_REF}"
    git -C "${THIRD_PARTY}/mujoco_menagerie" checkout "${MENAGERIE_REF}"
    git -C "${THIRD_PARTY}/mujoco_menagerie" pull --ff-only
fi

SRC="${THIRD_PARTY}/mujoco_menagerie/unitree_g1"
if [[ ! -d "${SRC}" ]]; then
    echo "[assets] ERROR: ${SRC} not found in cloned repo." >&2
    exit 1
fi

echo "[assets] syncing G1 assets into ${ASSETS}"
rsync -a --delete --exclude='.git' "${SRC}/" "${ASSETS}/"

echo "[assets] done."
echo "[assets] Key files:"
ls -lh "${ASSETS}"/*.xml 2>/dev/null || true
