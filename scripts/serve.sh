#!/bin/bash
set -e

# Environment variables for serving the trained policy.

# GPU devices to use.
export CUDA_VISIBLE_DEVICES=4

# Limit JAX memory to avoid OOM.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# ============================================================
# Experiment settings — update these to match your training.
# ============================================================

# Training config name (must match the --config-name used in train.sh).
CONFIG_NAME="pi05_xlerobot"

# Experiment name used during training (--exp_name in train.sh).
EXP_NAME="pick_and_place/29999"

# Base directory where checkpoints are stored (must match checkpoint_base_dir in the config).
# Default for pi05_xlerobot is ./outputs/checkpoints, so usually no need to change.
CHECKPOINT_BASE_DIR="./outputs/checkpoints"

# Full path to the checkpoint directory.
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

# Port for the WebSocket policy server.
PORT=8081

# ============================================================
# Launch the server.
# ============================================================

uv run scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CHECKPOINT_DIR}"
