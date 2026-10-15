#!/bin/bash
# One-time (and after every `git pull` that touches pyproject.toml/uv.lock): build or refresh the shared venv on /blue.
# Best run as a short interactive job (it imports torch and runs CPU tests):
#   srun --qos=xuefeng.liu-b --cpus-per-task=4 --mem=16gb --time=1:00:00 --pty bash slurm/setup.sh
#   bash slurm/setup.sh                              # core + whole-slide + GPU extras (the default)
#   WAILA_EXTRAS="--extra wsi" bash slurm/setup.sh   # CPU-only environment
set -euo pipefail
WAILA_SETUP=1 source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
uv sync --locked ${WAILA_EXTRAS:---extra wsi --extra gpu}
# Offline sanity check. Cap BLAS/torch threads: login nodes expose many cores but allow only a few, and an
# uncapped torch oversubscribes them (the CPU training tests in tests/test_train.py then crawl).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}} MKL_NUM_THREADS=${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
uv run --no-sync pytest -q --durations=5   # the first torch import from shared storage can take a few minutes
echo "venv ready: $UV_PROJECT_ENVIRONMENT"
