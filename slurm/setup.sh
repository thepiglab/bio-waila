#!/bin/bash
# One-time (and after every `git pull` that touches pyproject.toml/uv.lock): build or refresh the shared venv on /blue.
# Run on a login node, not inside array jobs.
#   bash slurm/setup.sh                              # core + whole-slide + GPU extras (the default)
#   WAILA_EXTRAS="--extra wsi" bash slurm/setup.sh   # CPU-only environment
set -euo pipefail
WAILA_SETUP=1 source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
uv sync --locked ${WAILA_EXTRAS:---extra wsi --extra gpu}
uv run --no-sync pytest -q          # offline sanity check of the checkout (dev group is included by default)
echo "venv ready: $UV_PROJECT_ENVIRONMENT"
