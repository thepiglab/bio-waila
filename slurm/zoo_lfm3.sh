#!/bin/bash
# LFM3 comparison on mmbu-public (docs/models/lfm3.md §4): the two new Liquid models against the incumbents.
#
#   decision:    dec_liquid_d1_2b (new; replaces d1-3B, which is not rerun)
#   generative:  gen_lfm3_vl_2b (new, no thinking)  vs  gen_gemma4_e2b_it, gen_lfm25_vl_3b
#
#   cd $WAILA_HOME && bash slurm/zoo_lfm3.sh 64     # smoke test: 64 public images per model
#   cd $WAILA_HOME && bash slurm/zoo_lfm3.sh        # full mmbu-public
#
# Chain: prefetch (CPU, burst QOS) → one B200 job per model (64 GB each, afterok prefetch) → leaderboard (afterany).
# Results: runs/zoo/<slug>/public_metrics.json and runs/zoo/leaderboard.md (which also lists earlier zoo runs).
# Needs transformers >= 5.19 in the venv (d1-2B's card; LFM3 ships remote code): checked before anything is submitted.
: "${WAILA_HOME:=$(cd "$(dirname "$0")/.." && pwd)}"
source "$WAILA_HOME/slurm/env.sh" || exit 1     # before `set -u`: env.sh reads unset variables (as in zoo_eval.sbatch)
set -euo pipefail
uv run --no-sync python - <<'PY'
import sys, transformers
from packaging.version import Version
v = Version(transformers.__version__)
sys.exit(0 if v >= Version("5.19") else f"transformers {v} < 5.19: d1-2B and LFM3-VL-2B need 5.19 (uv sync, then retry)")
PY
MODELS=dec_liquid_d1_2b,gen_lfm3_vl_2b,gen_gemma4_e2b_it,gen_lfm25_vl_3b \
  bash slurm/zoo_eval.sbatch submit "${1:-}"
