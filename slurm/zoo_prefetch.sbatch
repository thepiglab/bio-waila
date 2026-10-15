#!/bin/bash
#SBATCH --job-name=zoo-prefetch
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b         # burst QOS: CPU only, idle cores, low priority, outside the group's 1,640 GB memory cap
#SBATCH --cpus-per-task=4
#SBATCH --mem=12gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/zoo_%x_%j.log
#
# CPU-only stages of the model zoo (docs/experiments/zoo.md); the GPU is only used by slurm/zoo_eval.sbatch.
#   STAGE=prefetch     (default)  waila-zoo prefetch   weights → $HF_HOME, plus plan/eval_public.parquet. Gated repos (MedGemma …)
#                                  need the terms accepted on HF by the HF_TOKEN account; failures are listed in the log and do not
#                                  stop the other downloads (the matching GPU job then fails fast with HF_HUB_OFFLINE=1).
#   STAGE=leaderboard             waila-zoo leaderboard   runs/zoo/*/public_metrics.json → runs/zoo/leaderboard.{csv,md}
#
#   cd $WAILA_HOME && sbatch slurm/zoo_prefetch.sbatch                                  # prefetch everything enabled
#   sbatch --export=ALL,MODELS=gen_gemma4_e4b_it,dec_liquid_d1_3b slurm/zoo_prefetch.sbatch
#   sbatch --export=ALL,STAGE=leaderboard slurm/zoo_prefetch.sbatch
# `bash slurm/zoo_eval.sbatch submit [LIMIT]` chains prefetch → one GPU job per model → leaderboard.

: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
set -euo pipefail

case "${STAGE:-prefetch}" in
  prefetch)    uv run --no-sync waila-zoo prefetch ${MODELS:+--models "$MODELS"} ;;
  leaderboard) uv run --no-sync waila-zoo leaderboard && cat "$WAILA_RUNS/zoo/leaderboard.md" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
