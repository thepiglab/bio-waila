#!/bin/bash
#SBATCH --job-name=p2-metamap
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b        # burst QOS (CPU only)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=4:00:00
#SBATCH --output=logs/metalabel_map_%j.log
# Steps 2-3 of metadata labelling: map distinct metadata values to the closed vocabulary, then join labels back to every sample.
# Review $WAILA_PREP/metalabel/review.csv afterwards; put corrections in plan/overrides.csv, commit, pull, and re-submit this job.
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
uv run --no-sync waila-metalabel map   --out "$WAILA_PREP" && \
uv run --no-sync waila-metalabel apply --out "$WAILA_PREP"
