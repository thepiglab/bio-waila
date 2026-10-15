#!/bin/bash
#SBATCH --job-name=p2-metalabel
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b        # burst QOS (CPU only)
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/metalabel_%A_%a.log
# Step 1 of metadata labelling: profile every downloaded dataset's per-sample metadata (one shard per array task).
# Then, once all shards finish (steps 2-3 need every profile):
#   sbatch --dependency=afterok:<this job id> slurm/metalabel_map.sbatch
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
uv run --no-sync waila-metalabel profile --raw "$WAILA_RAW" --out "$WAILA_PREP" --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
