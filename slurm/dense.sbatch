#!/bin/bash
#SBATCH --job-name=p2-dense
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b        # burst QOS (CPU only)
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/dense_%A_%a.log
# Dense-annotation sidecars (masks / boxes / points) for the prepared images: src/bio_waila/data/dense.py.
# Needs prepare.sbatch to have finished. Per shard: fetch the annotation files paired with the media already in $WAILA_RAW
# (network: re-lists sources, reads zip directories, re-streams HF rows), then build the sidecars. After all shards:
#   sbatch --dependency=afterok:<this job id> --export=ALL,MERGE=1 --array=0-0 slurm/dense.sbatch
# ALL_SPATIAL=1 also covers datasets that may train teachers only; SKIP_FETCH=1 builds from what is on disk.
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
if [ -n "$MERGE" ]; then
  uv run --no-sync waila-dense build --out "$WAILA_PREP" --merge
else
  [ -n "$SKIP_FETCH" ] || uv run --no-sync waila-dense fetch --raw "$WAILA_RAW" --out "$WAILA_PREP" --shard "$SHARD" ${ALL_SPATIAL:+--all-spatial}
  uv run --no-sync waila-dense build --raw "$WAILA_RAW" --out "$WAILA_PREP" --shard "$SHARD" ${ALL_SPATIAL:+--all-spatial}
fi
