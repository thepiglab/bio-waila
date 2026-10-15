#!/bin/bash
#SBATCH --job-name=p2-prepare
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b        # burst QOS (CPU only)
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=96:00:00
#SBATCH --output=logs/prepare_%A_%a.log
# Decode, sample and convert images (<= 512 px JPEG), attach metalabel labels, drop near-duplicates of mmbu-public.
# Needs the labels from metalabel.sbatch + metalabel_map.sbatch. After all shards, merge (writes manifest.parquet + stats.md):
#   sbatch --dependency=afterok:<this job id> --export=ALL,MERGE=1 --array=0-0 slurm/prepare.sbatch
# PILOT=128 sbatch slurm/prepare.sbatch  -> writes $WAILA_WORK/prepared_pilot instead of the full $WAILA_PREP
# (for the whole pilot chain in one go use slurm/cpu_pilot.sbatch)
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
OUT=${PILOT:+$WAILA_WORK/prepared_pilot}; OUT=${OUT:-$WAILA_PREP}
if [ -n "$MERGE" ]; then
  uv run --no-sync waila-prepare --out "$OUT" --merge
else
  uv run --no-sync waila-prepare --raw "$WAILA_RAW" --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" ${PILOT:+--pilot $PILOT}
fi
