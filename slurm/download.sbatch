#!/bin/bash
#SBATCH --job-name=p2-download
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b        # burst QOS: idle cores, low priority, 4-day limit, outside the group's 1,640 GB memory cap
#SBATCH --array=0-15               # shard count = array size (SLURM_ARRAY_TASK_COUNT); change with: sbatch --array=0-7 ...
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=96:00:00
#SBATCH --output=logs/download_%A_%a.log
# Fetch the bounded sample of every dataset in plan/selection.jsonl into $WAILA_RAW/<id>/. Resumable: re-submit to
# continue; finished datasets are skipped.
#   PILOT=128 sbatch slurm/download.sbatch     fast first pass (~50k images); a plain sbatch later tops up to the full plan
# Check first that compute nodes can reach the internet (srun --qos=xuefeng.liu-b curl -sI https://zenodo.org); if not, run
# the same command on a login/data-transfer node:  SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=16 bash slurm/download.sbatch
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
uv run --no-sync waila-download --out "$WAILA_RAW" --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" --max-gb 20 ${PILOT:+--pilot $PILOT}
