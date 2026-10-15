#!/bin/bash
#SBATCH --job-name=p2-autolabel
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu          # investment QOS: GPUs need it
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/autolabel_%j.log
# OPTIONAL layer 3 after prepare + merge: model-based pseudo-labels for fields metadata could not settle.
# Needs the venv built with the gpu extra (bash slurm/setup.sh does this by default).
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
uv run --no-sync waila-autolabel scores    --prep "$WAILA_PREP"
uv run --no-sync waila-autolabel scores    --prep "$WAILA_PREP" --eval
uv run --no-sync waila-autolabel calibrate --prep "$WAILA_PREP"
uv run --no-sync waila-autolabel vlm       --prep "$WAILA_PREP" --max-rows 150000
uv run --no-sync waila-autolabel apply     --prep "$WAILA_PREP"
