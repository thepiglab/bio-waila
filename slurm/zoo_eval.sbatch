#!/bin/bash
#SBATCH --job-name=zoo-eval
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu           # investment QOS: group cap 20 GPUs / 210 CPUs / 1,640 GB; GPUs need it (burst is for CPU work)
#SBATCH --partition=hpg-b200        # same account / QOS / partition as slurm/b200.sbatch
#SBATCH --gpus=1                    # every zoo model fits one B200 (192 GB) in bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb                  # the submit helper overrides --mem / --gpus per model from configs/zoo.yaml
#SBATCH --time=12:00:00
#SBATCH --output=logs/zoo_%x_%j.log
#
# GPU stage of the model zoo (docs/experiments/zoo.md): scores ONE model on mmbu-public → $WAILA_RUNS/zoo/<slug>/.
# Weights must already be in $HF_HOME (slurm/zoo_prefetch.sbatch); the job runs with HF_HUB_OFFLINE=1.
#
#   cd $WAILA_HOME && bash slurm/zoo_eval.sbatch submit [LIMIT]
#       prefetch (CPU) → one GPU job per enabled model (own --mem / --gpus / job name, afterok prefetch) → leaderboard (CPU, afterany)
#       LIMIT (optional): score a random subsample of that many public images, for a smoke test
#   sbatch --export=ALL,MODEL=gen_medgemma_4b_it,LIMIT=64 slurm/zoo_eval.sbatch             # one model by hand
#   sbatch --array=0-22 slurm/zoo_eval.sbatch                                                 # or an array over the enabled models
#                                                                                             # (index into `waila-zoo list --porcelain`)

if [ "${1:-}" = submit ]; then
  : "${WAILA_HOME:=$(cd "$(dirname "$0")/.." && pwd)}"
  source "$WAILA_HOME/slurm/env.sh" || exit 1
  LIMIT=${2:-}; S=slurm/zoo_eval.sbatch
  JP=$(sbatch --parsable slurm/zoo_prefetch.sbatch)
  IDS=""
  while IFS=$'\t' read -r slug gpus mem; do
    J=$(sbatch --parsable --job-name="zoo-$slug" --dependency=afterok:$JP --gpus="$gpus" --mem="${mem}gb" \
        --export=ALL,MODEL="$slug",LIMIT="$LIMIT" $S)
    IDS="$IDS:$J"
  done < <(uv run --no-sync waila-zoo list --porcelain)
  JL=$(sbatch --parsable --dependency=afterany$IDS --export=ALL,STAGE=leaderboard slurm/zoo_prefetch.sbatch)
  echo "zoo: prefetch=$JP  eval=${IDS#:}  leaderboard=$JL   (watch: squeue -u \$USER; result: runs/zoo/leaderboard.md)"
  exit 0
fi

# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
if [ -z "${MODEL:-}" ]; then   # array mode: task i = i-th enabled model
  MODEL=$(uv run --no-sync waila-zoo list --porcelain | sed -n "$((${SLURM_ARRAY_TASK_ID:?set MODEL or submit an array} + 1))p" | cut -f1)
  : "${MODEL:?array index past the end of the enabled models}"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
args=(run --model "$MODEL")
[ -n "${LIMIT:-}" ] && args+=(--limit "$LIMIT")
uv run --no-sync waila-zoo "${args[@]}"
