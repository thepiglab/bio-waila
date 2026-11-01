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
# Models with their own serving engine (backend: systemone, i.e. dec_span_02) are not listed by `waila-zoo list --porcelain`
# and so are skipped here: use slurm/zoo_span.sbatch.
# Weights must already be in $HF_HOME (slurm/zoo_prefetch.sbatch); the job runs with HF_HUB_OFFLINE=1.
#
#   cd $WAILA_HOME && bash slurm/zoo_eval.sbatch submit [LIMIT]
#       every enabled model through `waila-zoo lanes`: LANES (default 4) lanes, each running one group of models at a time as
#       download (CPU) → one GPU job per model (own --mem / --gpus / job name) → delete the weights (CPU); then the next group.
#       Weights never pile up (at most one group per lane on disk) and at most LANES GPUs are in use. Leaderboard (CPU) at the end.
#       LIMIT (optional): score a random subsample of that many public images, for a smoke test
#   MODELS=a,b,c bash slurm/zoo_eval.sbatch submit [LIMIT]
#       the same chain for those slugs only, in that order (priority first); unknown or disabled slugs stop the submit
#   ACCOUNT=cai6734 CPU_ACCOUNT=cai5731 LANES=4 MODELS=... bash slurm/zoo_eval.sbatch submit
#       other allocations: GPU jobs on ACCOUNT's QOS, CPU jobs on CPU_ACCOUNT's regular QOS (default: ACCOUNT; no burst)
#   bash slurm/zoo_lfm3.sh [LIMIT]       LFM3 comparison: d1-2B, LFM3-VL-2B vs Gemma-4-E2B, LFM2.5-VL-3B
#   sbatch --export=ALL,MODEL=gen_medgemma_4b_it,LIMIT=64 slurm/zoo_eval.sbatch             # one model by hand (weights
#                                                                                             # downloaded first, deleted after)
#   sbatch --array=0-22 slurm/zoo_eval.sbatch                                                 # or an array over the enabled models
#                                                                                             # (index into `waila-zoo list --porcelain`;
#                                                                                             # needs every model downloaded: re-runs only)

if [ "${1:-}" = submit ]; then
  : "${WAILA_HOME:=$(cd "$(dirname "$0")/.." && pwd)}"
  source "$WAILA_HOME/slurm/env.sh" || exit 1
  LIMIT=${2:-}; S=slurm/zoo_eval.sbatch; P=slurm/zoo_prefetch.sbatch
  # ACCOUNT=<group>: GPU jobs on its QOS; CPU_ACCOUNT=<group> (default ACCOUNT): CPU jobs on its regular QOS, not burst.
  # Neither set: the #SBATCH lines of the two scripts.
  CPU=(); GPU=(); CA=${CPU_ACCOUNT:-${ACCOUNT:-}}
  [ -n "${ACCOUNT:-}" ] && GPU=(--account="$ACCOUNT" --qos="$ACCOUNT")
  [ -n "$CA" ] && CPU=(--account="$CA" --qos="$CA")
  PLAN=$(uv run --no-sync waila-zoo lanes ${MODELS:+--models "$MODELS"} --lanes "${LANES:-4}") || exit 1
  # Each lane takes its groups one after another: download (CPU) → score each model (GPU) → delete the weights (CPU).
  # A group's slugs reach the CPU jobs as MODELS in the environment: Slurm splits --export=VAR=a,b on commas.
  TAILS=""
  for lane in $(cut -f1 <<<"$PLAN" | sort -nu); do
    prev=""
    for group in $(awk -F'\t' -v l="$lane" '$1 == l {print $2}' <<<"$PLAN" | uniq); do
      rows=$(awk -F'\t' -v g="$group" '$2 == g' <<<"$PLAN")
      slugs=$(cut -f3 <<<"$rows" | paste -sd, -)
      first=${slugs%%,*}
      JD=$(MODELS=$slugs sbatch --parsable "${CPU[@]}" --job-name="zoo-get-$first" ${prev:+--dependency=afterany:$prev} \
           --export=ALL,STAGE=prefetch,STRICT=1 $P)
      last=""
      while IFS=$'\t' read -r _ _ slug gpus mem; do
        # a failed download cancels the group's GPU jobs (instead of leaving them pending), so the lane moves on
        last=$(sbatch --parsable "${GPU[@]}" --job-name="zoo-$slug" --dependency="afterok:$JD${last:+,afterany:$last}" \
               --kill-on-invalid-dep=yes --gpus="$gpus" --mem="${mem}gb" --export=ALL,MODEL="$slug",LIMIT="$LIMIT" $S)
      done <<<"$rows"
      prev=$(MODELS=$slugs sbatch --parsable "${CPU[@]}" --job-name="zoo-del-$first" --dependency=afterany:$last \
             --export=ALL,STAGE=purge $P)
      echo "zoo: lane $lane  $slugs  download=$JD  last score=$last  delete=$prev"
    done
    TAILS="$TAILS:$prev"
  done
  JL=$(sbatch --parsable "${CPU[@]}" --job-name=zoo-leaderboard --dependency=afterany$TAILS --export=ALL,STAGE=leaderboard $P)
  echo "zoo: leaderboard=$JL   (watch: squeue -u \$USER; result: runs/zoo/leaderboard.md)"
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
