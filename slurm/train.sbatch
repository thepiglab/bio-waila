#!/bin/bash
#SBATCH --job-name=p2-train
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%x_%j.log
# Teacher training on 1-4 B200s. TRAINER: clm | clm-llm-embed | clm-llm | jev. CONFIG and FOLD as below; extra
# overrides in SET (space-separated key=value, no commas). Multi-GPU runs use accelerate via torchrun (clm, jev).
#   sbatch --export=ALL,TRAINER=clm,CONFIG=configs/clm_siglip2.yaml,FOLD=0 slurm/train.sbatch
#   sbatch --gpus=4 --cpus-per-task=48 --mem=256gb --export=ALL,TRAINER=clm,CONFIG=configs/clm_siglip2.yaml,FOLD=0 slurm/train.sbatch
#   sbatch --array=0-3 --export=ALL,TRAINER=clm-llm-embed,CONFIG=configs/clm_llm_gemma4.yaml slurm/train.sbatch   # 4 embed shards
#   sbatch --export=ALL,TRAINER=jev,CONFIG=configs/jev_gemma4_e4b.yaml,FOLD=0,SET="train.formulation=F3" slurm/train.sbatch
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
: "${TRAINER:?TRAINER=clm|clm-llm-embed|clm-llm|jev}" "${CONFIG:?CONFIG=configs/...yaml}"
FOLD=${FOLD:-0}
NG=${SLURM_GPUS_ON_NODE:-1}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=name,memory.total --format=csv
launch() {  # $1 = module; rest = args
  local mod=$1; shift
  if [ "$NG" -gt 1 ]; then uv run --no-sync torchrun --standalone --nproc_per_node="$NG" -m "$mod" "$@"
  else uv run --no-sync python -m "$mod" "$@"; fi
}
case "$TRAINER" in
  clm)           launch bio_waila.train.train_clm --config "$CONFIG" --fold "$FOLD" ${SET:+--set $SET} ;;
  jev)           launch bio_waila.train.train_jev --config "$CONFIG" --fold "$FOLD" ${SET:+--set $SET} ;;
  clm-llm-embed) N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
                 uv run --no-sync waila-train-clm-llm embed --config "$CONFIG" --shard "$I/$N" ${SET:+--set $SET} ;;
  clm-llm)       uv run --no-sync waila-train-clm-llm train --config "$CONFIG" --fold "$FOLD" ${SET:+--set $SET} ;;
  *) echo "unknown TRAINER $TRAINER" >&2; exit 2 ;;
esac
