#!/bin/bash
#SBATCH --job-name=p2-smoke
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.log
# Run every real teacher for a few steps on a small slice of the prepared manifest, before committing GPU-days:
#   1. CLM-vision  SigLIP 2 So400m: zero-shot eval, a few stage-A/B steps, WiSE-FT, temperatures
#   2. CLM-LLM     Gemma-4-E4B: embed 256 images + label texts, train the heads
#   3. Jev         Gemma-4-E4B + LoRA: a few F2 steps, then a small evaluation (letter readout + tournament)
# Needs the prepared manifest ($WAILA_PREP/manifest.parquet) and HF access to the checkpoints (HF_TOKEN in .env
# if a model is gated). Outputs go to $WAILA_RUNS/smoke_*.
#   cd $WAILA_HOME && sbatch slurm/train_smoke.sbatch
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== 1/3 CLM-vision (SigLIP 2)"
uv run --no-sync waila-train-clm --config configs/clm_siglip2.yaml --limit 3000 --out "$WAILA_RUNS/smoke_clm" \
  --set train.epoch_size=2048 train.epochs_a=1 train.epochs_b=1 train.batch=64 train.max_steps=20 'train.wise_alphas=[0.5,1.0]'
echo "== 2/3 CLM-LLM (Gemma-4-E4B embeddings)"
uv run --no-sync waila-train-clm-llm embed --config configs/clm_llm_gemma4.yaml --limit 256 \
  --set cache="$WAILA_RUNS/smoke_clm_llm_cache" out="$WAILA_RUNS/smoke_clm_llm"
uv run --no-sync waila-train-clm-llm train --config configs/clm_llm_gemma4.yaml \
  --set cache="$WAILA_RUNS/smoke_clm_llm_cache" out="$WAILA_RUNS/smoke_clm_llm" train.epochs=3 train.epoch_size=4096 train.batch=256
echo "== 3/3 Jev (Gemma-4-E4B + LoRA)"
uv run --no-sync waila-train-jev --config configs/jev_gemma4_e4b.yaml --out "$WAILA_RUNS/smoke_jev" \
  --set train.examples_per_epoch=256 train.max_steps=10 train.eval_per_dataset=1 train.grad_accum=1
echo "smoke OK: see $WAILA_RUNS/smoke_*/*/metrics.json"
