#!/bin/bash
#SBATCH --job-name=p2-b200
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu           # investment QOS: group cap 20 GPUs / 210 CPUs / 1,640 GB; GPUs need it (burst is for CPU work)
#SBATCH --partition=hpg-b200        # verify: sinfo -p hpg-b200 ; sacctmgr show assoc user=$USER format=account,qos
#SBATCH --gpus=1                    # override per job: sbatch --gpus=4 ...  (1 = pilot runs, 4 = DDP/FSDP full runs)
#SBATCH --cpus-per-task=16          # JPEG decode is the bottleneck: ~12-16 CPU workers per GPU
#SBATCH --mem=96gb                  # group memory is the scarce resource: 96 GB per 1-GPU job, 256 GB for a 4-GPU job
#SBATCH --time=24:00:00
#SBATCH --output=logs/b200_%j.log
# Generic GPU launcher. RUN is whatever follows `uv run --no-sync` (an entry point or `python -m ...`); it is eval'd inside the
# job, so single-quote it to keep $VARS for the job. With more than one GPU it runs under torchrun instead: give `-m module args`
# or a script path (no "python"). No commas in RUN (sbatch --export splits on them).
#   sbatch --export=ALL,RUN='waila-autolabel scores --prep $WAILA_PREP' slurm/b200.sbatch
#   sbatch --export=ALL,RUN='python -m bio_waila.train.b0 --config configs/b0_mlp.yaml' slurm/b200.sbatch
#   sbatch --gpus=4 --cpus-per-task=48 --mem=256gb --export=ALL,RUN='-m bio_waila.train.jev --config configs/jev_f2.yaml' slurm/b200.sbatch
# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
: "${RUN:?set RUN, see the header of slurm/b200.sbatch}"
NG=${SLURM_GPUS_ON_NODE:-1}
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$NG" -gt 1 ]; then
  eval "uv run --no-sync torchrun --standalone --nproc_per_node=$NG $RUN"
else
  eval "uv run --no-sync $RUN"
fi
