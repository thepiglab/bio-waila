# Shared settings for every bio-waila Slurm job on HiPerGator. `source` it; never execute it.
#
#   One-time setup, on a login node:   cd <repo checkout> && bash slurm/setup.sh      (builds the venv on /blue)
#   Every sbatch script starts with:   source "$WAILA_HOME/slurm/env.sh"              (WAILA_HOME = the repo checkout)
#
# Always submit from the checkout (`cd $WAILA_HOME`): #SBATCH --output=logs/... is relative to the submit directory.
# Layout: <checkout>/{data,runs,logs,.venv} are gitignored; mmbu-public/ and data_discovery/ sit next to the checkout.
# Every path below can be overridden by exporting the variable first (that is how the scripts are tested off-cluster).
#
# Group xuefeng.liu: 20 GPUs / 210 CPUs / 1,640 GB investment cap (shared with the lab).
#   CPU jobs: --qos=xuefeng.liu-b (burst: idle cores, low priority, 4-day limit, outside the 1,640 GB memory cap).
#   GPU jobs: --qos=xuefeng.liu --partition=hpg-b200 (investment QOS; burst does not give GPUs).
# Credentials: $WAILA_HOME/.env (copy .env.example, chmod 600); loaded below. Never put secrets in this file.

export WAILA_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # the git checkout
if [ -f "$WAILA_HOME/.env" ]; then set -a; . "$WAILA_HOME/.env"; set +a; fi    # secrets (HF_TOKEN, KAGGLE_API_TOKEN), gitignored
export WAILA_WORK=${WAILA_WORK:-$WAILA_HOME/data}                         # everything big lives in the checkout, gitignored
export WAILA_DATA=${WAILA_DATA:-$WAILA_WORK}                                # paths.py reads WAILA_DATA/RAW/PREP/CACHE
export WAILA_RAW=${WAILA_RAW:-$WAILA_WORK/raw}
export WAILA_PREP=${WAILA_PREP:-$WAILA_WORK/prepared}
export WAILA_PUBLIC=${WAILA_PUBLIC:-$WAILA_HOME/../mmbu-public}           # eval-only copy of the public split (sibling of the checkout)
export WAILA_REGISTRY=${WAILA_REGISTRY:-$WAILA_HOME/../data_discovery/build/registry.json}  # waila-select only
export WAILA_RUNS=${WAILA_RUNS:-$WAILA_HOME/runs}                          # training outputs and checkpoints (gitignored)
export HF_HOME=${HF_HOME:-$WAILA_WORK/hf_cache}                            # keep HF caches off the 40 GB home quota
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WAILA_WORK/uv_cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$WAILA_HOME/.venv} # repo-local venv (gitignored), off the $HOME quota
export PATH="$HOME/.local/bin:$PATH"                                        # where the uv installer puts uv
command -v module >/dev/null 2>&1 && module load 7zip/24.09 >/dev/null 2>&1  # RAR/7z extraction backend for rarfile (skipped off-cluster)

mkdir -p "$WAILA_RAW" "$WAILA_PREP" "$WAILA_RUNS" "$WAILA_HOME/logs" "$HF_HOME"
command -v uv >/dev/null || { echo "bio-waila env: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; return 1 2>/dev/null || exit 1; }
cd "$WAILA_HOME" || return 1 2>/dev/null || exit 1

# Jobs never sync: concurrent `uv sync` from many array tasks would race on the shared venv. They run
# `uv run --no-sync waila-<command>` against the venv that slurm/setup.sh built.
if [ -z "$WAILA_SETUP" ] && [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/waila-select" ]; then
  echo "bio-waila env: no venv at $UV_PROJECT_ENVIRONMENT. Run once on a login node: bash $WAILA_HOME/slurm/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
