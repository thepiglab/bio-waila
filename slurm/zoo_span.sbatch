#!/bin/bash
#SBATCH --job-name=zoo-span
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu
#SBATCH --partition=hpg-b200        # PLACEHOLDER: respanai/span-02 needs a compute-capability 12.0 (sm_120a) GPU: RTX PRO 6000 Blackwell /
#                                   # RTX 5090. B200 (sm_100) is NOT supported. Replace partition / gres below with an sm_120 partition.
#SBATCH --gpus=1                    # PLACEHOLDER (see above), e.g. --gres=gpu:<sm_120 type>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/zoo_%x_%j.log
#
# Scores respanai/span-02 (registry slug dec_span_02) on mmbu-public: builds its Rust/CUDA engine from the HF snapshot if needed
# (needs cargo + a CUDA toolkit), starts `launch.sh` (0.0.0.0:8000) in the background, waits for GET /health, runs `waila-zoo run`,
# then stops the server. Not part of `slurm/zoo_eval.sbatch submit`. The snapshot must already be in $HF_HOME:
#   sbatch --export=ALL,MODELS=dec_span_02 slurm/zoo_prefetch.sbatch      # CPU, ~16 GB
#   cd $WAILA_HOME && sbatch slurm/zoo_span.sbatch                       # or --export=ALL,LIMIT=64 for a smoke test
# Env: NVCC (default /usr/local/cuda/bin/nvcc), PORT (default 8000), LIMIT, SPAN_API_KEY (if the server needs a bearer token).

: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1
set -euo pipefail
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PORT=${PORT:-8000}; NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}; export NVCC

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
if [ "$CAP" != "12.0" ]; then
  echo "ERROR: span-02's engine needs compute capability 12.0 (sm_120a: RTX PRO 6000 Blackwell / RTX 5090); this GPU is '$CAP'." >&2
  echo "       Edit the #SBATCH partition / gres at the top of this script to an sm_120 partition (B200 = sm_100 is unsupported)." >&2
  nvidia-smi --query-gpu=name,compute_cap --format=csv >&2
  exit 3
fi
command -v cargo >/dev/null || { echo "ERROR: cargo not found (load a Rust module / install rustup)" >&2; exit 4; }
[ -x "$NVCC" ] || { echo "ERROR: nvcc not found at $NVCC (set NVCC=/path/to/nvcc)" >&2; exit 4; }

SNAP=$(uv run --no-sync python -c "from huggingface_hub import snapshot_download as s; print(s('respanai/span-02', local_files_only=True))")
echo "span-02 snapshot: $SNAP"
cd "$SNAP"
if ! ls engine/target/release/dllm-server >/dev/null 2>&1; then
  echo "building the engine …"
  (cd engine && cargo build --release -p dllm-server)
fi

mkdir -p "$WAILA_HOME/logs"
bash launch.sh > "$WAILA_HOME/logs/span_server_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do   # up to 30 min for the model to load
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || { echo "ERROR: server exited; see logs/span_server_${SLURM_JOB_ID:-local}.log" >&2; exit 5; }
  sleep 10
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "ERROR: /health never came up" >&2; exit 5; }

cd "$WAILA_HOME"
args=(run --model dec_span_02 --base-url "http://127.0.0.1:$PORT")
[ -n "${LIMIT:-}" ] && args+=(--limit "$LIMIT")
uv run --no-sync waila-zoo "${args[@]}"
