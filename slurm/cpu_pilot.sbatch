#!/bin/bash
#SBATCH --job-name=p2-pilot
#SBATCH --account=xuefeng.liu
#SBATCH --qos=xuefeng.liu-b         # burst QOS: CPU only, idle cores, low priority, outside the group's 1,640 GB memory cap
#SBATCH --cpus-per-task=4
#SBATCH --mem=12gb                  # modest default; the `label` stage is overridden to 24gb by the submit helper
#SBATCH --time=24:00:00
#SBATCH --output=logs/pilot_%x_%A_%a.log
#
# CPU-only end-to-end PILOT of the data pipeline (no GPU, no model): ~64 images from each of the 407 planned datasets
# (about 26k images), labelled and prepared into $WAILA_WORK/prepared_pilot. Use it to find broken sources and bad labels
# before the full run. One script, four stages chained with job dependencies:
#
#   fetch    array 0..N-1   waila-download --pilot 64 --shard i/N, then waila-metalabel profile --shard i/N
#   label    one job        waila-metalabel map, then apply          (needs every shard's profile)
#   prepare  array 0..N-1   waila-prepare --pilot 64 --shard i/N --out $WAILA_WORK/prepared_pilot
#   merge    one job        waila-prepare --merge, then summary tables (label status, per-dataset outcomes, conflicts)
#
# One-time: bash slurm/setup.sh   (venv on /blue).   Submit from the checkout:
#
#   cd $WAILA_HOME && bash slurm/cpu_pilot.sbatch submit [N=8]     # runs the four sbatch commands below
#
# or by hand (N = 8 shards; PILOT and MAX_GB are optional, defaults 64 and 5):
#
#   N=8; S=slurm/cpu_pilot.sbatch
#   J1=$(sbatch --parsable --array=0-$((N-1)) --export=ALL,STAGE=fetch  $S)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-0 --mem=24gb --time=4:00:00 --export=ALL,STAGE=label $S)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 --array=0-$((N-1)) --export=ALL,STAGE=prepare $S)
#   sbatch --dependency=afterok:$J3 --array=0-0 --export=ALL,STAGE=merge $S
#
# Resumable: re-submitting a stage skips finished datasets. Datasets that fail to download are recorded in
# $WAILA_RAW/_status/ and skipped later, so the chain still completes; the merge summary lists them.
# Read the result: cat $WAILA_WORK/prepared_pilot/stats.md ; the full-run commands are the same without --pilot.
# Logs: <checkout>/logs/ (submit from the checkout; bash slurm/setup.sh creates it).

if [ "${1:-}" = submit ]; then   # helper: `bash slurm/cpu_pilot.sbatch submit [N]` from the checkout
  N=${2:-8}; S=slurm/cpu_pilot.sbatch
  J1=$(sbatch --parsable --array=0-$((N - 1)) --export=ALL,STAGE=fetch $S)
  J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-0 --mem=24gb --time=4:00:00 --export=ALL,STAGE=label $S)
  J3=$(sbatch --parsable --dependency=afterok:$J2 --array=0-$((N - 1)) --export=ALL,STAGE=prepare $S)
  J4=$(sbatch --parsable --dependency=afterok:$J3 --array=0-0 --export=ALL,STAGE=merge $S)
  echo "pilot chain: fetch=$J1 label=$J2 prepare=$J3 merge=$J4   (watch: squeue -u \$USER)"
  exit 0
fi

# Jobs start from the repo checkout: submit with `cd $WAILA_HOME` first (or export WAILA_HOME in ~/.bashrc).
: "${WAILA_HOME:=${SLURM_SUBMIT_DIR:?submit from the repo checkout or export WAILA_HOME}}"
source "$WAILA_HOME/slurm/env.sh" || exit 1

PILOT=${PILOT:-64}                     # images per dataset
MAX_GB=${MAX_GB:-5}                    # per-dataset download budget
OUT=$WAILA_WORK/prepared_pilot         # never mixes with the full $WAILA_PREP
i=${SLURM_ARRAY_TASK_ID:-0}; N=${SLURM_ARRAY_TASK_COUNT:-1}
set -euo pipefail

case "${STAGE:?set STAGE=fetch|label|prepare|merge (see the header)}" in
  fetch)
    uv run --no-sync waila-download --out "$WAILA_RAW" --pilot "$PILOT" --max-gb "$MAX_GB" --shard "$i/$N"
    uv run --no-sync waila-metalabel profile --raw "$WAILA_RAW" --out "$OUT" --shard "$i/$N"
    ;;
  label)
    uv run --no-sync waila-metalabel map   --out "$OUT"
    uv run --no-sync waila-metalabel apply --out "$OUT"
    echo "values to review (fill class, copy rows into plan/overrides.csv): $OUT/metalabel/review.csv"
    ;;
  prepare)
    uv run --no-sync waila-prepare --raw "$WAILA_RAW" --out "$OUT" --pilot "$PILOT" --shard "$i/$N"
    ;;
  merge)
    uv run --no-sync waila-prepare --out "$OUT" --merge          # manifest.parquet + stats.md (label status per head)
    echo; echo "== per-dataset outcome (prepare logs) =="
    cat "$OUT"/_logs/prepare_shard*.jsonl | uv run --no-sync python -c '
import collections, json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
last = {r["id"]: r for r in rows}                      # a re-run appends: keep the latest record per dataset
print(dict(collections.Counter(r["state"] for r in last.values())))
for st in ("not downloaded", "no media found", "empty"):
    ids = sorted(k for k, r in last.items() if r["state"] == st)
    if ids: print(f"{st} ({len(ids)}):", " ".join(ids))
'
    echo; echo "== download states =="
    uv run --no-sync python -c '
import collections, csv, glob, os
rows = [r for p in glob.glob(os.environ["WAILA_RAW"] + "/_status/status_shard*of*.csv") for r in csv.DictReader(open(p))]
print(dict(collections.Counter(r["state"] for r in rows)) or "no download status files")
'
    echo; echo "== per-dataset label coverage: $OUT/metalabel/apply_summary.csv; conflicts are ranked at the end of stats.md =="
    du -sh "$OUT/images" 2>/dev/null || true
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
