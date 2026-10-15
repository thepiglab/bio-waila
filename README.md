# bio-waila

**WAILA** ("What Am I Looking At") is a labeler for the MMBU Pass-2 biomedical-context fields. This repo (Python package `bio_waila`, console scripts `waila-*`) holds the training data and teacher models for an **MMBU Pass-2 labeler**: a closed-vocabulary classifier of biomedical-image metadata. The fields are modality, submodality, body part, stain, specimen and medical domain.

The labeler is a teacher, never the submitted model. Its labels train the end-to-end VLM we submit.

## Approach

- **Two teachers:**
  - a Jev-style decision model on Gemma 4;
  - a CLM-style classifier on SigLIP2.
- **Labels:** auto-labelled from each sample's own metadata, against MMBU's closed vocabulary.

The design is in [docs/OVERVIEW.md](docs/OVERVIEW.md); the training plan in [docs/experiments/training_plan.md](docs/experiments/training_plan.md);
the list of training datasets in [catalog/DATA.md](catalog/DATA.md).

## How it fits together

```
  ../data_discovery registry             plan/registry_fixes.json   (drop · route · labels · exclude_paths)
  5,120 dataset TOMLs                    plan/overrides.csv         (per-value label fixes)
            │                                      ▲   from metadata audits + image-level validation
            ▼                                      │   (docs/data/audits, docs/data/validation)
  ┌─ 1 SELECT ─── waila-select ───────────────────────────────────────────────────────────────┐
  │  420 datasets · ~873k planned images · mix follows the full MMBU benchmark · splits by    │
  │  dataset group · licence groups (permissive / nc, toggle `include_nc`)                    │
  └──────┬───────────────────────────────────────────────────────────┬────────────────────────┘
         │ plan/selection.jsonl, label_space.json                    └──► catalog/DATA.md, datasets.yaml
         ▼
  2 DOWNLOAD ─ waila-download ── bounded sample per dataset: zip range reads, HF parquet row groups,
         │                       tar reservoir, class shards; Kaggle/Zenodo/figshare/HF/OpenNeuro/…
         ▼
  3 LABEL ──── waila-metalabel ─ each sample's own metadata → MMBU's closed vocabulary
         │                       (CSV/JSON columns, folder + file names, DICOM tags, BIDS, HF fields)
         ▼
  4 PREPARE ── waila-prepare ─── decode (2D, DICOM/NIfTI slices, video frames, WSI tiles) → ≤512 px JPEG;
         │                       drop masks, blank frames, near-duplicates of mmbu-public
         ▼
  $WAILA_PREP/manifest.parquet ── one row per image; 6 fields, each hard │ multi │ candidates │ masked
         │
         ├────────────────────────┬────────────────────────┬─────────────────────────┐
         ▼                        ▼                        ▼                         ▼
  CLM-vision (primary)     CLM-LLM (side arm)       Jev                       zero-shot voters
  SigLIP 2 + per-field     frozen Gemma-4-E4B       Gemma-4-E4B + LoRA        Liquid d1-3B,
  heads, FLYP + WiSE-FT    embeddings + CLM heads   lettered options, F2/F3   stock Gemma letters
  (+ Moondream-tower arm)
         └────────────────────────┴────────────┬───────────┴─────────────────────────┘
                                               ▼
                        teacher ensemble → pseudo-labels for masked / candidate fields
                                               ▼
                student VLM (Gemma-4-E2B or LFM2.5-VL-3B) fine-tuned on Pass-2 JSON ──► submission

  Evaluation everywhere: leave-dataset-group-out folds (plan/folds.json), macro-averaged over held-out
  datasets; mmbu-public is evaluation-only (plan/eval_public.parquet) and never trained on.
```

| Where | What runs there | Entry point |
|---|---|---|
| Laptop | selection, catalog, per-dataset validation in a ≤30 GB cache, offline tests | `waila-select`, `waila-validate`, `uv run pytest` |
| Slurm cluster, CPU | full download → metalabel → prepare → merge, sharded | `bash slurm/cpu_pilot.sbatch submit 8`, `slurm/*.sbatch` |
| Slurm cluster, GPU | smoke test, then teacher training, pseudo-labelling, student | `sbatch slurm/train_smoke.sbatch`, `slurm/train.sbatch` |

## Layout

```
bio-waila/
├── pyproject.toml, uv.lock    uv-managed environment (Python 3.12)
├── src/bio_waila/
│   ├── paths.py               every location; override with WAILA_* env vars
│   ├── data/                  dataset assembly and auto-labelling
│   │   ├── vocab.py           label space + normalisers (shared by train and eval)
│   │   ├── rules.py           metadata value → closed-vocabulary matcher, DICOM/BIDS tables, label resolution
│   │   ├── files.py           file kinds, mask/junk filters
│   │   ├── select.py          registry → plan/selection.jsonl            (waila-select)
│   │   ├── download.py        bounded, resumable sampling per source    (waila-download)
│   │   ├── metalabel.py       per-sample metadata → labels              (waila-metalabel)
│   │   ├── prepare.py         decode → ≤512 px JPEG + labels manifest   (waila-prepare)
│   │   ├── eval_set.py        mmbu-public → eval parquet (eval only)    (waila-eval-set)
│   │   ├── autolabel.py       optional GPU fallback labeler             (waila-autolabel)
│   │   ├── cache.py           size-capped local cache                   (waila-cache)
│   │   └── validate.py        local per-dataset pipeline validation     (waila-validate)
│   ├── models/                clm_siglip (CLM-vision), clm_llm (CLM-LLM), jev_gemma (Jev), moondream_vision, zoo_models (zoo backends)
│   ├── train/                 labels, folds, data, losses; train_clm / train_clm_llm / train_jev   (waila-train-*)
│   └── eval/                  metrics macro-averaged over held-out datasets, temperature fitting, mmbu-public scoring (public.py, run.py), model zoo (zoo.py)
├── catalog/                   datasets.yaml + DATA.md: every training dataset (link, size, modality, description); built by scripts/build_catalog.py
├── plan/                      selection.jsonl, label_space.json, report.md (generated); registry_fixes.json, overrides.csv (hand-edited)
├── resources/                 taxonomy, MMBU vocabulary, alias maps, paper Fig. 3 numbers (self-contained copies)
├── configs/                   data.yaml + one YAML per teacher (clm_siglip2, clm_llm_gemma4, jev_gemma4_e4b, …)
├── slurm/                     env.sh, setup.sh, *.sbatch (CPU and GPU jobs)
├── docs/                      index in docs/README.md: overview, data/ (pipeline.md), experiments/, models/
├── scripts/                   build_catalog.py, merge_validation_fixes.py; audits/ (historical, import the old `pass2data`)
├── tests/                     offline pytest suite: matcher, files, vocab, select, pipeline + tiny end-to-end trainer runs
└── data/                      gitignored working data (raw/, prepared/, cache/)
```

## Setup

```bash
uv sync                    # core (CPU) environment
uv sync --extra wsi        # + OpenSlide for whole-slide images
uv sync --extra gpu        # + torch/transformers/open_clip (cluster GPU jobs)
uv run pytest
```

The registry used by `waila-select` lives next to this repo, at `../data_discovery/build/registry.json` (override with `WAILA_REGISTRY`). The public MMBU split is expected at `../mmbu-public` (override with `WAILA_PUBLIC`); it is used **for evaluation only**.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Pick datasets | `uv run waila-select` | `plan/selection.jsonl`, `report.md`, `label_space.json` |
| Fetch bounded samples | `uv run waila-download --shard i/n [--pilot 128]` | `$WAILA_RAW/<id>/` |
| Label from metadata | `uv run waila-metalabel profile\|map\|apply` | `$WAILA_PREP/labels/<id>.parquet`, `metalabel/review.csv` |
| Build images | `uv run waila-prepare --shard i/n` then `--merge` | `$WAILA_PREP/manifest.parquet`, `stats.md` |
| Eval set | `uv run waila-eval-set` | `plan/eval_public.parquet` |

**Corrections:** dataset-level fixes from the audits live in `plan/registry_fixes.json`: drop, route, labels, exclude_paths. Per-value label overrides live in `plan/overrides.csv`. General matcher fixes go in `src/bio_waila/data/rules.py` with a test. Edit those, never the registry. Details: [docs/data/pipeline.md](docs/data/pipeline.md).

## Train the teachers

Plan and rationale: [docs/experiments/training_plan.md](docs/experiments/training_plan.md). Three trainers share the label encoding
(`train/labels.py`), mode-aware losses (`train/losses.py`), dataset-group folds (`train/folds.py`, frozen to `plan/folds.json`) and
metrics macro-averaged over held-out datasets (`eval/metrics.py`):

| Teacher | Code | Config | Command |
|---|---|---|---|
| CLM-vision (SigLIP 2 + per-field heads, FLYP + WiSE-FT) | `models/clm_siglip.py`, `train/train_clm.py` | `configs/clm_siglip2.yaml` | `waila-train-clm --config … --fold 0` |
| CLM-LLM (frozen Gemma-4-E4B embeddings + CLM heads) | `models/clm_llm.py`, `train/train_clm_llm.py` | `configs/clm_llm_gemma4.yaml` | `waila-train-clm-llm embed --shard i/n`, then `train --fold 0` |
| CLM-vision arm: Moondream 3.1 tower | `models/moondream_vision.py` | `configs/clm_moondream31_vision.yaml` | `waila-convert-moondream --out …` once, then `waila-train-clm` |
| Jev (Gemma-4-E4B + LoRA, letter readout, F2/F3) | `models/jev_gemma.py`, `train/train_jev.py` | `configs/jev_gemma4_e4b.yaml` | `waila-train-jev --config … --fold 0` |

On the cluster: first `sbatch slurm/train_smoke.sbatch` (every real model for a few steps), then `slurm/train.sbatch` with
`TRAINER=clm|clm-llm-embed|clm-llm|jev` (1-4 GPUs; header has examples). Outputs land in `$WAILA_RUNS`.
`uv run pytest tests/test_train.py` runs all three end to end on tiny random models (CPU, no downloads).

## Evaluate on mmbu-public

mmbu-public (`../mmbu-public`, or `WAILA_PUBLIC`) is **evaluation only** and is never trained on. Each trainer scores its final
model on it automatically (`eval: {public: true}` in the config; skipped with a warning if the split is absent), and any finished
run can be scored on its own:

```bash
uv run waila-eval-set                                    # once; also built on demand → plan/eval_public.parquet
uv run waila-eval --run runs/clm_siglip2/fold0 [--limit 200] [--out DIR]
```

Both write `public_metrics.json` and `public_predictions.parquet` into the run directory and append the look to
`plan/public_looks.jsonl`. The headline number is the context score **C**: per image, the share of its annotated fields whose
top-1 class is in the target set (a stand-in for Pass 2), averaged over images; it is also split by modality and task type,
next to the usual per-field metrics (macro-averaged over MMBU tasks, and pooled). `waila-eval` supports CLM-vision, CLM-LLM
(needs the public images embedded by `embed`) and Jev (config saved as `config.json`).

## Benchmark off-the-shelf models (model zoo)

`waila-zoo` scores many released models zero-shot on mmbu-public Pass 2 and ranks them with the same context score C and
field metrics as the trained teachers (details: [docs/experiments/zoo.md](docs/experiments/zoo.md)). The registry is
`configs/zoo.yaml` (slug, HF id, mode, resources, licence/gating notes).

```bash
uv run waila-zoo list                                  # registry with enabled flags
uv run waila-zoo prefetch [--models a,b]               # CPU: download weights (HF_TOKEN) + build plan/eval_public.parquet
uv run waila-zoo run --model gen_medgemma_4b_it --limit 64    # GPU: smoke test; drop --limit for the full split
uv run waila-zoo leaderboard                           # runs/zoo/*/public_metrics.json → runs/zoo/leaderboard.{csv,md}
bash slurm/zoo_eval.sbatch submit [LIMIT]              # cluster: prefetch (CPU) → one GPU job per model → leaderboard (CPU)
```

Two modes: `generative` (image + the official Pass-2 prompt, greedy decode, parse the JSON, map strings onto the label space;
unmappable answers count as wrong) and `decision` (Jev-style lettered options per field, read from next-token logits;
Liquid d1 through its own `system_one` API). Gated models (MedGemma) need their terms accepted on Hugging Face first.

## Validate locally before using the cluster

```bash
uv run waila-validate --only cystods,papila --worker w1 --pilot 100 --max-gb 3
uv run waila-cache status          # stays under 30 GB (WAILA_CACHE_CAP_GB)
uv run waila-validate --summary    # table over every report in data/cache/reports/
```

For each dataset, `waila-validate`:
1. downloads a small sample;
2. runs metadata labelling and preparation;
3. writes `data/cache/reports/<id>.json`, with the metadata columns used, label modes and sources, conflicts, and sample images to check by eye;
4. deletes the raw download.

Nothing new is downloaded while the cache is over its cap.

## On a Slurm cluster

```bash
git clone <repo> && cd bio-waila
bash slurm/setup.sh                       # once: uv venv in the checkout (and after any pyproject.toml / uv.lock change)
bash slurm/cpu_pilot.sbatch submit 8      # CPU pilot, ~64 images per dataset: download, metalabel, prepare, merge
```

Submit from the checkout. Every job sources `slurm/env.sh` (paths; `data/`, `runs/`, `logs/` and `.venv/` live in the checkout and are gitignored) and runs `uv run --no-sync waila-<command>`. Set the account, QOS and partition in the `#SBATCH` headers for your cluster. The full-run chain (`download`, `metalabel`, `metalabel_map`, `prepare`, `autolabel`, `b200`) is described in [docs/data/pipeline.md](docs/data/pipeline.md) and [docs/experiments/b200_plan.md](docs/experiments/b200_plan.md). Update with `git pull`.

## Contributing

- `uv run pytest` runs the offline suite in a couple of seconds (no network, no registry, no real data). Add a regression case to `tests/` with every matcher or file-filter fix.
- Where things go: hand-edited plan data in `plan/`; run configs in `configs/`; one-off analysis in `scripts/`; documents in `docs/` (listed in `docs/README.md`).
- Do not commit anything under `data/`, tokens, or run outputs (see `.gitignore`).

## Credentials and data handling

- **All secrets go in `.env`:** `cp .env.example .env && chmod 600 .env`, then fill in `HF_TOKEN` and `KAGGLE_API_TOKEN`. It is gitignored and loaded automatically by the Python commands and the Slurm jobs. Variables already set in your shell take precedence.
- **Kaggle:** `KAGGLE_API_TOKEN` (KGAT_… token). `~/.kaggle/access_token` still works as a fallback.
- **Hugging Face:** `HF_TOKEN` (optional for downloads, needed for gated checkpoints).
- **Mendeley Data** blocks scripts: download in a browser into `$WAILA_RAW/<id>/archives/`.
- **No secrets, data or caches** are ever committed (see `.gitignore`).
- **mmbu-public** is evaluation-only. `prepare.py` drops near-duplicates of its images from training.
