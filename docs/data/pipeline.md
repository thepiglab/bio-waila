# Data pipeline: a maintainer's walkthrough

How a registry entry becomes a labelled training image, and where to fix things when it goes wrong. Commands are run with `uv run` from the repo root; on a Slurm cluster the same commands run through `slurm/*.sbatch` (see [Running on a Slurm cluster](#running-on-a-slurm-cluster)).

```
registry.json ──waila-select──▶ plan/selection.jsonl ──waila-download──▶ $RAW/<id>/
                                  plan/label_space.json                      │
                                                                             ▼
        $PREP/manifest.parquet ◀──waila-prepare──▶ $PREP/images/   ◀──waila-metalabel profile│map│apply
                 stats.md              (--merge)    (≤512 px JPEG)          $PREP/labels/<id>.parquet
```

Every location is in `src/bio_waila/paths.py` and can be overridden with a `WAILA_*` environment variable (`WAILA_RAW`, `WAILA_PREP`, `WAILA_PLAN_DIR`, `WAILA_REGISTRY`, `WAILA_PUBLIC`, ...). Nothing under `data/` is ever committed.

## The five steps

| # | Step | Command | Reads | Writes |
|---|---|---|---|---|
| 1 | **Select** | `waila-select` | registry (`$WAILA_REGISTRY`), `plan/registry_fixes.json`, `docs/data/audits/audit_*.jsonl` | `plan/selection.jsonl`, `label_space.json`, `report.md`, `disclosure.csv` |
| 2 | **Download** | `waila-download --shard i/n [--pilot 128]` | `plan/selection.jsonl` | `$RAW/<id>/` (`_status.json`, `_files.jsonl`, `archives/`, `hf_rows.jsonl`), `$RAW/_status/*.csv` |
| 3 | **Metalabel** | `waila-metalabel profile \| map \| apply` | `$RAW/<id>/`, `plan/overrides.csv` | `$PREP/metalabel/<id>/`, `value_map.csv`, `review.csv`, `$PREP/labels/<id>.parquet` |
| 4 | **Prepare** | `waila-prepare --shard i/n`, then `waila-prepare --merge` | `$RAW`, `$PREP/labels/`, mmbu-public (contamination filter only) | `$PREP/images/<id>/*.jpg`, `manifest/<id>.parquet`, `manifest.parquet`, `stats.md` |
| 5 | **Validate** | `waila-validate --only <id>` / `--summary` | the plan | `data/cache/reports/<id>.json`, `summary.csv` |

1. **Select** (offline). Filters the registry to datasets we may train on (license tier A, open or registration access, an MMBU modality, image data, a scriptable route), picks the purest datasets per MMBU submodality up to an image budget, and splits train/val by dataset group. Each plan row carries the dataset-level label target per field: `hard` (one value), `multi` (all values apply, domains), `candidates` (one of these, unknown which) or `masked` (no usable label). Corrections from `registry_fixes.json` are applied here.
2. **Download**. A bounded, resumable sample per dataset (`plan_images` files, at most `--max-gb`). Routes: HF, Zenodo, Figshare, Kaggle, OpenNeuro, Dataverse, direct URL; Mendeley and some Kaggle competitions must be fetched by hand into `$RAW/<id>/archives/`. Re-running continues; finished datasets are skipped.
3. **Metalabel**. Datasets already say what each image is (DICOM tags, CSV/JSON/HF columns, BIDS names, folder names), just not in MMBU's words. `profile` collects every metadata column and its distinct values per dataset; `map` maps each *distinct value* to a class with `rules.match_value` (a column is trusted only if at least 60% of its rows map to one field); `apply` joins the maps back to every file and reconciles them with the dataset-level label. The registry's `hard` and `multi` labels are never overridden: a disagreeing strong hit is logged as a `conflict` instead, which is how bad registry entries are found. `review.csv` lists frequent unmapped values in columns that look like Pass-2 fields.
4. **Prepare**. Extracts archives, finds media (images, DICOM, NIfTI, video, WSI, arrays; masks and macOS junk are skipped by `files.py`), samples slices/frames/tiles, writes 8-bit JPEGs with a long side of at most 512 px, drops duplicates and anything near-identical to an mmbu-public image, and attaches the labels from step 3 (falling back to header and path rules if metalabel was not run). `--merge` concatenates the per-dataset manifests and writes `stats.md`.
5. **Validate**. The same steps one dataset at a time inside a size-capped local cache (`waila-cache status`), for checking datasets on a laptop before using the cluster. It writes a per-dataset report: columns used, label modes and sources, conflicts, sample images. The raw download is deleted afterwards.

## Where does a fix go?

Pick the narrowest place that fixes the problem.

| Symptom | Fix in | Takes effect |
|---|---|---|
| Whole dataset is unusable, mislabelled at dataset level, has a wrong download ref, or contains extra folders that are not the dataset | `plan/registry_fixes.json`: `drop`, `route`, `labels`, `exclude_paths`, `force_include`, `reaudit` | re-run `waila-select` (then commit the regenerated `plan/`) |
| One metadata *value* maps to the wrong class (or an unmapped value needs a class) in one dataset or column | `plan/overrides.csv` (`dataset_id,column,value,field,class`; `*` is a wildcard for dataset or column; overrides always win) | re-run `waila-metalabel map` and `apply` |
| A keyword or code misfires across many datasets (`mammo` hits "Mammosomatotroph", `us` hits "ASC-US") | `src/bio_waila/data/rules.py`, plus a regression case in `tests/test_rules.py` | re-run `map` and `apply` |
| Masks, overlays or junk files are read as images, or real images are skipped | `src/bio_waila/data/files.py`, plus `tests/test_files.py` | re-run `profile`, `map`, `apply` and `prepare` |
| A label spelling is not canonical (`HES` vs `h&e`, a new species alias) | `src/bio_waila/data/vocab.py` or `resources/public_aliases.json`, plus `tests/test_vocab.py` | re-run `waila-select` and metalabel |
| Class lists or MMBU vocabulary change | `resources/` (`taxonomy.toml`, `pass2_vocabulary.json`, ...) | re-run `waila-select` |

Rules of thumb: never edit the registry (it lives in the sibling `data_discovery` workspace); per-dataset knowledge goes in the two data files (`registry_fixes.json`, `overrides.csv`), general knowledge goes in code with a test. Every `registry_fixes.json` entry should say why (the `drop` and `reaudit` values are the reason strings).

## Adding a dataset

1. Add it to the registry in `data_discovery` and rebuild `registry.json` (or, to try a dataset the registry already has but select skips, use `force_include`, and `route` for a manual download ref).
2. `uv run waila-select`. Check the row in `plan/selection.jsonl` (route, `plan_images`, `labels` per field) and `plan/report.md`.
3. Validate it alone: `uv run waila-validate --only <id> --pilot 100 --max-gb 3`. Open `data/cache/reports/<id>.json`: are the right columns trusted, are the label modes sensible, do the sample images match their labels, are there conflicts?
4. Fix what the report shows, using the table above, and re-validate with `--redo`.
5. Run `uv run pytest`, then commit the changes under `plan/`, `src/` and `tests/`.

## Running on a Slurm cluster

One-time on a login node: `git clone`, then `bash slurm/setup.sh` (builds the venv in the checkout with `uv sync --locked`). Afterwards, from the checkout, submit the scripts in `slurm/`: all of them `source slurm/env.sh` and call `uv run --no-sync waila-<command>`. CPU jobs use a low-priority burst QOS, GPU jobs the group's GPU QOS and partition (set in the `#SBATCH` headers).

- Pilot (about 64 images per dataset, CPU only, writes `$WAILA_WORK/prepared_pilot`): `bash slurm/cpu_pilot.sbatch submit 8`.
- Full run: `sbatch slurm/download.sbatch`, then `slurm/metalabel.sbatch` (array), `slurm/metalabel_map.sbatch`, `slurm/prepare.sbatch` (array), and the merge (`MERGE=1`, see the script header). Chain them with `--dependency=afterok:<jobid>`.
- Pick up a code or plan change with `git pull` (and `bash slurm/setup.sh` if `pyproject.toml` or `uv.lock` changed).

## Behaviour changes on 2026-10-07 (validation round R/N)

**Download**
- **Big zips are read remotely, not downloaded.** A zip over 200 MB is sampled member by member through HTTP range reads whenever the sample is under 30% of the zip's bytes, with 2× spare members kept for prepare's filters. The whole archive is downloaded only for small zips or when range reads fail. This cut validation downloads from 1.5–2 GB to tens of MB per dataset.
- **Class shards:** 3 or more comparable image archives (often one per class) are treated as shards, and the quota is spread across them (mll23: 17 class zips).
- **Tars:** reservoir sample over the first 10× n media members, instead of the first n.
- **Hugging Face parquet:** sampled from evenly spaced row groups across all files. A shuffle buffer only saw the head of class-sorted datasets.
- **Resilience:**
  - downloads resume after IncompleteRead;
  - range reads and zip opens back off on 403/429/503 (dataverse.no);
  - the byte budget counts extracted size, including `.mhd` data files.

**Prepare**
- **mmbu-public near-duplicates:** detected with a 256-bit dHash ≤ 36 plus a 32×32 thumbnail correlation ≥ 0.9, measured at 98.7% recall and 0.3% false positives. The old 64-bit dHash ≤ 4 dropped real images in 117 datasets.
- **Image reading:**
  - 16-bit multichannel TIFFs go through tifffile;
  - EXIF rotation is applied;
  - `.cri` is read as TIFF;
  - images more than 8:1 are dropped;
  - the blank test uses the brightest channel, so dim fluorescence is kept;
  - instance-label maps are treated as masks.
- **NIfTI:** slices along the acquisition axis when axial slices would be a strip, then resampled to square pixels.

**Metalabel**
- **Tables in sampled downloads:** a table qualifies if it covers ≥ 50% of the downloaded files, not only if ≥ 20% of its rows match. A sampled download can match only a few percent of a full-dataset table.
- **Overrides:**
  - Column ids are matched without their download container (`extracted/<archive>/`, `zip_members/<zip>/`, `files/`), so overrides survive a change of download strategy.
  - A hand override is not limited to the registry's candidate list.
  - An override for one field no longer blocks automatic matching of the other fields.
  - Overrides on 0/1 indicator columns are applied.
- **Indicator columns:** 0/1 columns named after a measurement (`*_area`, `*_size`, `*_volume`) are not indicators.
- **DICOM:** a plain `Modality=CT` no longer implies a CT subtype.
