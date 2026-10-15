# QA pass (repo restructure)

**Changed**
- **Tests** (`uv run pytest`: 185 pass, 3 xfail, about 2 s, offline): `test_rules.py` (audit regressions: Not A Brain, Fundus_Cardia, CT angiography, Mammosomatotroph, MR/T2W in typed columns, dates/IDs, LymphNode, hepatic flexure, NBI, DIC, BIDS), `test_files.py`, `test_vocab.py`, `test_select.py` (figshare/zenodo ids, route, eligible, implication rules), `test_eval_set.py`, `test_pipeline_smoke.py` (synthetic PNG/mask/CSV/NIfTI/DICOM dataset through metalabel profile/map/apply and prepare, incl. override and mmbu-public contamination drop).
- **Slurm**: all scripts `source $WAILA_HOME/slurm/env.sh` and run `uv run --no-sync waila-<cmd>`. `env.sh` honours overrides, exports `UV_PROJECT_ENVIRONMENT` in the checkout and fails fast without a venv. New `setup.sh` (one `uv sync --locked`, so array tasks never race on the venv) and `cpu_pilot.sbatch` (fetch, label, prepare, merge chain on burst QOS; `bash slurm/cpu_pilot.sbatch submit N`).
- **Docs**: `docs/README.md` index, `docs/data/pipeline.md`; stale paths fixed everywhere (pass2_dataset, research/, pass2data, `*.py` names); README layout, cluster and Contributing sections; `.gitignore` extended. `git status`: 77 files, 2 MB, no data or secrets.

**Bugs in src/ (not edited)**
- `eval_set.py`: the `# relative to ...` comment on the `rec = {` line swallowed `"dataset_id": "mmbu-public", "split": "eval"`, so eval rows lack both (`test_eval_rows_carry_dataset_id_and_split`, xfail).
- `metalabel apply` with no profiles: KeyError (`test_apply_with_nothing_profiled_is_a_noop`, xfail).
- `prepare --merge` with no manifests: ValueError (`test_merge_with_no_manifests_is_a_noop`, xfail).

**Notes**
- `src/` docstrings and comments still cite `research/` and `pass2data/`.
- `scripts/` still import `pass2data` and will not run.
- A folder named `CT/` is only a weak path hit (by design, audit item c); `MRI/` is strong.
- `plan/eval_public.parquet` holds public-split labels: decide whether it belongs in a pushed repo.
