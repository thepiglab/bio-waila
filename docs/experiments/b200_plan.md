# Pass-2 training on 4× B200: schedule

Date: 2026-10-07. This turns `experiment_plan.md` (budgeted in H100-hours) into a concrete schedule for **up to 4 B200s** on a shared Slurm cluster. The design and decision rules are unchanged: see [../OVERVIEW.md](../OVERVIEW.md) and [experiment_plan.md](experiment_plan.md).

## Assumptions (check these on day 0)

- **Speed:** 1 B200 ≈ **1.8 H100** for our bf16 workloads. On paper it is about 2.25× (dense BF16 about 2.25 PFLOP/s vs about 1 PFLOP/s), but we derate for data loading and small batches.
  - 192 GB of HBM per GPU means Gemma-4-E4B LoRA runs at large batch on one GPU.
  - Gemma-4-12B and 26B-A4B fit with FSDP across the 4 GPUs.
- **Access:** about 4 GPUs are realistically free in the group allocation, and group **memory is the binding constraint**. GPU jobs request 96 GB for one GPU or 256 GB for four (`slurm/b200.sbatch`). CPU jobs (download, metalabel, prepare) run on a low-priority burst queue and are all resumable. Expect queue waits on the GPU partition.
- **CPU throughput is the real bottleneck:** decoding JPEGs to 384 px is about 150–250 images/s per core. Give every GPU 12–16 CPU workers, and keep the prepared images on cluster storage (about 45 GB of ≤ 512 px JPEGs for 780k images).
- **Storage:** raw downloads are about 2–4 TB at a ≤ 20 GB-per-dataset budget (the real number comes from the dry run). Check the storage quota before starting the full download, and delete `raw/` archives after `src/bio_waila/data/prepare.py`.
- **GPU-hours:** the full matrix is about 660 H100-h ≈ **370 B200-h**. The minimum path is about 115 H100-h ≈ **65 B200-h**.
  - Four GPUs give 96 B200-h per wall-clock day, so the full plan takes about **4 days of GPU time and about 7–9 calendar days** including downloads, debugging and queue waits.

## The critical path is data, not GPUs

The GPUs would sit idle until images exist, so data runs in two waves on CPU nodes while the GPUs start on the first wave.

| Wave | Command | Size | Expected time |
|---|---|---|---|
| Pilot | `bash slurm/cpu_pilot.sbatch submit` (download, metalabel, prepare, merge as one dependency chain; ≤ 64 images per dataset, writes `prepared_pilot/`). The older `PILOT=128 sbatch slurm/download.sbatch` then `PILOT=128 sbatch slurm/prepare.sbatch` also works | ≤ 128 images per dataset, about 50k images, about 100–200 GB | hours (16 shards) |
| Full | `sbatch slurm/download.sbatch` (tops up the same folders; already-fetched files are reused), then `slurm/metalabel.sbatch`, `slurm/metalabel_map.sbatch`, then `slurm/prepare.sbatch` (merge with `MERGE=1`, see the script header) | about 780k images | 1–3 days, limited by Zenodo, Figshare and Kaggle throttling |

Run `uv run waila-metalabel profile`, `map` and `apply` on the pilot too (`cpu_pilot.sbatch` does): its `review.csv` shows which metadata values need a human, while the full download is still running.

## Code to write before the GPUs are useful (day 0–1)

None of these exist yet; the paths below are where they will live (`src/bio_waila/train/`, `src/bio_waila/eval/`), exposed as `waila-*` entry points or run through `slurm/b200.sbatch`. They are listed in the order they are needed.

1. **`src/bio_waila/eval/pass2.py`:** one metric script for every model. It reads a prediction parquet (per-field probabilities and top-1) and computes:
   - per-field accuracy and F1 on annotated fields only;
   - the judge-proxy C-hat;
   - ECE and Brier;
   - coverage at 95% precision;
   - consistency violations;
   - bootstrap CIs clustered by dataset.

   It runs against `plan/eval_public.parquet` (reporting only) and the held-out validation datasets.
2. **`src/bio_waila/eval/folds.py`:** freezes `plan/folds.json`. F0 is the official 52 validation datasets; F1 and F2 are two further dataset blocks; each block is split into calibration and report halves. Use split seed 17.
3. **`src/bio_waila/train/b0.py`:**
   - **B0:** cache SigLIP2-So400m features once, then train 6 heads (linear or MLP) with the four label-mode losses.
   - **B0-ft:** the same, with LoRA (or the last 8 blocks unfrozen) on the vision tower.
4. **`src/bio_waila/train/clm.py`:** a cosine classifier with label vectors initialised from text. Variants: shared query (C-F1/C-F2s), one query per field (C-F2q), and context-conditioned (C-F3).
5. **`src/bio_waila/train/jev.py`:** Gemma-4-E4B with the packed field blocks, attention masks `iso`, `fieldcausal` and `bidir`, a pointer head (softmax for single-label fields, sigmoid for multi-label), and F3 context modes TF, PC and SS.
   - Unit tests U1–U3 from the experiment plan run first. U1 checks that the packed independent mode equals 6 separate passes.
   - If Gemma 4's sliding-window layers reject custom masks, fall back to a prefix KV cache with batched option rows.
6. **`src/bio_waila/train/sft.py`:** the B2 student. Gemma-4-E4B (or E2B) learns to emit the exact Pass-2 JSON, with requested-field subsets sampled from the public annotation patterns, a key-order option, and Pass-1 replay.
7. **`src/bio_waila/eval/zeroshot.py`:** the zero-shot baselines.
   - B1-sig and B1-bio: SigLIP2 and BiomedCLIP with prompt templates.
   - B1-gem: stock Gemma-4 letter readout, Cygnet-style. Sum probability over every token that decodes to each letter, apply softcapping, and fit a temperature per field.

`src/bio_waila/data/autolabel.py` already holds working zero-shot scoring code that items 3, 4 and 7 can reuse.

## GPU schedule

"1×" means one GPU per job. Up to 4 such jobs run side by side on the 4 GPUs. "4×" means one job uses all four (DDP or FSDP).

### Day 1–2: stage 0 on the pilot (about 10 B200-h)

| GPU | Job | Est. B200-h |
|---|---|---:|
| 0 | B1-gem: stock Gemma-4-E4B letter readout on the pilot validation datasets and mmbu-public (forward only); fit per-field temperature | 1 |
| 1 | B1-sig and B1-bio zero-shot; cache SigLIP2 features for the pilot and public | 1 |
| 2 | B0-lin and B0-mlp on the cached features (minutes), then B0-ft LoRA on the pilot | 1.5 |
| 3 | U1–U3 mask unit tests for `src/bio_waila/train/jev.py`; judge-proxy calibration (local Gemma-4 judge on 800 sampled tuples) | 3 |

**Gate before stage 1a:**
- the unit tests pass;
- the judge-proxy kappa is ≥ 0.85;
- B0-mlp per-field accuracy is plausible.
- **Diagnostic:** B1-gem's zero-shot score is the number every trained model must beat.

### Day 2–4: stage 1a screening on the pilot (one fold, one seed, about 55 B200-h)

These are all 1× jobs, packed 4 at a time. Each runs 1.5–5 B200-h.

| Batch | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| A | J-F2 (Jev, independent) | J-F1c (Jev, all-at-once, cross-field) | C-F2q (CLM, query per field) | C-F1s (CLM, shared space) |
| B | J-F3-O2-TF (progressive, domain last, teacher-forced) | J-F3-O1-TF (domain first) | C-F3-O2-TF / C-F3-O2-PC | B2-p and B2-cf (JSON key order) |
| C | X-J out-of-fold cross-fit (3 J-F2 models, needed for PC) | J-F3-O2-PC | J-F1n (order-bias reference), J-A-head | C-A-text, C-A-tower, B0-ctx |
| D (stretch) | J-F3-O2-SS | J-F2sep (6 separate LoRAs) | J-A-ml, J-A-alias | B2-ff |

The scoreboard is `src/bio_waila/eval/pass2.py` on val-OOD. Apply decision rule D1 to choose what advances.

### Day 4–6: stage 1b confirmation (3 folds × 3 seeds, about 95 B200-h)

Advance about 8 runs:
- each family's independent (F2) reference;
- the best all-at-once (F1) and the best progressive (F3) run per family;
- B0-ft;
- the best B2 key order.

That is 8 runs × 9 (fold, seed) pairs ≈ 72 pilot-size jobs at about 1.2 B200-h, packed 4 at a time (about 24 wall-clock hours). This is the one pre-registered look at mmbu-public per advanced run. **Rules D2–D5** decide the formulation per family, whether the Jev or CLM branch is worth continuing, and whether to go to full data.

### Day 6–8: stage 2 full data (about 780k images, about 120 B200-h), only if rule D5 fires

| Job | GPUs | Est. B200-h |
|---|---|---:|
| Winning Jev formulation, Gemma-4-E4B, 2 epochs | 4× DDP | 25 |
| Winning CLM formulation, SigLIP2 | 1× | 7 |
| B0-ft full | 1× | 6 |
| Best B2 order, full | 4× DDP | 10 |
| Out-of-fold teachers (3-way group cross-fit of the winning Jev and CLM) | 4× / 1× | 45 |
| Pseudo-label inference over all 780k images (prefix-cached; F3 at 2–3× the cost) | 4× | 8 |
| Optional scale-up: Jev on Gemma-4-12B (FSDP) if E4B underfits | 4× FSDP | 20 |

Output: a pseudo-label parquet with per-field probability, an acceptance flag at the calibrated threshold, and the source teacher.

### Day 8–10: stage 3 transfer, the decisive test (about 80 B200-h)

Student runs, each a Gemma-4-E4B JSON fine-tune at about 9 B200-h on 4× DDP (about 2.5 wall-clock hours each):

| Run | Fine-tuned on |
|---|---|
| T0 | hard labels only |
| T1 | T0 + Jev pseudo-labels |
| T2 | T0 + CLM pseudo-labels |
| T3 | T0 + labels all three teachers agree on |
| T4 | T0 + soft top-3 targets |
| T5 | T0 + B0-ft pseudo-labels |
| T6 | T0 + its own confident predictions (self-training control) |

- **Seeds:** 2 for T0 and the best of T1/T2, 1 for the others.
- **Scoring:** judged C with a local Gemma-4 judge, plus Pass-1 task accuracy (T) from `mmbu-eval` on public.
- **Rule D6:** adopt a teacher only if its student beats T0 by ≥ 1 point of judged C and also beats T6.
- **Track 3 student:** repeat the winner on Gemma-4-E2B (about 6 B200-h).

## Minimum path if time or GPUs run short (about 65 B200-h, about 2 wall-clock days on 4 GPUs)

1. Stage 0.
2. Pilot runs: B0-mlp, B0-ft, J-F2, J-F1c, J-F3-O2-PC, C-F2q, C-F3-O2-PC, B2-p and B2-cf, at 1 fold and 2 seeds (about 35 B200-h).
3. Full-data T0, T5 and the better of T1/T2 (about 30 B200-h).

This answers the two questions that matter: which formulation teaches best, and whether a teacher helps the submission model at all.

## Utilisation rules

- **Fill GPUs with independent jobs.** Pilot runs are 1× jobs, and 4 run at once (job arrays work well: `--array=0-3` with a run table).
- **Multi-GPU only where it pays:** full-data Jev, B2 and students, or anything above 12B.
- **Keep the GPUs busy:** never leave one waiting on the CPU. Cache SigLIP2 features and Gemma image tokens per image once (both fit easily on cluster storage), and stream prepared JPEGs with ≥ 12 workers per GPU.
- **Save predictions, not just metrics:** every run writes per-image prediction parquet, the config and the git hash. Pseudo-labeling and the paired bootstrap need these; re-running inference is wasted GPU time.
- **Checkpoints:** save LoRA adapters only (tens of MB), plus the final full checkpoint for the student. Keep checkpoints under `$WAILA_RUNS/<run_id>/` (the checkout's gitignored `runs/`).
- **Public-split looks:** log every look at mmbu-public in `public_looks.jsonl`. The public split is reporting only.

## Day-0 checklist

1. Confirm B200 access and the GPU request syntax (`sinfo`). Adjust `slurm/b200.sbatch` if the partition or GRES name differs.
2. `git clone` this repo, run `bash slurm/setup.sh` (builds the uv venv in the checkout), and put `mmbu-public/` next to the checkout. Copy the Kaggle token file to `~/.kaggle/access_token` (chmod 600). `data_discovery/` is needed only to re-run `waila-select`.
3. Check that compute nodes reach the internet (`srun --pty curl -I https://zenodo.org`). If not, run downloads on a login or transfer node.
4. Launch the CPU pilot (`bash slurm/cpu_pilot.sbatch submit`), and in parallel write `src/bio_waila/eval/pass2.py`, `src/bio_waila/eval/folds.py`, `src/bio_waila/train/b0.py` and `src/bio_waila/eval/zeroshot.py`.
5. Start the full download as soon as the pilot looks sane.
