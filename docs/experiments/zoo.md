# Model zoo: zero-shot benchmark on mmbu-public Pass 2

`waila-zoo` (src/bio_waila/eval/zoo.py, backends in models/zoo_models.py, registry in configs/zoo.yaml) scores released models
without any training on the public MMBU split and produces the same numbers as the trained teachers: the context score **C**
(per image, the share of its annotated fields whose top-1 class is in the target set, averaged over images) and the per-field
metrics from `eval/public.py`. It answers "what does an off-the-shelf model already get?" and gives the baselines for the student.
Every run is logged in `plan/public_looks.jsonl` as `zoo:<slug>`.

## Commands

| Command | Where | What |
|---|---|---|
| `waila-zoo list [--porcelain]` | anywhere | the registry; `--porcelain` prints `slug<TAB>gpus<TAB>mem_gb` of enabled entries |
| `waila-zoo prefetch [--models a,b] [--strict]` | CPU | `snapshot_download` into `HF_HOME` (uses `HF_TOKEN`), builds `plan/eval_public.parquet` if missing. Gated or unauthorized repos are reported per model and the rest continue |
| `waila-zoo run --model SLUG [--limit N] [--batch B]` | GPU | scores one model; `--limit` takes a fixed-seed random subsample for smoke tests; works with `HF_HUB_OFFLINE=1` |
| `waila-zoo leaderboard` | CPU | `runs/zoo/*/public_metrics.json` → `runs/zoo/leaderboard.csv` and `.md` (C, mean field acc, per-field acc, n, mode, params, seconds per image, unmapped rate) |

Cluster: `bash slurm/zoo_eval.sbatch submit [LIMIT]` submits `slurm/zoo_prefetch.sbatch` (CPU), then one GPU job per enabled model
(own `--mem` / `--gpus` from the registry, `afterok` the prefetch), then a CPU leaderboard job (`afterany` all of them). Only the
per-model evaluation uses a GPU. `sbatch --array=0-N slurm/zoo_eval.sbatch` also works (task i = i-th enabled model).

Outputs per model, in `runs/zoo/<slug>/`: `public_metrics.json` (the standard report plus a `zoo` block: mode, parameters in
billions, seconds per image, unmapped / unknown rates, per-field mapping counts), `public_predictions.parquet` (per image and field:
prediction, targets, correct; plus `<field>_raw` / `<field>_map` for generative or `<field>_prob` for decision), and for generative
models `raw_outputs.jsonl` (raw text, parsed values, mapped class and how it was mapped).

## Mode A: generative

The image and the official Pass-2 prompt go to `AutoModelForImageTextToText` (bf16, `device_map="auto"`, chat template through the
processor, greedy decoding, `max_new_tokens` from the registry, default 384; batched, falling back to one by one when a processor
cannot pad). The prompt names **only the fields annotated for that image** (the non-masked ones in the eval manifest), by their
display names: body part, modality, stain, submodality, specimen, medical domain.

**Parsing.** Code fences are stripped and the first balanced `{...}` that holds metadata is used; trailing text, trailing commas,
smart quotes and single-quoted JSON are tolerated, as is a flat object without `predicted_metadata`. Keys are normalized
(`Body Part`, `body_part`, `medical_domain` → the field). A truncated reply falls back to regex scraping of `"key": "value"` pairs
(counted as a JSON failure in `json_ok_rate`). No usable JSON at all means every requested field is `parse_failed`.

**Mapping each string to a class** (`Mapper`), first hit wins:
1. exact: the shared `data/vocab.CANON` normalizer for the field, or a case/punctuation-insensitive match to a class name (also to
   the part of a name outside or inside a parenthesis, and with a trailing "imaging" / "microscopy" / "stain" / plural toggled);
2. alias: the MMBU alias map (`resources/public_aliases.json`) plus the hand-written synonyms in `EXTRA_ALIASES` (CT scan, MRI,
   Homo sapiens, ENT, …);
3. fuzzy: `difflib` ratio >= 0.88 against class names and aliases;
4. otherwise **unmapped**, which is wrong. `unknown` (and N/A, "none of the above", ...) is a decline: also wrong, but counted in
   `unknown_rate`, not in `unmapped_rate`.

Lists and `a|b` / `a; b` / `a, b` strings are tried whole first, then item by item; the first item that maps is the prediction
(top-1 is what C scores, in single- and multi-label fields alike). `unmapped_rate` = (unmapped + missing key + parse failure) / requested fields.

**Logits.** The predicted class gets logit +8 and every other class -8, so `score_public` sees a one-hot top-1. With no prediction
every logit is -8 except the first non-target class (-7): the argmax is then never correct, and the multi-label sigmoid readout in
`eval/metrics.py` never fires a positive. Calibration metrics (ECE) are meaningless for these runs.

## Mode B: decision (letter readout)

Per field, the class names become lettered options (`jev_gemma.build_prompt`: question, `A. name` lines, "Answer with the letter
of the best option only") and the model's next-token distribution is read: `P(letter)` sums the tokens that decode to that letter
(`GemmaBackend.letter_logits`), renormalized over the shown options. Fields are independent (no context from other fields) and
only images annotated for a field are scored. `GemmaBackend` was extended to take any HF image-text-to-text repo, a LoRA
`adapter` merged on a `base` (registry: `hf_id` = adapter, `base` = base model), `trust_remote_code`, and `device_map`; the Gemma-only
`max_soft_tokens` argument is dropped for other processors. Images are capped at 512 px on the long side (as in the Jev trainer).

**Many options.** Letters stop at 26, but body part has ~99 classes and submodality ~37. `jev_gemma.tournament_logprobs` handles
them: chunks of consecutive classes of at most `k_max` (registry `k_max`, default 26) are scored in round 1; the top
`finalists_per_chunk` (4, shrunk if needed so the finalists fit in one set) of every chunk are scored together in a final round.
Finalists keep their final-round log-probs; the other classes keep their within-chunk log-probs minus 10, so the output is a full
distribution whose argmax is the tournament winner. Body part costs 4 + 1 forward passes per image, submodality 2 + 1, the rest
1. Weakness: a class can lose its chunk to a letter bias, and the final round compares winners that never met. Output is
log-probabilities: the multi-label sigmoid in `eval/metrics.py` therefore reports top-1 only for body part and domain.

**Liquid d1** (`readout: d1`, d1-3B and d1-omni-600M): the model cards describe one API for both,
`model.system_one_batch([(state, {name: question}, images)])` with `{"type": "choice", "instructions", "criteria": {option: description}}`,
returning `{"answers": {name: {"choice", "confidence", "probabilities"}}}` (needs `trust_remote_code`; transformers >= 5.14 / 5.15). The image
is the whole state (`None`), the options are the class names (description = name), and `probabilities` become the log-probs. d1 has no
26-option limit, so all classes go in one question unless the entry sets `k_max` (then the same tournament is used).

## Registry (`configs/zoo.yaml`)

Keys: `slug`, `hf_id`, `mode`, `readout` (letters | d1), `base`, `trust_remote_code`, `dtype`, `gpus`, `mem_gb`, `max_new_tokens`,
`k_max`, `enabled`, `reason` (required when disabled), `notes` (licence, gating). `load_registry` validates every entry and rejects
unknown keys, duplicate slugs, bad modes, `k_max` above 26, and Chinese-origin / Qwen-family ids (project policy).

* **Generative, enabled:** MedGemma 4B, 1.5-4B, 27B (gated, auto-approved: accept the terms on the model page with the token's
  account before `prefetch`), Gemma 4 E2B / E4B / 12B / 26B-A4B, LFM2.5-VL 1.6B / 3B, Ministral-3-3B, Mistral-Small-3.2-24B, SmolVLM2-2.2B.
* **Decision, enabled:** stock Gemma 4 E4B / 12B / 26B-A4B (the letter-readout baseline), Liquid d1-3B and d1-omni-600M, Xor-26B-A4B,
  GEV-26B-Decide, the AJev 12B LoRA (on Gemma 4 12B), Torchcast decision 12B (cc-by-nc-4.0: evaluation only), Jev-Omni,
  Quyet-1.0-Large (31B). Several of these ship their own decision heads or calibration; the stock LM letter readout used here
  ignores those, so their numbers can understate them.
* **Disabled:** Llama-3.2-11B-Vision (manual gating by Meta), Phi-4-multimodal (custom code, not a stock image-text-to-text
  architecture), Winnow-12B (GGUF only, no safetensors), Blink-v0.3-26B-A4B-NVFP4 (NVFP4 kernels), rune-26b-a4b (safetensors repo private).

## Adding a model

1. Check the licence, that it is not Chinese-origin / Qwen-based, that it has safetensors weights and loads with
   `AutoModelForImageTextToText` (generative) or has a Gemma-style chat template (letter readout).
2. Append an entry to `configs/zoo.yaml` (`mem_gb`: 64 up to ~12B, 128 for 24-31B; one 192 GB GPU holds all of them in bf16).
   For a LoRA, `hf_id` is the adapter and `base` the base model. Disable with a `reason` if it cannot run.
3. `uv run waila-zoo prefetch --models <slug>`, then `uv run waila-zoo run --model <slug> --limit 64`, check
   `runs/zoo/<slug>/raw_outputs.jsonl` (unmapped answers suggest missing aliases in `EXTRA_ALIASES`), then the full run and `leaderboard`.
4. A model with a different calling convention needs a backend class in `models/zoo_models.py` (generative: `generate(images, prompts)`;
   decision: `logprobs(images, field, space, k_max)`; both expose `n_params`) and a branch in `load_backend`.

## Tests and caveats

`tests/test_zoo.py` is offline and CPU-only: registry validation, prompt/JSON parsing, mapping, the tournament with fake scorers,
leaderboard aggregation, and `run` end to end with fake backends on a fake mmbu-public. No real weights were loaded when this was
written: the model loading paths (`HFGenerator`, `LetterScorer`, `D1Scorer`), processor batching differences between model families,
LoRA merging for the AJev adapter and the d1 return format (dict vs list probabilities is handled, not observed) are untested until the
first GPU smoke run (`--limit 64`).
