# Model zoo: zero-shot benchmark on mmbu-public Pass 2

`waila-zoo` (src/bio_waila/eval/zoo.py, backends in models/zoo_models.py, registry in configs/zoo.yaml) scores released models
without any training on the public MMBU split and produces the same numbers as the trained teachers: the context score **C**
(per image, the share of its annotated fields whose top-1 class is in the target set, averaged over images) and the per-field
metrics from `eval/public.py`. It answers "what does an off-the-shelf model already get?" and gives the baselines for the student.
Every run is logged in `plan/public_looks.jsonl` as `zoo:<slug>`.

## Results (2026-11-01)

**Setup:**
- All 3,147 mmbu-public images, zero-shot, one run per model. One B200 per model, on HiPerGator account cai6734.
- Chain: download → score → delete (`zoo_eval.sbatch submit`).
- Generative answers were scored with the compound-splitting mapper (commit e65f956); runs scored before it were rescored.
- Only Gemma and Liquid models were run. The disabled entries (Mistral family, SmolVLM2, community decision fine-tunes) were not.
- Copy of `runs/zoo/leaderboard.md` on the cluster:

| model | mode | params (B) | C | mean acc | modality | submod. | body part | stain | specimen | domain | s/img | unmapped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dec_liquid_d1_3b | decision | 3.12 | **0.641** | 0.610 | 0.794 | 0.765 | 0.257 | 0.459 | 0.721 | 0.664 | 0.03 | – |
| dec_gemma4_e4b_it | decision | 7.94 | 0.590 | 0.569 | 0.829 | 0.727 | 0.385 | 0.480 | 0.412 | 0.580 | 0.28 | – |
| dec_gemma4_26b_a4b_it | decision | 25.81 | 0.587 | 0.580 | 0.790 | 0.697 | 0.296 | 0.582 | 0.512 | 0.606 | 0.43 | – |
| dec_gemma4_12b_it | decision | 11.96 | 0.584 | 0.575 | 0.736 | 0.776 | 0.348 | 0.559 | 0.465 | 0.568 | 0.30 | – |
| dec_liquid_d1_2b | decision | 1.92 | 0.514 | 0.455 | 0.749 | 0.666 | 0.210 | 0.145 | 0.433 | 0.526 | 0.02 | – |
| gen_gemma4_26b_a4b_it | generative | 25.81 | **0.443** | 0.458 | 0.617 | 0.289 | 0.555 | 0.559 | 0.102 | 0.624 | 0.68 | 0.21 |
| gen_gemma4_12b_it | generative | 11.96 | 0.441 | 0.457 | 0.644 | 0.328 | 0.527 | 0.559 | 0.103 | 0.582 | 0.72 | 0.28 |
| gen_medgemma_27b_it | generative | 27.43 | 0.436 | 0.438 | 0.557 | 0.361 | 0.527 | 0.448 | 0.191 | 0.544 | 1.00 | 0.16 |
| gen_lfm25_vl_3b | generative | 3.12 | 0.410 | 0.419 | 0.579 | 0.375 | 0.553 | 0.493 | 0.081 | 0.434 | 0.19 | 0.33 |
| gen_medgemma_1_5_4b_it | generative | 4.30 | 0.404 | 0.365 | 0.438 | 0.421 | 0.499 | 0.150 | 0.173 | 0.508 | 0.59 | 0.22 |
| gen_gemma4_e4b_it | generative | 7.94 | 0.389 | 0.393 | 0.592 | 0.275 | 0.493 | 0.423 | 0.002 | 0.575 | 0.62 | 0.22 |
| gen_lfm3_vl_2b | generative | 1.92 | 0.387 | 0.409 | 0.570 | 0.306 | 0.533 | 0.548 | 0.138 | 0.360 | 0.27 | 0.44 |
| gen_medgemma_4b_it | generative | 4.30 | 0.372 | 0.340 | 0.550 | 0.339 | 0.472 | 0.166 | 0.035 | 0.474 | 0.47 | 0.23 |
| gen_lfm25_vl_1_6b | generative | 1.60 | 0.344 | 0.354 | 0.543 | 0.182 | 0.555 | 0.436 | 0.002 | 0.407 | 0.15 | 0.38 |
| gen_gemma4_e2b_it | generative | 5.10 | 0.328 | 0.343 | 0.419 | 0.260 | 0.412 | 0.446 | 0.011 | 0.512 | 0.54 | 0.30 |
| dec_liquid_d1_omni_600m | decision | 0.59 | 0.280 | 0.240 | 0.492 | 0.099 | 0.254 | 0.011 | 0.077 | 0.508 | 0.02 | – |

**Re-sync to `a6280c5` (2026-11-02).** The table above was scored on the 2026-10-01 snapshot (`922064e`, 3,147 rows). `waila-zoo rescore --previous-public` re-scored every run on the current release on CPU.
- **Coverage:** 2,959 of 3,103 rows. The other 144 have no prediction yet: 33 new rows, 7 swapped images, and 104 images redrawn under the same file name, mostly chest X-ray segmentation rows. A GPU run of a model covers them.
- **Effect:** C moved by at most 0.011 and the order is unchanged. Rescored C:
  - **Decision:** d1-3B 0.639, Gemma-4-E4B 0.592, 26B-A4B 0.587, 12B 0.580, d1-2B 0.512, d1-omni-600M 0.275.
  - **Generative:**
    - Gemma 4: 12B 0.448, 26B-A4B 0.444, E4B 0.392, E2B 0.328.
    - MedGemma: 27B 0.434, 1.5-4B 0.395, 4B 0.366.
    - LFM: LFM2.5-VL-3B 0.405, LFM3-VL-2B 0.381, LFM2.5-VL-1.6B 0.333.

**Findings:**
- **Decision readout beats generation by 15–20 points of C.** The readout always picks a valid class, while 16–44% of generated answers map to nothing. d1-3B (3.1B, 0.03 s/image) is the best model overall.
- **Gemma 4's letter readout does not scale.** E4B, 12B and 26B-A4B land at 0.584–0.590.
- **Generative mode improves with size.** The top three are almost level: 26B-A4B 0.443, 12B 0.441 and MedGemma-27B 0.436. MedGemma beats Gemma 4 at neither size: MedGemma-4B 0.372 vs E4B 0.389, and 27B 0.436 vs 12B 0.441.
- **Specimen fails in every generative run (0.00–0.19).** The models name the sample ("tissue section", "retina/fundus") instead of the organism the label asks for. The decision readout does not have this problem (d1-3B 0.72), so this comes from how the prompt asks for specimen, not from what the models know.
- **Unmapped answers:**
  - Splitting compounds such as "chest/thorax" lifted Gemma-4-E2B from 0.274 to 0.328 and E4B from 0.367 to 0.389.
  - What stays unmapped is mostly vague: "medical imaging", "medicine", "light".
  - LFM3-VL-2B returned valid JSON for only 18% of images (every other model 95–100%), which accounts for much of its 44% unmapped. Its score partly measures output format.

**What this implies for the training plan.** None of this is adopted; the decisions stay with the user.
- **Free voter:** d1-3B (0.641) beats d1-2B (0.514) by 13 points. `models/lfm3.md` replaced d1-3B with d1-2B on Liquid's benchmarks; this measurement points the other way.
- **Jev teacher base:** the flat letter readout supports E4B as the cheap base. Scale may still matter after fine-tuning, which this does not test.
- **d1-2B fine-tune arm (`lfm3.md` §4 rule):** d1-2B is 7.6 points behind Gemma's letter readout, more than the ~3-point threshold. By the rule, the E4B LoRA keeps priority.
- **Track-3 student (`lfm3.md` §4 rule):** LFM3-VL-2B (0.387) is above Gemma-4-E2B (0.328), so by the rule it becomes the default student. The fallback LFM2.5-VL-3B scored higher still (0.410). LFM3's JSON failures (above) suggest a prompt or format fix may be worth more than the choice of base.

## Commands

| Command | Where | What |
|---|---|---|
| `waila-zoo list [--porcelain]` | anywhere | the registry; `--porcelain` prints `slug<TAB>gpus<TAB>mem_gb` of enabled entries |
| `waila-zoo prefetch [--models a,b] [--strict]` | CPU | `snapshot_download` into `HF_HOME` (uses `HF_TOKEN`), builds `plan/eval_public.parquet` if missing. Gated or unauthorized repos are reported per model and the rest continue |
| `waila-zoo run --model SLUG [--limit N] [--batch B] [--base-url URL] [--enable]` | GPU | scores one model; `--limit` takes a fixed-seed random subsample for smoke tests; works with `HF_HUB_OFFLINE=1`. A disabled entry is refused unless `--enable` is given (`--base-url`, systemone entries only, implies it) |
| `waila-zoo leaderboard` | CPU | `runs/zoo/*/public_metrics.json` → `runs/zoo/leaderboard.csv` and `.md` (C, mean field acc, per-field acc, n, mode, params, seconds per image, unmapped rate) |
| `waila-zoo rescore [--models a,b]` | CPU | Re-scores finished runs on the current eval set after a mapper change or an mmbu-public re-sync, without loading any model. Generative runs re-map their saved raw answers (`*_raw`). Decision runs keep their saved top-1 classes, so C and the field accuracies are exact but the probability-based metrics are dropped. Rows are matched by `unique_id` and image content: dropped rows are left out, and rows added or with a changed image are counted in `zoo.n_unscored` (the leaderboard's `unscored` column). Content is compared by the `image_sha1` the predictions carry. Older runs need `--previous-public <the copy they were scored on>`, because releases change pixels under the same file name: 104 rows between `922064e` and `a6280c5`. It rewrites `public_metrics.json` and `public_predictions.parquet`. The first report is kept as `public_metrics.before_rescore.json`, and `raw_outputs.jsonl` keeps the original mapping |
| `waila-zoo purge --models a,b` | CPU | deletes those models' weights (and a LoRA's base) from the HF cache through huggingface_hub's cache API, which also sweeps the shared blob store (`hub/blobs`) |
| `waila-zoo lanes [--models a,b] [--lanes 4]` | anywhere | the Slurm chain plan, `lane<TAB>group<TAB>slug<TAB>gpus<TAB>mem_gb`: models sharing a repo form one group; groups keep the given order and are dealt round-robin to the lanes |

Cluster: `bash slurm/zoo_eval.sbatch submit [LIMIT]` runs every enabled model **one group at a time per lane**: download its
weights (CPU, `STAGE=prefetch STRICT=1`) → one GPU job per model (own `--mem` / `--gpus` from the registry, `afterok` the download)
→ delete the weights (CPU, `STAGE=purge`, `afterany`) → the next group. Weights never pile up (at most one group per lane on disk),
at most `LANES` (default 4) GPUs are in use, and a failed download cancels its GPU jobs (`--kill-on-invalid-dep`) so the lane moves
on. A CPU leaderboard job runs after every lane. `MODELS=a,b` runs those slugs only, in that order (put the priority models first;
unknown or disabled slugs stop the submit). `ACCOUNT=<group>` puts the GPU jobs on that group's QOS; `CPU_ACCOUNT=<group>` (default: `ACCOUNT`)
puts the CPU jobs on that group's regular QOS, not burst. Example: GPUs on cai6734, CPUs on the CPU-only cai5731. `sbatch --array=0-N slurm/zoo_eval.sbatch` (task i = i-th enabled model) needs every model downloaded first,
so it is for re-runs only. `bash slurm/zoo_lfm3.sh [LIMIT]` is the LFM3 comparison (d1-2B, plus LFM3-VL-2B against Gemma-4-E2B and
LFM2.5-VL-3B; d1-3B is not rerun, since d1-2B replaces it; `docs/models/lfm3.md` §4); it first checks that the venv has transformers >= 5.19.

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

Lists, and compound strings (`a/b`, `a or b`, `a|b`, `a; b`, `a, b`), are tried whole first and then piece by piece, the
pieces only when the whole string is unmapped: "N/A" stays a decline. Added 2026-11-01: the first zoo runs left about 40% of
generative answers unmapped, many of them compounds such as "chest/thorax". The first piece that maps is the prediction (top-1
is what C scores, in single- and multi-label fields alike). In the multi-label fields (body part, domain), every mapped piece
is predicted: +8 on the first, +7 on the rest, so the F1 metrics see them all. `unmapped_rate` = (unmapped + missing key + parse
failure) / requested fields. After a mapper change, `waila-zoo rescore` re-maps the saved raw answers of every generative run on
CPU and rewrites its report, so no GPU is needed.

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
Without a `k_max`, `D1Scorer.score_rows` asks **all annotated fields of an image in one request**, each as a named `choice`
question over every class: d1-2B packs a state's questions after it (`d1_attention: set`, up to 16), which is how it was trained.
(Fixed 2026-10-27: the runner used to pass `k_max = 26` to every decision backend, so d1 ran tournaments on body part and
submodality although the registry set none.) d1-2B (LFM3-based) needs transformers >= 5.19.

## Registry (`configs/zoo.yaml`)

Keys: `slug`, `hf_id`, `mode`, `readout` (letters | d1), `backend` (systemone), `options`, `base`, `trust_remote_code`, `dtype`, `gpus`, `mem_gb`, `max_new_tokens`,
`k_max`, `enabled`, `reason` (required when disabled), `notes` (licence, gating). `load_registry` validates every entry and rejects
unknown keys, duplicate slugs, bad modes, `k_max` above 26, and Chinese-origin / Qwen-family ids (project policy).

* **Generative, enabled:** MedGemma 4B, 1.5-4B, 27B (gated, auto-approved: accept the terms on the model page with the token's
  account before `prefetch`), Gemma 4 E2B / E4B / 12B / 26B-A4B, LFM2.5-VL 1.6B / 3B, LFM3-VL-2B (remote code, thinking off).
* **Decision, enabled:** stock Gemma 4 E4B / 12B / 26B-A4B (the letter-readout baseline), Liquid d1-2B, d1-3B and d1-omni-600M.
* **Disabled by choice (2026-11-01):** only Gemma (Gemma 4 first, MedGemma lower priority) and Liquid models are run. Ministral-3-3B,
  Mistral-Small-3.2-24B and SmolVLM2-2.2B are not expected to be competitive. The community decision fine-tunes are also off: Xor-26B-A4B,
  GEV-26B-Decide, the AJev 12B LoRA, Torchcast decision 12B, Jev-Omni and Quyet-1.0-Large. Their entries stay for reference.
* **Memory:** `mem_gb` is host RAM, set from measured peaks. Weights load straight to the GPU, so the first runs used 2–9 GB. That
  gives 32 GB for models up to E4B, 48 GB for 12B and 64 GB for 26B-A4B / 27B.
* **Disabled, own engine:** respanai/span-02 (see its section below; run with `--base-url` or `slurm/zoo_span.sbatch`).
* **Disabled:** Llama-3.2-11B-Vision (manual gating by Meta), Phi-4-multimodal (custom code, not a stock image-text-to-text
  architecture), Winnow-12B (GGUF only, no safetensors), Blink-v0.3-26B-A4B-NVFP4 (NVFP4 kernels), rune-26b-a4b (safetensors repo private).

## respanai/span-02 (System One backend)

`dec_span_02` (Apache-2.0, a Gemma-4-26B-A4B fine-tune, image-capable) cannot be loaded with transformers or vLLM: the repo ships its own
Rust/CUDA engine (`engine/`, `launch.sh`) and exposes a TypeSafe System One HTTP API. The registry entry therefore has
`backend: systemone` and `enabled: false`; `SystemOneScorer` (models/zoo_models.py) is a plain HTTP client (mode `decision`).

**Hardware.** The engine's kernels need compute capability 12.0 (sm_120a: RTX PRO 6000 Blackwell or RTX 5090). B200 / H100 / A100 are not
supported. Building needs `cargo` and a CUDA toolkit: `(cd engine && NVCC=/usr/local/cuda/bin/nvcc cargo build --release -p dllm-server)`;
`bash launch.sh` then serves on 0.0.0.0:8000 (`GET /health` -> `{"ok": true}`).

**Self-host.** `waila-zoo prefetch --models dec_span_02` downloads the whole repo snapshot (~16 GB, no file filtering) on CPU. On a machine
with a suitable GPU: start the server, then `uv run waila-zoo run --model dec_span_02 --base-url http://HOST:8000 [--limit 64]`.
`slurm/zoo_span.sbatch` automates this (checks `nvidia-smi` reports compute_cap 12.0, builds the engine if needed, starts `launch.sh`, waits
for `/health`, runs the eval, stops the server); its partition / gres lines are placeholders to point at an sm_120 partition. It is not part of
`zoo_eval.sbatch submit`, and entries with a `backend` are not listed by `waila-zoo list --porcelain`.

**Hosted API.** The vendor also offers a hosted endpoint (`POST .../v1/decisions`, body field `"model": "span-02"`, same request shape).
That sends the evaluation images to a third party: check the mmbu-public licence and terms before using it. Set the `options` of the
registry entry (or a copy of it): `base_url`, `endpoint`, `model: span-02`; the bearer token comes from `SPAN_API_KEY` (`.env`), and the
`/health` check only warns on a 404.

**Protocol.** One request per image: `POST {base_url}{endpoint}` (default `/v1/systemone`) with
`{"state": {"task": "biomedical image metadata", "image": "Image 1"}, "questions": {field: {"type": "choice", "instructions": "What is the
imaging modality of Image 1?", "criteria": {class: class}}}, "images": [base64 JPEG data URI], "reasoning": {"mode": "auto"}}`; every field
annotated for that image is its own question (up to 679 options each, so no tournament / k_max). `answers[field].probabilities` become
log-probs (classes the server omits get a 1e-6 floor, then renormalized) and flow through the same `score_public` path as the other
decision backends. 529 / 429 / 5xx responses and connection errors are retried with exponential backoff; other error bodies
(`{"error": {"type", "message"}}`) fail the run with the message.

**Options** (registry `options:`): `base_url` (default `http://127.0.0.1:8000`), `endpoint`, `api_key_env` (`SPAN_API_KEY`), `model`,
`reasoning` (`auto` | `on` | `off`), `budget_ms`, `concurrency` (thread pool, 8), `timeout` (300 s), `retries` (6), `backoff`, `health`.
Cost note: reasoning `auto` may spend time thinking per request; `--limit 64` first.

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
