# Pass-2 labeler: final training plan

Date: 2026-10-07. This plan supersedes the "what to train" parts of `experiment_plan.md` and `b200_plan.md`. Those two still hold the detailed formulation matrix (F1/F2/F3), metrics and statistics; this file decides the models, the data use and the run order.

It rests on:
- the data pipeline as validated today (`docs/data/validation/`);
- four research notes from today:
  - [`models/med_encoders.md`](../models/med_encoders.md)
  - [`models/general_encoders.md`](../models/general_encoders.md)
  - [`models/decision_index_2026-10.md`](../models/decision_index_2026-10.md)
  - [`models/liquid_ai.md`](../models/liquid_ai.md)
- the earlier notes ([`models/backbones.md`](../models/backbones.md), [`clm_methods.md`](../models/clm_methods.md), [`jev_methods.md`](../models/jev_methods.md)).

I spot-checked the research claims that change decisions against the HF Hub API: d1-3B, d1-omni-600M, LFM2.5-VL, EmbeddingGemma 2, TIPSv2, C-RADIOv4 and MedSigLIP all exist, with the licenses and dates stated.

**Compute figures are estimates** (FLOP arithmetic at 0.8 PFLOP/s per B200), not measurements.

## 0. Decisions in one screen

| Role | Model | Why | Status |
|---|---|---|---|
| **CLM-style teacher (primary)** | **SigLIP 2 So400m/16**, fine-tuned FLYP-style with label-text-initialised heads | Apache-2.0, ungated, text tower, best price/accuracy (about 5 B200-h per 10-epoch run). There is still no SigLIP 3 or DINOv4. | build first |
| CLM second tower | **MedSigLIP-448** (Google, the MedGemma tower) | Medical pretraining helps most frozen, few-shot and on rare classes, which is where our long tail is. Gated (HAI-DEF), so **teacher only**. | bake-off arm |
| CLM third view | **DINOv3-L/16** (timm mirror) | Different training signal, strong on X-ray. Weak on EM and H&E. Fallbacks: PE-Core-L, C-RADIOv4-H. | bake-off arm |
| CLM-LLM side arm (the original CLM recipe) | frozen **Gemma-4-E4B** embeddings (layer at 60% depth, image-token mean plus last token) + per-field state heads + shared action head; **EmbeddingGemma 2** as the second encoder | CLM itself is a frozen Qwen3-8B plus two MLP heads (ineligible base). LLM embedders trail SigLIP-class encoders by 10–25 points on classification probes (MIEB; `models/clm_llm_vs_encoder.md`), so this is a cheap arm (about 1–2 B200-h). Keep it only if within ~3 points of SigLIP 2, or if it adds ≥ 1.5 points in combination. | code ready |
| Moondream 3.1 vision tower (CLM-vision arm) | Moondream 3.1's tower (SigLIP 2 So400m/14 fine-tuned inside the VLM, BF16) loaded into the SigLIP trunk, with SigLIP 2 text tower and heads (`waila-convert-moondream`, `configs/clm_moondream31_vision.yaml`) | Cheap bake-off arm: the only part of Moondream 3.1 that is usable without custom engine work. It lost alignment with SigLIP 2's text tower (pooled cosine 0.67), so stage A runs longer. **Moondream for Jev / CLM-LLM: not pursued**: kestrel exposes no logits, hidden states or local training; it would take 3–12 days (`models/moondream31_for_clm_jev.md`). | code ready |
| Microscopy / EM / endoscopy specialist | **BMC-CLIP-CF ViT-L/14** (Stanford, MIT, ungated) | The only ungated encoder trained on light, fluorescence and EM figures. | added only if those slices lag |
| **Jev-style teacher** | **Gemma-4-E4B-it**, letter readout plus LoRA for the prototype; **Gemma-4-26B-A4B-it** for scale | Best eligible evidence: Docto has E4B above MedGemma-1.5-4B and E2B as a decision base; 26B-A4B has the only strong non-Chinese *image* decision results (Rune: Image JevBench #2, ECE 22.9% → 6%). | build second |
| Free zero-shot voters | **Liquid d1-3B** (open since Oct 5); stock Gemma-4 letter readout | d1-3B is a ready-made image-capable decision model (3.1B, calibrated, 475 decisions/s). It costs nothing to add as a vote. | stage 0 |
| **Submitted student (Track 3)** | **Gemma-4-E2B vs LFM2.5-VL-3B**, decided by a head-to-head and by the organizers' parameter-counting ruling | E2B is 2.4B compute-active but **5.1B stored**. LFM2.5-VL-3B is 3.1B all-in (400M SigLIP2 tower). If stored weights count, E2B fails the 4B cap. | ask organizers now |
| Track 3 bridge tower | **EmbeddingGemma 2** vision tower | Exactly the E2B tower shape (768-d, 16 layers), retrained and text-aligned, Apache-2.0. The best path to put a medically adapted tower *inside* the E2B student. | stage 3 arm |

**Not chosen:**
- Moondream 3.1: no torch path for 3.1, FP8 Photon format.
- MedGemma as a base: gated, and lost to Gemma-4-E4B after identical Jev tuning.
- MetaEncoder-30B: auto-gated, LM lineage unverified; only an optional extra teacher.
- d1 as the student: it cannot generate text.
- Ministral 3, Phi, SmolVLM: no competitive evidence.
- All Chinese-origin models and anything built on them, for example ConceptCLIP, VisionFM, NV-Reason-CXR (Qwen-based), MMKD-CLIP.

**Rule 3 still holds:** the shipped artefact is one end-to-end VLM. Every teacher above exists only to produce labels for it.

## 1. The data and how each model uses it

### 1.1 What we have

| | Value |
|---|---|
| Plan | `plan/selection.jsonl`: **418 datasets, about 873k planned images**. 409 are permissive and 9 are in the non-commercial (nc) layer; 364 are training datasets and 54 are validation datasets. |
| Validation | Every selected dataset that can be downloaded here was validated on a pilot sample, with a human-style visual check of 8 images (V1–V7, R1–R4, N1–N4 reports). Fixes are in `plan/registry_fixes.json` (drop 161, route 38, labels 187, exclude_paths 86) and `plan/overrides.csv`. |
| Not downloadable here | About 18 Mendeley datasets (manual browser download), about 10 RAR datasets (need `unrar` or bsdtar on the cluster), 3 unsupported hosts (TCIA, GigaDB, BioStudies or ISIC) and 2 whole-slide-image datasets (`uv sync --extra wsi`). |
| Labels per image and field | `hard` / `multi` / `candidates` (partial label) / `masked`, plus source (`registry`, `meta`, `path`, `dicom`, `override`) and confidence. |
| Evaluation data | mmbu-public, **evaluation only** (`plan/eval_public.parquet`). Near-duplicates are removed from training with the strict test introduced today (256-bit dHash plus thumbnail correlation): 98.7% recall, 0.3% false positives. |

### 1.2 Splits

- **Folds: 5 leave-dataset-group-out folds, frozen once in stage 0** (group = dataset family, so that, for example, all ISIC derivatives sit in one fold).
  - Every number we report is per held-out dataset, then macro-averaged.
  - Image-level splits are banned: every image inherits its dataset's labels, so an image-level split measures dataset recognition.
- **Validation datasets:** the 54 `split: val` datasets in the plan are the early-stopping and calibration set. They are never used for training a teacher whose labels are later evaluated on them.
- **mmbu-public:** final yardstick only, through the judge-proxy scorer (stage 0). We never tune hyperparameters on it.

### 1.3 Licence toggle

- `configs/data.yaml: include_nc: false` is the default for everything that could reach the submission.
- The nc layer (isic-2019, pitvis-2023, cholecseg8k, pannuke, mhsma, …) is evaluated as **one ablation** (teacher trained with vs without it, on the same folds).
- It is used for submission-bound labels only if it helps and the organizers confirm nc data is acceptable for training a teacher.

### 1.4 Losses per label mode (shared by all teachers)

| Mode | Loss |
|---|---|
| `hard` | Softmax CE over the field vocabulary, label smoothing 0.1. |
| `multi` (body part, domain, multi-organ sets) | Sigmoid per class, ancestors as positives, ASL-style down-weighting of negatives. |
| `candidates` | Partial-label loss −log Σ_{c∈S} p(c), with zero-shot pre-filtering of S (PartialCLIP) and an entropy regulariser. |
| `masked` | No loss for that field. These images are what the teachers label later. |
| Field applicability | Given, as in the Pass-2 prompt. Stain loss applies only to microscopy rows. |

- **Sampling:** class-balanced over (dataset × label), with a per-dataset cap of 20k images per epoch (`--source-max` in select already caps MMBU-source datasets). Small datasets are up-weighted with a square-root rule, not to parity, to limit dataset memorisation.
- **Label-noise audit (stage 0):** relabel 300 held-out images per field by hand (Claude-assisted, human-checked). This gives a noise ceiling and decides whether `candidates`-mode labels are trusted at their stated confidence.

## 2. Teachers in detail

### 2.1 CLM-style teacher (SigLIP 2 family)

Recipe (sources in `general_encoders.md` §6.2):

| Stage | What happens |
|---|---|
| Labels | 4–8 prompts per label ("a {submodality} image of the {body part} …"), cached from the **frozen** text tower. Each label vector starts from the mean prompt embedding. |
| **A. Head warm-up** | Tower frozen. Per-field MAP query heads and label vectors are trained at LR 1e-3 for 1–2 epochs, with an L2 pull to the initial vector (stronger for labels seen in only 1–2 datasets). |
| **B. FLYP-style full fine-tune** | Logits exp(t_f)·cos(q_f, w_k), with prompts re-sampled each step. AdamW, LR 3e-5 at the top block, LLRD 0.8, weight decay 0.05, 5% warm-up then cosine, EMA 0.9998, 5–10 epochs at 384 px. No mixup, cutmix, hue or saturation jitter. Add JPEG and resampling augmentation to kill compression shortcuts. |
| **C. WiSE-FT** | Weight interpolation α ∈ {0.3, 0.5, 0.7, 1.0}, chosen on held-out dataset groups. |
| Calibration | Per-field temperature on validation datasets, plus an explicit "other/unknown" logit for abstention. |
| Variants of the winner | LoRA arm (r 32–64), 512 px, NaFlex-576 (aspect ratio and borders are metadata cues: ultrasound fan, endoscopy circle). |

The CLM teacher's job is:
1. to be the cheap, strong baseline (B0-ft in `experiment_plan.md`);
2. to score classes with **zero training datasets** through the text tower;
3. to pre-filter candidate sets for the partial-label loss.

### 2.2 Jev-style teacher (Gemma 4)

- **Readout:** letter readout (sum over every token that decodes to each letter; thinking off). An **option-count-dependent temperature** is fitted on our validation datasets.
  - Body part has about 100 options, so it is shown as a two-level menu: family (14) → atoms. Alternatively, a pointer head over option-token spans (`jev_methods.md`) replaces letters for that field.
- **Training:**
  - LoRA r16–32 on language tensors with the tower frozen first. Then a second run unfreezes the 150M E4B tower at LR ×0.1, because the towers were frozen in Gemma 4 pretraining and have never seen EM or fundus images.
  - Targets: 50% gold labels, 50% CLM-teacher distribution (Brier or KL), the recipe that recurs among the top Decision Index entries.
- **Formulations:** F2 (independent fields, one packed pass) and F3 (progressive, domain **last**), with context dropout and scheduled sampling. F1 is kept as the soft-flow control. All three cost the same compute (`experiment_plan.md` §1).
- **Image tokens:** 280 by default. A second run at 1,120 tokens for microscopy and pathology rows (Rune: +6 points at 4× tokens).
- **Scale run:** Gemma-4-26B-A4B-it, **frozen** letter readout first (deck31b showed frozen plus temperature is near the top), then one LoRA run if the frozen readout leaves a gap above 2 points to the E4B LoRA.

### 2.3 Zero-shot voters (no training)

- **d1-3B:** per-field `choice` questions with option descriptions as criteria. Image capped at 256 tokens per tile, so expect weakness on histology.
- **Stock Gemma-4-E4B and 26B-A4B letter readout:** the B1-gem baseline.
- **SigLIP 2 and MedSigLIP zero-shot with prompt ensembles:** the B1-sig baseline and the WiSE-FT endpoint.

These give the floor every trained teacher must beat, and cheap extra votes for the ensemble.

### 2.4 Teacher ensemble → pseudo-labels

For every training image and every field whose label is `masked` or `candidates`:
1. Combine calibrated teacher probabilities (weighted log-linear, weights fitted on validation datasets).
2. Keep the result as **hard** if p_max ≥ τ_f, where τ_f is chosen for at least 95% precision on held-out datasets. Keep it as a **soft target** if between τ_low and τ_f. Leave it masked otherwise.
3. `hard` and `multi` gold labels are never overwritten. Where a teacher disagrees with them confidently, the case is logged for review; that is how the remaining label bugs surface.

### 2.5 Batch composition, order and graded targets

Decided 2026-10-08. The code is in `train/sampling.py`, `train/similarity.py` and `train/losses.py`; ablation override sets are in `configs/ablations.yaml`.

**Mix everything in every batch, never one modality at a time.**
- The main loss scores each image against its field's whole vocabulary, and ViTs use per-sample LayerNorm. So batch-mates only affect gradient noise, and i.i.d. mixing gives the lowest-variance, unbiased gradient.
- Sequential modality schedules cause forgetting and per-field calibration drift. The multi-task literature (T5/mT5 mixing, multilingual MT) consistently favours mixing every step.

**Stratify on the label first, then the dataset (`sampler.mode: hierarchical`, the default).**
- Each draw picks a submodality at its planned share^τ (τ = 0.7), then a dataset within it ∝ n^0.5, then an image uniformly.
- `waila-select` already caps datasets and plans the per-submodality mix. The old √-dataset sampler double-balanced on top of that and pulled the mix off the benchmark (chest x-ray 10.7% → 7.6%, laparoscopy 8.8% → 6.5%).
- Spreading each label over many datasets fights the "label = dataset fingerprint" shortcut, which matters because we evaluate on held-out datasets.

**Reproducible by construction.**
- The global sequence of (row, augmentation seed) per epoch comes from the run seed, independently of the GPU count.
- `train.global_batch` is the unit: rank r takes its slice of every global batch, and every draw's augmentation is seeded. So 1-GPU and 4-GPU runs see the same examples in the same optimiser steps.
- Remaining nondeterminism: per-rank loss averaging before DDP, and cuDNN kernels.

**Graded similarity instead of binary targets.** Binary targets treat "chest x-ray vs digital radiography" like "chest x-ray vs H&E", and give no credit for overlapping body-part sets. `similarity.label_similarity` defines S_f:

| Field | Same class | Partial credit | Otherwise |
|---|---|---|---|
| submodality | 1 | 0.3 when both sit under one parent modality (taxonomy) | 0 |
| modality | 1 | 0.3 when the modalities share a submodality | 0 |
| body part | 1 | 0.5 within one of the 14 anatomical families | 0 |
| stain, specimen, domain | 1 | none | 0 |

These feed four mechanisms:
- **Label smoothing** (`train.smoothing_mode: hierarchy`, default; all teachers): the smoothing mass goes to similar classes instead of uniformly.
- **In-batch InfoNCE** (`train.loss: infonce` in CLM-LLM): targets T_ij = z_i · z_j with z = normalise(y S). This covers single labels, multi-label sets and candidate sets; CLM's binary group-masked loss is `soft_targets: false`.
- **Cross-field image-image term** (`train.cross_field_weight`; CLM-vision and CLM-LLM): targets average T_f over the fields both images label. Two images that share modality and body part but differ in submodality get partial credit. γ = 2 sharpens it so near-identical label sets dominate.
- **Curriculum:** no modality curriculum. The head warm-up is stage A, and candidate-set (partial-label) rows ramp 0 → 1 over the first 5% of steps (`train.cand_ramp_frac`). For Jev, fields are sampled inversely to how often they are labelled.

**Ablations** (one seed and one fold each; frozen-feature probes first, which take minutes):
- sampler: hierarchical vs plan mix vs dataset_sqrt vs flat classes;
- smoothing: uniform vs hierarchy;
- cross-field term on or off;
- for CLM-LLM: binary vs graded vs graded plus cross-field InfoNCE.

Keep a change only if it improves held-out-dataset macro accuracy beyond seed noise.

## 3. Student (the submission)

| Arm | Base | Data |
|---|---|---|
| T0 | E2B (and LFM2.5-VL-3B) | gold hard and multi labels only, Pass-2 JSON format |
| T5 | same | T0 plus B0-ft (CLM-only) pseudo-labels: the self-training control |
| T1 / T2 | same | T0 plus the best ensemble pseudo-labels (CLM plus Jev plus voters) |
| T-bridge | E2B with the EmbeddingGemma-2 tower swapped in (projector re-aligned) | T1 data |

- **JSON key order** is a free design choice, since each key is decoded after the earlier ones. The default is modality → submodality → body part → stain → specimen → domain (domain last). Domain-first is the O1 ablation.
- **Adoption rule:** a teacher is adopted only if its student beats T0 by at least 1 point of judged accuracy on held-out datasets **and** beats T5. Otherwise we keep the cheap CLM teacher.

## 4. Run order on 4× B200 (estimates)

| Stage | What | B200-h | Wall-clock on 4 GPUs |
|---|---|---|---|
| 0. Infrastructure | Freeze folds. WebDataset shards (512 px JPEG, the decode bottleneck). Judge-proxy scorer checked against a real LLM judge (κ ≥ 0.85). Noise audit. Zero-shot baselines: d1-3B, stock Gemma, SigLIP2 and MedSigLIP zero-shot. | ~10 | day 1 |
| 1a. Frozen-probe bake-off | Linear and MLP probes on cached features, per field, per slice. Encoders: SigLIP2 B/L/So400m, DINOv3-L, MedSigLIP-448, BMC-CLIP-CF, PE-Core-L, C-RADIOv4-SO400M, TIPSv2-So400m, EmbeddingGemma-2, Gemma-4-E2B tower, plus BiomedCLIP as a control. | ~15 | day 1–2 |
| 1b. CLM fine-tunes | Top 2–3 from 1a, full FT plus WiSE-FT. Then the winner's LoRA, 512 px and NaFlex variants. Mixture (SigLIP2 plus MedSigLIP logit fusion). nc ablation. 2 seeds. | ~45 | day 2–3 |
| 2. Jev teacher | E4B: F2, F3 (×3 context schemes) and F1 on one fold, then the winner on 5 folds. Tower-unfreeze run. 1,120-token run. 26B-A4B frozen readout, plus one LoRA run if warranted. | ~70 | day 3–5 |
| 3. Pseudo-labels plus student transfer | Ensemble labelling of about 870k images. T0/T5/T1 on E2B **and** LFM2.5-VL-3B (one seed each, then 3 seeds for the finalists). T-bridge. | ~90 | day 5–7 |
| **Total** | | **~230** | **about 7 days** |

**Minimum path (about 70 B200-h, about 2 days):**
1. Stage 0.
2. Stage 1a.
3. One SigLIP2-So400m FLYP run plus WiSE-FT.
4. d1-3B and stock-26B votes.
5. Pseudo-labels.
6. T0 vs T1 on the student allowed by the cap ruling.

That answers the only question that matters: do teacher labels make the submitted model better?

**Utilisation:**
- one run per GPU;
- frozen-feature extraction runs on GPU while CPU workers decode;
- CPU jobs on a low-priority (burst) queue;
- group memory is the binding limit, so CPU jobs request modest memory.

## 5. Questions to send the organizers now

1. **Parameter counting for Track 3:** do Gemma-4-E2B's per-layer-embedding tables (stored, not compute-active) count toward the 4B cap? This decides E2B vs LFM2.5-VL-3B.
2. **Licences:** is a model under the LFM Open License v1.0 acceptable as the submission (commercial use limited to under $10M annual revenue, with an exemption for qualified non-profits)? Is a student trained on labels from a HAI-DEF model (MedSigLIP) acceptable? Its terms treat outputs as not Model Derivatives, but other clauses cover distillation.
3. **Vision tower:** may the submitted VLM's vision tower be replaced (EmbeddingGemma-2 tower) or fine-tuned?
4. **Judging:** do partial-credit rules apply to multi-label body part and domain?
5. **Non-commercial data:** may it be used to train a teacher whose labels train the submission?

## 6. What changed from the earlier plan

- Medical encoders enter as a **second tower** (MedSigLIP-448), not a replacement. No single medical encoder covers endoscopy, EM and fluorescence. MedSigLIP's medical pairs are 98% histopathology patches. Evidence that medical encoders beat a fine-tuned SigLIP is thin, except frozen or few-shot.
- **Liquid d1-3B** is new: a free image-capable decision voter. **LFM2.5-VL-3B** is a real Track-3 student alternative if stored parameters count.
- **EmbeddingGemma 2** gives a text-aligned tower of exactly the E2B shape: the first credible way to transfer a medically adapted tower into the student, not only labels.
- Jev teacher for scale: **26B-A4B over 12B**. The Gemma-4-12B is encoder-free, so there's no tower to adapt, and the 26B-A4B has the image evidence.
- Data: 418 datasets and about 873k images, fully re-validated. The near-duplicate filter against mmbu-public was replaced; the old one dropped real images in 117 datasets.
