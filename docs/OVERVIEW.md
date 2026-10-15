# Pass-2 labeler: what to build and how to test it

Date: 2026-10-06. This synthesises the research reports under `docs/` (index: [README.md](README.md)):
- Jev: [models/jev_recipes_index.md](models/jev_recipes_index.md) and [models/jev_methods.md](models/jev_methods.md)
- CLM: [models/clm_methods.md](models/clm_methods.md)
- Data: [data/label_design.md](data/label_design.md), plus the audits in [data/audits/](data/audits/)
- Experiments: [experiments/experiment_plan.md](experiments/experiment_plan.md) and [experiments/b200_plan.md](experiments/b200_plan.md)
- Models: [models/backbones.md](models/backbones.md) and [models/medgpt_oss.md](models/medgpt_oss.md)
- How the data pipeline works and where fixes go: [data/pipeline.md](data/pipeline.md)

Nothing has been trained yet. The numbers below are counts from our own files, or come from those reports with their caveats.

## 1. The decision in one paragraph

Build the labeler as a **Jev-style decision model on Gemma 4**, with a **cheap CLM-style classifier on SigLIP2 beside it**.
- **Jev-style model:**
  - Start from the stock model before training anything.
  - Then train it in the competition's own JSON format, reading all fields in one pass, with a pointer or sigmoid readout for the large multi-label fields.
- **CLM-style classifier:** a cosine classifier whose label vectors start from text. It is a second voter and our only way to score classes with no training data.
- **Both are teachers, never shipped.** Rule 3 allows only one end-to-end VLM, so the shipped model is a Gemma-4 student fine-tuned on Pass-2 JSON from their labels.
- **Field order:** decode modality → submodality → body part → stain → specimen → domain, with domain **last**, not first.
- **Testing the three formulations:** all three are tested, but they cost the same compute. What actually differs is how much information flows between fields.

## 2. CLM vs Jev: which and why

| | Jev-style (decision model) | CLM-style (contrastive) |
|---|---|---|
| What it is for us | An image-capable LLM (Gemma 4) reads the image plus each field's options and outputs calibrated option probabilities. | An image embedding (SigLIP2) is scored against an embedding of each label's text, one softmax or sigmoid per field. |
| Evidence | A frozen Gemma-4-12B with letter readout is No. 1 of 106 on JevBench. A frozen Gemma-4-31B is rank 5 of 112 on the Decision Index. 7 of the top 13 open entries use Gemma 4. | CLM-8B ranks 101 of 112 with ECE 0.32. It uses a frozen Qwen3-8B (ineligible), it has no paper, and the multimodal version is unreleased. |
| Strength | Image and options interact, which helps fine distinctions. It also outputs the competition format. | Cheap: a frozen-feature baseline takes under 1 GPU-hour. Labels are text, so classes with zero training datasets can still be scored. That covers 6.3% of public body-part rows, 7% of stain rows and 3.7% of specimen rows. |
| Weakness | 4-5x the compute. No precedent for medical images or a tuned vision tower. Gemma-4 masking quirks. | A dual encoder loses on fine distinctions that depend on context. In practice it is a cosine classifier. |
| Role | Main teacher. | Second voter, unseen-class scorer and baseline. |

The backbone report found that **Pass 2 is a perception problem with a closed vocabulary, not a knowledge problem**:
- an 8B open model is within about 0.02 of frontier models on MMBU closed classification;
- general models beat medical ones on modality recognition (OmniMedVQA);
- frozen vision-tower probes beat a VLM's greedy answers on MedMNIST until the VLM gets supervised fine-tuning.

So we don't need a big "reasoning" LLM. The vision tower and the training data decide the result.

## 3. Your three formulations, made concrete

| | F1 all at once | F2 independent | F3 progressive |
|---|---|---|---|
| Jev | One sequence: image, then every field's question block and options. One pass gives probabilities for all six. Options may see earlier fields' option states. | Same packed sequence, but each field block sees only the image and its own options. This is mathematically identical to six separate passes (unit test U1), at the cost of one. | Coarse to fine: the accepted earlier answers are written into the context ("modality: x-ray imaging"), then the next field's options are shown, optionally pruned to compatible ones. |
| CLM | One image embedding against all 250 labels in one shared space (union loss). | One attention-pooled query per field over the image patches, with per-field losses. | Each field's query is conditioned on the embeddings of the labels already chosen (cross-attention, FiLM or concatenation). |
| Shipped VLM | — | — | **This is what the generative JSON already is.** Each key is decoded after the earlier ones, so the JSON key order is a free design choice to test. |

**F1 vs F2 costs the same compute.** The real variable is information flow between fields:
- none (F2);
- implicit and soft (F1);
- explicit and committed (F3).

F3's danger is exposure bias. With dataset-level labels, the gold context ("ophthalmology, fundus photography") nearly identifies the dataset. The model then learns "this context means this label" and stops looking at the image. So F3 is trained three ways and compared on held-out *datasets*:
- with predicted context (out-of-fold);
- with context dropout;
- with scheduled sampling.

**Order: domain last.** Measured on our data ([data/label_design.md](data/label_design.md) section 3):
- submodality fixes modality on 93% of rows;
- domain is the least transferable field: 62% accuracy on held-out datasets from (submodality, body part), against 92% in-sample;
- domain is mostly a function of submodality and body part, and the least visible field.

Committing to domain first would push its errors into every later field. So the default order is modality → submodality → body part → stain → specimen → domain. Domain-first is still in the matrix as variant O1, so the data can disagree with us.

## 4. Label design decisions (from data, not intuition)

- **Which fields to ask for.** The Pass-2 prompt lists the requested fields, so applicability is an input, not something to predict.
  - Modality, submodality, specimen and domain are annotated on 100% of public rows. Specimen is annotated for radiology too, as "human".
  - Stain appears only on microscopy: 572 of 874 microscopy rows, and never elsewhere.
  - Body part is missing on 309 rows: 278 stain-only microscopy rows and 31 clinical photos.
- **Domain is multi-label.** Parse the list, split each item on commas, lower-case, alias, deduplicate, and treat it as an unordered set.
  - "obstetrics and gynecology" is **one** class, and nothing is ever split on "and". "radiology, urology" is **two** classes.
  - This rule was verified on all 76 raw domain strings with zero unmapped atoms.
  - Train it as a sigmoid per domain; at inference keep the top one plus any domain above a threshold, at most 3.
- **Body part:** a multi-hot set of atoms, plus an optional 14-way family head.
  - The "|" lists are mostly hierarchy pairs like chest|lung (271 of 379 rows), and a few are inventories of several organs.
- **Hierarchy:** submodality is masked to the options legal under the predicted modality. The compatibility table is many-to-many: confocal microscopy sits under three modalities.
- **Registry fixes now applied in `src/bio_waila/data/select.py`:**
  - drop "radiology" when a specialty is present (108 datasets);
  - submodality implies body part (fundus → eye, dermoscopy → skin, chest x-ray → chest, mammography → breast, colposcopy → cervix);
  - cryo-ET and darkfield mean stain "unstained", and synthetic fluorescence means stain "synthetic", which was a bug before;
  - cell-line specimen becomes "human", following MMBU's convention;
  - sperm datasets get body part "sperm cells".
  - Result: hard body-part labels went from 140 to 260 datasets.

## 5. Data status

- **Plan size (updated 2026-10-07):** 418 datasets and about 873k planned images (409 permissive, 9 in the toggleable non-commercial layer), split 364 train and 54 validation datasets by dataset group. All were re-validated at image level (docs/data/validation). The final model plan is in `experiments/training_plan.md`. Per-submodality targets follow the full benchmark's mix, with an 8k floor.
- **mmbu-public:** evaluation only. Near-duplicates of it are removed from training.
- **MMBU-source datasets** (high overlap) are allowed, since this labeler is never submitted. 92 are in the plan.
- **Audits:** every selected dataset was audited from metadata only, in 9 audit batches (A, B, C1–C4, D, E). They found:
  - 95 datasets to drop: no pixels, masks only, restricted or embargoed, unsupported host, wrong record, unreadable format, or spectral or power Doppler with no matching class;
  - about 20 re-routes to better or moved sources;
  - about 75 label corrections;
  - about 30 path exclusions (synthetic subsets, model weights, mask trees, duplicate bundles);
  - a Figshare-ID truncation bug (fixed);
  - many matcher and pipeline fixes (applied).
  - All corrections live in `plan/registry_fixes.json`.
- **New datasets:** two discovery agents added 36 registry entries for rare classes. Stand-in entries (power Doppler, generic multichannel fluorescence) are kept out of the labels they don't truly match.
- **Remaining gaps:**
  - Specular microscopy: no permissive downloadable data exists.
  - Colonoscopy (645 planned images), gastroscopy (2,300), color doppler (3,240) and differential fluorescence (3,000) are small, but all above their full-benchmark shares (289, 197, 159 and 119 images).
- **Credentials:** Kaggle (26 datasets) works through `KAGGLE_API_TOKEN` in `.env` (or `~/.kaggle/access_token`), with per-file sampling and rate-limit backoff. Mendeley (18 datasets) needs a browser download into `raw/<id>/archives/`.

## 6. Backbones

| Role | Pick | Not chosen |
|---|---|---|
| Jev teacher | Gemma-4-E4B (prototype), then Gemma-4-12B or 26B-A4B | Moondream 3.1: no HF/torch path for 3.1, Photon/FP8 format, card excludes medical data, strengths are pointing and counting. DiffusionGemma: uncalibrated. MedGPT-OSS: 21B total with CLIP-L, gated, CC BY-NC, no modality/organ results. |
| CLM-style classifier | SigLIP2-So400m (fine-tuned against its own text tower), DINOv3-L as a second view | CLM-8B (Qwen backbone, poor calibration) |
| Shipped student | Track 3: Gemma-4-E2B (subject to the per-layer-embedding ruling). Track 2: Gemma-4-26B-A4B. | |

## 7. Experiments, in order

The full matrix, metrics, decision rules and statistics are in [experiments/experiment_plan.md](experiments/experiment_plan.md) (about 660 H100-hours). The minimal path, about 115 H100-hours, answers both questions that matter:

1. **Stage 0, infrastructure (about 20 H100-hours):**
   - freeze the dataset-grouped folds;
   - build a judge-proxy scorer and check it against a real LLM judge, requiring kappa ≥ 0.85;
   - write the unit tests (U1: packed independent mode equals 6 separate passes);
   - run baselines B1-gem (stock Gemma-4 letter readout, no training), B1-sig (SigLIP2 zero-shot) and B0 (frozen SigLIP2 with linear and MLP heads).
2. **Pilot, about 128 images per dataset (about 50k images), one fold, two seeds:**

   | Run | Model |
   |---|---|
   | B0-ft | SigLIP2 with LoRA and 6 heads |
   | J-F2, J-F1c, J-F3-O2-PC | Jev |
   | C-F2q, C-F3-O2-PC | CLM |
   | B2-p, B2-cf | direct Gemma-4 JSON fine-tune: prompt order, coarse-to-fine order |

3. **Transfer test:**

   | Run | Fine-tuned on |
   |---|---|
   | T0 | hard labels only |
   | T5 | T0 plus pseudo-labels from B0-ft |
   | T1 or T2 | T0 plus pseudo-labels from the best Jev or CLM model |

   **Adopt a teacher only if** its student beats T0 by at least 1 point of judged C *and* beats the self-training control.

The answer is not fixed in advance. If the plain SigLIP2 fine-tune (B0-ft) is within noise of Jev and CLM as a teacher, we keep the cheap one.

## 8. Open questions and risks

- **Organizers:**
  - Do Gemma-4-E2B's per-layer embeddings count toward the 4B cap?
  - Is a swapped or tuned vision tower allowed in the harness?
  - Do the judge's partial credits apply to multi-label domain and body part?
- **Label noise:** every image inherits its dataset's label. A stage-0 audit re-labels 300 held-out images per field to estimate the noise.
- **Long tail:**
  - stain and specimen: the median class appears in only 1 training dataset;
  - body part: 45 of 102 classes have fewer than 3.
  - Held-out-dataset validation cannot measure those classes. Text-labelled models (CLM, Jev) are the only ones that can score unseen classes.
- **Engineering:** Gemma 4's sliding-window layers versus custom attention masks. Fall back to a prefix KV-cache with batched option rows.
