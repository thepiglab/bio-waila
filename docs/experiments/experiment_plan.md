# Pass-2 labeler experiment plan: all-at-once vs independent vs progressive, for Jev-style and CLM-style models

Date: 2026-10-06. Scope: experiments only (recipes are covered by other notes). Everything here is a plan; no number below is a measured result, except the label statistics in section 0, which were computed from `plan/eval_public.parquet` and `plan/label_space.json`.

## 0. Framing, facts that shape the design

**What is and is not submitted.** The submission is one end-to-end generative VLM that emits Pass-2 JSON in a single pass (rules 3 and 5: no ensembles, cascades, auxiliary models, repeated or iterative prompting). So the Jev-style and CLM-style models are teachers, labelers and probes. They are never submitted, which means:
- Teachers may be ensembles, multi-pass, F3 sequential, beam, permutation-TTA. Only the student must be one pass.
- What transfers to the submission is (a) label coverage and precision for the masked/candidate rows, (b) calibrated confidence used to filter or weight pseudo-labels, (c) the field order and format chosen for the JSON (a generative VLM is already an F3-with-teacher-forcing chain, see 1.3 and section 2), (d) soft targets.
- The experiments therefore have two questions: Q1 which formulation is the best teacher (accuracy, calibration, coverage at 95% precision), and Q2 whether any teacher stage beats direct SFT of the VLM (B2) on the real Pass-2 score.

**Facts from the local files (public split, n = 3,147; train plan: 393 datasets, about 763k images, 340 train / 53 val datasets split by group).**
| Item | Value |
|---|---|
| Annotated fields per public row | 4 fields: 170 rows; 5 fields: 2,687; 6 fields: 290 |
| Annotated rate | modality 100%, submodality 100%, domain 100%, specimen 96.9%, body part 90.2%, stain 16.8% (528 rows; 415 light microscopy, 82 fluorescence, 23 EM, 8 digital pathology) |
| Multi-label | body part avg 1.17 labels (max 5); domain avg 1.19 (max 3); 8 rows have 2 modalities and 2 submodalities |
| Class counts (train vocab / appearing in public) | modality 11/11; submodality 37/35; body part 102/65; stain 39/17 (plus "uniform test slide", not in vocab); specimen 38/20; domain 23/23 |
| Public rows with a label that has zero training datasets | submodality 12 (differential fluorescent, specular); body part 178 of 2,838 (6.3%); stain 37 of 528 (7.0%); specimen 112 of 3,048 (3.7%); domain 0 |
| Label supervision in train plan | stain is masked for 325 of 393 datasets; body part is `candidates` for 222 datasets and masked for 27; modality is hard for 377; domain is `multi` for all |
| Fields requested at test time | The Pass-2 prompt lists the annotated fields. So the model is told which fields to answer; no "is stain applicable" head is needed. The requested-field set is an input, and loss and scoring are masked by it. |

**Indicative dependency statistics (public labels, in-sample majority-vote lookups, so optimistic).** submodality given modality 0.77; domain given modality 0.60; domain given submodality 0.71; domain given submodality and body part 0.93 (118 cells over 3,147 rows, so partly memorisation); body part given submodality 0.60; specimen given submodality 0.91; stain given submodality 0.53. Reading: domain is mostly a function of (submodality, body part) and is the weakest-evidence field to put first, while specimen and submodality are cheap given the modality. Caveat on rule 6: these numbers use public labels to motivate ordering. Before the order variants are frozen, recompute the same table on train-registry labels only and cite that one; keep the public numbers out of any decision record.

**Backbones (all non-Chinese origin).**
- Jev-style torso: Gemma-4-E4B (already used by `src/bio_waila/data/autolabel.py`; at most 4B active, vision tower included in the budget); fallback Gemma-3-4B-it if the E4B text-only attention masking is awkward. Verify parameter counts against rule 3 when freezing.
- CLM-style: SigLIP2 So400m (Google) image tower; label tower is the SigLIP2 text tower (default) or EmbeddingGemma-300M (ablation). Both are shown to be within the 12B cap if later merged.
- B0/B1: SigLIP2 So400m frozen, BiomedCLIP (Microsoft) for zero-shot.
- B2 student: the same Gemma-4-E4B base as the Jev torso, so teacher and student share a vision tower and tokenizer.


## 1. Exact definitions of F1 / F2 / F3

Notation: fields `f` in {modality (M), submodality (S), body_part (B), stain (T), specimen (P), domain (D)}; options `O_f` (sizes 11, 37, 102, 39, 38, 23 = 250 in total); `R(x)` the requested-field set (input); label modes from `README.md` (`hard`, `multi`, `candidates`, `masked`). Single-label fields: M, S, T, P. Multi-label fields: B, D (and, rarely, M and S on 8 public rows; treated as single-label in training, multi in eval via any-match).

Common loss per field, applied only if `f in R(x)` and the label mode is not `masked`:
- `hard` single-label: cross-entropy over the (possibly pruned) option set.
- `candidates` single-label: `-log sum_{c in cand} p_c` (partial-label).
- `multi` (B, D): per-option binary cross-entropy on sigmoid pointer logits (reference), positives = values, negatives = all other options in `O_f`. For B, negatives are down-weighted (0.3) because dataset-level multi labels are incomplete (a chest film labelled "chest" is also "lung"). Alias groups (cell/cells, breast/breast tissue, colon/colon tissue, lung/lung tissue, blood/blood cells) share positives with weight 0.5 (alias smoothing, ablation A-alias).
- `candidates` for multi fields: noisy-or, `-log(1 - prod_{c in cand}(1 - p_c))`, no negative terms on candidates.
- Loss weight per field w_f = 1, normalised by number of requested annotated fields in the row, to mirror the per-example normalisation of C.

### 1.1 Jev-style (cross-encoder with option tokens, pointer head)

**Sequence layout (one image).**
```
[BOS] <task prompt> <image tokens> [CTX block, F3 only] [FIELD BLOCK_f for each f in R(x), in layout order]
FIELD BLOCK_f = <F_f> "field: body part. choose all that apply." <Q_f>
                { <o> option text </o> } x |O_f|
```
- Readout: the hidden state at `</o_j>` (last layer, concatenated with a mid layer through a small gate) is the option key. `<Q_f>` is a query token placed after the header (it sees image and header and, if the mask allows, the field's options).
- Pointer head (reference): `s_j = (W_k h_{o_j}) . (W_q h_{Q_f}) / sqrt(d) + b_{f}`, `W_k`, `W_q`: d to 256, about 1M parameters, shared across fields with a learned field bias `b_f` (so a new field needs only a new header string). Ablation A-head: scalar head `s_j = w . MLP(h_{o_j})`.
- Single-label fields: softmax over `s_j` within the field (Jev `Choice`). Multi-label fields: sigmoid of `s_j` per option (Jev `Noul`, yes/no per option); an optional second head predicts k in {1, 2, 3+} for top-k decoding (ablation A-ml).
- Body part, 102 options: about 5 tokens each with markers, around 520 tokens for the block. All six blocks together are about 1.3k option tokens; with 256 image tokens and the prompt the F1 sequence is about 1.9k tokens, well inside the context. No windowing needed. Option order is re-shuffled per sample in training and fixed (alphabetical) or averaged over 4 permutations at teacher inference.
- Position ids: every option in a field block starts at the same position offset (parallel-context style); blocks are offset sequentially. This plus the mask below makes outputs invariant to option order.

**Attention mask (the main F1/F2 lever).** Three mask tiers, implemented as a 4-D boolean mask (PyTorch SDPA or FlexAttention; sequences are about 2k tokens so cost is minor).
| Mask | Options in a field see | Fields see each other? |
|---|---|---|
| `iso` | image, prompt, own header, own tokens only (no other option) | no |
| `fieldcausal` | as `iso` plus all option states of earlier fields in layout order | one way, earlier to later |
| `bidir` | image, prompt, all option tokens of all fields (prefix-LM style) | yes |
Naive concatenation (`naive`: plain causal over all options, fixed order) is kept only as an order-bias reference.

**F2 (independent): `iso` mask, field blocks each see only their own options.** Because the image prefix is shared and positions are reset, this is mathematically the same as six separate forward passes each containing the image plus that field's block. Unit test U1 asserts equality of logits (max abs diff below 1e-3 in bf16) between the packed `iso` pass and six real independent passes; we then train and serve F2 in the packed form (cost equals F1, no 6x image recompute). A genuinely separate set of six per-field models (`J-F2sep`) is kept as a single reference run to measure multi-task interference in the shared LoRA.

**F1 (all-at-once): one pass, entire schema shown, probabilities for every requested field at once.** Reference F1 is `fieldcausal` with layout order coarse to fine (domain, modality, submodality, body part, specimen, stain); `bidir` is a variant. Because option text is constant across images, cross-field visibility carries no direct information; it matters only because option states are already image-conditioned, so a body-part option can read a modality option's state ("this image looks like x-ray"): an implicit, soft, single-pass chain with no commitment. Acceptance in F1: per-field threshold `tau_f` on max probability (single-label) or per-option thresholds (multi-label), calibrated on held-out datasets (section 4).

**F3 (progressive): coarse to fine with commitments.** A sequence of up to six passes. Pass k state = image + CTX block (text of accepted labels so far, e.g. "modality: x-ray imaging; submodality: chest x-ray") + only field k's options (pruned by compatibility). The CTX block is full text, so the same model weights serve any context length.
- Order variants: O1 domain, modality, submodality, body part, specimen, stain (coarse by semantic breadth, the user's example); O2 modality, submodality, body part, specimen, stain, domain (visual first, domain last because domain is largely implied by body part and submodality); O3 confidence-adaptive (always ask the field with the highest F2-estimated confidence next, the "easy first" order; stretch run).
- Option pruning by compatibility tables (from train-registry datasets plus `taxonomy.toml`): hard prune keeps options with support in at least 1 training dataset given the context (smoothed: keep top-3 unpruned F2 options as an escape hatch); soft prior adds `log(p(c|ctx)+eps)` to the logit. Diagnostic: prune recall (share of gold labels kept in the pruned set) on held-out folds; if below 99% use soft prior only.
- Acceptance policies at inference (all evaluated on every trained F3 model; no extra training): P1 greedy commit (always append top-1, needed for C scoring); P2 confident-only (append only if p at least `tau_f`, else append nothing, which is why context dropout is trained); P3 soft context (append top-3 labels with probabilities as text); P4 beam B=3 over the chain with chain-rule marginalisation `p(y_k|x) = sum_ctx p(ctx|x) p(y_k|x,ctx)`.
- Training context: TF teacher forcing (gold context); PC predicted context (out-of-fold predictions from the matching F2 model, 3-fold cross-fit on train datasets grouped by dataset, so errors look like test errors); SS scheduled sampling (gold replaced by the model's own detached top-1 with probability ramping 0 to 0.5 over training, requires online rollouts, about 3x cost); all variants add context dropout (each earlier field masked with p = 0.3) and, for TF, confusion-matrix corruption (gold replaced by a draw from the F2 validation confusion with rate equal to F2 val error). Packed TF training is a single pass: gold-answer tokens of earlier fields are inserted after each block and later blocks may attend to them.
- Exposure-bias hypothesis (explicit): under dataset-level labels the gold context (e.g. "ophthalmology, fundus photography") nearly identifies the dataset, so a TF model can learn "context implies label" and under-use the image; it scores well on in-dataset validation and degrades on unseen datasets and under predicted context. Predicted: TF < PC ~ SS on held-out datasets; TF >= PC on same-dataset validation.

### 1.2 CLM-style (dual encoder)

Image tower `g` (SigLIP2 So400m, LoRA or last-N-blocks) gives patch tokens `P` and a pooled embedding `u`. Label tower `h` embeds each option text `"<field>: <label>"` (with synonyms from the alias map; ablation A-text adds a one-sentence LLM-written definition). Score `s(x,c) = exp(t) * <u_f, v_c> + b_f` with `v_c = normalise(h(c))` cached per field.
- **C-F1 (all-at-once): shared space, one pooled `u`, one loss over the union of all 250 options.** Two losses: multi-positive sigmoid (SigLIP-style, per image-option binary; naturally gives calibrated per-option probabilities for every field, handles multi-label) and union softmax with multiple positives (one per annotated field). The union makes labels of other fields into negatives ("chest" competes with "chest x-ray"), which probes whether a single cross-field embedding space helps or interferes.
- **C-F2 (independent per field).** Two sub-variants. (i) `C-F2s`: single `u`, per-field grouped softmax: this is identical to the per-field restriction of C-F1 at inference and identical in loss to a sum of per-field cross-entropies, so it is a collapse case; one run to verify and to give the per-field-normalisation baseline. (ii) `C-F2q` (the true F2): field-conditioned query, `u_f = MAP(P; q_f)` where `q_f` is the label-tower embedding of the field name fed to a one-layer attentive pooling (cross-attention) over patch tokens, plus a per-field linear projection; the image tower runs once, six cheap pooling heads run on top. This is the dual-encoder analogue of "one pass per field", and it lets body part look at local structure while modality uses global features.
- **C-F3 (progressive).** Context labels condition the query: `q_f = emb(field) + sum_{g before f} W_g v_{ŷ_g}`. Conditioning variants: `cat` (MLP over [u; v_ŷ1; ...]), `film` (gamma, beta from mean of context label embeddings modulating `u_f`), `xattn` (context label embeddings are extra query tokens in the attentive pool, giving label-conditioned localisation, e.g. "retina" plus "fundus" attends to the disc). Soft context uses the expected embedding `sum_c p_c v_c`. Training context: TF, PC (out-of-fold from `C-F2q`), SS (almost free here, heads are tiny), plus the same context dropout and corruption as 1.1. Pruning: same compatibility tables as the logit mask or soft prior.
- Dual-encoder property to exploit: option embeddings are text, so classes with zero training datasets (6.3% of public body-part rows, 7% of stain rows, 3.7% of specimen rows) can still be scored. A-text and the label-tower choice determine how well. Every method that has a fixed learned weight per class (B0) cannot score them at all.

### 1.3 Where the formulations collapse and where they differ
| Setting | F1 vs F2 | F3 vs F2 |
|---|---|---|
| Frozen features + independent heads (B0) | identical (heads share nothing but the features); joint MLP trunk is the only coupling | n/a unless context embeddings are added to the head input |
| CLM, single `u`, per-field grouped softmax | identical in function and loss | differs only through the context-conditioned query |
| CLM, field-conditioned query (`C-F2q`) vs shared `u` + union loss | differs: query conditioning and normalisation | |
| Jev, `iso` mask | identical in function to F2 (unit test U1); F1 = F2 | |
| Jev, `fieldcausal` / `bidir` | differs: implicit cross-field sharing through image-conditioned option states | |
| F3 with context dropout = 1 or a model that ignores the CTX block | | collapses to F2 |
| F3 with full-chain beam and exact marginalisation | approaches a joint model, the ideal F1 | |
| B2 generative JSON, keys in prompt order | | is F3 with TF in prompt order (autoregressive chain), with free-text answers |
So the design does not compare "one pass vs many passes" for compute reasons (compute is equal under caching); it compares **information flow between fields**: none (F2), implicit and soft (F1-fieldcausal/bidir; C-F1 union), explicit and committed (F3 with TF, PC or SS).

## 2. Baselines (exact)
- **B0-lin / B0-mlp.** Frozen SigLIP2 So400m (patch16, 384 px; pooled embedding and mean patch token concatenated, 2 x 1152), cached once for the 763k train images and 3,147 public images. Six heads trained jointly on cached features (linear; or MLP 2304-2048-GELU-dropout 0.2): CE for hard, partial-label for candidates, BCE for multi, masked fields dropped; class-balanced sampling by dataset; label smoothing 0.05; temperature scaling per field after training. By construction F1 = F2 here. B0-ctx adds the gold-or-predicted context to the head input (label embedding concatenated) to test F3 without any text model.
- **B0-ft.** B0 with the ViT fine-tuned (LoRA r=32 on all blocks or last 8 blocks unfrozen; the better one at pilot), 6 heads, same losses. Expected to be the strongest plain predictor and the main competitor for CLM and Jev.
- **B1-sig / B1-bio.** Zero-shot, no training: SigLIP2 (and BiomedCLIP) image-text similarity with per-field prompt templates ("a medical image of {label}" for modality and submodality, "a {label} specimen" etc.; 3 templates averaged), per-field softmax over `O_f`, optional pruning by compatibility. B1-gem: Gemma-4-E4B zero-shot Jev-style option scoring (log-likelihood of each option text given the Pass-2 style question, restricted to `O_f`; no pointer head) to measure what the LoRA and pointer head add.
- **B2-sft.** The submission format: Gemma-4-E4B (LoRA r=64 plus unfrozen projector; vision tower frozen at pilot, last blocks at full) fine-tuned to emit the exact Pass-2 JSON for the exact prompt (random requested-field subsets drawn from the public annotation-pattern frequencies, fields listed in alphabetical order as in the prompt) with a short templated `perception_rationale`. Variants: B2-p (keys emitted in prompt order), B2-cf (keys emitted in the fixed order domain, modality, submodality, body part, specimen, stain, whatever the prompt order is: JSON key order is free), B2-ff (fine to coarse, domain last), optional B2-rat (rationale first). Targets use hard and multi labels only: for a row whose field is masked or `candidates`, that field is simply not requested in the training prompt (and so absent from the JSON target), so B2 is the clean "no teacher" reference; teacher-based variants are the T runs (section 4, stage 3).

## 3. Metrics

All metrics are computed per field on rows where the field is annotated (`z_ik = 1`), exactly as the challenge scores, on three evaluation sets: **val-OOD** (held-out datasets, section 4), **public** (3,147 rows), and for the student **public judged** (real LLM judge). Public is reporting only (section 4 rule on looks).

**3.1 Per-field quality.**
- Single-label fields: top-1 accuracy, macro-F1 over classes present, and accuracy by class-frequency bucket (head: at least 10 train datasets, mid: 3-9, tail: 1-2, unseen: 0). For M and S, also the modality-submodality joint accuracy.
- Multi-label fields (B, D): `hit@1` (top-1 in reference set), set-F1 and exact-set match under the decoding policy (top-1 plus any option with p at least `tau_multi`, `tau_multi` fit on val), and mAP over options.
- Classes are compared after alias-merging (cell/cells, breast/breast tissue, colon/colon tissue, lung/lung tissue, blood/blood cells, "tissue" variants); the unmerged number is also logged.
- Slices that must be reported for every run: by submodality group (microscopy, endoscopy, radiology, optical), unseen-class rows, within-dataset-varying rows (rows whose label differs from the dataset's majority label), and rows where stain is requested.

**3.2 Judge-proxy C score (C-hat).** Exactly the challenge definition: `c_i = (1/|A_i|) sum_{k in A_i} m_ik`, `C-hat = mean_i c_i`, `A_i` the annotated fields. Credit `m_ik` is computed by a proxy to be calibrated against a real LLM judge.
- `C-hat-strict`: single-label field 1 if prediction (after the vocab alias map in `src/bio_waila/data/vocab.py`) is in the reference set else 0; multi-label field Jaccard of predicted and reference sets.
- `C-hat-soft`: synonym pairs 1.0, parent/child or related pairs 0.5 (hierarchy table built from `taxonomy.toml`: lung vs lung tissue, eye vs retina, abdomen vs abdominal organs), multi-label field soft-F1.
- Calibration of the proxy (stage 0 gate): sample 800 (field, prediction, reference) tuples stratified by field and by error type from B2 outputs and classifier errors; score them with a local Gemma-4 judge (rules: judges are interchangeable at kappa 0.96) and with Sonnet-5 on a 200-tuple subset using a reconstructed rubric ("semantic equivalence, score in [0,1]"). Report Pearson r of C-hat vs judge-C over resamples of 500 rows and Cohen kappa at the 0/1 level; gate: kappa at least 0.85 for the soft proxy, otherwise rebuild the credit table from the judge sample (fit credit by pair type). Classifier outputs and B2 free text are always compared under the same judge (classifier prediction strings go through the same judge prompt) when the question is "classifier vs B2".
- Also report C-hat if the 'unknown' policy is used (never beneficial, but check B2 does not emit it).

**3.3 Calibration.**
- ECE (15 equal-mass bins, top-label confidence), per field, plus classwise-ECE over all option marginals for multi-label fields; Brier score (multi-class sum for single-label, mean over options for multi-label); NLL.
- Both pre- and post-temperature (T fit per field on the calibration half of held-out datasets). The gap pre vs post, and the ECE on a different set of datasets than the one T was fit on, measure shift robustness. Reliability diagrams on OOD datasets only.
- Selective prediction: risk-coverage curve and AURC per field.

**3.4 Coverage at 95% precision (the pseudo-labeling metric).**
- For each field the threshold `tau_f` is the smallest value whose accepted set on the calibration datasets has precision at least 0.95 with a one-sided lower confidence bound (cluster bootstrap over datasets, 5th percentile of precision at least 0.95).
- Reported: coverage (fraction of rows accepted), realised precision at that `tau_f` on the held-out report datasets and on public (does the threshold transfer?), and dataset-level coverage (share of datasets with at least 50% rows accepted). A formulation with high coverage at a threshold that fails to hold precision on the other half is penalised: the metric used for decisions is coverage subject to realised precision at least 0.94 on the report half.
- Joint acceptance for F3: a row's later field label is accepted only if all earlier committed fields that it was conditioned on were accepted (chain acceptance) under P2; also report marginal acceptance.

**3.5 Consistency violations.**
- Hard: submodality not in the allowed set of the predicted modality per `taxonomy.toml` (the table is many-to-many: brightfield microscopy is allowed under light microscopy and digital pathology).
- Soft: (submodality, body part), (modality, stain), (submodality, specimen) pairs with zero support in the training registry (counted at dataset level); stain on non-microscopy rows; "x-ray imaging" with brain; and so on.
- Rate per 1,000 rows and rate among correct-field rows (violations that have an incorrect field are separated from violations among rows whose other fields are right). F3 with pruning is near zero by construction, so it is always reported together with accuracy; an F3 run with zero violations and lower accuracy is a loss, not a win.

**3.6 Latency and throughput.** On one H100 80 GB (and B200 if scheduled), bf16, batch 1 and batch 32, image resolution fixed, warm cache, median and p95 of end-to-end ms per image and images per second:
- F1 one pass; F2 packed (equal) and, for reference, six real passes with prefix KV cache; F3 sequential 1-6 passes with prefix KV cache (image prefix computed once), greedy and beam B=3; CLM image tower plus cached label embeddings; B0; B2 autoregressive generation of about 150 tokens.
- Projected wall-clock to label all 763k images and share of a day of one GPU.

**3.7 Transfer to the VLM (the decisive metric).**
- Student runs T0-T6 (section 4, stage 3) are scored on public with the Pass-2 prompt and a local Gemma-4 judge: real `C` (judged), plus per-field judged accuracy. To bound forgetting, the Pass-1 task score T is measured with the `mmbu-eval` harness on public for every student, with a fixed Pass-1 replay mixture in the SFT data; the headline is `S = 0.8 T + 0.2 C` (and the Track-3 efficiency factor is constant at fixed parameters), alongside C alone.
- Report: delta C and delta S of each student against T0 (same images, hard labels only) and against T-self (B2 self-training control: same extra images, labels from B2's own confident predictions).
- Teacher-side "label supply" diagnostics: number of newly supervised (image, field) pairs, estimated pseudo-label precision (from held-out datasets), and share of pseudo-labels on classes with zero hard-label datasets.

**3.8 Secondary diagnostics.** Option-order sensitivity (fraction of predictions that change under permutation of option layout, Jev only), dataset-identifiability probe (accuracy of a linear probe of final features on dataset ID within held-out datasets), accuracy vs nearest-neighbour similarity to training images, robustness to image perturbations (JPEG q=40, 50% downscale, grayscale, border crop, text overlay removal).

## 4. Protocol

### 4.1 Data and splits
- **Never train, select thresholds, or fit temperature on mmbu-public.** Public rows are reporting only. Enforced by an assertion in the data loader (`dataset_id != "mmbu-public"` and no row whose `unique_id` appears in the eval parquet) and by the `src/bio_waila/data/prepare.py` dHash filter (at most 4 bits). Added in stage 0: a SigLIP2-embedding near-duplicate scan (cosine at least 0.95) of public against train, with the share of public rows that have a near neighbour reported and those train images dropped.
- Note on rule 6 (no "optimizing using MMBU evaluation examples"): `src/bio_waila/data/autolabel.py` and `README.md` currently say thresholds are calibrated on public rows. This plan instead calibrates on held-out train-registry datasets (val-OOD); recommend changing `src/bio_waila/data/autolabel.py` accordingly or disclosing it. Public is looked at only for pre-registered looks (below), all logged in an append-only file (`public_looks.jsonl`, one line per run) so the number of looks is auditable.
- **Held-out datasets.** Group splits only (the `related_ids` groups from `src/bio_waila/data/select.py`), never image-level. Folds:
  - `F0`: the official 53 validation datasets held out.
  - `F1`, `F2`: two further disjoint blocks of about 50 datasets drawn from the 340 train datasets, stratified by submodality, with the constraint that every class with at least 3 datasets keeps at least 1 train dataset per fold; train on the rest (the F0 datasets are allowed in the training side of F1 and F2).
  - Within each held-out block, the datasets are split into calibration half (A) and report half (B) at dataset level (swap and average for the final numbers); temperature scaling and `tau_f` are fit on A and reported on B and the other way round.
  - Each training side also keeps a 2% image-level slice per dataset for early stopping (ID-val), never used for decisions.
  - Metrics on classes that never appear in a fold's training side are reported separately as "unseen" and excluded from the headline.
- **Pilot data.** At most 128 images per dataset (about 49k images over 393 datasets), with a floor of 300 images per submodality; per-dataset sampling is uniform over images; the full-data stage uses the planned 763k with per-dataset cap 2,000 (6,000 for rare classes) and dataset-balanced sampling (probability proportional to the square root of dataset size).
- **Augmentation.** Random resized crop (scale 0.6-1), no flips (laterality and orientation cues matter for chest and body part), mild brightness and contrast, JPEG compression; no hue or saturation jitter (stain colours are the label); random erase of border and corner text overlays. Resolution: SigLIP2 384 px; Gemma-4 default image-token budget (256 tokens) for Jev and B2.
- **Prompting.** One fixed Jev prompt per field (task description plus field description from `label_space.json`), frozen before stage 1a; no prompt tuning on public.

### 4.2 Seeds and reproducibility
- Seed set {0, 1, 2} controls head and LoRA initialisation, data order, option shuffling and context-dropout masks; folds are fixed by split seed 17 and never change after stage 0.
- Stage 1a: 1 fold (F0) x 1 seed (screening only; gaps smaller than 2 x the stage-1b seed std of that family are not interpreted). Stage 1b and later: 3 folds x 3 seeds for finalists (stage 1b), 2 seeds at full scale for the top-2 finalists (the cost of a third seed is only paid if the CIs overlap).
- Every run writes: git hash of the repo, the config, split hash, per-image predictions and probabilities (parquet; required for paired bootstrap, calibration and pseudo-labeling), and a metrics JSON. Uniform eval script `src/bio_waila/eval/pass2.py` (to be written) takes only the prediction parquet, so all families go through identical metric code.

### 4.3 Compute budget (estimates, assume H100 80 GB at about 35% MFU; B200 about 1.7x faster)
- Jev (Gemma-4-E4B LoRA, F1 and F2 sequence about 1.9k tokens, pilot 42k images x 3 epochs): about 3 H100-h per run; F3 TF packed about 3; F3 PC about 3 plus one shared 3-way out-of-fold F2 cross-fit (about 6); F3 SS about 8; full data (763k, 2 epochs) about 45 H100-h per run; teacher inference over 763k images (F2 packed, forward only) about 7 H100-h, F3 with prefix cache 2-3x that.
- CLM (SigLIP2 So400m LoRA plus text tower; pilot): about 1.2 H100-h per run; full data about 12.
- B0: feature extraction about 1.5 H100-h plus CPU decode (I/O bound); heads minutes. B0-ft: about 1.5 pilot, about 10 full. B2: about 2 pilot, about 17 full.
- Totals (rounded): stage 0 about 20 H100-h; stage 1a about 100; stage 1b about 170; stage 2 about 220 (incl. pseudo-labeling); stage 3 about 150. About 660 H100-h, plan with 15% contingency (about 760); hard cap 800, no stage starts unless the previous stage's decision rule fired. 1 GPU per run (DDP over 4 GPUs for the full-data Jev and B2 runs).

### 4.4 Stages
**Stage 0 (infrastructure, no claims): about 20 H100-h.** Freeze folds, vocab and prompts; run `src/bio_waila/eval/pass2.py` on B1-sig, B1-bio, B1-gem, B0-lin, B0-mlp (pilot features); unit tests U1 (packed `iso` equals six independent passes), U2 (option-permutation invariance of `iso`/`fieldcausal` pooled over 100 images), U3 (TF packed F3 equals sequential TF), U4 (metric code against a hand-computed 50-row example); label audit (an LLM or human re-labels 300 random held-out images per field to estimate dataset-level label noise, per-field; gives the ceiling on val accuracy and flags fields where val-OOD and public disagree); judge-proxy calibration (3.2); near-duplicate scan. Gate: kappa(proxy, judge) at least 0.85; U1-U4 pass; B0-mlp pilot reproduces plausible per-field accuracy (sanity floor).

**Stage 1a (screening, 1 fold, 1 seed): about 100 H100-h.** All pilot rows of the matrix. Outputs: per-run table of val-OOD Ĉ-soft, per-field accuracy, ECE, coverage@95P, violations, latency. No public evaluation yet except for B2/B0-ft as sanity (logged as 2 looks).

**Stage 1b (confirmation, 3 folds x 3 seeds): about 170 H100-h.** Advance to 1b: the F2 reference of each family (always), the best run of each of F1 and F3 per family (best F3 = best order x training-context combination under the best acceptance policy), B0-ft and the best B2 order. Everything on all three folds and three seeds, paired bootstrap, public evaluation once per advanced run (the pre-registered public look).

**Stage 2 (full data, 763k images): about 220 H100-h.** Only if D5 fires: the winning Jev and CLM formulations, B0-ft and the best B2 trained at full scale; pseudo-label all masked and candidate fields of the 763k train images with out-of-fold teachers (3-way group cross-fit for rows whose own dataset is in the teacher's training set; teacher trained on other datasets' labels). Output: pseudo-label parquet with per-field probability, acceptance flag at `tau_f`, source teacher.

**Stage 3 (transfer to the VLM): about 150 H100-h.** T0-T6 (matrix); 2 seeds each for T0, T1/T2 (whichever teachers are best), 1 seed for the others.

### 4.5 Experiment matrix

Legend: Stage in {0, 1a, 1b, 2, 3}. H = est. H100-hours per run at the stage-1a size (stage-2 sizes in 4.3). "Tests" names the question each run answers.

| Run ID | Family | Formulation and key settings | Stage | H | Tests |
|---|---|---|---|---:|---|
| B1-sig | zero-shot | SigLIP2 so400m, per-field softmax over O_f, 3 templates | 0 | 0.3 | floor; value of text-only knowledge; unseen classes |
| B1-bio | zero-shot | BiomedCLIP, same templates | 0 | 0.3 | domain-specific CLIP floor |
| B1-gem | zero-shot | Gemma-4-E4B option log-likelihood, no pointer head | 0 | 0.5 | what the pointer head and LoRA add |
| B0-lin | frozen feat. | SigLIP2 frozen, 6 linear heads | 0 | 0.1 | cheap ceiling; F1=F2 case |
| B0-mlp | frozen feat. | same, MLP heads | 0 | 0.1 | head capacity |
| B0-ctx | frozen feat. | MLP heads + gold-or-predicted context embedding, order O1 and O2 | 1a | 0.2 | F3 effect with no text model; TF vs PC exposure bias |
| B0-ft | fine-tune | SigLIP2 ViT LoRA/last-8, 6 heads | 1a/1b/2 | 1.5 | strongest plain predictor and teacher |
| J-F2 | Jev | `iso` mask, scalar-per-field softmax (single-label) and sigmoid (multi-label), O layout alphabetical, reference | 1a/1b/2 | 3 | F2 reference |
| J-F2sep | Jev | six separate per-field LoRAs, `iso` | 1a | 12 | interference in shared LoRA; upper bound on independence |
| J-F1c | Jev | `fieldcausal` mask, layout O1 | 1a/1b/2 | 3 | implicit cross-field sharing (F1 proper) |
| J-F1b | Jev | `bidir` mask over all options | 1a | 3 | bidirectional option encoding |
| J-F1n | Jev | naive causal concatenation, fixed order | 1a | 3 | option-order bias baseline |
| J-A-head | Jev | scalar head instead of bilinear pointer | 1a | 3 | head design |
| J-A-ml | Jev | multi-label as softmax soft target plus k head instead of sigmoid | 1a | 3 | multi-label formulation (B, D) |
| J-A-alias | Jev | alias smoothing on B/P labels | 1a | 3 | alias handling |
| J-F3-O1-TF | Jev | order O1, teacher forcing, packed | 1a/1b | 3 | order O1 and TF |
| J-F3-O1-PC | Jev | O1, out-of-fold predicted context plus dropout | 1a/1b | 3+6 | exposure bias fix |
| J-F3-O1-SS | Jev | O1, scheduled sampling | 1a | 8 | online rollouts |
| J-F3-O2-TF | Jev | O2 (domain last), TF | 1a/1b | 3 | order O2 and TF |
| J-F3-O2-PC | Jev | O2, PC | 1a/1b | 3 | |
| J-F3-O2-SS | Jev | O2, SS | 1a | 8 | |
| J-F3-O3 | Jev | confidence-adaptive order on the best of the above | 1a (stretch) | 4 | easy-first order |
| C-F1s | CLM | shared `u`, union multi-positive sigmoid loss | 1a/1b/2 | 1.2 | F1: shared space, per-option probs |
| C-F1m | CLM | shared `u`, union softmax, multi-positive | 1a | 1.2 | normalisation |
| C-F2s | CLM | shared `u`, per-field grouped softmax | 1a | 1.2 | collapse case F1 = F2 |
| C-F2q | CLM | field-conditioned attentive-pool query, per-field loss | 1a/1b/2 | 1.4 | F2 proper |
| C-F3-O1-TF / PC | CLM | order O1, `xattn` conditioning, TF or PC | 1a (1b best) | 1.4 | |
| C-F3-O2-TF / PC / SS | CLM | order O2, `xattn`, TF or PC or SS | 1a (1b best) | 1.4 | |
| C-F3-O2-film, C-F3-O2-cat | CLM | O2, PC, FiLM or concatenation conditioning | 1a | 1.4 | conditioning type |
| C-A-text | CLM | label text plus one-sentence definition | 1a | 1.2 | text tower and unseen classes |
| C-A-tower | CLM | EmbeddingGemma-300M label tower | 1a | 1.4 | label tower |
| B2-p | VLM SFT | Gemma-4-E4B, JSON in prompt order | 1a/1b/2 | 2 | direct SFT reference |
| B2-cf | VLM SFT | JSON keys fixed coarse-fine (domain first) | 1a/1b | 2 | generative F3 order |
| B2-ff | VLM SFT | JSON keys fixed fine-coarse (domain last) | 1a/1b | 2 | |
| B2-rat | VLM SFT | rationale first (stretch) | 1a (stretch) | 2 | does evidence-first help |
| X-J / X-C | cross-fit | 3-way group out-of-fold J-F2 / C-F2q models | 1a | 6 / 3 | provides PC contexts and OOF pseudo-labels |
| T0 | transfer | B2-best on hard labels only | 3 | 17 | control |
| T1 | transfer | T0 + pseudo-labels from best Jev at `tau_f` | 3 | 17 | Jev teacher |
| T2 | transfer | T0 + pseudo-labels from best CLM | 3 | 17 | CLM teacher |
| T3 | transfer | T0 + agreement of Jev, CLM and B0-ft (all agree, joint precision higher) | 3 | 17 | ensemble teacher (teacher side is unconstrained) |
| T4 | transfer | T0 + soft top-3 sequence targets from best teacher (weights = teacher p) | 3 | 17 | calibration helps the student |
| T5 | transfer | T0 + pseudo-labels from B0-ft | 3 | 17 | does Jev/CLM beat the plain classifier as a teacher |
| T6 | transfer | T0 + B2's own confident predictions on the same extra rows (self-training) | 3 | 17 | control for extra-data effect |

### 4.6 Config sketches (base config plus per-run overrides)

Every run is `base + overrides`; the matrix IDs map to the overrides below. Names are placeholders for a Hydra/YAML layout in `configs/`.

```yaml
# base_common.yaml
data:
  train_manifest: $WAILA_PREP/manifest.parquet      # 763k (stage 2) or pilot subset (<=128/dataset)
  fold: F0                      # F0|F1|F2 ; held-out datasets from plan/folds.json (split seed 17)
  cap_per_dataset: {pilot: 128, full: 2000}
  sampler: sqrt_dataset_balanced
  fields: [domain, modality, submodality, body_part, specimen, stain]
  requested_fields: sample_from_public_pattern()    # 4/5/6 fields; stain only if microscopy and annotated
  label_modes: {use: [hard, multi, candidates], skip: [masked]}
  forbid_dataset_ids: [mmbu-public]
eval:
  sets: [val_oodA, val_oodB, public]      # public: log to public_looks.jsonl, stage-1b+ only
  calib: temperature_per_field_on_A
  thresholds: {precision: 0.95, bound: cluster_bootstrap_lower5}
train:
  seeds: [0, 1, 2]
  epochs: {pilot: 3, full: 2}
  amp: bf16
  early_stop: id_val_loss
```

```yaml
# base_jev.yaml  (extends base_common)
model:
  torso: google/gemma-4-E4B              # verify active and total params for Track 3 when frozen
  image_tokens: 256
  lora: {r: 32, alpha: 64, targets: [q,k,v,o,gate,up,down], lr: 2.0e-4}
  vision: {train: false}                 # stage 2: unfreeze last 4 blocks, lr 2e-5
  pointer_head: {type: bilinear, dim: 256, field_bias: true, lr: 1.0e-3}
  readout_layers: [last, mid]
layout:
  field_order: O1                        # domain, modality, submodality, body_part, specimen, stain
  mask: iso                              # iso | fieldcausal | bidir | naive
  position_reset_per_option: true
  option_shuffle: {train: true, eval: fixed_alpha, teacher_tta_permutations: 4}
heads:
  single: {fields: [modality, submodality, specimen, stain], type: softmax}
  multi:  {fields: [body_part, domain], type: sigmoid, neg_weight: {body_part: 0.3, domain: 1.0}}
loss: {partial_label: candidates_logsum, multi_candidates: noisy_or, field_weight: per_row_normalised}
f3:
  enabled: false
  context_train: tf            # tf | pc | ss
  ctx_dropout: 0.3
  corruption: confusion_matrix   # tf only
  prune: {mode: soft_prior, hard_escape_topk: 3, table: compat_train_registry}
  ss: {p_start: 0.0, p_end: 0.5}
  infer_policy: P1               # P1 greedy | P2 confident-only | P3 soft top3 | P4 beam3 marginal
```

```yaml
# base_clm.yaml  (extends base_common)
model:
  image_tower: google/siglip2-so400m-patch16-384   # LoRA r=32 or last 8 blocks
  label_tower: siglip2_text                         # frozen first 2 epochs, then lora
  label_text: "{field}: {label}"                    # + aliases; A-text adds definitions
  query: field_cond_attn_pool                       # shared | field_cond_attn_pool
  proj_dim: 1152
loss:
  type: union_sigmoid        # union_sigmoid | union_softmax | per_field_softmax
  temperature_init: 10.0
  bias_init: -10.0
  partial_label: candidates_logsum
f3:
  enabled: false
  conditioning: xattn        # xattn | film | cat
  context_train: tf
  ctx_dropout: 0.3
  soft_context: false
  prune: {mode: soft_prior}
```

```yaml
# base_b0.yaml / base_b2.yaml
b0:  {features: siglip2_so400m_frozen, concat: [pooled, mean_patch], head: linear|mlp,
      mlp: {hidden: 2048, dropout: 0.2}, label_smoothing: 0.05, ft: {mode: none|lora_r32|last8}}
b2:  {model: google/gemma-4-E4B, lora: {r: 64}, train_projector: true, vision_train: false,
      prompt: pass2_official_template, output: json_with_templated_rationale,
      key_order: prompt|coarse_fine|fine_coarse, rationale_first: false,
      requested_only_supervised: true, pass1_replay_fraction: 0.3}
```

Per-run overrides (all other keys inherit):

| Run | Overrides |
|---|---|
| B1-sig, B1-bio | zero-shot loader; templates=3; `prune=false` (also reported with `prune=soft_prior`) |
| B1-gem | torso=gemma-4-E4B, no LoRA, option log-likelihood scoring |
| J-F2 | `mask=iso`, `f3.enabled=false` |
| J-F2sep | six runs of J-F2 each with one field (heads.single/multi limited to one field), merged at eval |
| J-F1c | `mask=fieldcausal` |
| J-F1b | `mask=bidir` |
| J-F1n | `mask=naive`, `position_reset_per_option=false`, `option_shuffle.eval=fixed_alpha` |
| J-A-head | `pointer_head.type=scalar` |
| J-A-ml | `heads.multi.type=softmax_soft_target_plus_k` |
| J-A-alias | `loss.alias_smoothing=0.5` for body_part and specimen |
| J-F3-O{1,2}-{TF,PC,SS} | `f3.enabled=true`, `layout.field_order=O1\|O2`, `f3.context_train=tf\|pc\|ss`, `mask=iso` (each pass sees only its own options), PC needs `X-J` out-of-fold predictions |
| J-F3-O3 | `layout.field_order=adaptive` (next field = highest F2-confidence remaining) on best-known context_train |
| C-F1s | `query=shared`, `loss.type=union_sigmoid` |
| C-F1m | `query=shared`, `loss.type=union_softmax` |
| C-F2s | `query=shared`, `loss.type=per_field_softmax` |
| C-F2q | `query=field_cond_attn_pool`, `loss.type=per_field_softmax` |
| C-F3-O{1,2}-{TF,PC,SS} | C-F2q plus `f3.enabled=true`, order, context_train, `conditioning=xattn` |
| C-F3-O2-film / cat | as C-F3-O2-PC with `conditioning=film\|cat` |
| C-A-text | `label_text` with one-sentence definition (written once by an LLM, reviewed, frozen) |
| C-A-tower | `label_tower=embeddinggemma-300m` |
| B0-ctx | B0-mlp plus context embedding input (`ctx=gold\|pred`, order O1\|O2) |
| B2-p / cf / ff | `key_order=prompt\|coarse_fine\|fine_coarse`; B2-rat: `rationale_first=true` |
| T0 | B2-best, `labels=hard_multi_only` |
| T1 / T2 / T5 | T0 plus `pseudo=jev\|clm\|b0ft`, accept `p >= tau_f` (OOF teacher, P2 policy for F3 teachers), requested fields extended to the newly labeled ones |
| T3 | `pseudo=agree(jev,clm,b0ft)` |
| T4 | `pseudo=jev_soft_top3`, loss = sum of teacher-weighted sequence cross-entropies over the top-3 option strings |
| T6 | `pseudo=self`, confident B2 predictions with the same acceptance rule |

### 4.7 Decision rules (pre-registered before stage 1a)

Primary selection metric: dataset-balanced `C-hat-soft` on val-OOD (report half B, averaged with the swap), because it is the challenge definition normalised by dataset. Practical-significance margin: 0.5 points of C-hat (equals about 0.1 point of the final base score at the 0.2 weight); smaller differences are ties and resolve toward the simpler and cheaper formulation (order: F2, F1, F3; then CLM, B0-ft, Jev).

- **D1 (advance from 1a):** a run advances if it is the family's F2 reference, or within the top-3 of its family, or within 1.0 point of the family best on C-hat-soft with better ECE or coverage@95P.
- **D2 (formulation per family, H1 and H2):** F1 beats F2 if the paired cluster-bootstrap 95% CI of delta C-hat-soft over datasets is above +0.5 on val-OOD and public delta point estimate is at least -0.25; same rule for F3 vs F2 and F3 vs F1. For F3, the winner must also keep ECE no worse than +0.01 vs F2 and chain-accepted coverage@95P not lower. F3 policy choice (P1-P4), order (O1 vs O2) and training context (TF/PC/SS) are picked by the same rule within F3 on F0 and verified on F1 and F2.
- **D3 (per field):** the teacher may combine formulations field by field: use X for field f if it wins f under D2 with Holm adjustment (6 fields x 2 comparisons); otherwise use the simplest formulation within the tie margin.
- **D4 (Q2a, is the Jev/CLM stage even worth it as a predictor?):** best-of(J, C) vs B2-best and B0-ft on public judged C and per field. If B2-best is within 1 point of C-hat on every field the classifier adds nothing as a predictor, and its only possible value is label supply (D6). Unseen-class rows are reported separately: a formulation that scores at least +10 points over B0-ft on unseen-class rows (text-defined options) keeps its place for that reason alone.
- **D5 (go to stage 2, full data):** go if the best Jev or CLM beats B0-ft by at least +0.5 on C-hat-soft with CI above 0, or by at least +5 points of weighted coverage@95P at precision at least 0.94, or by at least +10 points on unseen-class rows. Otherwise B0-ft is the teacher and the Jev/CLM branch is closed.
- **D6 (Q2b, transfer):** a teacher is adopted for the final labeling if its student beats T0 by at least +1.0 point of judged C (equals +0.2 on the base score) with 95% CI above 0 and loses at most 0.5 points of T. Student comparisons must also beat T6 (the self-training control) on C, otherwise the gain is just more training on more images. If T5 (B0-ft teacher) is within noise of T1/T2, adopt B0-ft (cheapest, simplest). T3 (agreement) is adopted if it beats the best single teacher at equal or lower precision cost.
- **D7 (JSON order for the final model):** B2-cf vs B2-ff vs B2-p by the same paired test on judged C; independent of the classifier results, since the output order is a free design choice of the generative student.

### 4.8 Statistics
- **Bootstrap.** Cluster bootstrap with B = 10,000 resamples: resample whole datasets (val-OOD) then images within datasets; for runs compared in a pair, use identical resample indices (paired). On public (no dataset id) cluster by pseudo-source: k-means on SigLIP2 embeddings (k about 150, plus exact `unique_id` groups), report both pseudo-cluster and image-level CIs and treat the pseudo-cluster one as the decision CI (image-level CIs are too tight because MMBU contains near-duplicate images from the same source).
- **Seeds.** For multi-seed results, hierarchical bootstrap: resample seeds then datasets; report mean and seed std of each metric. Stage-1a screening differences below 2 x the stage-1b seed std for that family are not interpreted.
- **Tests.** Paired differences in per-field accuracy: exact McNemar on public as a complement (not a decision rule); ECE and coverage differences from paired cluster bootstrap; Holm-Bonferroni across the pre-registered hypotheses H1 (F1 vs F2), H2 (F3 vs F2), H3 (J vs C vs B0-ft), H4 (teacher pseudo-labels raise judged C), each at 6 fields and all-fields C-hat; everything else is labeled exploratory.
- **Power.** Public n = 3,147 gives per-field accuracy SE of about 0.5 to 0.8 points (image-level) but the stain field has n = 528 (SE about 2 points) and body-part tail classes have a handful of rows; stain, tail body-part and unseen-class conclusions are descriptive. The paired difference SE is smaller than the unpaired one when the two models make correlated errors, but test the correlation instead of assuming it (report the disagreement rate).
- **Calibration CIs.** ECE bias grows at small n; compare ECE at fixed n via subsampling to the smallest field and report the bootstrap CI of the paired difference, and the Brier decomposition.
- **Look budget on public.** At most one look per advanced run in stage 1b (about 20 looks), one per stage-2 and stage-3 run; the count is reported in the final write-up.

## 5. Risks, confounds, and how the design controls them

| Risk | Why it matters here | Control |
|---|---|---|
| **Dataset-level labels are noisy per image.** Every image inherits its dataset's label (e.g. body part "chest|lung", stain, a mixed OCT/fundus dataset). Val-OOD accuracy is agreement with noisy labels, not truth. | Differences between formulations can be buried in label noise; candidate and multi fields are the worst. | Label audit in stage 0 (300 images per field re-labeled) gives per-field noise and a ceiling; metrics are also reported on `hard` rows only; public (clean human labels) is the confirmatory set; partial-label losses for `candidates`; negative down-weighting for multi-label B. Ablation: train with and without `candidates` rows. |
| **Dataset-style leakage / shortcut.** A model can identify the dataset (scanner, overlay, JPEG style) and output the dataset's label. Worse for F3 under teacher forcing (gold context nearly identifies the dataset). | Inflates in-distribution scores, hides the F3 exposure-bias failure, and gives optimistic pseudo-label precision. | Group splits only, dataset-balanced sampling, overlay removal and mild augmentation, dataset-identifiability probe, accuracy on within-dataset-varying rows, perturbation robustness, nearest-neighbour similarity strata, and `PC`/`SS`/context dropout in F3. Reported metric is OOD datasets, never ID-val. |
| **Exposure bias in F3.** Training on gold context and testing on predicted context. | Error cascades from modality or submodality to body part, specimen and stain. | TF/PC/SS comparison, context dropout, confusion corruption, soft and beam inference policies P3/P4, chain-acceptance reporting. D2 requires F3 to win on OOD folds, not on ID-val. |
| **Option-order and position bias (Jev).** Causal LMs favour early or recent options; with 102 body-part options this is large. | Spurious differences between F1/F2/F3 due to layout, and unstable pseudo-labels. | `iso` mask with position reset (order invariant by construction, test U2), per-sample shuffling, `J-F1n` as the measured baseline for order bias, 4-permutation TTA allowed on the teacher only, option-order sensitivity metric. |
| **Class imbalance and long tail.** two submodalities have 43-51 train datasets (brightfield, structural MRI) while 18 have at most 6, 20 body-part classes have none, stain and specimen have many singletons; "human" is 92% of specimen rows. | Accuracy is dominated by the head; gains on the tail go unseen, and calibration is only measured on head classes. | Macro-F1 and bucketed accuracy, sqrt dataset-balanced sampling, unseen-class slice, per-class-bucket ECE, coverage@95P reported per field and per bucket. Specimen: also report accuracy excluding "human" rows. |
| **Missing-not-at-random supervision.** Stain is masked on 325 of 393 datasets; body part is masked or ambiguous for 249; the labelled datasets are not a random sample. | A head can learn "stain is labeled iff dataset X", and thresholds fit on labeled datasets may not hold for the unlabeled majority. | Report on public (stain at 17% of rows) and on val-OOD datasets with hard stain labels separately; pseudo-label precision estimated on held-out datasets with hard labels only; acceptance thresholds are conservative (95% with a lower bound). |
| **Closed-vocabulary vs open generation.** The judge accepts synonyms and gives partial credit; a classifier forced to choose among 102 body parts may lose credit that free text would earn, or gain by choosing the exact string. | Classifier ranking can differ from the real C ranking. | Judge-proxy C-hat with alias merging and soft credit, calibrated to a real judge (stage 0 gate); classifier vs B2 comparisons use the same judge; the primary question for transfer is judged C of the student (3.7), not classifier accuracy. |
| **Pseudo-label confirmation bias.** The teacher's errors become the student's labels; if the teacher trains on the same images, its labels are over-confident. | T1-T5 would look better than they are, and errors correlate with the student's own. | Out-of-fold teachers (3-way group cross-fit), T6 self-training control, T3 agreement of three different teachers, acceptance only above `tau_f` with a lower bound on precision, evaluation only on public (never used for labeling). |
| **Public is not the hidden test, and is used repeatedly.** Hidden set is larger, with a different mix; repeated public looks are implicit tuning (rule 6). | Over-fitting to public, and a possible disclosure problem. | Selection on held-out train-registry datasets, public looks capped and logged, thresholds and temperature never fit on public (recommend changing `src/bio_waila/data/autolabel.py`), final decisions disclosed in the write-up. |
| **Unfair compute and capacity comparisons.** Jev-style uses a 4B torso, CLM about 0.4B vision plus text, B0-ft about 0.4B, F1/F2 have equal compute only because of the packed implementation. | A bigger torso wins for reasons unrelated to formulation. | Formulation comparisons are within a family at fixed parameters, data, epochs and LoRA rank; cross-family comparisons (H3) report cost (H100-h, latency) next to C-hat; D5 requires a margin for the more expensive model. |
| **Judge and proxy noise.** The official judge is Sonnet-5; the proxy and a local Gemma-4 judge can disagree at the margins. | Small effects (0.5 points) can flip. | Proxy calibration gate (kappa at least 0.85), the 0.5-point practical margin, repeat judge scoring of transfer runs with two judges (Gemma-4 and Sonnet-5 on a 500-row subset). |
| **Engineering risk in custom masks.** Gemma-4 mixes sliding-window and global attention layers, per-layer embeddings, and image-token bidirectional attention. | A wrong mask silently turns F2 into F1 or leaks options. | Unit tests U1-U3 before any training; one run of J-F1n and J-F2 compared on identical inputs; fall back to prefix-KV-cache plus batched option rows (equivalent for `iso`) if FlexAttention does not support the layer pattern. |
| **Rules.** Submission must be a single end-to-end generative VLM, no auxiliary models. | A Jev or CLM head cannot be shipped; F3 multi-pass inference cannot be shipped. | Plan treats them as teachers only; the shipped quantity is a B2 student. Teacher ensembles and multi-pass inference are allowed because they are off the submission path. Training-data disclosure per rule 6 (179 medium-overlap datasets in `plan/disclosure.csv`) is unaffected. |

### Minimum viable path if time or compute runs short
Stage 0, then only: B0-mlp, B0-ft, J-F2, J-F1c, J-F3-O2-PC, C-F2q, C-F3-O2-PC, B2-p, B2-cf at one fold and two seeds (about 60 H100-h), then T0, T5, and the better of T1 or T2 (about 55 H100-h). That answers D4 and D6 (does a teacher help the VLM) and the main F1/F2/F3 contrast at about 15% of the full budget.
