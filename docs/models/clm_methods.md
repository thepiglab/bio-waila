# CLM and contrastive image-label training for the Pass-2 classifier

Date: 2026-10-06. Scope: (1) CLM in depth, (2) contrastive/closed-set image classification methods and medical results, (3) dual vs cross encoder, (4) a concrete CLM-style design for our six fields on a non-Chinese base.

Conventions: numbers come from primary sources I fetched (CLM README and code, arXiv PDFs read as text) unless marked. `[UNVERIFIED]` = not confirmed from a primary source. `[EST]` = my own estimate, not a source number. CLM's blog (contrastive-lm.notion.site) is a JavaScript page and returned nothing through fetch, so scaling-law figures and the full benchmark tables are only known through the README and press summaries. There is no arXiv paper; the README's BibTeX entry is a Notion blog (`note={Notion Blog}`).

Hard constraint reminder: CLM-8B itself sits on a frozen Qwen3-8B, so it is unusable as a base. Everything below uses it only as a method reference.

---

## 1. CLM in depth

### 1.1 What it is (verified from README, `heads.py`, `train/finetune.py`, `docs/FINETUNING.md`)

| Item | Finding |
|---|---|
| Authors / release | Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Re, Azalia Mirhoseini (Stanford / NVIDIA / Hazy Research). Announced 2026-09-23 (README `/v1/models` lists release_date 2026-09-19). |
| Paper | None. Blog on Notion plus GitHub `Contrastive-LM/CLM` (Apache-2.0) and HF `Contrastive-LM/CLM-v0.1-8B` (Apache-2.0). |
| Encoder | Frozen Qwen3-8B served by vLLM in pooling mode. **Last-token pooling** gives a 4096-d vector. Nothing inside the LLM is trained. |
| What is trained | Two separate MLP projection heads, one for "state" and one for "action". README: each is "a 20M-parameter trainable projection head" (4096 -> width -> ... -> 512, GELU, optional LayerNorm/residual, depth and width configurable in the checkpoint `cfg`). Plus a scalar `logit_scale`. The released head file is about 75 MB. |
| Score | `exp(logit_scale) * cos(state_head(s), action_head(a))`, with the scale clamped at 100. `logit_scale` is learnable and initialised to log(1/0.07). |
| Pooling / inputs | State head sees `context + question` as prose (README: "never JSON, the heads are trained on prose"). Action head sees each candidate text verbatim (for `Choice`, the option's description, or its key if empty). |
| Caching | Embeddings (and projections) of actions and revisited states are cached in a pre-reserved GPU arena. Actions are encoded once, so inference is one encoder pass per fresh state plus a dot product per cached candidate. Measured on an RTX 4090: about 28 ms for a new state, about 0.6 ms for a cached one. |
| Output | `Choice` = softmax over cosine scores (optional temperature divides the logits). `Score` = expected rubric level. `Noul` = a yes/no probability, built from the two-option "true"/"false" criteria. `confidence` = top probability minus the mean of the others. |

### 1.2 Loss

- **Bidirectional InfoNCE** over an in-batch B x B similarity matrix, with temperature learned via `logit_scale`:
  `L = -(1/2B) sum_i [ log softmax_j(s_i . a_j / tau)[i] + log softmax_j(a_i . s_j / tau)[i] ]`.
- **Group masking.** In `_clm_loss`, entries where two rows share the same code (the same task or the same action) are set to -inf. This removes false negatives from class collisions (state/action pairs that are actually the same label). It differs from FLYP, which deliberately keeps collisions (section 2.1).
- **Hard negatives (stage 2).** The state-to-action direction gets extra explicit negatives per state, `exp(s.a+/tau) / (exp(s.a+/tau) + sum_k exp(s.h_k/tau))`. The hard negatives are *synthetic wrong answers written by Gemini 2.5 Flash-Lite* ("semantically similar but incorrect"), not mined from the model's own errors. Held-out eval uses one gold answer plus 10 hard negatives per question.
- **Closed-set variant (`choice` task in `finetune.py`).** The fine-tuning script has `--loss infonce` (default; bidirectional in-batch InfoNCE over the batch's distinct option texts) and `--loss softce` (softmax over each question's own options only). It supports soft targets (annotator distributions) as well as hard targets. This is the closest existing code to our setting, and it already handles soft/multi-positive targets. Defaults: AdamW, OneCycle with 10% warmup and cosine decay, lr `2e-3*sqrt(1024/width)*sqrt(batch/1024)` for the clm task, 5e-4 for choice, 20 epochs, batch 2048 for clm and 256 for choice.

### 1.3 Data pipeline and stage ablations (README)

1. **Pre-training:** about 60M Nemotron DQA question-answer pairs (question = state, answer = action). Released as `Contrastive-LM/CLM-v0.1-Pretrain-Nemotron` (62.5M rows).
2. **Mid-training:** about 30M synthetic hard negatives, added to the InfoNCE denominator.
3. **Post-training:** about 1M agent trajectory steps (Agent Data Protocol, Endless-Terminals, LiteCoder-Terminal-SFT), mixed 40% Nemotron replay and 60% agentic. Replay keeps hard-negative top-1 at 68.5% (vs 69% before); agentic-only training for the same steps drops it to 56.2%.
4. **Why two stages.** On about 100K held-out questions with 10 hard negatives each: pre-training alone gets 52.1% top-1; a short mid-training stage reaches 69.2%; hard negatives from the start peak at 62.4% and then overfit. So hard negatives work as a refinement on top of a broad pre-training, not a substitute for it.
5. **Scaling.** Test InfoNCE falls as a power law in compute, data, head size and encoder size. Encoder size gives the strongest gains. Optimal head size is about linear in training tokens (about 310 tokens per parameter). Compute for the released run is not disclosed [UNVERIFIED].

Note that the heads are trained on pre-computed embeddings: `train/finetune.py` loads cached encoder embeddings (`embed_shard.py` / `merge_embeddings.py` output), so fine-tuning is MLP-only training, typically minutes.

### 1.4 Evaluation against Jev (README + press summaries; the third-party numbers are not independently checked)

- Zero-shot, CLM-8B is "on par" with Jev on computer use, gaming and tool calling, with up to 9x lower latency, and 13x at about 1k candidates.
- Where the dual encoder loses zero-shot: BFCL v4 tool calling 95.2% vs Jev 99.2%, and WikiRacing 26/30 vs 30/30 (VentureBeat and aigenapps summaries).
- Where it wins after light fine-tuning, as a best-of-N verifier: DeepSWE 81.6% (31/38 held-out) and Terminal-Bench 2.1 87.6% (30 held-out tasks), at 4.1 to 5.7x lower latency. Press cites Jev at 71.1% and 83.1%; the README instead says Jev "fails to serve as a verifier" and scores below pass@1. These conflict, so treat the Jev side as [UNVERIFIED].
- Stated limitation (aigenapps quoting the post): "A ranker cannot return an answer that was not in the set you gave it." When all candidates are wrong, CLM returns the least-wrong one with no failure signal. It also needs stable action sets to benefit from caching.

### 1.5 Multimodal CLM-35B and follow-ups

- The HF card says "A multimodal CLM-35B ... is coming in early October." As of today (2026-10-06) the HF org page lists only 2 models (CLM-v0.1-8B, deepswe-clm-heads-8k) and 2 datasets; no CLM-35B. The README roadmap lists "Vision and multimodal support: images, video" as future work. Backbone is unstated; a Qwen-family base is likely [UNVERIFIED]. Do not plan around it.
- I found no replication or follow-up on a non-Chinese backbone or on images. The head format supports other encoders (`cfg.hidden_size`; "a head only makes sense with the encoder and pooling it was trained against"), but nobody has published one.

### 1.6 What CLM actually teaches us

1. The core is *cached, disaggregated two-tower scoring*, trained on only two MLP heads over frozen features. That is a few GPU-minutes once embeddings are cached.
2. CLM needs 60M+30M pairs because its action space is open-ended text. Ours is about 250 labels per whole vocabulary. In that regime a dual encoder collapses to a **cosine classifier whose weight matrix is generated from label text** (section 4.2). The gains that remain are the text prior, hierarchy sharing and open-vocab extension, not raw accuracy.
3. Transferable tricks: group-masked InfoNCE, soft/multi-positive targets, learned temperature, last-token or attention pooling, replay when moving to a harder objective, hard negatives as a second-stage refinement.
4. The "no none-of-the-above" limitation applies to us directly (section 4.6).

---

## 2. Contrastive classification with images: what the literature says

### 2.1 Fine-tuning a CLIP/SigLIP-style model for closed-set classification

| Method | Idea | Evidence |
|---|---|---|
| **Zero-shot with prompt ensembles** | Average text embeddings of several prompts per class, cosine, softmax. MedSigLIP's zero-shot protocol is exactly this (one or more prompts per class, averaged, softmax over classes). | MedGemma report: zero-shot AUCs on CXR/derm/fundus/path. BiomedCLIP "over-predicts all labels" on highly imbalanced data in zero-shot (Sadman et al. 2025), so calibration is needed. |
| **Linear probe (LP)** | Logistic regression on frozen image embeddings. | MedSigLIP linear probes beat zero-shot by 9.8% on 5-class diabetic retinopathy, and reach about the level of Path Foundation (1.9% behind on average). |
| **Text-initialised head (zero-shot head init)** | Initialise the classifier rows from the text embeddings of class names. | WiSE-FT uses it. PartialCLIP finds semantic-text init "consistently" beats random init in all three partial-label settings. LIFT (long tail) uses a cosine classifier. |
| **Full fine-tuning** | Update everything with CE. | Raises ID accuracy but distorts features: on 10 shift datasets FT is +2% ID but -7% OOD vs linear probing (Kumar et al., LP-FT paper). |
| **LP-FT** | Fit the head first, then fine-tune all. | LP-FT: +1% ID, +10% OOD over FT on those datasets. |
| **FLYP ("finetune like you pretrain")** | Continue the *pretraining contrastive loss* with class names as prompts; updates both towers; samples prompts from a distribution; **ignores class collisions** in the batch. | ImageNet ID/OOD (ViT-B/16, Table 1): FT 81.4 / 54.8, LP-FT 81.8 / 60.5, FLYP 82.6 / 60.2; with ensembling FLYP 82.9 / 63.2 vs LP-FT 82.1 / 61.8. 32-shot SST2: FLYP 68.0 vs FT 63.4. Masking collisions hurt PatchCamelyon by 1.3%, so "exactly match the pretraining loss" mattered. |
| **WiSE-FT** | Interpolate weights of the zero-shot and fine-tuned models: `theta = (1-a) theta_0 + a theta_ft`. | +4 to 6 pp under distribution shift and +1.6 pp ImageNet over prior work, no extra compute. |
| **LIFT (long tail)** | Lightweight fine-tuning (adapters) plus a text-initialised cosine classifier plus logit adjustment. | Heavy FT *hurts tail classes*: ImageNet-LT 72.9% (full FT) vs 73.5% (LIFT); iNat 46.4 vs 48.5. Reason: train/test feature distributions of tail classes become inconsistent under heavy FT. |

Read-across for us:
- Our held-out split is by dataset group, which is a *large* distribution shift. That is the regime where LP-FT, WiSE-FT and FLYP matter most.
- Many of our classes are tail classes: body_part has a median of 3 training datasets per class with 45 of 102 classes under 3; stain has a median of 1 with 25 of 39 under 3; specimen median 1 with 23 of 38 under 3 (computed from `plan/label_space.json`). LIFT's finding applies directly.

### 2.2 Sigmoid vs softmax, multi-label

- **SigLIP loss** is a pairwise sigmoid with a learnable temperature `t` and bias `b`, initialised at t' = log 10 and b = -10 so training starts near the "mostly negative" prior. It beats softmax below batch size about 16k and saturates around 32k; there is no all-gather normalisation. SigLIP 2 keeps this loss and adds decoder (LocCa) and self-distillation/masked-prediction losses, with four sizes (B 86M, L 303M, So400m 400M, g 1B) and NaFlex variable-resolution variants.
- For *multi-label* tasks the softmax in CLIP makes classes compete ("global feature tends to be dominated by the most prominent class"). MuMIC (Booking.com) swaps in a tempered-sigmoid BCE: 90.1% macro mAP across 32 majority classes. DualCoOp (CLIP with positive/negative prompts) is the other reference. [Both from search summaries plus the MuMIC PDF; I did not extract the DualCoOp numbers.]
- Practical rule: softmax for single-label fields (a proper distribution, class competition is desired), sigmoid with a learned per-field bias for multi-label fields.

### 2.3 Hierarchical and structured labels

- **Level-restricted contrastive learning** (Tao et al., June 2026, on BioCLIP): when one contrastive batch mixes taxonomic levels, a species and its genus become false negatives of each other. Restricting each contrast to categories from the same level, with group-balanced sampling, improves hierarchical consistency and raises iNat21 average accuracy across levels by 30.47% over the baseline. Direct lesson for a shared label space spanning six fields: **never contrast labels from different fields**; the denominator is per field (and per level inside body_part).
- **ProTeCt** (UCSD): CLIP and prompt-tuned CLIP often predict the correct leaf and the wrong coarse level; adds a prompt-tuning loss calibrating consistency across tree cuts (metrics HCA, MTA). Direct lesson: measure modality-vs-submodality consistency (the submodality's parent must equal the predicted modality).
- **HAPrompts** (hierarchy-aware prompts, "better mistakes") for zero-shot CLIP: only abstract read via search summary.

### 2.4 Partial / noisy dataset-level labels

- **PartialCLIP** (He et al., 2025): 11 foundation models x 13 PLL algorithms x 8 datasets x 3 scenarios. Findings: with CLIP ViT-B/16 the gap between PLL algorithms shrinks (CRDPLL beats CC by 29.3% on CIFAR-100 with a ResNet backbone but by only 0.4% with CLIP ViT-B/16); accuracy barely moves with ambiguity (CC accuracy drops 11.1% when ambiguity goes 0.1 -> 0.7 with the CNN, but only 0.2% with fine-tuned CLIP); text-embedding classifier init helps in all settings; **pre-filtering the candidate set with zero-shot CLIP** gives large gains (e.g. +81.7 points on one CIFAR setting). PEFT (VPT-deep, AdaptFormer) was best among methods; and when the candidate set gets too large, PartialCLIP can fall below plain zero-shot CLIP.
- Our `candidates` mode is *dataset-level*, not instance-level: every image of a dataset shares one candidate set. Identifiability comes only from *other* datasets that name the label singly. The standard loss `-log sum_{c in S} p(c)` (CC / PRODEN family) is consistent but allows the model to put all mass on one candidate for the whole dataset. Mitigate with a class-prior or entropy regulariser, zero-shot pre-filtering inside the candidate set, and by evaluating only on hard-labelled datasets.

### 2.5 Medical contrastive models

| Model | Size / data | Notes and numbers |
|---|---|---|
| **BiomedCLIP** (Microsoft) | ViT-B/16 + PubMedBERT; PMC-15M (15M figure-caption pairs from 4.4M papers) | Broad image-type coverage (radiology, pathology, microscopy figures, charts). Batch 4k optimal; context length 256 helped (R@1 69.0 -> 73.5). Trained on 16 A100 or V100. Zero-shot strong on RSNA/LC25000/TCGA-TIL. Over-predicts labels on imbalanced data (zero-shot). |
| **UniMed-CLIP** (MBZUAI) | ViT-B/16 (MetaCLIP init) + BioMedBERT; 5.3M pairs over 6 modalities (X-ray, CT, MRI, US, pathology, fundus) | **+12.61 absolute over BiomedCLIP averaged over 21 zero-shot datasets with 3x less data.** Key trick: LLM (GPT-4o) turns label triplets *(category, anatomy/organ, modality)* into several template captions per image, randomly sampled at train time. 10 epochs on 16 A100-40GB, about 10 hours, lr 5e-5, global batch 2048. This is almost exactly our "metadata labels -> text" situation. |
| **MedSigLIP** (Google) | SigLIP-400M tuned on about 33M medical image-text pairs (635k across modalities + 32.6M histopathology patches), 448 px | Medical data mixed at **2% weight** with the original WebLI data to retain general ability. CXR zero-shot +2.0% over ELIXR-based CXR Foundation (+7.1% on fractures); beats Derm Foundation linear probes on 79-condition dermatology. After swapping the encoder, the Gemma LM "needed to be re-adapted" (MedGemma pretraining with general + medical data). Test modalities: CXR, derm, fundus, histopathology only. License is HAI-DEF terms, not Apache [UNVERIFIED]. |
| **PMC-CLIP**, others | | Not read in detail. |
| Modality / organ classification | | I found no clean head-to-head benchmark of *modality + organ* classification across these models. The closest are the 21-dataset UniMed zero-shot suite and an anatomy-detection CLIP fine-tune (Kakkar et al.: +47.6% over PubMedCLIP baseline on whole-body MR/CT organ lists). Modality is almost always an easy task for these models; expect the hard fields to be stain, specimen and body_part tails. |

---

## 3. Dual encoder vs cross-encoder: where the dual encoder loses

Setup: a CLM-style model is a dual encoder (image and label embedded separately). A Jev-style model feeds state *and options* through one network (a cross encoder, with a pointer/routing head). VentureBeat: "Jev and Laya primarily support caching the state representation, while CLM's dual-encoder architecture lets it compute and cache state and action embeddings independently", which implies Jev-style options sit in the state's context.

Evidence for where each wins:

1. **Option semantics must interact with state.** In CLM's own zero-shot comparison the dual encoder trails on tool calling (95.2 vs 99.2) and open-ended WikiRacing (26/30 vs 30/30). It leads only after fine-tuning (verifier tasks).
2. **Expressiveness limits of single-vector dual encoders.** Weller et al. (ICLR 2026, "Theoretical Limitations of Embedding-Based Retrieval") show the number of distinct top-k subsets a d-dimensional embedding can return is bounded; even k=2 fails on the LIMIT dataset; cross-encoders and multi-vector (MaxSim) models do not share the limit. This binds when the *set of possible queries* is combinatorial; it does not bind for about 40 to 100 atomic labels per field.
3. **Compositional binding.** CLIP-style dual encoders struggle with attribute-object binding and relations (Winoground-type benchmarks) and multi-object scenes; global embeddings are dominated by the prominent object. Relevant to us only for *multi-region body_part* and multi-label domain, not for modality, stain or specimen.
4. **Dual encoders can be competitive when cached and trained for the task** ("In defense of dual-encoders for neural ranking", ICML 2022; abstract only). Retrieval-style reranking evidence is text-centric.
5. **Generative VLM "cross-encoders" are not better at plain classification.** Zhang et al. 2024: LLaVA/GPT-4V underperform CLIP on ImageNet-style classification despite using CLIP as the encoder. Linear probing of LLaVA-1.5-7B reaches 77.1% on ImageNet (vs 85.2% for its CLIP tower) while generation gets only 47.6%; fine-tuning with ImageNet data lifts generation to 85.7%. The information is in the latent space and is decoded only with enough *classification data in training*. They also found class frequency during training correlates with per-class accuracy.

Verdict for Pass-2:
- Modality, submodality, stain, specimen are mostly *global texture, colour and acquisition-physics* cues. Accuracy is bounded by the vision tower and by data (the dataset-level label noise), not by image-label interaction. A dual encoder (= cosine classifier) will not be worse than a cross-encoder here, and a cross-encoder would be much more expensive and harder to calibrate.
- A dual encoder loses (a) when labels are not a closed set the model was trained on, (b) for multi-region or relational outputs, (c) when per-label spatial attention matters. The middle path (label-conditioned attention pooling, section 4.3) addresses (b) and (c) for almost no cost. Open-vocabulary extension (a) is the only reason to keep a text tower; the final VLM generating JSON handles it natively.

---

## 4. A concrete CLM-style design for the six fields

### 4.1 Image tower (non-Chinese) and how much to unfreeze

| Candidate | Size / license | Pros | Cons |
|---|---|---|---|
| **SigLIP 2 So400m/14** (Google) | 400M; license believed Apache-2.0 [UNVERIFIED] | Aligned text tower; sigmoid loss; strong dense features; NaFlex variants; used as VLM tower (PaliGemma 2, Gemma 3 line) | General web data; medical microscopy coverage uncertain |
| **MedSigLIP-448** (Google) | 400M; HAI-DEF terms [UNVERIFIED] | Medical tune with replay; aligned text tower; 448 px | Test coverage CXR/derm/fundus/path only; non-pathology microscopy probably weak |
| **BiomedCLIP ViT-B/16** | 86M; MIT [UNVERIFIED] | Broadest biomedical figure coverage (PMC-15M), cheap, own text tower | 224 px, older; weaker features than SigLIP 2 |
| **Perception Encoder (PE-Core)** (Meta) | L/G sizes; license believed Apache-2.0 [UNVERIFIED] | Aligned text tower; paper's key finding: "best visual embeddings are not at the output of the network", so read an intermediate layer; PE-lang is built for VLMs | Large for G; medical coverage unknown |
| **DINOv3 ViT-L** (Meta) | distilled from a 7B teacher; custom license [UNVERIFIED] | Best frozen dense features; Gram anchoring against feature drift | No text tower, so label vectors must be learned or aligned |
| **Gemma 4 vision tower** | about 150M (E2B/E4B) or about 550M (26B-A4B, 31B); Apache-2.0; variable aspect ratio; token budgets 70/140/280/560/1120 (model card: lower budgets for classification). The 12B "Unified" model is encoder-free. | If the final VLM is Gemma 4, the classifier's tower *is* the submission tower, so there is no transfer gap | No paired text tower; encoder lineage (SigLIP-derived or not) not stated in the sources I found [UNVERIFIED]; encoder-free 12B has nothing to adapt |

Recommendation: two roles.
1. **Teacher/labeler** (offline, no size limit): the best of {SigLIP 2 So400m, MedSigLIP-448, PE-Core L} chosen by a frozen-feature probe on the dataset-group validation split. Probe first, which costs hours, not days, and avoids guessing between them.
2. **Student = the final VLM's own tower**, trained with the same objective plus distillation from the teacher.

How much to unfreeze (ordered; stop when the held-out-dataset metric stops improving, never use in-distribution validation):
- **Stage A, frozen:** per-field attention-pool head and label vectors only (LP). Read features from a mid-late layer or a learned weighted sum of the last 4 layers (PE finding; cheap ablation).
- **Stage B, partial:** unfreeze the top 25 to 50% of blocks (or LoRA r=16 to 64 on all attention/MLP), layer-wise lr decay about 0.75, body lr about 1e-5 and head lr about 1e-3 (lr values are conventional starting points, not from a source). Evidence: FT distorts features (LP-FT paper), heavy FT hurts tails (LIFT), PEFT beat linear probes in PartialCLIP.
- **Stage C, weight interpolation:** WiSE-FT the result with the stage-A weights, alpha chosen on held-out dataset groups (typical sweet spot is mid-range; tune).
- **Drift regulariser while tuning:** cosine distance between pooled (and a patch subset of) features from the tuned tower and a frozen copy on a replay set of unlabeled images (general + biomedical). This is a cheap analogue of MedSigLIP's 98%-general replay and DINOv3's Gram anchoring.
- Augmentation caution: avoid hue/saturation jitter (it destroys stain identity); use crop/flip/blur/mild contrast only.

### 4.2 Label side: frozen text tower vs learned label vectors vs EmbeddingGemma

- **Frozen aligned text tower** (SigLIP 2 / MedSigLIP / PE / BiomedCLIP): zero-shot prior, open-vocab, prompt ensembles per label. Weak for strings like "h2b-mrfp1" or "nuclear and membrane fluorescent labels".
- **EmbeddingGemma** (308M, 768-d with MRL truncation, Gemma 3 encoder-decoder initialisation, bidirectional attention): rich descriptions (definition, synonyms, parent path) embedded once and cached. Not aligned to any image tower, so a CLM-style label head must learn the alignment from only about 250 distinct "actions". That head is effectively a lookup table, so use EmbeddingGemma as an *initialiser/regulariser*, not a free text tower.
- **Learned label vectors** (a prototype matrix per field): the standard closed-set solution, and what a dual encoder reduces to here.

Recommended hybrid (my synthesis, supported by WiSE-FT, PartialCLIP, LIFT):
1. Build each label's text once, as K = 4 to 8 prompts or descriptions (UniMed-style variation: "a {modality} image of the {body part}, stained with {stain}").
2. Initialise label vector `w_k = normalise(P . mean_prompt_embedding)` where `P` is a linear map fitted (ridge) from the text tower (or EmbeddingGemma) space to the image tower's embedding space on the training pairs.
3. Train `w_k` freely at head lr, with an L2 pull to the init (stronger for labels from 1 to 2 datasets) and, for hierarchy, `w_k = w_text_k + w_parent(k) + delta_k` so tail classes share strength with parents.
4. During image-tower tuning optionally keep the frozen text tower in the loop, FLYP-style, with stochastic prompt sampling (FLYP updated both towers). Do not do this if the image tower will be swapped for Gemma 4's, which has no text tower.

### 4.3 One shared embedding space vs one head per field

Use a **shared trunk with field-specific query heads and one shared label space, with contrastive denominators restricted to each field**:
- One MAP/attention-pooling head with F = 6 learned field queries (SigLIP's MAP head, and Query2Label/CLIP-Decoder-style designs [from memory, not re-fetched]) over the patch tokens yields six embeddings `q_f` of 512 d. This avoids the single-global-vector bottleneck for multi-label fields.
- Label vectors for all fields live in one space and carry a field tag in their text ("stain: h&e"), so a text prior transfers. Negatives for field f are only labels of field f (level-restricted contrast; Tao et al. 2026).
- Each field has its own learned temperature `tau_f` (and bias `b_f` for sigmoid fields). Init `1/tau = 14` for softmax fields and (t', b) = (log 10, -10) for sigmoid fields.
- If you want strictly one global embedding (CLM-like), keep a single query and add a small per-field linear adapter `W_f`; expect it to be worse on multi-label fields.

### 4.4 Losses per label mode

Common: cosine logits `s_{f,k} = exp(t_f) cos(q_f, w_k)`.

| Mode | Fields | Loss |
|---|---|---|
| **hard** | modality, submodality, stain, specimen (single-valued) | Softmax CE over the field's full vocabulary (equals CLM's `softce`). Full vocabulary in the denominator already contains every negative, so "hard negative mining" reduces to an optional margin on confusable pairs taken from the validation confusion matrix. One-directional is enough; if you add the label-to-image direction (CLM/FLYP style), use class-balanced batches and either mask or ignore collisions (FLYP: ignoring was better on PCam; CLM: masking). |
| **multi** | domain (23), body part (102) | Sigmoid BCE per label (SigLIP init), with ASL-style negative down-weighting. Because dataset metadata lists are incomplete, treat unlisted labels as *weak* negatives (weight 0.1 to 0.3), and add ancestors of listed body parts as positives (chest implies thorax region). |
| **candidates** | any field | Partial-label loss `-log sum_{c in S} p(c)`. Add (a) zero-shot or teacher pre-filtering of S (PartialCLIP), (b) a prior-balancing term to stop a dataset collapsing onto one candidate, (c) optional PRODEN-style self-training once the model is decent. |
| **masked** | any field | No loss term. Instead, unlabeled images take part in teacher distillation (soft teacher targets) and in the drift loss. |

Long tail: class-balanced sampling over (dataset x label), logit adjustment with `log prior_f` (LIFT uses LA), a dataset-capped loss weight, and an explicit "dataset shortcut" check (leave-one-group-out validation).

How multi-label body part and domain work in a contrastive model: each field query `q_f` is scored against all field labels independently (sigmoid), so there is no competition. At inference threshold with a per-label calibrated threshold or take the top-k above a score; domain is also correlated with (modality, body part, specimen) and is probably the noisiest field (dataset-level, often not visible), so give it low loss weight and consider predicting it from the other fields' marginals via a tiny MLP.

### 4.5 Joint vs independent vs progressive

| Formulation | Implementation in a contrastive model | Pros | Cons |
|---|---|---|---|
| **Independent per field** | Six separate softmax/sigmoid heads from section 4.3; losses summed with field weights | Simplest; each mode handled natively; no error propagation | No consistency (CT + H&E possible); fields can't help each other beyond the shared trunk |
| **Joint** | "Actions" are cached embeddings of *observed label tuples* (modality, submodality, stain, specimen) built as `e_tuple = g(sum or concat of field vectors)` or from a tuple text; softmax over the finite tuple set (M tuples, cached like CLM's actions). Partial and candidate labels become marginalisation: `-log sum_{tuples compatible with the known/candidate fields} p`, masked fields = sum over all values. Multi-label body part and domain stay as sigmoid side heads | Consistency by construction; elegant handling of candidates and masked; CLM's caching applies | Cannot output an unseen tuple; M may be in the thousands with sparse coverage; hard to add the multi fields |
| **Progressive** | Tree: modality -> submodality (children of the modality) -> stain (only for microscopy) -> specimen. Train with teacher-forced gold parent; condition by restricting the child label set to the parent's children and optionally by adding the parent's label vector to the child query (FiLM-style). At inference marginalise over the top-k parents (beam) to avoid error propagation | Matches the final VLM's JSON generation order; natural for stain being microscopy-only | Error propagation unless marginalised; more code |

My recommendation: **a hybrid.** Predict submodality as the tree leaf (37 classes, softmax) and get modality as the sum of its children (guaranteed consistent; this collapses the modality/submodality joint). Stain and specimen get independent softmax heads gated by the predicted modality (stain loss only for microscopy; use an explicit "not applicable" class and the masked mode otherwise). Body part and domain are sigmoid heads. Add a cheap consistency penalty/readout (e.g. zero out stains incompatible with the predicted modality) rather than a full joint model. No head-to-head evidence exists for these formulations in the literature I read; run the three as an ablation on the dataset-group validation split.

### 4.6 Calibration

- Softmax fields: train with CE (a proper scoring rule), then fit one temperature per field on held-out dataset groups (the 53 validation datasets) and check on `mmbu-public` (which the README says is used only to evaluate/calibrate). CLM itself exposes only a global temperature knob and defines confidence as top minus mean of the others.
- Because training labels are dataset-level, models learn dataset identity and become overconfident on unseen datasets. Evaluate ECE per held-out dataset, not pooled; ensemble 3 to 5 seeds or WiSE-FT interpolants; use label smoothing about 0.05 and conformal prediction sets for `candidates`-heavy fields.
- Sigmoid fields: per-label Platt/isotonic on held-out data; start from the SigLIP bias init so the prior is already near the base rate.
- "None of the above": CLM's own weakness. Add an explicit `other/unknown` class per field and train it with outlier-exposure images (datasets outside the vocabulary, or images from classes you withhold in leave-class-out rounds), or threshold on max cosine / energy score.
- Jev's headline is calibration by proper scoring rules (Brier); reuse that idea by reporting Brier alongside accuracy per field.

### 4.7 GPU-hours (all `[EST]`, A100-80GB-equivalent, 763k planned images, about 10 epochs)

| Step | Estimate |
|---|---|
| Frozen-feature extraction, one tower, 763k images (So400m at 384 px is about 0.65 TFLOPs per image, so about 5e17 FLOPs, about 1.5 h compute but decode-bound) | 3 to 6 h per tower |
| Probe / label-vector training on cached features (the CLM regime) | under 1 h |
| Partial tuning (top 25 to 50% blocks or LoRA), So400m at 384 px, 10 epochs | 15 to 30 h |
| Full fine-tune, So400m at 384 px, 10 epochs (about 1.5e19 FLOPs at 40% MFU is about 33 h before overhead) | 30 to 60 h |
| Same at 224 to 256 px | one third of the above |
| ViT-B/16 at 224 (BiomedCLIP size) full fine-tune | 2 to 6 h |
| Gemma 4 about 150M tower, small token budget | under 10 h |
| Anchors | UniMed-CLIP: ViT-B/16 + text tower, 53M samples, 10 h on 16 A100-40GB (about 160 A100-h; data-loader-limited); BiomedCLIP: 16 A100 or V100 for 8 to 32 epochs over 15M pairs |
| Ablation grid (3 formulations x 2 towers x 2 unfreeze levels) | about 150 to 400 A100-h in total, which fits a handful of 8-GPU days |

### 4.8 Transfer into the final VLM

1. **Same tower.** If the submission uses Gemma 4's tower, train the classifier directly on that tower (section 4.1 student). Then export the tuned tower weights into the VLM.
2. **Different tower.** Use the teacher's soft per-field distributions as distillation targets for the student (covers masked and unlabeled images) plus pooled-feature distillation.
3. **Projector re-alignment is required after any tower change.** MedGemma precedent: tower tuned with 2% medical + 98% original data, then the language model was "re-adapted for this new vision encoder" with general + medical pretraining data. Plan: (i) projector-only warm-up, (ii) projector + LM LoRA on the JSON task mix plus general image-text replay.
4. **Do not unfreeze the tower during VLM SFT.** Prismatic VLMs (Karamcheti et al.) found fine-tuning the visual backbone during VLM training "dramatically degrades performance", especially localisation (p = 0.00381), and that single-stage training matches multi-stage. If any tower update is wanted, use a tiny lr plus the feature-drift loss to the stage-B tower.
5. **Train the VLM on Pass-2 JSON with class-balanced sampling.** Zhang et al. show decoding comes from exposure: per-class VLM accuracy tracks class frequency, and adding classification data fixes it (LLaVA-1.5-7B ImageNet generation 47.6% -> 85.7%). The classifier then serves as the teacher whose labels and soft targets fill the JSON for masked fields.
6. **Auxiliary head option.** Keep the six field heads on the pooled vision tokens as an auxiliary loss during VLM training; it is cheap and anchors the visual representation.
7. **Checks after each stage:** a held-out-dataset probe of the tower, plus a general VQA/caption regression suite, to catch drift early.
8. Track-3 note (from our memory notes): the vision tower's parameters count towards the size limit, so a 550M Gemma 4 tower is a budget decision, not only an accuracy one.

---

## Comparison table

| Approach | What is trained | Hard / multi / candidates / masked | Open vocab | Calibration | Compute | Held-out-dataset robustness | Verdict |
|---|---|---|---|---|---|---|---|
| Frozen CLIP/SigLIP zero-shot (prompt ensemble) | Nothing | Hard only (argmax); multi via thresholds; no partial | Yes | Poor (BiomedCLIP over-predicts on imbalanced data) | None | High but low accuracy on stain/specimen | Baseline and pre-filter for candidates |
| Linear probe / MLP head on frozen features (multi-head CE) | Heads | Native for all modes with custom losses | No | Good after temperature scaling | Under 1 h after caching | High (features unchanged) | Strong cheap baseline; run first |
| **CLM-style dual encoder on frozen features** (text-init label vectors, per-field query heads) | Heads + label vectors | All modes as in 4.4 | Partial (via text init) | Good (CE + temperature) | Under 1 h | High | **Recommended core**; equals a cosine classifier with a text prior |
| FLYP (contrastive FT of both towers) | Both towers | Hard; collisions ignored | Yes | Medium | 10s of A100-h | Best OOD among FT methods in the paper | Use for stage B if a text tower is kept |
| LP-FT / WiSE-FT | Head then tower; interpolate | Any | No | Medium | 10s of A100-h | Better than plain FT by 7 to 10 OOD points (LP-FT) | Use as the tuning recipe, with WiSE-FT at the end |
| Joint tuple contrastive (CLM "actions" = label tuples) | Heads | Marginalisation handles candidates/masked; multi fields separate | Only seen tuples | Good | Under 1 h | High | Consistent but brittle; try as an ablation |
| Progressive / hierarchical (parent-restricted) | Heads + conditioning | Natural for tree fields | No | Good if marginalised over parents | Under 1 h | High | **Recommended for modality -> submodality -> stain/specimen** |
| Jev/Clef-style cross-encoder (options in context, pointer head) | LoRA + head on a VLM | Choice / Score / Noul | Yes | Designed for proper scoring | Many A100-h; base models mostly Qwen | Unknown | Not needed for closed fields; the final VLM covers this role |
| Generative VLM only (JSON from SFT) | LM + projector | Via SFT targets | Yes | Weak (no probabilities) | Big | Depends on exposure | The submission; train it with classifier labels and balanced sampling |

---

## Open questions and risks

- Dataset-level labels mean many classes (stain and specimen especially) are one-dataset classes, so a model can score well by recognising the dataset. Held-out groups cannot test these classes; consider merging them into parent families for reporting and use the text-init + L2-pull to keep them tied to semantics.
- Licences of MedSigLIP, DINOv3, PE, SigLIP 2 and BiomedCLIP are from memory; check them before relying on a tower. They were not fetched.
- Gemma 4's encoder lineage and whether its weights can be extracted and trained standalone with an attention-pool head: check on the HF card and transformers code before committing.
- CLM-35B multimodal may still appear; if it does, its backbone is most likely Qwen-family [UNVERIFIED], so it will remain off-limits.

## Sources

CLM
- https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- https://huggingface.co/Contrastive-LM (org page: 2 models, 2 datasets, no CLM-35B on 2026-10-06)
- https://github.com/Contrastive-LM/CLM (README, `src/clm/heads.py`, `train/finetune.py`, `docs/FINETUNING.md`, read from raw.githubusercontent.com)
- https://contrastive-lm.notion.site (blog; not readable via fetch)
- https://venturebeat.com/technology/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests
- https://aigenapps.com/papers/research/clm-8b-jev-language-model-shape
- https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en (no-paper note)
- Prior notes: data_discovery/research/jev_clm_research.md

Contrastive fine-tuning and classification
- FLYP: https://arxiv.org/abs/2212.00638
- WiSE-FT: https://arxiv.org/abs/2109.01903
- LP-FT: https://arxiv.org/abs/2202.10054
- SigLIP: https://arxiv.org/abs/2303.15343
- SigLIP 2: https://arxiv.org/abs/2502.14786
- LIFT (long tail): https://arxiv.org/abs/2309.10019
- PartialCLIP: https://arxiv.org/abs/2506.05027
- MuMIC (tempered sigmoid multi-label): https://arxiv.org/abs/2211.05232
- Level-restricted contrastive learning: https://arxiv.org/abs/2606.21838
- ProTeCt: https://arxiv.org/abs/2306.02240
- Perception Encoder: https://arxiv.org/abs/2504.13181
- DINOv3: https://arxiv.org/abs/2508.10104
- EmbeddingGemma: https://arxiv.org/abs/2509.20354

Medical
- BiomedCLIP: https://arxiv.org/abs/2303.00915
- UniMed-CLIP: https://arxiv.org/abs/2412.10372
- MedGemma / MedSigLIP: https://arxiv.org/abs/2507.05201
- BiomedCLIP on imbalanced radiology: https://arxiv.org/abs/2506.14136
- CLIP anatomy detection on MR/CT: https://arxiv.org/abs/2405.20735

Dual vs cross encoder, VLM transfer
- Theoretical limits of embedding retrieval: https://arxiv.org/abs/2508.21038
- In defense of dual-encoders (abstract only): https://research.google/pubs/in-defense-of-dual-encoders-for-neural-ranking/
- Winoground: https://arxiv.org/abs/2204.03162
- Why are VLMs bad at image classification: https://arxiv.org/abs/2405.18415
- Prismatic VLMs: https://arxiv.org/abs/2402.07865
- Gemma 4 model card: https://huggingface.co/google/Gemma-4-E4B ; https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know
- Moondream 3 preview (vision encoder not disclosed): https://moondream.ai/blog/moondream-3-preview

Project files read: README.md, plan/label_space.json (class counts and datasets-per-class), plan/report.md.
