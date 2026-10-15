# CLM on an LLM backbone vs a contrastive vision encoder (SigLIP 2) for the Pass-2 classifier

Date: 2026-10-08. Question: for a CLM-style closed-vocabulary classifier over six fields (modality 11, submodality 37, body part ~100 multi-label, stain, specimen, domain), is it better to start from frozen (or lightly tuned) multimodal-LLM embeddings, as CLM does with frozen Qwen3-8B plus last-token pooling plus two MLP heads, or from a contrastive vision encoder such as SigLIP 2 with its text tower?

Builds on `clm_methods.md`, `general_encoders.md` and `med_encoders.md` in this folder; it does not edit them. Conventions: numbers come from sources I fetched in this session (HF cards and configs, the transformers source on `main`, arXiv HTML/PDF text) unless marked. `[UNVERIFIED]` = not confirmed from a primary source. `[EST]` = my own arithmetic. `[INELIGIBLE]` = Chinese-origin model or built on one; used as evidence only. Hard constraint: no Chinese-origin models or models built on them.

---

## 0. Recommendation

**Choose (c), both as bake-off arms, but weight them very unequally.** SigLIP 2 So400m with its text tower stays the primary and the expected winner. The LLM-CLM arm is a cheap, frozen-features-only side experiment with two members. It is worth running because (i) the submission is a Gemma 4 VLM, so how much usable class information the Gemma 4 stack holds is a decision input, and (ii) it costs about one GPU-hour per variant once features are cached. It is not worth a LoRA or full-tune effort unless the frozen arm gets close.

Why not (b), LLM-CLM only: every independent comparison I found puts frozen or contrastively tuned LLM embeddings 10 to 25 points below CLIP/SigLIP on image-global classification (section 2), and under distribution shift the LLM embedders lose even where they win in-distribution (VLM2Vec: ImageNet-R 70.2 vs OpenCLIP 83.0, ObjectNet 29.5 vs 51.4). Our evaluation is on held-out datasets.

Why not (a), SigLIP 2 only: two things SigLIP 2 cannot tell us are the Gemma-stack ceiling and whether a Gemma-family text encoder gives better label priors. Both are answerable cheaply.

**The two LLM-arm members, in order:**

| Arm | What | Why | Prior (my expectation, not a result) |
|---|---|---|---|
| **B1: Gemma-4-E4B hidden states** (`google/gemma-4-E4B-it`, also try the base `google/gemma-4-E4B`), frozen, heads only | The literal CLM recipe on a non-Chinese 4.5B-effective LLM. Cache pooled states from several layers, then train image-side and label-side MLP heads | It is the submission's own backbone family. Also gives a control: pooled Gemma vision-tower output (pre-projector) vs LLM hidden states tells us whether the LLM adds anything | Below SigLIP 2 by several points on held-out datasets; roughly equal to its own 150M tower |
| **B2: EmbeddingGemma 2** (`google/embeddinggemma-2`, 744M, Apache-2.0, ungated), frozen, linear probe plus cosine to label text | A contrastively trained, text-aligned, Gemma-4-tower-shaped encoder; image and label text share one 768-d space. Zero-shot sanity check for free | Cheapest LLM-flavoured arm (about 1 TFLOP/image at the default budget, `[EST]`). Its text tower is also the best candidate label-text encoder for the SigLIP 2 arm's heads (the LLM2CLIP idea; section 2.5) | Between Gemma-4-E4B and SigLIP 2; its image quality is not documented against dedicated encoders |

**Which layer and pooling for B1 (hypotheses to ablate; nothing here is measured on our task):**
- Cache, per image, the pooled states at layers {10, 16, 21, 26, 32, 38, 42} (E4B has 42 layers; the last 18 share KV). Two readouts per layer: (1) **mean over image-token positions** (the robust one: mean beat final-image-token in the Qwen2.5-VL probe study and tied with last token in the LLaVA probe study), and (2) **last prompt token** (the CLM readout). Evidence for mid-depth over final-layer: section 3.
- Expect the best layer near 50 to 70% depth (about 21 to 29). Use the final layer only as a reference. Normalise each layer's features (per-feature standardisation or RMSNorm without affine) before the heads, because raw residual-stream states have outlier dimensions (engineering judgement, not sourced).
- A learned weighted sum over the cached layers (the `clm_methods.md` 4.1 stage-A idea) is the cheap final choice.

**Prompt template for the image "state" (B1):** image first, then text (the Gemma 4 card says to put image content before text). Use the chat template with `add_generation_prompt=True` so the last token is the model-turn opener, which is the CLM-style readout position. Three templates to compare, all cheap:
- P0 (E5-V style): `Summary above image in one word:` — E5-V's exact image prompt is `Summary above image in one word:`, last-token pooling on the final layer (E5-V model card).
- P1 (metadata, recommended default): `Describe this image as imaging metadata: modality, submodality, anatomy/body part, stain, specimen. Answer in a few words:`
- P2 (no task text): `Describe the image.`
P1 vs P0 is untested for medical images [UNVERIFIED]; the prompt only matters for the last-token readout, not for the image-token mean.

**Label side (CLM "action"):** embed each label's description with the same Gemma-4-E4B text-only (same layer, last-token pooling, a separate action head), or skip the head and use learned label vectors initialised from those embeddings. The `clm_methods.md` hybrid (text init, L2 pull, hierarchy sum) still applies. Section 5 has the code.

**Decision rules (set before running):**
1. Report all arms on the dataset-group validation split (not in-distribution), per-field macro-F1, per-modality recall (EM, H&E/IHC, fundus, OCT, endoscopy), rare-class recall.
2. Drop an LLM arm if it is more than 3 macro-F1 points below the SigLIP 2 frozen probe and adds under 1.5 points when its features are concatenated with SigLIP 2's.
3. Only if B1 is within about 3 points of SigLIP 2, spend LoRA effort on the LLM (r=16 to 64, top half of layers) with the LP-FT recipe from `clm_methods.md` section 2.1. Do not contrastively tune the LLM on retrieval-style data: VLM2Vec-LoRA's linear-probe score fell to 62.0 against 74.5 for text-only-trained E5-V on MIEB (section 2.1).
4. Do not plan around CLM-35B or MetaEncoder-30B (section 1).

---

## 1. CLM and Meta updates since 2026-10-06

| Item | Status as of 2026-10-08 (checked) |
|---|---|
| Multimodal CLM-35B | **Not released.** HF API for `author=Contrastive-LM` lists only `CLM-v0.1-8B` (created 2026-09-21) and `deepswe-clm-heads-8k` (2026-09-22). The `CLM-v0.1-8B` card still says "A multimodal CLM-35B ... is coming in early October." The GitHub README still lists "Vision and multimodal support: images, video" under roadmap; the latest commit I could see is 2026-10-05 ("Revise citation title in README"). No backbone is stated anywhere. A Qwen-family base remains likely but is [UNVERIFIED]. |
| Non-Qwen CLM heads | None. README: "`CLM_v0.1-8B.pt`, Qwen3-8B backbone, last-token pooling". The head format (`cfg.hidden_size`) would allow another encoder, but nobody has published one. |
| **MetaEncoder-30B** (Meta, 2026-09-30) | Exists and is the closest thing to the promised multimodal CLM. Contrastive fine-tune of `meta-models/Muse-Glimmer-30B` (dense, about 29.6B, 52 layers, frozen PE ViT-G/14 perception encoder of about 1.8B, up to 4096 visual tokens; "distilled from Muse Spark"; Apache-2.0; developer Meta Superintelligence Lab). Last-token pooling, L2-normalised (`pooling_type='LAST'`). "Task + candidates" interface; the card recommends listing the closed-set options in the prompt. Results on the card: MMEB V3 image 0.7897, video 0.6486, visdoc 0.8149; ImageNet-1K 85.5% top-1 with vLLM and 84.5% with the HF class, on **200 images** ranked against all 1,000 labels. The card is gated (auto), so I read only the public part. |
| What MetaEncoder tells us | A 30B contrastive LM matches but does not clearly beat its own contrastive tower: PE-Core-G zero-shot is 85.4 on the full ImageNet-1K val (`general_encoders.md`), against 84.5 to 85.5 on a 200-image subset for the 30B model. Small-sample caveat; not a clean comparison. It is also far too large to submit. It remains a possible offline labeller only. |

---

## 2. Evidence: LLM-based embedders vs CLIP/SigLIP-class dual encoders

### 2.1 Head-to-head numbers (verified unless marked)

| Source | Models | Result | Reading |
|---|---|---|---|
| **MIEB** (arXiv 2504.10471, ICCV 2025; 130 tasks, 50 models) | Few-shot (16-shot) linear probe: SigLIP-so400m-384 84.6, EVA02-CLIP-bigE 86.0, SigLIP-L-384 83.7; **E5-V 74.5, Voyage-multimodal-3 71.3, VLM2Vec-LoRA 62.0** | Zero-shot classification: SigLIP-so400m 70.8; **E5-V 50.0, Voyage 58.2, VLM2Vec-LoRA 46.3**. Clustering NMI: EVA02 92.4; E5-V 70.1, VLM2Vec-Full 70.7 | The cleanest evidence: CLIP/SigLIP lead LLM embedders by 10 to 25 points on linear-probe and zero-shot classification. MLLM embedders win on visual-text and multilingual tasks. Contrastive retrieval-style tuning of the LLM (VLM2Vec) hurt linear-probe quality relative to the text-only-trained E5-V. |
| **MMEB v1** (VLM2Vec paper, Table 2 and 5) | Classification average: CLIP 42.8, OpenCLIP 47.8, SigLIP 40.3, E5-V 21.8, UniIR 42.1, **VLM2Vec (Phi-3.5-V) 54.8** | In-distribution (trained on) classification: ImageNet-1K 65.6 vs OpenCLIP 63.5; SUN397 72.7 vs 68.8; VOC2007 88.6 vs 52.4; N24News 79.5 vs 38.6. **OOD classification: ImageNet-R 70.2 vs OpenCLIP 83.0; ObjectNet 29.5 vs 51.4; ImageNet-A 19.3 vs CLIP 25.5; Country-211 13.0 vs 19.2**; Place365 42.6 vs 37.8 | The MMEB classification wins are largely in-distribution: 10 of the 20 training datasets are classification sets. On held-out classification sets the LLM embedder lost 4 of 5 comparisons to CLIP/OpenCLIP. This is the regime our eval is in. |
| **LLaVE** (arXiv 2503.04812) | MMEB Classification: VLM2Vec (LLaVA-OV-7B) 63.5, **LLaVE-7B 65.7**, LLaVE-2B 62.1; MMRet 56.0. Overall IND 75.0, OOD 64.4 | One epoch of training (more epochs overfit and hurt OOD); vision encoder frozen; 17 h on 8 A100-40GB for the 2B model | Best fine-tuned numbers I could verify, all on MMEB in-distribution classification; all of the best variants here are [INELIGIBLE] (see 2.2). |
| **Revisiting MLLMs: image classification** (arXiv 2412.16418) | Qwen2-VL ImageNet-average 95.0, LLaVA-OV 94.1, LLaVA-1.5 75.3, SigLIP 95.1; fine-grained: Qwen2-VL 91.2 vs OpenAI CLIP 85.6 | Generation-based classification, not embeddings. Improvement credited to a better LLM, more training data, and SigLIP-class towers | A modern MLLM can match SigLIP when it generates the label itself after instruction tuning on classification data; this is about the VLM arm, not about frozen embeddings. Qwen-based leaders are [INELIGIBLE]. |
| **Gemini Embedding 2** (arXiv 2605.27295; closed Google model, Gemini-initialised, bidirectional, mean pooling) | Image-to-text retrieval R@5 on MicroVQA: **Gemini Emb 2 79.3** vs CLIP-L/14-336 46.7, SigLIP 2 Giant/16-384 33.3, SigLIP 2 L 27.4, TIPS-G 20.0 | Long-caption microscopy retrieval, not classification | A strong LLM-initialised embedder can beat SigLIP 2 by a wide margin on microscopy *semantics*. Not available to us, and not a classification probe. EmbeddingGemma 2 is a different, much smaller model. |
| **EmbeddingGemma 2 card** | MIEB-lite task-type mean 64.64 (card says it "outperforms the base and so400m versions of SigLIP"; per-type numbers and SigLIP scores are not given in the card, and the press summary I read gave no figures either); MMEB v2 image 57.28 (Hit@1), VisDoc 67.84, Video 50.67, overall 59.01 | Mean over task types, not linear-probe classification | The only Google non-Chinese data point, and aggregate. Whether it beats SigLIP 2 on linear-probe classification is unknown [UNVERIFIED]. |
| **MetaEncoder-30B card** | See section 1 | 200-image subset | Roughly equal to its own contrastive tower, not better. |

### 2.2 Model inventory (backbone, eligibility, what is known)

| Model | Backbone | Eligible? | Notes |
|---|---|---|---|
| E5-V (2024) | LLaVA-NeXT-8B (Llama-3-8B); tower [UNVERIFIED] | Yes by lineage as far as I know | Text-only contrastive training; prompt `Summary above image in one word:`, last token, final layer. MIEB LP 74.5. |
| VLM2Vec v1 | Phi-3.5-V-4B (Microsoft), LLaVA-NeXT-7B | Yes | Last token, final layer; prompt `Instruct: {task}\nQuery: {q}`. MMEB classification 54.8 (Phi), 61.2 (LLaVA-NeXT-HR). |
| MM-Embed (NVIDIA) | LLaVA-NeXT-Mistral-7B based [UNVERIFIED] | Probably | Numbers not fetched. |
| mmE5 | Llama-3.2-11B-Vision based [UNVERIFIED] | Probably | Numbers not fetched. |
| **EmbeddingGemma 2** | Gemma 4 style: 270M text (24 layers, 512-d) + 170M vision (768-d, 16 layers, 12 heads) + 300M audio; mean pooling; MRL 768/512/256/128 | **Yes** (Google, Apache-2.0, ungated) | See 2.3. |
| **MetaEncoder-30B** | Muse Glimmer 30B + PE ViT-G | Yes by what I could read (Meta, "distilled from Muse Spark"); gated | See section 1. |
| LLaVE (0.5B/2B/7B) | LLaVA-OV with Qwen2 LM; the 2B model uses Aquila-VL-2B (BAAI) | **[INELIGIBLE]** | Best MMEB v1 numbers. |
| MMRet / MegaPairs | BAAI | **[INELIGIBLE]** | |
| VLM2Vec-V2 | Qwen2-VL-2B | **[INELIGIBLE]** | |
| jina-embeddings-v4 | Qwen2.5-VL-3B (verified from the jina paper) | **[INELIGIBLE]** | |
| GME, B3, MoCa, UniME (Qwen variants), Ops-MM-embedding, LamRA | Qwen2-VL / Qwen2.5-VL based (from memory) | **[INELIGIBLE]** | I did not fetch numbers. Evidence only if needed. |
| Gemini Embedding 2 | Gemini (closed) | Not available | |

I found no Gemma-, Llama- or Mistral-based multimodal embedder with MMEB classification numbers that is both eligible and fetched, other than VLM2Vec-Phi-3.5-V, E5-V and EmbeddingGemma 2.

### 2.3 EmbeddingGemma 2 in detail (card, `config.json`, HF API, all read 2026-10-08)

- Created 2026-09-14 on the Hub, Apache-2.0, ungated, 744,371,512 parameters (BF16). `EmbeddingGemma2Model`, `model_type: embedding_gemma2`, `transformers_version: 5.18.0.dev0` (newer than the 5.5.0.dev0 used for Gemma 4 E4B; check your transformers version).
- Text backbone: 24 layers, hidden 512, intermediate 2048, 4 heads, 5:1 local:global attention, vocab 262,144, `hidden_size_per_layer_input: 512`, projection 512 to 768. Sentence-Transformers pooling config: **mean pooling**, `include_prompt: true`.
- Vision tower: `gemma4_vision`, 768-d, 16 layers, 12 heads, patch 16, pooling kernel 3, `default_output_length` 280. Identical in shape to the E2B/E4B tower (both are 768-d, 16 layers per the Gemma 4 report and configs); `general_encoders.md` verified the weights differ from E2B's.
- Image token budget: 70/140/280 (default)/560/1120 soft tokens; the shared context is 8,192 tokens, so about 29 images at the default budget and about 114 at a low budget. Use **bfloat16 or float32, never float16** (NaN or silent degradation).
- Training: the card says only that the data covers web text, code, images, video and audio, with a January 2025 cutoff, cross-modal pairs, and short task-instruction prefixes (text only; "pass images ... without any prefix"). **Loss, initialisation and distillation are not stated.** The sibling Gemini Embedding 2 report uses contrastive training with distillation, a bidirectional transformer initialised from Gemini, and mean pooling plus a random linear projection; whether EmbeddingGemma 2 is distilled from it is [UNVERIFIED]. A web summary claiming so is not a primary source.
- Task prefixes (text): `task: classification | query: {content}` (symmetric), `task: search result | query: {query}`, documents as `title: {title} | text: {content}` or `title: none | text: ...`. Which prefix is best for short label strings matched against images is unknown [UNVERIFIED]; test no prefix, the classification prefix, and the document format.
- The quoted limits: at 256-d quality is "close to lossless", at 128-d multimodal quality drops substantially.

### 2.4 LLM embedder vs dual encoder after fine-tuning

- Closest evidence on "can the LLM catch up with the tower after supervised tuning" is Zhang et al. (arXiv 2405.18415): LLaVA-1.5-7B generation accuracy on ImageNet 47.6 rises to **85.7 after fine-tuning on classification data**, equal to CLIP-L's 85.2 linear probe, while the frozen-LLM probe was 77.1 (average token) or 76.9 (last token). BLIP-2-2.7B: probes 81.4 (average) or 80.3 (last) against 86.5 for the EVA ViT-G tower.
- So: fine-tuning the whole VLM reaches tower-level, not above it. No source I found shows an LLM-embedder beating a SigLIP-class tower on closed-set classification of held-out domains after fine-tuning. In MMEB the margins come from training on the benchmark's own classification sets.
- LLaVE reports that more than one epoch lowered out-of-domain scores. Treat any LLM-tuning arm as high-risk for held-out datasets.

### 2.5 The LLM as the *label-text* encoder (a different use of the same idea)

LLM2CLIP (arXiv 2411.04997) couples an LLM, via a caption-contrastive-tuned text side and a lightweight adaptor, to a pretrained CLIP vision tower; the abstract reports gains over EVA02 and SigLIP 2, including linear-probe classification. [Only the abstract and search summary were read; numbers not extracted.] It supports the cheap hybrid: keep the SigLIP 2 image tower, and try EmbeddingGemma 2's text tower for label descriptions, aligned to the image space by the ridge map in `clm_methods.md` 4.2. This is the best-motivated way for an LLM-class model to help the primary arm.

---

## 3. Frozen MLLM hidden states as features: layer and pooling

| Source | Model(s) | Finding |
|---|---|---|
| Zhang et al., "Why are VLMs bad at image classification" (2405.18415) | LLaVA-1.5-7B, BLIP-2-2.7B | Probing the last token or the average token "results in much better performance than probing other token positions"; average slightly ahead (77.1 vs 76.9; 81.4 vs 80.3). Probe stays 6 to 8 points below the vision tower (85.2; 86.5). |
| Tan et al., "The LLM Bottleneck" (2505.24840), Appendix C.2 and Fig. 3 | Qwen2.5-VL-7B ([INELIGIBLE] model, method evidence only) | Linear probes at vision-encoder output, projector output and final-LLM-layer image-token states give about the same CUB-200 / iNat21 accuracy; **mean pooling over image tokens "consistently outperforms" selecting the final image token**; probes beat the VLM's own generation. Visual information survives the LLM; decoding it is the bottleneck. |
| Fu et al., "Hidden in plain sight" (2506.08008) | Prismatic VLMs (Vicuna, CLIP/SigLIP/DINOv2 towers) | VLMs underperform their own towers on vision-centric tasks, but the representations do not degrade through projector and LLM layers; mid-to-late LLM layers (their layers 4 to 6 on correspondence tasks) were strongest and **the final LLM layer showed the largest drop**; fine-tuning the LLM helped more than tuning the tower or projector. |
| "How Multimodal LLMs Solve Image Tasks" (2508.20279, COLM 2025) | LLaVA-style and Qwen2-VL | Layer-wise probing shows phases (grounding, lexical integration, semantic reasoning, decoding); fine-grained category information peaks in **middle layers**. I could only read a model-generated summary of this one; treat details as [UNVERIFIED]. |
| Skean et al., "Layer by Layer" (2502.02013) | Text LLMs, some vision models | Mid-depth layers often beat the final layer on 32 text-embedding tasks. |
| E5-V recipe | LLaVA-NeXT-8B | Final-layer, last-token, with the "one word" prompt, but only because the model was trained on text pairs to make that readout meaningful. A frozen, untrained model does not give a good last-token embedding without a prompt; E5-V's untuned-style results are poor on classification (MMEB 21.8). |

Reading for B1:
1. The tower already holds nearly everything the LLM states hold; the LLM can only help via text-aligned semantics. So the **control readout** (mean-pooled Gemma vision-tower output, before `embed_vision`) is mandatory in the experiment.
2. Final-layer last-token is the weakest choice for a frozen causal LM. Prefer a mid-layer image-token mean, plus a mid-layer last-prompt-token readout, and let the head choose.
3. Gemma 4 E4B is causal for images (`use_bidirectional_attention` is `None` for the small models, so the standard causal mask applies; I read this in `modeling_gemma4.py`). Image tokens therefore cannot see the prompt; only the tokens after them can read the image. That is a reason to keep a prompt after the image and to read trailing prompt tokens as well as the image-token mean.

---

## 4. Medical evidence

No paper I found compares MLLM-embedding probes against CLIP/SigLIP/MedSigLIP on modality, organ or pathology tagging at the scale of our label space. The pieces that exist:

| Source | Result | Reading |
|---|---|---|
| "How Medical VLMs Underutilize Their Vision Encoders: A Dermatology Perspective" (2609.36557) | On five dermatology datasets, a few-shot linear probe on MedSigLIP beats the full MedGemma VLM (generation) by **10.26 points on average** (range 7.72 to 13.18); even 8-shot probes beat it. The paper does not probe MedGemma's LLM hidden states. | Same gap as the general-domain studies: tower probe > VLM output. Whether LLM hidden states close the gap is untested. |
| MLLM4PUE (2502.07221), pathology | LLaVA-NeXT-based embedder (last token, prompt `Summarize above <type> in one word:`) trained on pathology pairs beats PLIP, PathCLIP and QuiltNet zero-shot on patch classification: DigestPath-Signet weighted F1 0.665 vs PLIP 0.329 (Openpath data), 0.854 vs PathCLIP 0.361 (PathCap data); SICAP 0.402 vs QuiltNet 0.164 | Positive for the LLM embedder, but zero-shot, narrow (pathology), and the CLIP baselines are weak zero-shot models. Not a comparison against a probed SigLIP 2 or MedSigLIP. |
| Gemini Embedding 2 on MicroVQA retrieval (section 2.1) | 79.3 vs SigLIP 2 G 33.3 | LLM-initialised embedders can handle microscopy semantics far better than SigLIP 2 for caption retrieval; closed model. |
| OmniMedVQA (2402.09181) | Generic LVLMs recognise modality at about 49.5% (Qwen-VL-Chat), 49.6% (LLaVA-v1.6-7B), BLIP-2 68.2%, InstructBLIP 75.3% (older generation) | Generation-based recognition is weak and varies widely; one reason we do not trust the VLM's own output for tagging. |
| MedGemma report (via `med_encoders.md`) | MedSigLIP linear probes beat zero-shot by up to 9.8 points; after swapping the encoder the LM needed re-adapting | Supports probing the tower, and training heads rather than trusting text prompts. |
| Pelvic MRI sequence tagging (via `med_encoders.md`) | After fine-tuning, general and medical encoders both reach at least 93% | Series-type tagging is learnable by any strong tower once trained; the pretraining flavour matters little after tuning. |

Summary: nothing supports an LLM-hidden-state advantage on modality, organ or stain tagging; the supporting medical evidence is retrieval/zero-shot for LLM embedders, and probing evidence favours the vision tower.

---

## 5. Costs and implementation

### 5.1 Tokens per image

Gemma 4 vision: patch 16, 3x3 pooling after the tower, so the tower sees 9x the soft-token count in patches.

| Budget (soft tokens) | Patches through the tower | About the pixel area (aspect preserved) | LLM sequence with about 40 prompt tokens |
|---|---|---|---|
| 70 | 630 | about 400 x 400 | about 112 |
| 140 | 1,260 | about 570 x 570 | about 182 |
| **280 (default)** | 2,520 | about 800 x 800 | about 322 |
| 560 | 5,040 | about 1,140 x 1,140 | about 602 |
| 1120 | 10,080 | about 1,600 x 1,600 | about 1,162 |

(Images smaller than the budget get fewer tokens; the processor expands the placeholder to `boi + image_token * n + eoi` with n computed after the aspect-preserving resize. Pixel areas are `[EST]` from budget x 9 x 256.) For comparison, SigLIP 2 So400m/16 at 384 px is 576 tokens through the tower; at 512 it is 1,024. The Gemma 4 card says lower budgets suit classification, captioning and video; start at 140, test 70 and 280.

### 5.2 Compute for 870k images (`[EST]`, my arithmetic; B200-equivalent at an effective 0.8 PFLOP/s as in `general_encoders.md`, dense bf16; real wall-clock is probably 1.5 to 3x higher and data-loader bound)

Assumptions: Gemma-4-E4B LLM about 9 GFLOP per token (2 x 4.5B effective; PLE lookups are free), vision tower 0.3 GFLOP per patch plus attention, SigLIP 2 So400m at 384 about 0.65 TFLOP per image (from `clm_methods.md` 4.7).

| Model | Per image | 870k images | Ideal time |
|---|---|---|---|
| SigLIP 2 So400m/16, 384 px | about 0.65 TFLOP | about 5.7e17 | about 12 min (decode-bound: 3 to 6 h per `clm_methods.md`) |
| EmbeddingGemma 2, budget 70 | about 0.2 TFLOP | about 1.9e17 | about 4 min |
| EmbeddingGemma 2, budget 280 | about 1.0 TFLOP (tower dominates: 2,520 patches) | about 8.7e17 | about 18 min |
| Gemma-4-E4B, budget 70 | about 1.2 TFLOP | about 1.0e18 | about 21 min |
| Gemma-4-E4B, budget 140 | about 2.0 TFLOP | about 1.8e18 | about 38 min |
| Gemma-4-E4B, budget 280 | about 3.7 TFLOP | about 3.2e18 | about 67 min |
| Gemma-4-E4B, budget 560 | about 7.3 TFLOP | about 6.3e18 | about 2.2 h |
| Gemma-4-E4B, budget 1120 | about 15 TFLOP (attention starts to matter) | about 1.3e19 | about 4.6 h |

So the whole LLM arm (B1 at 140 plus B2 at 280, one pass each) is about 1 to 2 B200-hours of compute, a few hours of wall-clock with a decode-bound loader. Feature storage for B1, five layers x two readouts x 2,560-d fp16, is about 51 KB per image, about 45 GB for 870k images `[EST]`; cut to three layers or store a 50k-image pilot first. EG2 at 768-d fp16 is 1.5 KB per image. Memory: E4B in bf16 is about 16 GB (8B with embeddings); the 0.3B audio tower can be dropped (set `audio_config=None`; `Gemma4Model.__init__` handles it, loading may warn about unexpected keys) [UNVERIFIED that it loads cleanly].

### 5.3 Is LoRA on the LLM needed?

- Frozen-plus-heads is the right first step: the literature says the information is already in the (frozen) states; what is missing is decoding it (Zhang et al.; LLM Bottleneck; Fu et al.), and a head trained on cached features is exactly that.
- Tuning the LLM contrastively on retrieval-style data worsens probe quality (VLM2Vec-LoRA 62.0 vs E5-V 74.5 on MIEB), and more than one epoch hurt OOD (LLaVE).
- LoRA only if B1 is within about 3 points of SigLIP 2 on held-out datasets, and then as LP-FT (head first), top half of layers, early stopping on held-out dataset groups. No source shows LoRA on the LLM lifting held-out classification above a SigLIP-class tower.

### 5.4 Getting pooled embeddings from Gemma-4-E4B (checked against `transformers` `main`: `modeling_gemma4.py`, `processing_gemma4.py`, `modeling_auto.py`, and the E4B-it `config.json`/`processor_config.json`; the code below is a sketch I have not run)

Facts verified from source and config:
- `AutoModel` maps `gemma4` to `Gemma4Model` (vision tower + `embed_vision` + `language_model` + optional audio). `Gemma4ForConditionalGeneration` is the head-bearing class (`AutoModelForImageTextToText` / `AutoModelForMultimodalLM`). `gemma4_text` maps to `Gemma4TextModel`; `gemma4_vision` to `Gemma4VisionModel`. There is also a `gemma4_unified` family (the encoder-free 12B).
- E4B text config: 42 layers, hidden 2560, 8 heads, intermediate 10240, `hidden_size_per_layer_input` 256, `num_kv_shared_layers` 18, sliding window 512 with a full-attention layer every 6th, `use_bidirectional_attention: null`, `final_logit_softcapping` 30. Vision: 768-d, 16 layers, 12 heads, patch 16, pooling kernel 3, default 280 tokens.
- `Gemma4Model.forward(input_ids, pixel_values, attention_mask, mm_token_type_ids, image_position_ids, ...)` returns `Gemma4ModelOutputWithPast` with `image_hidden_states` and (via `output_hidden_states=True`) the per-layer `hidden_states`. Whether `hidden_states` has length 43 (embeddings + 42 layers) and whether the last entry is post-final-norm should be checked once [UNVERIFIED].
- The processor (`Gemma4Processor`) returns `pixel_values`, `image_position_ids` and `mm_token_type_ids` (`return_mm_token_type_ids` defaults to True). The image budget is `max_soft_tokens` in {70, 140, 280, 560, 1120} (default 280 in `processor_config.json`; `patch_size` 16, `pooling_kernel_size` 3, `image_seq_length` 280). The processor reads `images_kwargs["max_soft_tokens"]` when counting tokens, so pass `max_soft_tokens=...` as a kwarg to the processor call (or set `processor.image_processor.max_soft_tokens`); confirm once that the number of `image_token_id` positions in `input_ids` matches. `image_token_id` is 258880.
- `Gemma4Model.get_image_features(pixel_values, image_position_ids)` gives the projected vision output (`pooler_output`, a tuple split per image). The raw tower output is `Gemma4VisionModel(...).last_hidden_state` (pre-projector), the control readout.
- Use bf16. Left padding for batched last-token pooling (the HF docs use `padding_side="left"`).

Sketch:

```python
import torch
from transformers import AutoProcessor, AutoModel

MID = "google/gemma-4-E4B-it"            # also try "google/gemma-4-E4B" (base)
proc = AutoProcessor.from_pretrained(MID, padding_side="left")
model = AutoModel.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()  # Gemma4Model
IMG_ID = model.config.image_token_id     # 258880

PROMPT = ("Describe this image as imaging metadata: modality, submodality, anatomy/body part, "
          "stain, specimen. Answer in a few words:")

def embed_images(pils, budget=140, layers=(10, 16, 21, 26, 32, 38, 42)):
    msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            for _ in pils]
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = proc(text=texts, images=[[p] for p in pils], return_tensors="pt",
               padding=True, max_soft_tokens=budget).to("cuda")
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                          # tuple, index 0 = embeddings (verify len == 43)
    m = (inp["input_ids"] == IMG_ID).unsqueeze(-1)  # image-token mask, (B, T, 1)
    feats = {}
    for L in layers:
        h = hs[L].float()
        feats[f"mean_img_L{L}"] = (h * m).sum(1) / m.sum(1)   # mean over image tokens
        feats[f"last_L{L}"] = h[:, -1]                        # left padding: last position = model-turn opener
    return feats
```

Control readout (no LLM): `model.vision_tower(pixel_values=inp["pixel_values"], pixel_position_ids=inp["image_position_ids"]).last_hidden_state`, then average over the non-padding positions (`pixel_position_ids != -1`); remember the tower output is pre-3x3-pooling in the sense of the patch grid (the pooler is separate), so check the shapes.

Heads (CLM): image-side MLP (2,560 to width to 512, GELU, optional LayerNorm/residual) per field or per field-query; label-side MLP the same; scores `exp(t_f) * cos(...)`, softmax per field (sigmoid for body part and domain), exactly as in `clm_methods.md` 4.3 to 4.4. Standardise each cached feature (the intermediate residual stream has outlier dimensions) before the heads.

**Embedding the label texts with the same model (CLM "action" side):** text-only forward through the same `Gemma4Model` (no pixel values), with the same chat wrapper and pooling.

```python
def embed_labels(texts, layers=(21, 26, 32, 42)):
    msgs = [[{"role": "user", "content": [{"type": "text", "text": f"Imaging label: {t}"}]}] for t in texts]
    s = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    tok = proc(text=s, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model(**tok, output_hidden_states=True, use_cache=False)
    return {f"last_L{L}": out.hidden_states[L][:, -1].float() for L in layers}
```

Use rich label text (name, synonyms, parent path, one-line definition; several variants per label, UniMed-style) and cache once; with about 250 labels this takes seconds. Read the label at the same layer index as the image readout for the shared head, or let the label head pick its own layer. As in `clm_methods.md` 4.2, treat these as the *initialiser* of learned label vectors, not as a free text tower, because the two heads must learn the alignment from only about 250 distinct "actions".

**EmbeddingGemma 2 (arm B2):**

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.bfloat16})
img_emb = m.encode({"image": [pil1, pil2]})        # one embedding per input; image-only input form [UNVERIFIED]
lab_emb = m.encode(["chest radiograph", "H&E stained histology"], prompt_name="Classification")
```

The card's example shows `model.encode({"text": ..., "image": [...]})` for interleaved input; image-only dict input, how to set the token budget in Sentence-Transformers (use the `transformers` processor's `max_soft_tokens` instead if it is not exposed), and which prompt works best for label strings are all unverified. Output: 768-d, mean-pooled, L2-normalised, MRL to 512/256/128. Run (1) zero-shot cosine to label text as a sanity check, (2) a linear/MLP probe on image embeddings, (3) the CLM-style heads on top.

### 5.5 Experiment order (about one GPU-day including loading)

1. Pilot: 50k images stratified by dataset, B1 at budget 140 with P1, seven layers x two readouts + vision-tower control. Fit a simple per-field logistic probe with grouped-by-dataset CV. Pick the layer and pooling. (About 10 min of compute.)
2. P0/P1/P2 prompt comparison and budget 70/140/280 on the same pilot.
3. Full 870k extraction for the winning B1 config and for B2 at 280. Train CLM heads. Compare with the SigLIP 2 probe from `general_encoders.md` section 7, and with concatenation.
4. Apply the decision rules in section 0.

---

## 6. Open questions

- Whether EmbeddingGemma 2's vision tower was initialised from the Gemma 4 E2B/E4B tower, how it was trained, and its MIEB per-task-type numbers (linear probe and zero-shot classification separately) are all undocumented in the card.
- `Gemma4Model.forward(..., output_hidden_states=True)` hidden-state count, whether the last entry is normalised, whether `audio_config=None` loads cleanly, and whether `max_soft_tokens` passes through `Gemma4Processor.__call__` to the image processor: confirm with a 5-line test.
- Prompt effect on medical images for a frozen Gemma-4-E4B: no evidence; P0 vs P1 is an empirical question.
- MetaEncoder-30B card is gated; the ImageNet figure is from 200 images and the model card text I could read was partial. Muse Glimmer's lineage beyond "distilled from Muse Spark" is [UNVERIFIED].
- Numbers for MM-Embed, mmE5, B3, MoCa, UniME and Ops-MM-embedding were not fetched. The main conclusion does not depend on them: the verified MIEB and MMEB OOD results already show the sign of the gap.
- CLM-35B: check again before the bake-off starts. If a multimodal CLM appears on a Gemma, Llama or Mistral base, it would join arm B as a drop-in.

## Sources

CLM and Meta
- https://huggingface.co/Contrastive-LM and `https://huggingface.co/api/models?author=Contrastive-LM`
- https://huggingface.co/Contrastive-LM/CLM-v0.1-8B (README lines on CLM-35B)
- https://github.com/Contrastive-LM/CLM (README raw; commits via the GitHub API)
- https://huggingface.co/facebook/meta-encoder (public part) and https://huggingface.co/meta-models/Muse-Glimmer-30B

Gemma family
- https://huggingface.co/google/embeddinggemma-2 (README, `config.json`, `1_Pooling/config.json`, HF API)
- https://huggingface.co/google/gemma-4-E4B and `gemma-4-E4B-it` (`config.json`, `processor_config.json`)
- https://github.com/huggingface/transformers `src/transformers/models/gemma4/{modeling,processing}_gemma4.py`, `models/auto/modeling_auto.py`
- Gemma 4 Technical Report: https://arxiv.org/pdf/2607.02770 (150M tower, frozen in pretraining, budgets 70/140/280/560/1120; E4B 4.5B effective, 8B total)
- Gemini Embedding 2: https://arxiv.org/pdf/2605.27295 (mean pooling, Table 4 specialised-domain retrieval)

LLM embedders and benchmarks
- MIEB: https://arxiv.org/html/2504.10471v1
- VLM2Vec / MMEB: https://arxiv.org/html/2410.05160v2
- LLaVE: https://arxiv.org/html/2503.04812
- E5-V: https://arxiv.org/abs/2407.12580 and https://huggingface.co/royokong/e5-v
- jina-embeddings-v4 (Qwen2.5-VL-3B backbone): https://arxiv.org/html/2506.18902
- LLM2CLIP: https://arxiv.org/abs/2411.04997

Probing and layers
- Why are VLMs bad at image classification: https://arxiv.org/html/2405.18415
- Revisiting MLLMs, image classification: https://arxiv.org/html/2412.16418
- The LLM Bottleneck: https://arxiv.org/pdf/2505.24840
- Hidden in plain sight: https://arxiv.org/html/2506.08008v1
- How Multimodal LLMs Solve Image Tasks: https://arxiv.org/pdf/2508.20279 (summary only)
- Layer by Layer: https://arxiv.org/abs/2502.02013

Medical
- Medical VLMs underutilize vision encoders (dermatology): https://arxiv.org/html/2609.36557v1
- MLLM4PUE / PMEB: https://arxiv.org/html/2502.07221v1
- OmniMedVQA: https://arxiv.org/pdf/2402.09181 (via search summary)
- MedGemma / MedSigLIP: https://arxiv.org/abs/2507.05201 (via `med_encoders.md`)
