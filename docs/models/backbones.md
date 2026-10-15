# Which backbone for the Pass-2 metadata classifier (CLM-style or Jev-style)?

*Research date: 2026-10-06. Gating, license, safetensors totals: HF Hub API, read today. Architecture: model cards, `config.json` files, the Gemma 4 and Ministral 3 reports. FLOP and throughput numbers are my own arithmetic (labelled ESTIMATE), not measurements. [UNVERIFIED] = no primary source read. Prior notes this builds on: `data_discovery/research/non_chinese_bases.md`, `mmbu_paper_notes.md`, `jev_clm_research.md`. Hard constraint applied throughout: no Chinese-origin models or models built on them.*

## 0. Bottom line

1. **Pass-2 is a perception problem with a closed vocabulary, not a knowledge problem.** Modality, submodality, stain and (mostly) body part are texture, colour, field-of-view and format cues. Specimen and medical domain are semantic, but mostly follow from (modality, body part, content) and are bounded by dataset-level label noise. The evidence in section 2 says three things:
   - In-domain supervised data moves the metric far more than LM size or knowledge.
   - Frozen vision-tower features already separate the classes. The LM mostly adds decoding or alignment overhead.
   - Medical or knowledge-heavy tuning without broad perception data does not help.
2. **Answer to "vision reasoner vs knowledge LM vs pure encoder":** the default labeler should be a **pure vision encoder trained from pretrained weights (SigLIP2-So400m, second view DINOv3-L), with its own text tower supplying the label-name prior.** That is a CLM-style model, and it is the cheapest and the most likely to win. LM knowledge helps in only three places:
   - label-name priors for the many rare classes (the text tower of SigLIP2 already gives this);
   - the semantic fields (domain, specimen);
   - a Jev-style teacher that reads *privileged text* (the dataset-card description) next to the image when labeling training images.
3. **Moondream 3.1 is the wrong tool for this job.** Its edge is detection, pointing and counting. It excludes medical data, has no HF or torch path for the 3.1 checkpoint, and ships FP8 experts in Photon format. Its encoder is SigLIP-So400m-shaped, so the useful part is available as plain SigLIP2 at lower cost.
4. **Transfer to the submission VLM requires adapting the submission VLM's own tower.** Gemma 4 and Ministral 3 both train the LM with the vision encoder frozen (Gemma 4 report, section 2.2; Ministral 3 report), so neither tower has ever been adapted to microscopy, endoscopy or fundus images. That is the real opportunity.
   - Track 3: **Gemma-4-E2B** (150M tower) is the primary pick, with the PLE-counting question open.
   - Track 2: **Gemma-4-26B-A4B** (550M, 27-layer, 1152-d tower).
5. **Recommended order of work** (about 2 days before committing to any LM torso): a frozen-feature bake-off on grouped-by-dataset CV:
   - SigLIP2-So400m probe;
   - plus DINOv3-L;
   - Gemma-4-E2B pooled-hidden-state probe;
   - then E2B Jev-LoRA only if the LM torso shows a gain.

## 1. Candidate table

Totals, gating and license are from the HF API today unless noted. "Active incl. vision" is the Track-3 number. Rows marked (prior) rely on `non_chinese_bases.md` for architecture facts; I re-checked only the API fields.

| Candidate | Org / country | Total / active incl. vision | Vision encoder, resolution handling | LM origin | License, gating | transformers / vLLM | Medical numbers | Bolt-on difficulty (pointer head or embeddings) |
|---|---|---|---|---|---|---|---|---|
| **Gemma-4-E2B(-it, base)** | Google, US | 5.12B total (BF16). Compute-active about 2.42B: einsums 1,870M + embedder 400M + vision 150M. PLE table 2,340M is a lookup; audio 305M; drafter 76M (Gemma 4 report, Table 1) | 150M ViT: 16 layers, 768-d, patch 16, 2D RoPE. Variable aspect ratio, token budget 70/140/280/560/1120. Image dimensions divisible by 48 (patch 16 x pool 3). Vision weights frozen in pretraining | Gemma 4 (Google) | Apache-2.0, **not gated** | `Gemma4Model` returns hidden states with no LM head. `Gemma4ForConditionalGeneration`; `AutoModelForMultimodalLM`. vLLM: yes (card) | MedXpertQA-MM 23.5 (card) / 22.5 at 280 tokens (report Table 12) | Low. `Gemma4Model` gives `last_hidden_state` and `image_hidden_states` directly. PLE quirk, section 3 |
| Gemma-4-E4B | Google | 7.996B total; 4.5B effective (over 4B) | same 150M tower | Gemma 4 | Apache-2.0, not gated | same | MedXpertQA-MM 28.7 | Low. Fine as a labeler (no size cap); not Track-3 eligible |
| **Gemma-4-26B-A4B(-it, base)** | Google | 25.8B total (HF); 3.8B active (LM 2.8B + 550M tower + embedder; the report gives 24.5B/2.8B experts) | **550M ViT: 27 layers, 1152-d, 16 heads, MLP 4304, patch 16** (same shape as SigLIP-So400m; whether it is SigLIP2-derived is [UNVERIFIED]). Same budgets. `use_bidirectional_attention="vision"` | Gemma 4 MoE. Config: 30 layers, hidden 2816, **128 experts top-8, expert dim 704, plus a parallel dense MLP 2112**. Sliding window 1024 | Apache-2.0, not gated | `Gemma4*`, vLLM (card). Router logits are exposed | MedXpertQA-MM 58.1 (card) / 55.7 at 280 tokens | Medium. Same API as E2B; MoE fine-tuning quirk (section 3) |
| Gemma-4-12B | Google | 11.96B dense | **Encoder-free:** 48x48x3 patches into one 35M matmul plus 2D position embeddings | Gemma 4, trained from scratch unified | Apache-2.0, not gated | yes | MedXpertQA-MM 48.7 (47.4 at 280) | Low for a Jev torso, but there is **no tower to adapt or transfer** |
| Gemma-4-31B | Google | 31.27B dense | 550M ViT | Gemma 4 | Apache-2.0, not gated | yes | MedXpertQA-MM 61.3 | Low but heavy (62.5GB bf16). Labeler only; over 27B means API for submission |
| **MedGemma 4B / 1.5-4B / 27B** | Google | 4.30B / 4.30B / 28.8B | **MedSigLIP** (about 0.4B vision), 448 px fixed | Gemma 3 4B / 27B | HAI-DEF (`other`), **auto-gated** | yes | Strong: MMBU UC closed 0.439 / 0.403 vs Gemma-3-4B 0.303 (paper Table 2) | Low technically. Fails the ungated rule and the 4B cap for a submission. As a labeler it is allowed to generate data; see section 5 |
| Moondream 3.1 9B-A2B | M87 Labs / Moondream, US | 9.27B total (HF: BF16 1.22B + **FP8-E4M3 8.05B** + F32 5M); about 2.0-2.23B active; whether vision is inside "2B" is **unverified** | 27-block 1152-d ViT (SigLIP-So400m shape), patch 14, **378 px crops, up to 12; one 27x27 = 729-token map (+BOS) with bidirectional prefix attention** (preview torch code; 3.1 assumed the same). Region head for boxes | MoE (24 layers, experts from layer 4, 64 experts top-8, expert dim 1024), drop-upcycled from Moondream 2 (parent LM origin [UNVERIFIED]) | Moondream Model License 1.0 (`other`); no hosted-inference or fine-tuning resale; not gated | **No HF, no vLLM.** Photon/kestrel only (`kestrel==0.9.1`), as in `mmbu-eval/servers/moondream`. The GitHub repo has a PyTorch implementation (`moondream/torch`: text, vision, region, lora) for the **3-preview** layout; the 3.1 Photon tensor names do not match | Card excludes medical data | **High.** Needs a 3.1-to-torch weight loader and FP8 dequant, plus custom LoRA or MoE code |
| **Ministral-3-3B / 8B / 14B** (Instruct, Base, Reasoning) | Mistral AI, France | 3.85B / 8.92B / 13.95B. The Instruct repos are FP8 (3B: 3.03B F8 + 0.82B BF16); **Base repos are BF16** | **Pixtral-style ViT 410M copied from Mistral Small 3.1 and frozen: 24 layers, 1024-d, patch 14, native variable resolution up to 1540 px, 2x2 merge (28 px/token). A new projector per size** | Pruned and distilled from Mistral Small 3.1; Instruct distilled from Mistral Medium 3 (report). No Qwen use stated | Apache-2.0, not gated per API | `Mistral3ForConditionalGeneration` in transformers 5; vLLM >= 0.12 | None found | Low. Dense, standard decoder, no PLE or MoE |
| Mistral-Small-3.2-24B | Mistral AI | 24.0B | Pixtral ViT | Mistral | Apache-2.0, not gated | yes | none | Low; labeler or Track-2 only |
| Pixtral-12B-2409 | Mistral AI | about 12.4B | Pixtral ViT 400M | Mistral Nemo | Apache-2.0 | yes | none | Old; superseded by Ministral 3 |
| Llama 3.2-Vision 11B / Llama 4 Scout | Meta, US | 10.7B / 108.6B (17B active) | ViT-H-like tiles / MetaCLIP-based | Llama | **manual gate**, llama3.2 / other | yes | n/a | Gated. Skip |
| Phi-4-multimodal | Microsoft, US | 5.57B total (about 4.6B excluding audio, [UNVERIFIED]) | SigLIP-400M, 448 dynamic crops | Phi-4-mini 3.8B | MIT, not gated | yes | none | Over the Track-3 cap; no advantage as a labeler |
| Molmo2-O-7B | Ai2, US | 7.76B | SigLIP So400m/14-384 (prior), up to about 8 crops of 378 | **Olmo-3-7B** (Molmo2-4B/8B are Qwen3: excluded) | Apache-2.0, not gated | remote code | none | Medium: custom code; pointing prior is irrelevant here |
| Granite-4.0-3B-Vision | IBM, US | 3.997B | SigLIP2 so400m-patch16-384, tiled (prior) | Granite 4.0 Micro 3B | Apache-2.0, not gated | yes | none | Low; 3.997B leaves no headroom for Track 3 |
| Nemotron Nano 12B v2 VL | NVIDIA, US | 13.18B | RADIO-family ViT, 512 tiles | Nemotron-H hybrid (NVIDIA) | NVIDIA Open License, not gated | custom | none | **Policy risk:** the card says part of the post-training data was relabeled with Qwen2.5-72B annotations (data, not weights). Ask the UF policy owner |
| Llama-3.1-Nemotron-Nano-VL-8B | NVIDIA | 8.7B | C-RADIOv2-H | Llama-3.1-8B | NVIDIA + Llama licenses | custom | none | Llama terms attached; skip |
| SmolVLM2-2.2B | Hugging Face | 2.247B | SigLIP So400m/14-384, 384 tiles | SmolLM2-1.7B | Apache-2.0 | yes | none | Low. Weak perception (prior, [UNVERIFIED]); useful as an ablation |
| LFM2.5-VL-1.6B / LFM2-VL-3B | Liquid AI, US | 1.60B / 3.00B | **SigLIP2 NaFlex (So400m shape) 400M**, native up to 512 px | LFM2 (Liquid) | `lfm1.0` (`other`), not gated | yes | none | Low. The best fit for "SigLIP2-adapted tower drops in", section 4. License terms [UNVERIFIED] |
| **DiffusionGemma-26B-A4B-it** | Google, US | 25.8B total, 3.8B active | Gemma 4 550M tower, same budgets | Gemma 4 MoE converted to block diffusion (256-token canvas, up to 48 steps) | Apache-2.0, not gated | `DiffusionGemmaForBlockDiffusion`; vLLM, SGLang | MMMU-Pro 54.3 | High. Hidden-state or logit access at fixed canvas positions is **not documented**. The Prem "structured read" is a community hack. Not recommended |
| **SigLIP2** So400m (p14/384, p16/256/384/512, naflex), g-opt, L, B | Google, US | So400m 1.136B with text tower (vision about 0.4B); g-opt 1.87B | Fixed 384/512, or NaFlex variable aspect (up to about 256-1024 patches) | n/a | Apache-2.0, not gated | yes (`SiglipModel`) | Not medical; same family as the MedSigLIP tower | **Lowest.** Image and text towers are aligned; 0.46-0.58 TFLOP/image |
| DINOv3 (L/16, H+, 7B) | Meta, US | L 303M; 7B 6.72B | Patch 16, RoPE, variable resolution | n/a | **manual gate**, DINOv3 license | yes | Beats BiomedCLIP/CT-Net on several tasks but degrades on WSI, EM, PET (arXiv 2509.06467, prior) | Image-only: needs a separate text or label side. The gate is a smaller problem for a labeler than for a submission |
| Perception Encoder PE-Core (B/L/G) | Meta, US | L 0.3B, G about 1.9B (prior) | Fixed 336/448 | n/a | Apache-2.0, not gated | via timm / open_clip | no medical benchmark found | Low: CLIP-style with a text tower. A candidate second view |
| MedSigLIP-448 | Google, US | 0.878B (two towers) | 448 fixed | n/a | HAI-DEF, **auto-gated** | yes | CXR, derm, ophth, path zero-shot and probe results (card) | As a labeler or oracle feature only; outputs are not Model Derivatives (section 5) |
| BiomedCLIP | Microsoft, US | about 0.15B | ViT-B/16 at 224 | PubMedBERT | MIT, not gated | open_clip | PMC-15M figure-caption coverage; over-predicts on imbalanced data (prior) | Lowest; cheap baseline |
| EmbeddingGemma-300M (text side) | Google, US | 0.303B | text only | Gemma | `gemma`, **manual gate** | yes | n/a | Usable only as a frozen label-embedding space for CLM (labels, not shipped) |
| Gemini Embedding 2 | Google | API only; preview since 2026-03-10 | Natively multimodal; 3072-d MRL; up to 6 images per request | n/a | API terms | n/a | n/a | Cannot fine-tune. Sending about 1M source images to an API is a data-terms and cost problem. At most a zero-shot sanity view on a small sample |

## 2. Evidence: does LM knowledge help beyond the vision encoder?

There is **no published head-to-head on a Pass-2-like task** (modality + submodality + body part + stain + specimen + domain across about 5,000 datasets). The pieces below are indirect. Treat the conclusion as strong on direction and weak on magnitude.

| Source | What it shows | Implication |
|---|---|---|
| **MMBU paper** (arXiv 2606.06696, local text in `_sources/`) | Perception is the bottleneck across 17 models. Closed ungrounded classification: GPT-5.4-mini 0.533, Qwen3-VL-32B 0.530, InternVL3.5-8B 0.517; so an 8B open model sits within about 0.02 of the frontier. Medical adaptation: MedGemma-4B +0.136 over Gemma-3-4B, OctoMed-7B +0.133 over Qwen2.5-VL-7B, but Lingshu-32B -0.057 and LLaVA-Med about 0. Medical models beat non-medical on submodality only 62% of the time (mean F1 lead 0.08). Adaptation success tracks data size (+0.2 pp win rate per 1M examples, R2 0.47) more than model size | LM scale and "medical knowledge" help little. **Broad, in-domain perception data** is what helps. Narrow or caption-only medical tuning can hurt |
| **OmniMedVQA** (CVPR 2024, arXiv 2402.09181) | Modality recognition: InstructBLIP 70.6, MedVInT 59.8, BLIP-2 57.5, RadFM 21.3. General-domain BLIP-2 had the best average over all medical LVLMs "by a large margin". Medical models were not clearly better on modalities that look like natural images | Modality and anatomy recognition is low-level visual. Medical LM knowledge adds nothing for it without matching visual training. (2024, old models; direction only) |
| **MedBridgeRL** (arXiv 2603.01301, MedMNIST-v2, Qwen2.5-VL-7B family) | Linear probes on the **frozen vision tower** are far above the VLM's greedy answer accuracy: OrganA probe about 85 vs VLM Acc@1 14. After medical SFT on in-distribution labels the VLM reaches 76. Some tasks stay far below the vision-only reference (MedViT-v2) | The tower already holds most of the signal. **Supervised in-distribution training closes the VLM-to-probe gap; LM knowledge does not.** The same data run through a head on the tower is cheaper |
| **Frozen-embedding and DINOv3 benchmarks** (arXiv 2606.11606, 2509.06467) | General-purpose DINOv3 features rival BiomedCLIP and CT-Net frozen, except on WSI, EM, PET. CLS probes lose small-lesion signal | OK for global properties such as modality and organ. EM and WSI need fine-tuning. Small-lesion loss is irrelevant for metadata |
| **MedXpertQA-MM scaling in Gemma 4** (card/report) | E2B 23.5, E4B 28.7, 12B 48.7, 26B-A4B 58.1, 31B 61.3 | This benchmark rewards LM reasoning and scale. It says little about modality tagging, which saturates early. Do not read it as a Pass-2 predictor |

**Where an LM torso should help (hypotheses to test, not facts):**
- *Rare classes.* About half of the 35 public submodalities have under 25 public rows, and some have 2-4 (confocal laser endomicroscopy, multiphoton, specular). A label-name prior transfers to those classes. SigLIP2's own text tower provides one.
- *Domain and specimen.* "Oncology" or "pulmonology" is a function of body part, finding and modality. A multi-head model over (modality, body part) captures most of that, and the rest is label noise.
- *Privileged-context teacher.* For labeling training images we also have the dataset TOML text (description, modalities, body parts, stains). A VLM-torso Jev can read that text beside the image and choose per-image labels inside the dataset's allowed set. That is useful for mixed-modality archives. It cannot be used at Pass-2 test time, because Pass 2 gives only the image.
- *Field consistency* (stain only for microscopy, hierarchy of submodality under modality) is handled by hierarchical masking in a plain multi-head model.

## 3. Jev conversion: effort, risk, throughput, memory

Jev-style here means: drop the LM head, put the options in the prompt after the image, and score each option's hidden state against the hidden state at an `<answer>` position. The Strands Decider recipe is a rank-16 LoRA plus a pointer head of just over 1M parameters on Qwen3.5-2B (its card). CLM-8B is a frozen backbone with last-token-pooled embeddings and two small projection heads, trained with bidirectional InfoNCE; action embeddings are cached.

**ESTIMATE basis.** FLOPs per image = 2 x (LM compute params x tokens + vision params x patches). Throughput assumes about 0.8 PFLOP/s effective on a B200 and about 0.35 on an H100 (about 35% MFU of dense BF16). Jev-style adds about 1,300 option tokens (about 250 labels x about 5 tokens) per image; CLM-style adds about 30 text tokens. Real HF code with MoE routing may be 2-5x slower. Neither mode is a bottleneck: labeling 1M images takes hours on one GPU.

| Backbone | LM-head removal and pointer-head wiring | LoRA and quirks | Image tokens | Jev-style, 1M images (B200 / H100, ESTIMATE) | CLM-style, 1M images (B200 / H100, ESTIMATE) | bf16 weights | Effort / risk |
|---|---|---|---|---|---|---|---|
| **Gemma-4-E2B** | Use `Gemma4Model` (no head). Read hidden states at option-marker and `<answer>` positions | Dense. **PLE:** per-layer inputs combine a token-ID lookup and a context projection; for multimodal positions with no input IDs only the context projection is used (HF doc). Check how image-token and option-token positions are handled, and consider precomputing per-layer inputs. 20 of 35 layers share KV. The 2.34B PLE table can live on CPU | 70-1120 per image (use 280, or 560 for microscopy) | 2.8 h / 6.3 h | 0.8 h / 1.7 h | 10.2GB (PLE 4.7GB of it) | **Low / low-medium.** About 2-3 days to a working pointer head |
| Gemma-4-E4B | Same | Same, bigger PLE table | same | about 2x E2B | about 2x | 16GB | Low / low-medium |
| **Gemma-4-26B-A4B** | Same API; MoE adds router logits | **128 experts top-8, expert dim 704, plus a parallel dense MLP.** Whether PEFT can target the fused expert weights is [UNVERIFIED]; the safe plan is LoRA on attention and the dense MLP only, with experts frozen. Watch routing drift with a short, option-heavy sequence, and load-balancing off. HF MoE speed is [UNVERIFIED] | 70-1120; 550M tower | 4.5 h / 10.4 h | 1.7 h / 3.8 h | 51.6GB (one B200, or 2 x H100-80 for training) | **Medium / medium.** About 1 week, mostly MoE debugging |
| DiffusionGemma-26B-A4B | Needs fixed-position reads inside a denoising canvas; undocumented | Block-diffusion training; no PEFT recipe known | same as 26B | n/a | n/a | 51.6GB | **High / high. Not recommended** |
| **Ministral-3-3B / 8B / 14B (Base, BF16)** | `Mistral3Model` hidden states; standard | Dense, no PLE or MoE; plain LoRA. Tower is frozen in the release (ours to unfreeze). Instruct repos are FP8: start from the **Base** (BF16) repos | about 324 at 512 px (28 px/token); native up to 1540 px | 3B 3.8 h / 8.6 h; 8B 8.8 h / 20 h | 3B 1.1 h / 2.5 h; 8B 2.2 h / 5.1 h | 7.7 / 17.8 / 27.9GB | **Lowest for a dense VLM torso** (about 1-2 days). The 3B is the only Track-3-sized option beside Gemma E2B |
| Moondream 3.1 | `moondream/torch` has the forward for the 3-preview layout; for 3.1 write a Photon-to-torch loader and dequantize FP8 experts | Custom MoE (64 experts top-8) and a hand-rolled LoRA; the Moondream `lora.py` fetches variants from the Moondream API | 729 + BOS per image after the multi-crop merge; up to 12 crops in the tower | 1 crop 3.3 h / 7.6 h; 4 crops 4.2 h / 9.5 h | 1.4 h / 3.1 h; 4 crops 2.2 h / 5.0 h | about 18.5GB if dequantized to BF16 (9.3GB native) | **High / high.** About 1-2 weeks, with licence and harness risk. Strengths (box, point, count) do not matter for Pass 2 |
| SigLIP2-So400m + text tower (CLM) | No LM: pooled image embedding (MAP head) vs text-tower label embeddings, sigmoid or softmax loss | Full or last-N-block fine-tune; LoRA optional | 576-729 patches | n/a | 0.2 h / 0.4 h (encoder only) | 2.3GB (fp16, with text tower) | **Lowest / lowest.** Hours to a baseline |
| DINOv3-L + head | Image-only features; labels via a cosine classifier | Gated (manual) | 1024 patches at 512 px | n/a | 0.2 h / 0.5 h (L); 4.8 h / 10.9 h (7B) | 0.6GB (L) | Low |

**Memory and training.** LoRA training at about 1.6k tokens per sample, with gradient checkpointing, fits one 80GB GPU for E2B, E4B and the Ministral models. The 26B-A4B needs one B200 (180GB) or two H100s. Per-sample cost for a Jev-style E2B step is about 3 x 8 = 24 TFLOP, so 2M samples is about 17 B200-hours (ESTIMATE).

**Cost of the Jev option block.** Putting the label list after the image costs 4-5x the compute of the CLM form, because the option tokens must see the image. If the options were a fixed prefix before the image, their KV could be cached once, but then the option hidden states could not attend to the image and the pointer head would degrade toward a CLM bi-encoder. The two forms trade accuracy of option-image interaction against cost; at 1M images either is affordable.

## 4. Ranked recommendations

### (a) CLM-style labeler / teacher (contrastive, image vs label embeddings)

1. **SigLIP2-So400m (p16 at 384/512 or NaFlex), contrastive fine-tune against its own text tower with per-field label prompts.** Second view: **DINOv3-L/16**, concatenated or ensembled. Apache-2.0 and ungated for SigLIP2; the DINOv3 gate is acceptable for a labeler. Cheap, matches the closed-vocabulary evidence, and gives the rare-class label-name prior for free.
2. **Gemma-4-E4B (or E2B) as a CLM-8B-style dual encoder:** last-token-pooled image-conditioned state plus label embeddings from the same LM, projection heads, InfoNCE. This is the direct test of "does an LM torso add anything over (1)?". It is worth running only as a probe first (frozen hidden states, hours).
3. **PE-Core-L/G**, if a third view is wanted. No medical benchmark found.
4. **MedSigLIP-448 as a labeling-only oracle feature.** Gated, but the HAI-DEF terms say outputs are not Model Derivatives (section 5). Not for the submission.
5. BiomedCLIP, as a cheap baseline. EmbeddingGemma-300M, only if a frozen label-embedding space is wanted. Gemini Embedding 2: skip.

### (b) Jev-style labeler / teacher (VLM torso with pointer head)

1. **Gemma-4-E2B or E4B, prototype stage.** Low risk, same family and tower as the Track-3 student, so any behavior transfers. Use E4B for teacher quality if E2B underperforms.
2. **Gemma-4-26B-A4B, scale-up stage.** Best quality among practical ungated non-Chinese options (MedXpertQA-MM 58.1), a 550M tower, and the family of the Track-2 student. MoE LoRA is the main risk. Fall back to attention + dense-MLP LoRA.
3. **Ministral-3-8B Base (or 14B).** The simplest dense torso, native resolution up to 1540 px, BF16 Base available. Use it if Gemma MoE or PLE work stalls. The 3B is the only other Track-3-sized option.
4. Not recommended: DiffusionGemma (undocumented position access), Moondream 3.1 (format, loader, irrelevant strengths, no medical data), Nemotron (Qwen-annotation data; policy question), MedGemma (gated, over the 4B cap; fine only as a zero-shot labeling probe).

Expect the Jev torso to beat (a)1 only if the privileged-context idea or the rare-class prior is worth more than the added cost. Measure with grouped-by-dataset CV, per-class recall on rare classes, and calibration (ECE).

### (c) Backbone whose adapted vision tower transfers into the submission VLM

The adapted tower must be the tower of the model you will submit. Rankings by track:

| Track | Rank | Backbone | Why | Caveats |
|---|---|---|---|---|
| **3** | 1 | **Gemma-4-E2B** (150M tower, about 2.42B compute-active, multiplier about 1.05) | Own tower, never adapted (frozen in pretraining), variable aspect ratio, HF-native. Add an auxiliary classification/distillation loss during SFT on the labels from (a) or (b), then drop the aux heads | Counting of PLE (2.34B) and audio (305M) is unresolved; the rules count vision. If PLE counts, P is about 4.76B and E2B is ineligible. Ask |
| 3 | 2 | **Ministral-3-3B** (410M Pixtral tower; 3.85B; multiplier about 1.004) | Dense, no counting ambiguity, native resolution | No headroom under 4B, so the tower cannot grow. Weaker per size (no medical numbers found) |
| 3 | 3 | **LFM2.5-VL-1.6B** (SigLIP2 NaFlex 400M; 1.60B; multiplier about 1.10) | Its tower is SigLIP2-NaFlex-So400m-shaped, so a SigLIP2-adapted classifier tower could drop in with a new projector, subject to a shape check [UNVERIFIED]. Highest efficiency multiplier | `lfm1.0` license terms to read. Small LM. Possible weaker base accuracy offsets the multiplier (MMBU notes: a model 2B-class needs within about 7% of a 4B one) |
| 3 | alt | E2B LM with a **SigLIP2-So400m tower swapped in** (P about 2.7B, multiplier about 1.04, i.e. -1% vs E2B) | Largest perception capacity (about 0.4B vs 150M) | Requires a new projector and an alignment run, plus custom model code; the harness may not allow it. Open question 2 |
| **2** | 1 | **Gemma-4-26B-A4B** (550M, 27-layer 1152-d tower, 3.8B active) | Same family as the T3 student, so tower knowledge can also be distilled across; ungated, Apache | MoE LoRA. A strong base lowers the delta term (non-medical base; low `Acc_base` helps) |
| 2 | 2 | **Ministral-3-14B** (or 8B) | Dense and simple | Higher base accuracy lowers the delta |
| 2 | 3 | Molmo2-O-7B | SigLIP tower plus Olmo-3 (not Qwen) | Remote code; tower family [UNVERIFIED] as SigLIP2 |

**Labels vs tower adaptation.** The lowest-risk way to use the classifier is still "labels only": build the Pass-2 SFT set from the teacher's calibrated predictions and train the VLM end to end. Tower pre-adaptation or auxiliary distillation is an ablation on top. The MedGemma result (a medically adapted SigLIP tower plus LM tuning, +0.10 to +0.14 closed accuracy over Gemma-3-4B in the MMBU paper) hints that tower adaptation matters, but how much of that gain comes from the tower is [UNVERIFIED].

## 5. Trade-offs and open questions for the organizers

1. **PLE counting.** Does "active parameters including vision" for Track 3 count Gemma-4-E2B's per-layer embedding table (2.34B)? Is the audio encoder (305M) excluded when unused? Is the speculative drafter (76M) ignored?
2. **Tower replacement.** Is a modified architecture (a swapped vision tower plus new projector) allowed in the harness? Does it need `trust_remote_code` support?
3. **Labels from a gated model.** May we use a gated model (MedSigLIP, MedGemma) to *label* training images if we disclose it? HAI-DEF terms: "Outputs are not deemed Model Derivatives". The rule on ungated models is about the submission, but the disclosure rule (rule 6) still applies.
4. **Pass-2 scope.** Are all six fields always requested? How is the list-valued `medical domain` scored (any vs all)? This sets how much the semantic fields (where an LM may help) matter.
5. **FP8 and BF16.** Moondream 3.1 and the Ministral Instruct repos ship FP8. Does the harness evaluate the BF16 upcast?
6. **UF policy.** Is Nemotron VL acceptable given Qwen2.5-72B-annotated post-training data? (UF question, not an MMBU one.)

## 6. Cheap experiments that would settle the ranking (about 2 days)

On grouped-by-dataset CV over the registry (never split by image; see `non_chinese_bases.md` section 3.2):
1. Frozen probes: SigLIP2-So400m, DINOv3-L, their concatenation, PE-Core-L, BiomedCLIP, MedSigLIP (oracle). Per-field macro-F1 and rare-class recall.
2. Frozen **Gemma-4-E2B and Ministral-3-3B pooled-hidden-state probe** (image plus a fixed instruction): compute is hours. If it does not beat (1) by a clear margin, an LM torso is not worth the Jev engineering.
3. Only then: E2B Jev-LoRA with and without dataset-card text in the prompt; and a SigLIP2 fine-tune with and without the text-tower label prior. Report the gap against the registry-fold score on `mmbu-public` once, as a final check, not for selection.

## 7. Sources

- Gemma 4 technical report (Table 1 params, section 2.1 vision, section 2.3 encoder-free, Tables 10 and 12): https://arxiv.org/abs/2607.02770 (read from the PDF)
- Gemma 4 E2B card: https://huggingface.co/google/gemma-4-E2B-it
- Gemma 4 `config.json` (E2B, 26B-A4B), read today: https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/config.json
- transformers Gemma 4 docs (`Gemma4Model`, PLE, token budgets, MoE, bidirectional vision attention): https://huggingface.co/docs/transformers/model_doc/gemma4
- DiffusionGemma-26B-A4B card: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- Ministral 3 report (vision encoder copied and frozen, teachers): https://arxiv.org/abs/2601.08584
- Ministral-3-3B card and configs: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- Moondream 3.1 card and release notes: https://huggingface.co/moondream/moondream3.1-9B-A2B, https://moondream.ai/blog/moondream-3-1-beyond-benchmarks
- Moondream PyTorch implementation (3-preview layout): https://github.com/m87-labs/moondream/tree/main/moondream/torch
- Local: `mmbu-eval/servers/moondream/server.py` and `configs/runs/moondream31-glimmer.sh` (kestrel shim)
- Strands Decider 2B (pointer head, LoRA rank 16): https://strandsagents.com/blog/introducing-strands-decider/
- CLM-v0.1-8B card (last-token pooling, projection heads, frozen backbone): https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- HF Hub API for every model row, 2026-10-06: https://huggingface.co/api/models/<repo>
- MMBU paper text, local: `data_discovery/research/_sources/mmbu_paper_arxiv_2606.06696.txt`
- OmniMedVQA: https://arxiv.org/abs/2402.09181
- MedBridgeRL / "When Does RL Help Medical VLMs?": https://arxiv.org/abs/2603.01301
- Does DINOv3 set a new medical vision standard: https://arxiv.org/abs/2509.06467
- Frozen embeddings discard small-lesion signal: https://arxiv.org/abs/2606.11606
- HAI-DEF terms (outputs not Model Derivatives): https://developers.google.com/health-ai-developer-foundations/terms
- Gemini Embedding 2 announcement: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/
- Nemotron Nano 12B v2 VL model overview (Qwen2.5-72B annotations): https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-nano-12b-v2-vl
- Prior notes: `data_discovery/research/non_chinese_bases.md`, `mmbu_paper_notes.md`, `jev_clm_research.md`, `models.md`
