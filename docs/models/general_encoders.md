# General-purpose vision encoders for the Pass-2 CLM-style classifier (state of 2026-10-07)

*Research date: 2026-10-07. Gating, license, parameter totals, tags and release dates: HF Hub API, read today (the `curl` calls are reproducible: `https://huggingface.co/api/models/<repo>`). Architecture: model cards, `config.json` files, safetensors headers, arXiv abstracts or HTML. FLOP and wall-clock numbers are my own arithmetic, labelled ESTIMATE. [UNVERIFIED] = no primary source read. Builds on `backbones.md` and `clm_methods.md` in this folder; it does not edit them (section 8 lists what it corrects). Hard constraint applied throughout: no Chinese-origin models and nothing built on them; origin is stated per row.*

## 0. Bottom line

1. **The existing call holds, with refinements: primary = SigLIP 2 So400m/16 with its own text tower; second view = DINOv3-L/16.** I found nothing newer that beats SigLIP 2 on the combination we need (aligned text tower, at least 400M vision params, Apache-2.0, ungated, proven fine-tuning recipes). **There is no SigLIP 3**: the `google/` HF org has no SigLIP release after 2025-02-21 (listing sorted by creation date, read today), and a web search for "SigLIP 3" returns nothing.
2. **New since the earlier notes, worth knowing about:**
   - **TIPSv2** (Google DeepMind, CVPR 2026, Apache-2.0, ungated, B/L/So400m/g, text towers included). This is the closest "SigLIP-class successor" from Google, but it is tuned for dense patch-text alignment, and I found no ImageNet or classification table, so it is a probe candidate, not a primary.
   - **C-RADIOv4** (NVIDIA, Jan 2026, distilled from SigLIP2-g + DINOv3-7B + SAM3; NVIDIA Open Model License, ungated). One forward pass gives SigLIP2-like and DINOv3-like features. It is the best single-model alternative to the "pair", but it is partly redundant with our primary, because SigLIP2-g is one of its teachers.
   - **EmbeddingGemma 2** (Google, 2026-09-14, Apache-2.0, ungated). It contains a **text-aligned vision tower with exactly the Gemma-4-E2B tower shape** (768-d, 16 layers, 12 heads, about 168M params). I compared three weight tensors against `google/gemma-4-E2B`: the shapes are identical and the weights are different, so it is a contrastively retrained tower. This is the only text-aligned encoder in the Gemma 4 tower shape. It is the most interesting new option for the Track-3 transfer story (section 5.3).
   - **MetaEncoder-30B** (Meta, 2026-09-30, Apache-2.0, auto-gated). It is a contrastively fine-tuned 30B multimodal LM (Muse Glimmer, with a PE-Core ViT-G/14), shipped as a CLM-style "task + candidates" encoder. It is the multimodal CLM we had been waiting for, but it is not a vision encoder and it is far too large to submit. It is a possible labeler/teacher only (section 5.4).
3. **The Gemma 4 vision towers are not SigLIP 2 weights.** The 550M tower has the So400m width, depth, heads and MLP width (27 layers, 1152-d, 16 heads, MLP 4304), but it is a decoder-style block with a gated MLP, extra norms and 2D RoPE. The parameter arithmetic fits: a 412M SigLIP-shape ViT plus the third MLP matrix (27 x 1152 x 4304 = 134M) is 546M, versus the stated 550M. SigLIP2 checkpoints cannot be loaded into it. Anything learned in the SigLIP2 classifier must reach the Gemma tower through labels, distillation or a linear map, not weights (section 5.3).
4. **Medical evidence for general encoders is thin and mostly old-generation.** The solid facts are listed in section 3. They point the same way: general encoders are strong on radiographs, dermatology, endoscopy and fundus-like global tasks. They are weak on whole-slide pathology and electron microscopy, which are exactly the microscopy submodalities where our held-out datasets will hurt. No public paper compares SigLIP2, PE-Core or C-RADIO against medical encoders on modality/anatomy/stain tagging.
5. **Budget:** with the arithmetic in section 6.1, SigLIP2-So400m full fine-tuning on 880k images costs about 0.2 B200-days for 10 epochs at 384 px and about 0.4 at 512 px (ESTIMATE). One B200-day per run is more than enough; spend the surplus on seeds, the LoRA arm, and WiSE-FT sweeps rather than on larger models.

## 1. What I checked, and what is not there

| Question | Finding |
|---|---|
| SigLIP 3? | None. `google/` has SigLIP 2 (2025-02-21) and, newer, TIPS v1 (2026-08-19) and v2 (2026-04-09), EmbeddingGemma 2 (2026-09-14), Gemma 4 (2026-03). No SigLIP repo is newer than 2025-02-21 |
| DINOv4? | None found. Meta's newest vision repos (by creation date): `meta-encoder` (30B, 2026-09-30), MoEViE (Aug 2026), Sapiens2 (Apr 2026), EUPE (Mar 2026), Pixio (Dec 2025). The newest DINOv3-labelled repo is a depth head (2026-03) |
| Perception Encoder v2 / PE-Core update? | No. The `PE-*` repos are dated 2025-04 to 2025-07 (the `pe-av` and `pe-a-frame` repos, Dec 2025 to Jan 2026, are audio-video) |
| AIMv2 successor? | None found. AIMv2 HF repos were last touched 2025-07-08. The newer Apple vision release is LensVLM-9B (2026-09-21, a VLM, not an encoder) |
| New OpenAI CLIP? | None. The `openai/` org has no vision model newer than 2023 |
| TULIP (Berkeley)? | Paper (arXiv 2503.15485): SigLIP-initialised, B/16, So/14, G/16; IN-1k zero-shot 79.5 (B), 85.0 (So); RxRx1 linear probe 9.8 vs 4.6 for SigLIP. Weights are advertised on the project page; **I found no HF repo** and did not verify the license. Treat as [UNVERIFIED] availability, and it is SigLIP v1-based, so skip |
| LAION / DataComp open_clip | Nothing new: `laion/CLIP-ViT-bigG-14-laion2B` and `-H-14-laion2B` (MIT), DataComp-XL L/14 (MIT). All are behind SigLIP2 and PE on ImageNet zero-shot and are older. Apple's DFN5B-CLIP-ViT-H-14-378 is under the Apple research-only license |
| Anything from October 2026? | Only `meta-encoder` (2026-09-30, updated 2026-10-07) and `embeddinggemma-2` (2026-09-14). Both are covered below |

## 2. Candidate table

"Params" is the vision tower unless stated; "total" includes the text tower. Origin = the organisation that trained it. Everything in this table is non-Chinese. Benchmarks are the best general number I could verify for that model.

| Model | Origin | Params | Resolution | Text tower | License, gating | HF loader | Best general benchmark | Medical evidence |
|---|---|---|---|---|---|---|---|---|
| **SigLIP 2 B/16** (also B/32) | Google, US | 86M (375M total with text) | fixed 224/256/384/512; NaFlex | yes (Gemma tokenizer, 256k vocab) | Apache-2.0, **ungated** | `AutoModel` / `SiglipModel`; NaFlex needs `Siglip2Model`. open_clip: `hf-hub:timm/ViT-B-16-SigLIP2-384` | IN-1k zero-shot 79.1 (256 px) | none direct |
| **SigLIP 2 L/16** | Google, US | 303M (882M total at 512) | fixed 256/384/512 (no NaFlex) | yes | Apache-2.0, ungated | same | 82.5 (256 px) | none direct |
| **SigLIP 2 So400m/14 and /16** | Google, US | about 400M (1.136B total). Config: 27 layers, 1152-d, MLP 4304, 16 heads | /14: 224/384; /16: 256/384/512; **NaFlex** (default 256 patches; 128-1024) | yes (27-layer 1152-d, 256k vocab; about 0.7B of the total) | Apache-2.0, ungated | same; NaFlex: `google/siglip2-so400m-patch16-naflex` | 83.4 (/16, 256 px) | the family MedSigLIP and Pelvis-SigLIP build on (section 3) |
| **SigLIP 2 g-opt/16** | Google, US | about 1.0B (1.87B total). 40 layers, 1536-d | 256 / 384 | yes | Apache-2.0, ungated | same | 84.5 (256 px) | none direct |
| **DINOv3** S/S+/B/L/H+ (ViT, patch 16, RoPE), 7B, ConvNeXt T/S/B/L; `sat493m` variants for L and 7B | Meta, US | 21M / 29M / 86M / **303M** / 840M / 6.72B | any resolution (multiples of 16); timm default 256 | **no** (dino.txt exists for ViT-L only; its weights are on GitHub, I did not find them on HF) | **DINOv3 License** (custom): commercial use allowed; military, ITAR and weapons uses prohibited; redistribution must carry the license; academic papers must acknowledge. `facebook/dinov3-*`: **gated, manual**. The `timm/vit_large_patch16_dinov3.lvd1689m` mirror is **not gated** but the license still binds | `AutoModel` (`dinov3_vit`); timm | linear-probe ADE20k 55.9 mIoU (7B); global tasks "match or exceed SigLIP 2 and PE" (paper text; I did not extract the IN-1k table) | Section 3: best or near-best on chest X-ray; **weak on WSI, EM, PET**; no monotone scaling |
| **Perception Encoder, PE-Core** T/S/B/L/G | Meta, US | B 0.09B, L 0.32B, **G 1.88B** (paper Table 4) | B 224, L 336, G 448 (T/S 384) | yes | **Apache-2.0**, ungated (HF tags and GitHub `LICENSE.PE` read) | `perception_models` repo; open_clip `hf-hub:timm/PE-Core-L-14-336`, `timm/PE-Core-bigG-14-448` | IN-1k zero-shot B 78.4 / L 83.5 / G 85.4 (paper Table 5; SigLIP2-g-opt 85.0) | none found |
| PE-Lang, PE-Spatial (L/G, also B/S/T for Spatial) | Meta, US | same trunks | 448 (Spatial B/S/T: 512) | **no** (alignment-tuned for VLMs / dense) | Apache-2.0, ungated | `perception_models`; timm `vit_pe_lang_*`, `vit_pe_spatial_*` | PE-Lang: DocVQA 94.6; PE-Spatial: COCO box mAP 66.0 | none. For metadata tagging prefer PE-Core. The paper's claim that the best features sit in intermediate layers applies to PE-Core |
| MetaCLIP 2 (worldwide) S/B/M/L/H/giant | Meta, US | L/14 1.08B total; H/14-378 1.86B total; giant 3.63B total | 224-384 | yes | **CC-BY-NC-4.0**, ungated | `transformers` (`metaclip-2-*`) | beats English-only ViT-H/14 by 0.8 on IN-1k zero-shot; CVQA 57.4, XM3600 64.3 (abstract) | none. **Non-commercial: flag.** Only worth it if multilingual label text mattered, which it does not |
| **C-RADIOv4** SO400M / H (also 1D-H 1.45B) | NVIDIA, US (teachers: SigLIP2-g-384 [Google], DINOv3-7B [Meta], SAM3 [Meta]) | 412M (HF 431M) / 631M (HF 652M) | any, multiples of 16, tested 256-2048; best at about 1024 | via a SigLIP2 adaptor (the report gives zero-shot numbers; adaptor and tokenizer API: [UNVERIFIED]) | **NVIDIA Open Model License**, ungated; "ready for commercial/non-commercial use" | `AutoModel(..., trust_remote_code=True)` | IN-1k zero-shot 82.01 / 83.09, kNN 85.75 / 86.68, ADE20k 55.1 / 55.2 (card table) | none (report mentions no medical or microscopy use) |
| **TIPSv2** B/L/So400m/g (14) | Google DeepMind, US | 86M / 303M / 412M / 1.1B (text 110M / 184M / 448M / 389M) | 448 | yes | Apache-2.0, ungated | `AutoModel(..., trust_remote_code=True)`; timm `vit_so400m_patch14_reg1_tipsv2.webli` | abstract only: "on par or better than recent vision encoders" over 9 tasks and 20 datasets; I did not find an IN-1k table | none |
| **EmbeddingGemma 2** (text + vision + audio, one 768-d space) | Google DeepMind, US | vision about 168M (gemma4_vision, 768-d, 16 layers, 12 heads); text 270M; total 740M | token budget 70/140/280 (default)/560/1120 | yes (shared embedding space, MRL 768/512/256/128) | Apache-2.0, **ungated** | `sentence-transformers` or `transformers` (use bf16, not fp16) | MIEB-lite 64.64, MMEB v2 image 57.28 (card) | none |
| **Gemma 4 vision towers** (from E2B/E4B; from 26B-A4B/31B) | Google, US | 150M / 550M | variable aspect ratio, 70-1120 soft tokens, patch 16 pooled 3x3 | **none** | Apache-2.0, ungated | `Gemma4VisionModel.from_pretrained("google/gemma-4-E2B-it")` | none standalone. Frozen during Gemma 4 pretraining (report 2.2) | VLM level only (MedXpertQA-MM in `backbones.md`) |
| AIMv2 L/H/1B/3B (patch 14; 224/336/448) plus a `-lit` L with text | Apple, US | L 0.3B, 1B (1.24B on disk), 3B (2.7B) | 224-448; `native` variant | only `aimv2-large-patch14-224-lit` (437M total) | **Apple AMLR: research-only, no commercial use, derivatives research-only** (license text read) | `AutoModel(..., trust_remote_code=True)` | IN-1k frozen trunk 89.0 (1B-448), 89.5 (3B) | arXiv 2505.19779: AIMv2-H ranked best on all four medical sets, fully fine-tuned or frozen (section 3) |
| MoEViE B/L/H | Meta, US | total 0.5B/1.7B/3.5B; active 0.1B/0.3B/1.1B | 224/384/448 | yes | **CC-BY-NC-4.0** | custom `moe_vie` + open_clip fork; CUDA and Triton only | IN-1k zero-shot 79.3 / 83.6 / 85.1 | none. No gain over PE-Core or SigLIP2, plus NC and custom kernels: skip |
| EUPE T/S/B (ViT-16, ConvNeXt) | Meta, US | under 100M | any, multiples of 16 | no | FAIR non-commercial research license | repo + torch hub | multi-teacher distilled small encoders | none. Skip |
| OpenVision 1/2/3 | UC Santa Cruz-led, US (authors from several universities and NVIDIA) | up to H/14, So400m/14 | up to 448 | v1: yes. v2: caption decoder | Apache-2.0 | open_clip | not checked | none. Trained on LLaVA-recaptioned DataComp; the recaptioner's lineage is [UNVERIFIED]. Skip |
| MetaEncoder-30B | Meta, US | 29.78B (PE-Core ViT-G/14 1.8B + 30B-class LM) | up to 4096 visual tokens | yes (it is the LM) | Apache-2.0 plus a usage policy; **gated, auto** | custom `modeling_metaencoder.py`; vLLM pooling path | 85.5% top-1 on 200 ImageNet images against all 1,000 labels (card, vLLM path); MMEB V3 image 0.790 | none. Labeler candidate only |
| *References, not general:* MedSigLIP-448 | Google, US | 878M total | 448 | yes | HAI-DEF (`other`), gated auto | `AutoModel` | n/a | CXR, derm, fundus, histopath (card) |
| *Reference:* BiomedCLIP, UniMed-CLIP | Microsoft, US; MBZUAI, UAE | ViT-B/16 | 224 | yes | MIT / [UNVERIFIED] | open_clip | n/a | see `clm_methods.md` 2.5 |

Not considered, by rule: BAAI EVA-CLIP, any Qwen/InternVL/Shanghai-AI-Lab encoder, Jina-CLIP (EVA tower), FG-CLIP and similar Chinese releases.

## 3. Evidence on medical transfer for general encoders

| Source | What it shows | Use for us |
|---|---|---|
| **Does DINOv3 set a new medical vision standard?** (arXiv 2509.06467) | DINOv3 frozen with a linear probe can beat BiomedCLIP and CT-Net on some tasks. NIH-14 chest X-ray AUC: DINOv3-L 0.7865 vs BiomedCLIP 0.7771; RSNA pneumonia: BiomedCLIP 0.8831 vs DINOv3-L 0.8708. Performance peaked at 512 px and fell at larger sizes. **Pathology (Camelyon16/17 WSI): UNI and CONCH far ahead (UNI 0.965 vs DINOv3-S 0.840); DINOv3 about the level of a generic ResNet50.** The abstract also names EM and PET as failure cases and says DINOv3 "does not consistently follow the scaling law" in the medical domain | Do not fine-tune DINOv3-7B. Use L. Expect EM and H&E/IHC to be the weak submodalities for general encoders. A pathology encoder as a third view for stain and specimen is worth a probe (it is not general-purpose, so it is out of scope here) |
| **MedMNIST+** (arXiv 2404.15786) and **Rethinking foundation models on MedMNIST** (arXiv 2501.14685) | End-to-end > linear probing > kNN (about 82.8 / 76.7 / 72.1 average accuracy, MedMNIST+). ViTs beat CNNs. DINO and DINOv2 are the strongest in LP and kNN; EVA-02 weak in LP; SAM weak. Resolution gains plateau between 128 and 224 px | Old generation (no SigLIP2, PE, DINOv3). Direction: fine-tune the whole encoder; don't expect resolution to be the lever for global metadata |
| **Fine-tuning natural-domain foundation models** (arXiv 2505.19779) | CoCa, AIMv2, DINOv2, CLIP, SAM2, VMamba, MAE on CBIS-DDSM, ISIC2019, APTOS2019, CheXpert. "AIMv2 achieved the highest performance across both frozen and unfrozen configurations in all four datasets"; DINOv2 and SAM2 close. At 224 px with AdamW, LR searched in 1e-5..1e-3. Margins are small (CheXpert AUC 0.819 vs 0.814) | AIMv2 is good but research-only licensed. Margins of 0.5 pt are within what fine-tuning noise would erase |
| **Pelvis-SigLIP** (MICCAI 2026 workshop) | Female pelvic MRI **series/sequence** retrieval: zero-shot fails for PubMedCLIP, BiomedCLIP, MedSigLIP and SigLIP-2 alike (14.5-29.4% sequence accuracy, below the 36.2% majority baseline). After end-to-end training all four reach 93% or more; MedSigLIP best at 95.1% | The closest thing I found to our task (acquisition-type tagging). It says zero-shot prompts are not enough for submodality, that fine-tuned general SigLIP-2 is at most about 2 points behind fine-tuned MedSigLIP (all four models reach 93% or more; I did not see the per-model SigLIP-2 figure), and that medical pretraining gives a small edge after fine-tuning. Single paper, single anatomy |
| **MedGemma / MedSigLIP report** (arXiv 2507.05201) | MedSigLIP is SigLIP-400M continued on 33M medical pairs (635k general-modality + 32.6M histopathology patches) with medical data at **2% mixing weight** to keep general ability; 896 px training, 448 px released. **The report contains no comparison against unmodified SigLIP-400M** (I checked) | The public record does not quantify what medical tuning adds over the general tower. MedBridgeRL and MMBU in `backbones.md` are the only indirect evidence |
| **Self-supervision drives representational convergence** (arXiv 2607.20274) | 18 image encoders, 650k chest radiographs. Models trained with label supervision converge least. A linear map fitted on 4,096 unlabeled radiographs transferred 14-finding classifiers at 0.987 of original performance | Two uses: (i) SSL and contrastive encoders are partly redundant views, so an ensemble of SigLIP2 and DINOv3 may give less than expected; (ii) a cheap **linear map between towers** could carry the classifier from the SigLIP2 tower to the Gemma tower (hypothesis, tested on chest X-ray only) |
| **Pathology benchmarks** (e.g. arXiv 2407.06508, 2408.15823) | Pathology-specific encoders (CONCH, Virchow2, UNI) outperform general ones on tile and slide tasks | Same message as DINOv3 above |

What I did not find: any paper that reports SigLIP2, PE-Core, TIPS or C-RADIO on medical modality, anatomy, or stain recognition, or on dataset-held-out medical classification. "Strongest general encoder for medical metadata" is therefore an inference from the general benchmarks plus the evidence above, and has to be settled by our own grouped-by-dataset probe (section 7).

## 4. Why not the alternatives

- **PE-Core-G (1.88B) is slightly better than SigLIP2-g-opt on IN-1k (85.4 vs 85.0) but costs about 1.6 B200-days to fully fine-tune for 10 epochs (ESTIMATE, section 6.1)**, so it breaks the budget. PE-Core-L (83.5 IN-1k, Apache, text tower) is the best fallback second view if the DINOv3 gate or license is a problem.
- **C-RADIOv4** is attractive operationally (one pass, any resolution), but its SigLIP2 and DINOv3 knowledge is distilled, so it is not independent of our two chosen encoders. Its license (NVIDIA Open Model) is also another custom license to track. Include it in the frozen bake-off; do not plan around it.
- **TIPSv2** may be a better *feature* extractor for small or local findings than SigLIP2, but our fields are global. It is the one to add if the probe shows SigLIP2 failing on microscopy.
- **AIMv2** (best on the 4-dataset medical comparison) is research-only. Only use it if the organisers confirm a research license is acceptable for the submitted model's lineage. A labeler's weights are not shipped, but the disclosure rule still applies; check.
- **MetaCLIP 2 and MoEViE** are CC-BY-NC with no accuracy edge.
- **EmbeddingGemma 2's** image quality (MMEB v2 image 57.28) is not comparable to dedicated image encoders; its value is shape and family, not accuracy.

## 5. Recommendation

### 5.1 Primary backbone

**SigLIP 2 So400m/16, fixed 384 px (576 tokens), with its text tower providing the label-prompt prior, full fine-tuning with layer-wise LR decay.** Reasons: Apache-2.0, ungated; text tower included (label-text-initialised heads without extra machinery); family with demonstrated medical adaptability (MedSigLIP, Pelvis-SigLIP); the best price/accuracy point in the table (about 0.2 B200-days for a 10-epoch run). Variants to test in this order:

1. 384 px fixed (baseline).
2. **NaFlex** at 576 tokens (`google/siglip2-so400m-patch16-naflex`), because image aspect ratio and borders are themselves metadata cues (ultrasound fan, endoscopy circle, wide fundus, EM). NaFlex exists for B and So400m only (not L or g-opt). Which is better for our fields is unmeasured [UNVERIFIED].
3. 512 px fixed (`siglip2-so400m-patch16-512`), roughly double the cost, for EM, fundus and WSI-tile cases.
4. SigLIP2-L/16 at 384 or 512 as a scaling check, and g-opt/16 at 384 (about 0.47 B200-days) only if So400m clearly beats L.

### 5.2 Second view

**DINOv3-L/16 at 512 px** (about 0.28 B200-days for 10 epochs, ESTIMATE). Reasons: different pretraining signal from SigLIP2 (self-supervised, no text), best-documented medical results among general encoders (section 3), cheap. Caveats: manual gate on `facebook/dinov3-*` (use the ungated `timm/vit_large_patch16_dinov3.lvd1689m` mirror, same license); it has no text tower, so label vectors come from the primary's text embeddings through a fitted linear map (`clm_methods.md` 4.2) or are learned free; weak on EM and H&E tiles. Combine at the logit level after per-field temperature scaling, or concatenate pooled features into one head; compare both.

**Fallbacks.** If the DINOv3 license is a problem: PE-Core-L/14-336 (Apache, text tower). If a single model must do both jobs: C-RADIOv4-H. If SigLIP2 fails on microscopy: TIPSv2-So400m.

### 5.3 Transfer to the Gemma 4 student

- **Weights do not transfer from SigLIP2 to a Gemma tower.** Transfer is via labels (the main plan), soft-label distillation, or an aux head on the student's tower (`clm_methods.md` 4.8).
- **The Gemma-4-E2B tower (150M, 768-d, 16 layers) can be fine-tuned as a CLM-style classifier itself** with a MAP head and label vectors, then put back in the VLM. Cost: the tower pools 3x3 *after* the ViT, so the default 280 soft tokens means 2,520 patches through the tower. Use a 70-140 token budget for classification (630-1,260 patches, ESTIMATE about 0.08-0.15 B200-days for 10 epochs at 150M).
- **EmbeddingGemma-2's vision tower is the text-aligned member of that family.** Same shape as the E2B tower; weights verified different on three tensors (layer 0 MLP up-projection, layer 7 q-projection, layer 15 MLP down-projection). So it is a retrained tower, with a text tower in the same 768-d space, Apache-2.0 and ungated. Proposal: fine-tune it with the same CLM-style recipe as the primary and compare against the E2B-tower classifier. If it matches SigLIP2-B-level accuracy, it is the best Track-3 "bridge". Putting it into the E2B VLM would still need the projector re-aligned (the E2B `embed_vision` projection was trained for the original tower), and whether its init was the E2B tower is [UNVERIFIED].
- **Linear-map transfer (hypothesis from arXiv 2607.20274):** fit a ridge map from Gemma-tower pooled features to SigLIP2 pooled features on unlabeled biomedical images, then reuse the SigLIP2 classifier through it as a distillation target.

### 5.4 Labeler option: MetaEncoder-30B

It is not a general vision encoder, but it is the multimodal CLM-style model we were waiting for, from a US lab, with the PE-Core ViT-G/14 inside it. A frozen-embedding probe (task text plus candidate label descriptions, two small heads on cached embeddings) is the CLM recipe of `clm_methods.md` section 1. ESTIMATE: 880k images x about 300 visual tokens through 30B parameters is about 1.6e19 FLOPs, or about 6 B200-hours at 0.8 PFLOP/s. Access is auto-gated. Lineage of Muse Glimmer's LM beyond "Meta Superintelligence Lab" is [UNVERIFIED], so check the no-Chinese-base rule before using it. It cannot be a submission component. Run it only as an extra teacher after the cheap bake-off.

## 6. Compute and fine-tuning recipe

### 6.1 Budget (ESTIMATE)

Basis: forward FLOPs per image = 2 x params x tokens + 4 x layers x tokens^2 x width; training = 3 x forward (no activation-checkpointing recompute; add about 33% if used); 880k images; 0.8 PFLOP/s effective per B200 (same assumption as `backbones.md`). One B200-day = 6.9e19 FLOPs.

| Config | Tokens | Train TFLOP / image | 10 epochs, full FT | One frozen pass over 880k |
|---|---|---|---|---|
| SigLIP2-B/16 @384 | 576 | 0.33 | 1.0 h (0.04 d) | 0.03 h |
| SigLIP2-L/16 @384 / @512 | 576 / 1024 | 1.15 / 2.17 | 3.5 h (0.15 d) / 6.6 h (0.28 d) | 0.12 / 0.22 h |
| **SigLIP2-So400m/16 @384** | 576 | 1.61 | **4.9 h (0.20 d)** | 0.16 h |
| SigLIP2-So400m NaFlex, 256 tokens | 256 | 0.68 | 2.1 h (0.09 d) | 0.07 h |
| **SigLIP2-So400m/16 @512** | 1024 | 3.03 | **9.3 h (0.39 d)** | 0.31 h |
| SigLIP2-g-opt/16 @384 | 576 | 3.70 | 11.3 h (0.47 d) | 0.38 h |
| **DINOv3-L/16 @512** | 1024 | 2.17 | **6.6 h (0.28 d)** | 0.22 h |
| DINOv3-H+/16 @512 | 1024 | 5.68 | 17.3 h (0.72 d) | 0.58 h |
| DINOv3-7B @512 | 1024 | (frozen only) | not trainable in budget (5.5 d) | **4.4 h** |
| PE-Core-L/14 @336 | 576 | 1.20 | 3.7 h (0.15 d) | 0.12 h |
| PE-Core-G/14 @448 | 1024 | 12.5 | 38 h (1.6 d): over budget | 1.3 h |
| Gemma-4 150M tower, 70 / 280 soft tokens | 630 / 2520 patches | 0.63 / 3.20 | 1.9 h (0.08 d) / 9.8 h (0.41 d) | 0.06 / 0.33 h |
| Gemma-4 550M tower, 280 soft tokens | 2520 patches | 10.7 | 33 h (1.4 d): over budget | 1.1 h |

Throughput implied for So400m@384: about 500 images/s, so image decoding, not the GPU, is likely the bottleneck for large medical images. Pre-resize to shards (for example 512 px JPEG quality 95 in WebDataset) once. LoRA saves about one third of the compute (no weight gradients), mainly memory.

**Plan inside one B200-day per run:** each run is far below the cap. Spend the headroom on (a) three seeds of the primary, (b) the LoRA arm, (c) a 512-px and a NaFlex variant, (d) WiSE-FT sweeps (evaluation only, minutes).

### 6.2 Starting recipe (values marked *derived* are my choices; sourced values carry a citation)

**Labels.** Build K = 4-8 prompts per label per field ("a {modality} image of the {body part} ...", UniMed-CLIP style) and cache their text-tower embeddings. Init each label vector `w_k` from the mean prompt embedding (zero-shot head init; used by WiSE-FT [Wortsman 2022] and found best in PartialCLIP [He 2025]). Keep the text tower frozen: the vocabulary is about 250 labels, so updating the 0.7B text tower mostly learns a lookup table, and freezing keeps the zero-shot endpoint exact for WiSE-FT.

**Stage A: head warm-up (LP-FT's first half).** Image tower frozen; train the per-field MAP query heads and label vectors (L2 pull to init, stronger for labels seen in 1-2 datasets), lr 1e-3, wd 0, 1-2 epochs. LP-FT beats plain fine-tuning by about 1 pt ID and 10 pt OOD on the paper's shift sets (Kumar 2022, arXiv 2202.10054); FLYP-style initialisation from the text head gives the same protection.

**Stage B: FLYP-style fine-tuning.** Loss = the field's softmax CE (or sigmoid for multi-label fields) over the full field vocabulary with logits `exp(t_f) * cos(q_f, w_k)`, where `w_k` is re-sampled each step from the K cached prompts. That equals FLYP's contrastive loss with the whole vocabulary in the denominator, and keeps class collisions in (FLYP: ignoring collisions beat masking; PCam -1.3% when masked). Reported FLYP gains: ImageNet ID/OOD 82.6/60.2 (LP-FT 81.8/60.5; plain FT 81.4/54.8); +4.2% OOD over plain FT across 7 shifts (arXiv 2212.00638).

| Setting | Start value | Source / status |
|---|---|---|
| Optimizer | AdamW, bf16, grad-clip 1.0 | derived; MedMNIST+ and the CLIP fine-tuning paper both use AdamW |
| Batch | 512-1024, class-balanced over (dataset x label) | derived (`clm_methods.md` 4.4) |
| Image-tower peak LR (top block) | **3e-5**, grid {1e-5, 3e-5, 1e-4} | derived; brackets: SigLIP2-So400m fine-tune for IQA used backbone 5e-6 (head 1e-4, batch 12, 512 px, arXiv 2509.17374); MedMNIST+ used 1e-4; the CLIP fine-tuning paper used 4e-4 at batch 2048 with LLRD 0.65 (ViT-L/14) |
| Layer-wise LR decay | **0.8** (27 layers), grid {0.7, 0.8, 0.9}; for L (24 layers) 0.75-0.85 | derived; CLIP fine-tuning paper (arXiv 2212.06138): 0.6 for B/16, 0.65 for L/14. Those are at much higher base LR, so they do not transfer directly |
| Head / label-vector LR | 1e-3, no LLRD | `clm_methods.md` 4.1 (conventional) |
| Weight decay, warmup, schedule | 0.05; 5% of steps; cosine to 0 | CLIP fine-tuning paper: wd 0.05, warmup 5-10 epochs of 30-50 |
| Epochs | 5-10 over 880k | derived; the CLIP paper used 30-50 on ImageNet (1.28M) |
| EMA | 0.9998 on weights, evaluate the EMA | CLIP fine-tuning paper (arXiv 2212.06138) |
| Label smoothing | 0.1 for softmax fields; none for sigmoid | CLIP fine-tuning paper (softmax); derived |
| Drop-path / mixup / cutmix | 0-0.1 / **off** / **off** | CLIP paper removed mixup and cutmix; mixup would also blend stain identity |
| Augmentation | RandomResizedCrop scale (0.7, 1), flips, small rotation, blur, brightness/contrast, JPEG and down-up resampling (kills dataset-specific compression cues). **No hue or saturation jitter, no grayscale** | `clm_methods.md` 4.1; derived |
| Resolution | 384 fixed first; then 512 and NaFlex-576 | derived; DINOv3 chest X-ray peaked at 512 px (arXiv 2509.06467); MedMNIST+ plateaued past 128-224 |
| Multi-label fields (body part, domain) | sigmoid with SigLIP bias init (t' = log 10, b = -10), ASL-style negative down-weighting, ancestors as positives | SigLIP (arXiv 2303.15343); ASL (arXiv 2009.14119); `clm_methods.md` 4.4 |
| Candidate-set labels | `-log sum_{c in S} p(c)` with zero-shot pre-filtering of S, class-prior balancing and entropy regulariser | CC loss (arXiv 2007.08929, Feng 2020); PRODEN self-training (arXiv 2002.08053); PartialCLIP pre-filtering (arXiv 2506.05027) |
| Validation | leave-dataset-group-out, per held-out dataset, never image-level | `clm_methods.md` |

**Stage C: robustness interpolation.** WiSE-FT on the image tower and label vectors: `theta = (1 - a) theta_zeroshot + a theta_ft`, sweep a in {0.3, 0.5, 0.7, 1.0} on held-out dataset groups (+4 to 6 pt under shift in the WiSE-FT paper, arXiv 2109.01903). If three or more seeds exist, also try Model Stock (layer-wise averaging of two fine-tuned models, ECCV 2024, arXiv 2403.19522), and Lipsum-FT (arXiv 2404.00860, which regularises with random text guidance) as a second robust-fine-tuning arm, but only if WiSE-FT leaves a visible held-out gap.

**LoRA arm.** Same data and loss; rank 32-64 on QKV, output and both MLP matrices of every block, alpha = 2r, dropout 0.05, LR 2e-4 to 5e-4 (derived; typical LoRA LRs run about 5x higher than full FT: one source quotes 5e-4 vs 1e-4 [from search snippets only]; the SigLIP2 IQA study used rank 4 on q,k and matched full fine-tuning within noise). Evidence on LoRA vs full: "LoRA Learns Less and Forgets Less" (Biderman 2024, arXiv 2405.09673; LLM experiments): LoRA underperforms full fine-tuning in most settings but regularises better outside the target domain. For ViTs I have only secondary snippets that lower-rank updates forget less [UNVERIFIED]. Our decision rule: pick the arm with the better held-out-dataset macro-F1, and always run WiSE-FT on both. My prior: full fine-tuning with LLRD wins (880k images, large domain shift for EM and histology), LoRA is a free regulariser worth one run.

**Calibration.** Per-field temperature scaling on held-out dataset groups (not pooled), ECE per held-out dataset, three seeds or WiSE-FT interpolants for the ensemble, explicit "other/unknown" class (`clm_methods.md` 4.6).

## 7. Cheapest experiments that settle the ranking (about 1.5 days of B200)

1. **Frozen probe bake-off** on grouped-by-dataset CV (CLS + mean-pooled patch tokens, mid-late layer, concatenation of the last 4 layers): SigLIP2 B/L/So400m/g; DINOv3 L/H+/7B (7B only here, 4.4 h per pass); PE-Core L/G; C-RADIOv4 SO400M/H; TIPSv2 So400m; EmbeddingGemma 2 vision; Gemma-4-E2B and -26B-A4B towers (70 and 280 tokens); MedSigLIP and BiomedCLIP as oracle references. Report per-field macro-F1, per-modality recall (especially EM, H&E/IHC, fundus, OCT, endoscopy), and rare-class recall.
2. **Zero-shot SigLIP2 with prompt ensembles** as the WiSE-FT endpoint and as the PartialCLIP-style pre-filter for candidate-set datasets.
3. **Full fine-tune of the top 2-3 encoders** with section 6.2, one seed, grouped-CV; then LoRA arm, NaFlex and 512-px variants of the winner.
4. **Gemma-bridge test:** E2B-tower classifier vs EmbeddingGemma-2-tower classifier vs SigLIP2-B classifier at equal budget, to see whether a Gemma-shaped tower can approach the SigLIP2 score.

## 8. Corrections and updates to the two existing notes

| Where | Said | Now |
|---|---|---|
| `clm_methods.md` 4.1 table | SigLIP 2 license "believed Apache-2.0 [UNVERIFIED]"; PE license "believed Apache-2.0 [UNVERIFIED]" | Both verified Apache-2.0, ungated (HF API, `LICENSE.PE`). DINOv3 terms verified (section 2) |
| `clm_methods.md` 4.1 and `backbones.md` section 1 | Gemma 4 tower "encoder lineage not stated"; 550M tower "same shape as SigLIP-So400m (SigLIP2-derived?)" | Same width, depth, heads, MLP width, but a decoder-style block with a gated MLP, 2D RoPE plus a learned 2D position table; params reconcile at 546M. SigLIP2 weights do not load. Lineage of the weights is still unstated [UNVERIFIED] |
| `clm_methods.md` 4.1 Gemma row | "No paired text tower" | EmbeddingGemma 2 provides a text-aligned tower of the same shape as the E2B tower |
| `backbones.md` row `EmbeddingGemma-300M` | gated, text only | Superseded for our purposes by `google/embeddinggemma-2` (Apache-2.0, ungated, multimodal, 768-d) |
| `backbones.md` section 4(c) "LFM2.5-VL ... a SigLIP2-adapted tower could drop in" | applies to SigLIP2-shaped VLM towers only | Still true for LFM2; does not apply to Gemma 4 |
| `backbones.md` PE-Core row | "no medical benchmark found" | Still true today (re-searched) |
| Recommendation | SigLIP2-So400m primary, DINOv3-L second | Confirmed. Added fallbacks (PE-Core-L, C-RADIOv4-H, TIPSv2) and the Gemma-shaped bridge (EmbeddingGemma 2) |

## 9. Unverified or open

- Numeric claims not extracted from primary tables: DINOv3 IN-1k linear numbers and the dino.txt zero-shot table (the HTML returned a truncated section 7.3); TIPSv2 benchmark tables; C-RADIOv4 adaptor names for text alignment.
- Whether NaFlex or fixed resolution is better for our fields.
- Whether the EmbeddingGemma-2 vision tower was initialised from the E2B tower.
- Lineage of the Muse Glimmer language model (MetaEncoder-30B) with respect to the no-Chinese-base rule.
- TULIP weight availability and license; OpenVision recaptioner lineage.
- ViT-specific LoRA-vs-full forgetting evidence (only secondary snippets seen).
- Every FLOP and hour number in section 6.1 is arithmetic, not a measurement; real HF code with dynamic NaFlex padding or the Gemma-4 tower may run 1.5-3x slower.
- Organizer questions already open in `backbones.md` section 5 (PLE counting, tower replacement allowed or not) decide whether the Gemma-bridge path is usable at all.

## 10. Sources

HF Hub API/cards, read 2026-10-07 (`https://huggingface.co/<repo>`): `google/siglip2-*`, `google/tipsv2-*`, `google/embeddinggemma-2`, `google/gemma-4-E2B`, `google/gemma-4-26B-A4B`, `google/medsiglip-448`, `google/medgemma-1.5-4b-it`, `facebook/dinov3-*`, `timm/vit_large_patch16_dinov3.lvd1689m`, `facebook/PE-Core-*`, `timm/PE-Core-*`, `facebook/metaclip-2-*`, `facebook/MoEViE-*`, `facebook/EUPE-*`, `facebook/meta-encoder`, `meta-models/Muse-Glimmer-30B`, `nvidia/C-RADIOv4-SO400M`, `apple/aimv2-*`, `UCSC-VLAA/openvision*`.

Papers and pages:
- SigLIP 2: https://arxiv.org/abs/2502.14786 ; HF blog https://huggingface.co/blog/siglip2
- DINOv3: https://arxiv.org/abs/2508.10104 ; repo and license https://github.com/facebookresearch/dinov3
- Perception Encoder: https://arxiv.org/abs/2504.13181 ; license https://github.com/facebookresearch/perception_models
- MetaCLIP 2: https://arxiv.org/abs/2507.22062
- C-RADIOv4: https://arxiv.org/abs/2601.17237
- TIPSv2: https://arxiv.org/abs/2604.12012
- AIMv2: https://arxiv.org/abs/2411.14402 ; license text at `https://huggingface.co/apple/aimv2-1B-patch14-448/resolve/main/LICENSE`
- TULIP: https://arxiv.org/abs/2503.15485
- Gemma 4 report: https://arxiv.org/abs/2607.02770 ; transformers docs https://huggingface.co/docs/transformers/model_doc/gemma4
- dino.txt: https://arxiv.org/abs/2412.16334
- Does DINOv3 set a new medical vision standard: https://arxiv.org/abs/2509.06467
- MedMNIST+: https://arxiv.org/abs/2404.15786 ; foundation models on MedMNIST: https://arxiv.org/abs/2501.14685
- Fine-tuning natural-domain foundation models for medical classification: https://arxiv.org/abs/2505.19779
- Pelvis-SigLIP: https://papers.miccai.org/miccai-2026-sat/CAPI_WOMEN_019.html
- MedGemma report: https://arxiv.org/abs/2507.05201
- Representational convergence in medical encoders: https://arxiv.org/abs/2607.20274
- Pathology benchmarks: https://arxiv.org/abs/2407.06508 ; https://arxiv.org/abs/2408.15823
- CLIP fine-tuning recipe: https://arxiv.org/abs/2212.06138
- SigLIP2 fine-tuning for IQA (full vs LoRA recipe): https://arxiv.org/abs/2509.17374
- FLYP https://arxiv.org/abs/2212.00638 ; WiSE-FT https://arxiv.org/abs/2109.01903 ; LP-FT https://arxiv.org/abs/2202.10054 ; Model Stock https://arxiv.org/abs/2403.19522 ; Lipsum-FT https://arxiv.org/abs/2404.00860
- LoRA Learns Less and Forgets Less: https://arxiv.org/abs/2405.09673
- ASL: https://arxiv.org/abs/2009.14119 ; CC loss: https://arxiv.org/abs/2007.08929 ; PRODEN: https://arxiv.org/abs/2002.08053 ; PartialCLIP: https://arxiv.org/abs/2506.05027 ; SigLIP: https://arxiv.org/abs/2303.15343
- Earlier notes: `backbones.md`, `clm_methods.md` in this folder.
