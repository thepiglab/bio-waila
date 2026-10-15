# MedGPT-oss (University of Florida / Lehigh): backbone assessment for MMBU Pass-2 and submission

*Research date: 2026-10-06. Sources read today: arXiv 2603.00842v2 (full PDF text), HF API and model card for `UFNLP/MedGPT-oss`, HF API for `openai/gpt-oss-20b`, HF dataset cards for II-Medical-Reasoning-SFT, ReasonMed, PubMedVision, MedTrinity-25M, plus local notes `data_discovery/research/mmbu_exclusion_list.md` and `docs/models/backbones.md`. [UNVERIFIED] = no primary source read. Search budget used: 2 WebSearch calls. The weights, `config.json`, `README.md` raw file and `modeling_medgpt_oss.py` are behind HF auto-gating (HTTP 401), so I could not read the code or the config; architecture facts come from the paper and the model card.*

## 0. Bottom line

- **Real, public, and from UF**, but a poor fit for us. It is **not eligible for Track 2** (HF repo is `gated: auto`, license is **CC BY-NC 4.0**, so it fails "public and ungated"). It is **not eligible for Track 3** (21.2B total params, over the 12B cap; active params would be about 3.9B, so only the active cap is met).
- **Architecture is a plain LLaVA-style bolt-on: gpt-oss-20B (OpenAI, MoE) + OpenAI CLIP-ViT-L/14@336 + 2-layer MLP.** Everything is US-origin at the weight level. The one gray area is the **training data**: several instruction-tuning text sets are distilled from Qwen and DeepSeek models (section 3.5). That matters only if the organizers read "built on" to include distillation data.
- **No Pass-2-relevant evidence.** The paper reports no modality, organ, stain or VQA-RAD/PathVQA/PMC-VQA/OmniMedVQA/MMBU numbers. Its vision benchmarks are SLAKE, MedFrameQA, MMMU-Med, MedXpertQA-MM. Its own Figure 1 ablation found that vanilla CLIP beat BiomedCLIP, SigLIP and MedSigLIP as the tower, which is mild evidence *against* "biomedical-pretrained tower" and says nothing clean about tower tuning.
- **Use as a labeler or teacher: no.** **Use as a CLM image tower: no.** **Value to us:** a data-recipe reference (staged alignment, joint tower+LM tuning in mid-training).

## 1. Identification

| Item | Finding |
|---|---|
| Name | **MedGPT-oss** (paper title: "MedGPT-oss: Training a General-Purpose Vision-Language Model for Biomedicine"). Also written MEDGPT-OSS-20B in the paper. |
| Name collisions | Unrelated community text-only models on HF: `Tonic/med-gpt-oss-20b`, `justinj92/MedGPT-OSS-20B` (2025-08), plus GGUF quants. These are LM fine-tunes with no vision. Do not confuse them with the UF model. |
| Paper | arXiv **2603.00842**, cs.CL. v1 2026-03-01, **v2 2026-09-21**. "Technical Report", described by the authors as a work-in-progress report. [arxiv.org/abs/2603.00842](https://arxiv.org/abs/2603.00842) |
| HF repo | **`UFNLP/MedGPT-oss`**. Created 2026-04-22, last modified 2026-05-13. 3 downloads and 1 like at read time. Tags: `custom_code`, `image-text-to-text`, `medgpt_oss`. |
| Org | UFNLP on HF = UF Department of Health Outcomes & Biomedical Informatics (HOBI), the GatorTron group. Lead contact Yonghui Wu (yonghui.wu@ufl.edu). Other UFNLP models are the GatorTron family. |
| GitHub | None found. The paper gives no URL, and the HF org page links only the personal GitHub `yonghuiwuf`. The paper says it releases "the complete training recipe ... and a rigorous evaluation harness". I could not locate that harness repo. [UNVERIFIED] whether it exists publicly. |
| Authors | Kai Zhang (Lehigh, now Stanford Radiation Oncology), Zhengqing Yuan (Notre Dame), **Cheng Peng, Mengxian Lyu, Ziyi Chen (UF HOBI)**, Songlin Zhao (Lehigh), Yanfang Ye (Notre Dame), Wei Liu (Mayo Clinic), **Ying Zhang (UF Research Computing)**, Kaleb E Smith (NVIDIA), Lifang He (Lehigh), Lichao Sun (Lehigh, corresponding), **Yonghui Wu (UF HOBI, corresponding)**. First/lead group is Lehigh (Sun lab, BiomedGPT lineage); UF contributes data/compute and the HF release. |
| Funding / compute | University of Florida compute, ARPA-H PARADIGM, NSF. Trained on 8x NVIDIA B200 (DeepSpeed ZeRO-3). |
| License | **CC BY-NC 4.0** on the model (HF API `license: cc-by-nc-4.0`). The click-through terms prohibit commercial use, clinical decision-making and deployment in clinical systems. The paper text says "permissive licensing" and the arXiv page is CC BY 4.0, but that applies to the paper, not the weights. |
| Gating | **`gated: auto`** (agreement form, auto-approved). |
| Size on disk | 21,232,236,224 BF16 params, 9 safetensors shards, about 40 GB. |

## 2. Architecture

| Component | Detail | Source |
|---|---|---|
| LM | **gpt-oss-20B**, full model (OpenAI, US, Apache-2.0). MoE, 20.9B total (HF API 20,914,757,184) and 3.6B active. Uses the harmony chat format. | paper section 3.1; gpt-oss card |
| Vision encoder | **OpenAI CLIP-ViT-L/14@336px**, about 0.3B. Vanilla CLIP, not a medical checkpoint. | paper section 3.1 |
| Projector | **2-layer MLP** (CLIP features to LM embedding space). The intro says "linear projector", section 3.1 says two-layer MLP. The paper is inconsistent. | paper |
| Resolution handling | Dynamic tiling: up to **12 tiles + thumbnail**, 336 px tiles, **downsample ratio 0.5** (pixel-shuffle style, InternVL-like convention). Tokens per tile not stated. About 144 per tile if 24x24 patches are shuffled by 0.5 [UNVERIFIED, my arithmetic]. | paper section 3.3 |
| Context | YaRN scaling factor 32, theta 150k, 4,096 to 131,072 positions. Training max sequence length 32,768. | paper section 3.3 |
| Total params | **21.23B**. Delta over gpt-oss-20b = 317.5M, which matches CLIP-L (about 304M) + small projector. | HF API arithmetic |
| Active params incl. vision | About 3.6B + 0.32B = **about 3.9B** (my estimate). | arithmetic |
| HF class | Custom code: `MedGptOssModel`, `modeling_medgpt_oss.py`, `configuration_medgpt_oss.py`. Needs `trust_remote_code`, transformers >=4.55. Card says one 48 GB GPU, or two 24 GB. | HF card |

**Eligibility:**

| Track | Rule | MedGPT-oss | Verdict |
|---|---|---|---|
| Track 3 | at most 4B active and at most 12B total, incl. vision | about 3.9B active (ok), **21.2B total (fail)** | **Ineligible** |
| Track 2 (<27B) | public and ungated on HF | 21.2B (ok), **gated: auto, NC license (fail)** | **Ineligible as released** |

A re-trained clone from ungated Apache-2.0 gpt-oss-20b + CLIP would be Track-2-size-legal, but see section 5(c).

## 3. Training

### 3.1 Stages (paper section 3.2, Table 1, Table 2)

| Stage | Trainable | Data | Wall-clock (8x B200) |
|---|---|---|---|
| 1. Pretraining / short-context alignment | LLM frozen; **vision encoder and projector trainable** (paper text). Projector LR 1e-3. | about **939K** image-caption pairs: PMC-OA (477,078 filtered), Quilt-1M (359,725 filtered), ROCOv2 (38,861 filtered), BIOMEDICA derm+surgery (63,775). Single-panel only, artifacts and overlaid text removed, LLM-assisted cleaning. | 8 h 27 min |
| 2. Mid-training / long-context | **All parameters** (tower, projector, LM) | about **19M** samples: MedTrinity open subset (18.5M), PubMedVision alignment (646,759), MIMIC-CXR single (22,747) + multi-view/longitudinal (146,893). | 240 h 24 min |
| 3. Instruction tuning | All parameters | about **7M** pairs: text-only (about 5.3M) + multimodal VQA. | 86 h 26 min |

Total about 335 h x 8 GPUs = about **2,700 B200-GPU-hours** (my arithmetic).

**Internal inconsistency on the tower:** the Stage-1 prose says the visual encoder is trainable. Table 2 says "Vision encoder: Frozen in stage 0-1; jointly trained in stage 2". The paper also calls the first stage "stage 0". Treat "tower frozen in stage 1, tuned in stages 2-3" as the likely reading. [UNVERIFIED which is correct.]

### 3.2 Image data (Table 1)

- **Radiology:** ROCOv2, MIMIC-CXR (+ MIMIC-CXR-VQA 363,598), SLAKE (9,835), VQA-RAD (1,793), GEMeX-ThinkVG (202,383).
- **Pathology:** Quilt-1M, Quilt-VQA (107,131).
- **Mixture:** PMC-OA, BIOMEDICA subsets, **MedTrinity open subset (18.5M)**, PubMedVision, PMC-VQA (188,229), OpenMM-Medical (88,996), GMAI-Reasoning10K (9,965).

### 3.3 Text-only data

II-Medical-Reasoning (2.2M), Infinity-Instruct-Gen (1.48M, general chat), ReasonMed (1.1M), MedMCQA, HealthCareMagic-100k, MedQA, MedExpQA, PubMedQA, AlpaCare-MedInstruct-52k, MedQuAD, HuatuoGPT-o1 data, MedReason, Medical-R1-Distill-Data (22K, "distilled from DeepSeek-R1"), MedThoughts-8K, iCliniq-10K.

### 3.4 Overlap risk with MMBU and with excluded sources

- MMBU excluded (by construction) the training sets reported by LLaVA-Med, MedGemma, Lingshu and OctoMed (`data_discovery/research/mmbu_exclusion_list.md`). Of MedGPT-oss's image sources, the following are named in that list and so are **not** among MMBU's 410 sources: PMC-OA, ROCOv2, Quilt-1M / Quilt-LLaVA, PubMedVision, MIMIC-CXR, MIMIC-CXR-VQA, SLAKE, VQA-RAD, PMC-VQA. Name-level overlap with MMBU is therefore low.
- **Not in that list:** MedTrinity-25M (18.5M samples in Stage 2). Its HF card says it aggregates **27 public collections** (CheXpert, MIMIC-CXR-JPG, PadChest, NIH-CXR, TCGA, NCT-CRC-HE-100K, Breast Histopathology, BraTS24, CT-RATE, DeepLesion, plus PMC-VQA/PathVQA/VQA-RAD/SLAKE). Several of these are plausible upstream originals of MMBU sources (NCT-CRC-HE, TCGA derivatives). **Image-level overlap with MMBU is possible for MedTrinity-derived data** [UNVERIFIED which MMBU sources]. The same name-vs-image caveat in the exclusion note applies. It matters only if we *train on* MedGPT-oss outputs or weights, which we are not planning to do.
- BIOMEDICA (derm + surgery) and GEMeX-ThinkVG, GMAI-Reasoning10K, OpenMM-Medical are not in the exclusion list. They are literature-derived or reasoning-QA sets, so overlap is probably low [UNVERIFIED].
- Contamination controls reported by the authors: "prevent overlap with evaluation data" against their own benchmarks only. MMBU is not mentioned.

### 3.5 Chinese-origin check (constraint d)

| Component | Origin | Verdict |
|---|---|---|
| LM gpt-oss-20B | OpenAI, US | OK |
| Vision tower CLIP-L/14-336 | OpenAI, US | OK |
| Projector | Random init (no pretrained projector described) | OK |
| Pipeline code | Tiling + 0.5 downsample + thumbnail follow the InternVL convention. Paper cites LLaVA-OneVision-1.5 and Qwen-VL as architectural inspiration. | Code-style influence only. Cannot read `modeling_medgpt_oss.py` (gated). [UNVERIFIED] whether any InternVL code was reused. |
| Instruction data | **II-Medical-Reasoning**: Qwen3-235B-A22B synthetic answers (873K), DeepSeek-R1 traces, Qwen2.5-32B-Instruct classifier. **ReasonMed**: Qwen2.5-72B, DeepSeek-R1-Distill-Llama-70B and HuatuoGPT-o1-70B for the CoT paths, Qwen2.5-72B as verifier (Lingshu team). **Medical-R1-Distill-Data** and probably **MedThoughts-8K**: DeepSeek-R1 distilled. **Infinity-Instruct** (BAAI), **HuatuoGPT-o1 data**, **PubMedVision** (CUHK-SZ, but generated with GPT-4V), GMAI-Reasoning10K (Shanghai AI Lab) | **Gray area.** No Chinese weights are used, but a large share of the text SFT data is generated by Qwen and DeepSeek models. Under a strict reading of "built on", this would need an organizer ruling. [UNVERIFIED whether the rule covers distillation data.] |

## 4. Results (all from the paper's own harness, not copied from other papers; accuracy %)

**Vision benchmarks (Table 4):**

| Benchmark | MedGPT-oss | MedGemma-27B | Lingshu-32B | Hulu-Med-32B | OctoMed-7B |
|---|---|---|---|---|---|
| MedXpertQA-MM (2,000 q) | **49.23** | 30.90 | 31.43 | 34.35 | 25.60 |
| SLAKE (in-domain) | 71.53 | 55.98 | 72.24 | 69.14 | 65.07 |
| MedFrameQA | **63.01** | 47.63 | 61.01 | 62.82 | 42.82 |
| MMMU-Med dev | **61.49** | 47.43 | 59.43 (Lingshu trained on dev) | 57.71 | 47.65 |
| MMMU-Med-Pro 4-opt | 52.34 | 45.80 | 52.67 | 52.45 | 44.62 |
| MMMU-Med-Pro 10-opt | 39.94 | 36.71 | 43.45 | 37.41 | 23.07 |

**Text-only (Table 5):** MedQA 70.11, PubMedQA 57.81, MedMCQA 62.53, MedXpertQA-text 25.38 (best), MMLU-Med 72.59, Medbullets 68.71 (best). MedGemma-27B beats it on MMLU-Med (80.65) and MedMCQA (65.48).

**Other:** MIMIC-CXR report generation RadGraph-F1 0.189 / RaTEScore 0.522 / 1/RadCliQ 0.803, third behind Hulu-Med and Lingshu. ICL: Patient-trial 1-shot 55.60 (best), Impression RaTEScore 47.2.

**Not reported:** VQA-RAD, PathVQA, PMC-VQA, OmniMedVQA, GMAI-MMBench, MMBU, **any modality / organ / stain / specimen recognition**. VQA-RAD, PMC-VQA and SLAKE are *training* data, so the paper would not be a clean test of them anyway.

**Versus Gemma 4 (cross-protocol, indicative only).** Per `backbones.md`, the Gemma 4 model cards report MedXpertQA-MM of 28.7 (E4B) and 58.1 (26B-A4B). MedGPT-oss scores 49.2 in its own harness. The prompt format and harness differ, so the numbers are not directly comparable [UNVERIFIED comparability]. No head-to-head with Gemma 4 or MMBU exists. Each baseline in the paper is the largest published checkpoint of its family (27B-32B), so the "outperforms larger models" claim is on reasoning-heavy MCQ, not perception.

**Figure 1 (encoder ablation, GPT-oss-20B + LoRA, BiomedGPT corpus of 1M pairs):** vanilla CLIP beat BiomedCLIP, SigLIP and MedSigLIP on medical multimodal benchmarks (numbers only in the figure; I could not extract them from the text). This is the only tower evidence in the paper. It compares encoders after light adaptation, not frozen vs tuned tower, so there is **no ablation isolating the benefit of tuning the tower in Stage 2**.

## 5. Relevance to us

**(a) Pass-2 labeler or teacher (Jev-style): not recommended.**
- It was never trained on a closed-vocabulary modality, organ, stain or specimen task, and the paper offers no number that predicts such performance. Its recipe is caption plus VQA plus long-form reasoning.
- Conversion cost is high: 21B MoE (gpt-oss has 32 experts, top-4 [UNVERIFIED from card; the API config showed `experts_per_token: 4`]), harmony chat format with analysis/final channels, custom `trust_remote_code` class, MXFP4-origin weights re-saved as BF16 (40 GB).
- Gemma-4-26B-A4B is ungated, has native `Gemma4ForConditionalGeneration` support and a 550M tower, so it is the cheaper and better-supported MoE torso. Gemma-4-E4B is the cheapest labeler.
- If a second-opinion zero-shot labeler is wanted for audits, the 48 GB single-GPU footprint makes it usable, but I would not spend time on it.

**(b) Vision encoder as CLM image tower or as evidence: weak on both counts.**
- The tower is a CLIP-L/14-336 whose weights were updated with an LM loss through a gpt-oss decoder. It has no matching text tower in the release, so it is not a contrastive CLM backbone. We would have to train a text tower or head from scratch.
- Weights are NC-licensed and gated. SigLIP2-So400m (plus its text tower) remains the better CLM base.
- As evidence that biomedical tower tuning helps: the paper tunes the tower in the mid-training stage but never ablates it. The only ablation (Fig. 1) favors vanilla CLIP over medical-pretrained encoders in that setting. Our `backbones.md` thesis (Gemma 4 and Ministral 3 keep towers frozen, so tower adaptation to microscopy, endoscopy and fundus is the opportunity) is neither confirmed nor refuted. At most it shows that tower+LM joint tuning on 19M medical image-text pairs yields a strong OOD MCQ model.

**(c) Submission candidate for Track 2: no as released.**
- Gated and NC fail the "public, ungated" rule, whatever the size.
- Rebuilding our own is possible (Apache-2.0 gpt-oss-20b + MIT-licensed OpenAI CLIP, recipe is public) but costs about 2,700 B200-GPU-hours for the paper's recipe, uses data with Chinese-model-distilled text, relies on MedTrinity (possible MMBU image overlap), and is gated by the unverified harness release. Gemma-4-26B-A4B already has a native vision stack, 3.8B active and an ungated Apache-2.0 release. Net: not worth it.
- It is a useful *reference point* for what a staged recipe gets on MedXpertQA-MM (49.2 in its harness).

**(d) Chinese-origin component:** none in weights (OpenAI LM + OpenAI CLIP + new projector). **Data-level gray area:** Qwen3, Qwen2.5, DeepSeek-R1 and HuatuoGPT-o1 distilled text in the SFT mix (section 3.5). Pipeline code is InternVL-style [UNVERIFIED].

**(e) Practical.**
- Access: an HF access click-through gets the 40 GB.
- Inference: transformers >=4.55 with `trust_remote_code`. **No vLLM support documented**, and the custom `medgpt_oss` model type will not load in stock vLLM. [UNVERIFIED, config not readable.]
- A single 48 GB GPU is enough (about 40 GB BF16, plus activations for up to 12 tiles).
- If we ever want it as a baseline row in the report, run it via transformers on one H100/B200 and cite the NC license.

## 6. Verdict table

| Criterion | MedGPT-oss | Gemma-4-E4B | Gemma-4-26B-A4B | SigLIP2-So400m (classifier tower) |
|---|---|---|---|---|
| Origin | OpenAI LM + OpenAI CLIP, UF/Lehigh recipe | Google | Google | Google |
| Chinese-origin (weights) | None | None | None | None |
| Chinese-origin (training data) | **Gray** (Qwen/DeepSeek-distilled SFT text) | Not disclosed [UNVERIFIED] | Not disclosed [UNVERIFIED] | None known |
| Total / active params | 21.2B / about 3.9B | 8.0B / 4.5B | 25.8B / 3.8B | about 0.9B (tower alone, both towers) |
| Track 3 eligible | **No** (21.2B total) | No (4.5B active) | No | n/a (not a VLM) |
| Track 2 eligible | **No** (gated, NC) | Yes | Yes | n/a |
| License / gating | CC BY-NC 4.0, auto-gated | Apache-2.0, ungated | Apache-2.0, ungated | Apache-2.0, ungated |
| Medical vision evidence | Strong OOD MCQ (MedXpertQA-MM 49.2, own harness) | MedXpertQA-MM 28.7 (card) | 58.1 (card) | none reported for medical |
| Perception / modality-recognition evidence | None | None | None | strong general zero-shot |
| Pass-2 labeler (Jev) | Poor (MoE + harmony + custom code, no evidence) | **Best cheap torso** | Good MoE torso | n/a |
| CLM image tower | No (no text tower, NC) | No | No | **Yes (default)** |
| Tower tuned on biomedical data? | Yes (joint tuning, 19M + 0.9M pairs), not ablated | No (frozen in pretraining) | No | No |
| Inference support | transformers custom code only | transformers + vLLM | transformers + vLLM | transformers |
| Action | Cite as related work, optional baseline, do not build on | Primary Track-3 / labeler | Primary Track-2 | Primary CLM tower |

## Sources

- Paper: [MedGPT-oss, arXiv 2603.00842](https://arxiv.org/abs/2603.00842) (v2 PDF read in full, Tables 1-6, Fig. 1-2 text).
- HF model: [UFNLP/MedGPT-oss](https://huggingface.co/UFNLP/MedGPT-oss) (card, API: gated auto, cc-by-nc-4.0, 21,232,236,224 BF16 params, `MedGptOssModel`). Org page: [UFNLP](https://huggingface.co/UFNLP).
- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (HF API: 20,914,757,184 params; card: 3.6B active, Apache-2.0, harmony).
- Dataset provenance: [II-Medical-Reasoning-SFT](https://huggingface.co/datasets/Intelligent-Internet/II-Medical-Reasoning-SFT), [ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed), [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision), [MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M).
- Local: `data_discovery/research/mmbu_exclusion_list.md`, `docs/models/backbones.md`, `data_discovery/research/models.md` (already lists MedGPT-oss as "T2 only", which is now corrected: gated + NC means not even Track 2).
