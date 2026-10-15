# Liquid AI models for the biomedical-image metadata classifier

Researched 2026-10-07. Sources are the Liquid blog and HF model cards, plus the raw data files of the HF Jev Decision Index space (`data/index.json`, snapshot 2026-10-07T18:25Z). Anything I could not confirm is marked [UNVERIFIED].

## 1. What "D1" is

**"d1" is an exact name, not a mangling.** It is Liquid AI's first *decision model*, in the Jev / System One sense.

- It reads a state (text, JSON, images or a mix) plus typed questions, in one forward pass, and emits **zero output tokens**. Answers are read from the model's distribution over the options.
- Question types (from the d1-3B card):
  - `noul`: yes/no, returns P(yes).
  - `choice`: a `{name: description}` dict, returns the argmax plus per-label `probabilities` and `confidence`.
  - `score`: an ordered list of 2-10 levels.
- API: `model.system_one(state, questions, images=[...])` and `system_one_batch([...])`.
- Timeline (dates differ between sources):
  - d1 text API: announced around Sep 22-29, 2026. Sources conflict: llmreference says Sep 22, datanorth says Sep 29, OpenRouter says `d1-20260930`.
  - d1 vision added: Oct 5, 2026 (llmreference).
  - **Open weights `LiquidAI/d1-3B` and `LiquidAI/d1-omni-600M`: appeared on HF Oct 7, 2026.** The HF org page listed them as "updated ~7 hours ago". The earlier blog said "API-only, open weights soon".
- Hosted API: $0.04 per 1M input tokens, nothing for output. Images cost 1.5 tokens per 32x32 patch (1024x1024 = 1,536 tokens). Up to 8 images per request. Reachable via the Liquid console, d1.liquid.ai, Vercel AI Gateway and OpenRouter (text-only at the time of the blog).
- The hosted model's size is not disclosed [UNVERIFIED]. The claim "first model to outperform Jev" is for the hosted d1. The open d1-3B does NOT beat Jev (see section 6).

Sources:
- https://www.liquid.ai/blog/d1-decision-model
- https://www.liquid.ai/blog/d1-open
- https://huggingface.co/LiquidAI/d1-3B
- https://huggingface.co/LiquidAI/d1-omni-600M
- https://www.llmreference.com/model/d1/liquid
- https://huggingface.co/LiquidAI (org listing)

### Open d1 variants

| | **d1-3B** | **d1-omni-600M** |
|---|---|---|
| Total params | 3.12B | 587M |
| Base | LFM2.5-VL-3B (decoder-only) <- LFM2.5-2.6B-Base | LFM2.5-Encoder-350M (bidirectional) |
| Vision tower | SigLIP2 NaFlex shape-optimized, 400M | SigLIP2, 94M |
| Other | none | 112M audio encoder (17-layer FastConformer); 381M shared trunk |
| Inputs | text, JSON, images | text + image *or* audio |
| Context | 32,768 | 16,384; multimodal requests capped at 896 tokens (benchlm) |
| License | LFM Open License v1.0 (`lfm1.0`) | same |
| Support | Transformers >=5.14 with `trust_remote_code=True`, vLLM, SGLang, llama.cpp (day-one), GGUF and w8a8 variants on HF | similar, GGUF available |
| Jev Decision Index | 48.57 (card, v0.2.1); 48.99 on the live board | 15.95 (card); 17.9 on the live board |

**Speed (d1-3B card):**
- RTX 4090: 8 ms for a single question, 17 ms with a 384px image, 475 decisions/s batched.
- AMD MI325X: 1,106/s batched.
- Apple M5 Pro: 62 ms for a 384px image, 78/s.
- Jetson Orin Nano: 202 ms for an image.

**d1-3B vision (11 public benchmarks), mean 74.1 vs 73.9 for the LFM2.5-VL-3B base:**

| Benchmark | Score |
|---|---|
| MMBench | 84.9 |
| AI2D | 79.9 |
| MMStar | 59.9 |
| MME | 82.1 |
| POPE | 88.5 |
| BLINK | 59.2 |
| CV-Bench | 82.1 |
| VL-RewardBench | 65.0 (base 50.9) |
| VisualWebBench | 71.4 |

**d1-3B text mean:** 77.1 on 8 benchmarks, including PubMedQA 66.0.

There is **no medical-image result** on the card.

## 2. Recent Liquid AI release landscape

| Model | Date | Notes |
|---|---|---|
| LFM2 / LFM2-VL (450M, 1.6B, 3B) | 2025 | Tech report arXiv 2511.23404. |
| Liquid Nanos (LFM2-350M/1.2B-Extract, RAG, etc.) | Sep 2025 | Task-specific small models. |
| LFM2.5 family (1.2B Base/Instruct/JP/VL/Audio, 350M) | early 2026 | Pretraining extended from 10T to 28T tokens, then RL. |
| LFM2.5-VL-450M and LFM2.5-VL-1.6B | 2026 | Exact dates [UNVERIFIED]. |
| LFM2.5-230M | Jun 25, 2026 | |
| LFM2.5-2.6B (text, agentic) | Aug 4, 2026 | |
| **LFM2.5-VL-3B** | Aug 12, 2026 | 3.1B total. |
| LFM2.5-VL-3B-DSpark | Sep 24, 2026 | 279.5M speculative-decoding drafter. Up to 2.66x faster on H100 (SGLang), 3.13x on M5 Max (MLX-VLM). |
| **d1 (API)** | Sep 22-Oct 5, 2026 | |
| **d1-3B and d1-omni-600M (open weights)** | Oct 7, 2026 | |

**LFM3:** I found no release. The HF org listing shows none, and a Manifold market asks "when will LiquidAI LFM3 be released".

## 3. Hard-constraint check (no Chinese-origin models)

- **LM backbone:** Liquid's own LFM2 hybrid. The tech report says it was developed "without relying on other vendors' pretrained base models". LFM2.5-2.6B-Base was trained on 34T tokens.
- **Vision encoder:** SigLIP2 NaFlex, a Google model (So400m 400M for the 3B and 1.6B, Base 86M for the 450M, 94M in d1-omni). It is not Qwen or InternViT derived. Qwen and InternVL appear in Liquid's tables only as comparison baselines.
- **Residual risks, not verifiable from public docs:**
  - Distillation or synthetic data from third-party teachers during post-training: [UNVERIFIED].
  - Liquid's own vision training data: [UNVERIFIED].

**Verdict: passes the "no Chinese-origin base" rule on everything published.**

Caution on the Jev board: the third-party entries `LFM2.5-2.6B-RLCD` (6.78) and `CLM-v0.1-8B` (7.41) are not Liquid releases. CLM-v0.1-8B is built on **Qwen3-8B-Base**, so it is banned if you were considering "CLM-style" checkpoints off the shelf.

## 4. License (LFM Open License v1.0)

From `LiquidAI/d1-3B/LICENSE`. This is the same `lfm1.0` license used on the LFM2.5-VL cards. Quotes are from the fetched text:

- Commercial Use is defined as "any use of the Work for direct or indirect commercial advantage or monetary compensation."
- Threshold: "annual revenue of 10 million United States dollars ($10,000,000) or more."
- "The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity not exceeding the Threshold."
- Non-profit carve-out: "The Threshold shall not apply to a Qualified Non-Profit Organization's use of the Work or a Derivative Work for Non-Commercial or Research Purposes."
- Derivative works must retain copyright, patent, trademark and attribution notices, and carry attribution via NOTICE, docs or a display.
- Termination is automatic on non-compliance, and you must delete all copies and derivatives.
- Not gated (no click-through seen; the repos were fetchable anonymously).

**Practical read:** a university or non-profit doing research is fine. Entities with $10M or more in annual revenue cannot use it commercially without a separate license.

**To check before committing:** whether the MMBU challenge requires submitted weights to be under a permissive or OSI license. That rule is outside this research [UNVERIFIED]. Gemma-4 is Apache-2.0 (E2B card), which is cleaner.

## 5. Candidate models for our task

### LFM2.5-VL-3B (best Liquid candidate for Track 3)

- **Params:** 3.1B total, in a safe position under the 4B cap with about 0.9B headroom.
  - LM 2.6B: 30 layers (22 double-gated short-conv + 8 GQA), hidden 2048, 32 heads / 8 KV heads, vocab 128K.
  - Vision: SigLIP2 So400m NaFlex, 27 layers, hidden 1152, patch 16, about 400M.
- **Image handling:**
  - Native up to 512x512, otherwise non-overlapping 512x512 tiles (max 10) plus a resized thumbnail.
  - Pixel-unshuffle x2 connector. `min_image_tokens=64`, `max_image_tokens=256`.
  - The tech report quotes 128-256 tokens per image, and about 2.8K tokens for heavily tiled inputs.
  - **Concern for us:** fine histology and radiology detail may be lost at 64-256 tokens. Set `max_image_tokens` and tiling deliberately and test.
- **Context:** 32,768.
- **Support:** native `Lfm2VlForConditionalGeneration` in transformers (config shows no `auto_map`, so no remote code), vLLM, SGLang, llama.cpp, MLX, ONNX.
- **Fine-tuning:**
  - Official Colab notebooks for SFT with TRL (LoRA) and with Unsloth (LoRA).
  - **LEAP Finetune**, Liquid's open-source stack: SFT, LoRA, DPO, GRPO, text, vision and MoE, GGUF export.
  - Docs say "text, vision" support but do not name LFM2.5-VL-3B specifically [UNVERIFIED].
- **Speed:** 228 tok/s on M5 Max using under 3.3GB; 20 tok/s on Galaxy S26 Ultra; about 11K tok/s throughput on H100 at high concurrency. DSpark drafter gives up to 2.66x more.
- **Languages:** 16. **License:** `lfm1.0`.
- **Benchmarks:** these are Liquid's self-reported numbers from the HF card. The Gemma columns were measured by Liquid, so treat them as vendor-chosen.

  | Benchmark | LFM2.5-VL-3B | Gemma4 E2B | Gemma4 E4B |
  |---|---|---|---|
  | MMStar | 63.3 | 57.9 | 61.9 |
  | MME | 73.1 | 54.9 | 68.1 |
  | RealWorldQA | 73.1 | 56.2 | 61.8 |
  | MathVista | 68.5 | 62.1 | 52.9 |
  | ChartQA | 81.3 | 43.5 | 41.9 |
  | OCRBench v2 (EN) | 47.5 | 44.7 | 48.7 |
  | POPE | 88.7 | 84.0 | 86.9 |
  | MMMU Pro | 30.5 | 34.5 | 39.1 |
  | MM-IF Eval | 60.6 | 64.5 | 66.7 |
  | BLINK | 61.5 | 51.8 | 56.4 |
  | MuirBench | 58.3 | 40.7 | 48.9 |

  - Other scores: ScreenSpot-v2 80.7, RefCOCO 87.9, DocVQA 91.1 (blog).
  - Secondary sources report MMMU(val) 48.4, OCRBench v1 84.2 and MMBench-dev-en 78.18 [UNVERIFIED; not on the card I fetched].
  - **No medical benchmarks for either family.**

### LFM2.5-VL-1.6B and LFM2.5-VL-450M

- **LFM2.5-VL-1.6B:** 1.6B total (1.2B LM + 400M SigLIP2 So400m). Context 32K, vocab 65K, 8 languages. MMStar 50.67, OCRBench v2 41.44, RealWorldQA 64.84, multilingual MMBench 65.9.
- **LFM2.5-VL-450M:** 350M LM + 86M SigLIP2 Base. MMBench-dev-en 60.91, OCRBench 684. The card warns it is "not well-suited for knowledge-intensive tasks or fine-grained OCR".
- Both are candidates for a cheap teacher or ablation, not for the final submission, where capacity matters.

### Gemma-4 comparison

| | Gemma-4-E2B | Gemma-4-E4B | LFM2.5-VL-3B |
|---|---|---|---|
| Effective params | 2.3B | 4.5B | n/a (3.1B all-in) |
| **Total params incl. PLE embeddings** | **5.1B** | **8B** | 3.1B |
| Vision encoder | about 150M, aspect-preserving, 70-1,120 tokens/image | same | 400M, 64-256 tokens/image plus tiles |
| Audio encoder | about 300M | about 300M | none |
| Context | 128K | 128K | 32K |
| License | Apache-2.0 | Apache-2.0 | LFM Open v1.0 ($10M threshold) |
| Gemma-reported MMMU-Pro (vision) | 44.2 | 52.6 | 30.5 (Liquid-measured) |

**Flag for the Track-3 cap.**
- Gemma-4-E2B is "2.3B effective" but **5.1B total** with embeddings, and E4B is 8B total.
- If the Track-3 cap of 4B total parameters counts all stored weights, E2B may already exceed it. This is the biggest finding of this research for our current pick.
- Check how the organizers count parameters: stored versus effective, and whether the PLE tables count [UNVERIFIED].
- If they count stored weights, LFM2.5-VL-3B (3.1B) and LFM2.5-VL-1.6B are among the few strong VLMs that clearly fit.

## 6. Jev Decision Index: Liquid entries

Data is from `huggingface.co/spaces/multimodalart/jev-decision-index/data/index.json`, generated 2026-10-07T18:25Z, 114 models. The headline is `public_skill`. Hardware: 1 x RTX PRO 6000. The panel is mostly text.

| Model | Public skill | Rank of 114 | Notes |
|---|---|---|---|
| Jev 1.13.0 (reference) | 57.96 | | |
| Top: Torchcast Decision 27B | 65.1 | 1 | |
| Perplexity Decider v1.1 27B | 62.25 | 2 | |
| **LiquidAI d1-3B** | **48.99** | **31** | Full fine-tune, 3.12B. By area: knowledge .243, language .565, retrieval .528, tools .745, arts .389. Calibration: acc .702, ECE .047. Median latency 16 ms. |
| **LiquidAI d1-omni-600M** | **17.9** | 92 | 587M. Median 10.9 ms. |
| Third-party LFM2.5-2.6B-RLCD | 6.78 | 107 | monotykamary, not Liquid. |
| Third-party LFM2.5-350M-RLCD | 1.43 | 114 | notnotsamuel, not Liquid. |

Gemma-based comparison points on the same board:

| Entry | Public skill |
|---|---|
| Winnow-12B | 50.66 |
| AJev (Gemma 4 12B) | 51.66 |
| Winnow-E4B (E4B LoRA, 8.0B served) | 40.86 |
| Gevva E4B | 32.44 |
| Gevva E2B | 28.53 |

So d1-3B is the best sub-4B-served-size decision model I saw. It beats every Gemma-E2B/E4B-based decision fine-tune on the board by 8 points or more. It is still about 9 points under Jev itself, and the card's "best under 10B" claim holds only for open models on this board.

**The separate `vision.json` board (11 vision tasks) does not list d1-3B.** Its entrants are 20 community models, led by JEV-27B-VL at 69.8. So there is no independent vision-decision score for d1 yet.

## 7. Fit for our task

### Role (a): teacher labeler. d1-3B is a strong fit.

- Its interface matches our task: an image, a per-field `choice` question, and the option list as `criteria` descriptions. You get calibrated probabilities over a closed vocabulary.
- It fits our planned thresholding or abstention step, and a large corpus is cheap to label.
- Throughput: 475 decisions/s batched on a 4090, so millions of labels are tractable. Zero generated tokens means no JSON parsing failures.
- Gaps:
  - No medical-image validation. It is trained and benchmarked on general vision plus PubMedQA text.
  - Image tokens are capped at 256 per tile, which is lossy for histology.
  - Open fine-tuning of d1 itself is undocumented. `trust_remote_code` is required, and the training code is not released [UNVERIFIED]. Expect to calibrate with a held-out labeled slice rather than fine-tune it.
- Recommendation: run d1-3B zero-shot on a few hundred labeled images per field and compare against our current teacher and Gemma-12B-based options. Use it as an additional vote or prior. Check the license against how teacher labels are redistributed.

### Role (b): submitted end-to-end VLM. d1 itself cannot do it; LFM2.5-VL-3B is a credible alternative.

- d1 is **not a chat or generation model** ("does not write text"). It cannot produce Pass-1 free text or Pass-2 JSON.
- LFM2.5-VL-3B (the same base) can generate text. Via TRL or LEAP LoRA/full SFT it can emit both Pass-1 text and Pass-2 JSON.
- Why consider it over Gemma-4-E2B:
  - 3.1B fits the 4B cap with headroom, whereas E2B is 5.1B stored.
  - A 400M vision tower is larger than Gemma's 150M, and the hybrid conv LM is fast.
  - Liquid's own tables show it ahead of E2B/E4B on perception tasks.
- Why not switch on benchmarks alone:
  - It is behind on MMMU-Pro (30.5 vs 34.5 for E2B and 39.1 for E4B).
  - 32K context vs 128K, and 16 languages.
  - The license has the $10M threshold (fine for academia, a risk if the challenge requires open licensing).
  - No medical results for either family.
  - Liquid's Gemma numbers are vendor-measured. Gemma's own published MMMU-Pro (44.2 and 52.6) is far above Liquid's measurement (34.5 and 39.1), so protocol differences matter.
- **Suggested next step:** a bake-off on our own held-out slice with identical LoRA SFT (same data and steps) for LFM2.5-VL-3B, LFM2.5-VL-1.6B and Gemma-4-E2B. Measure per-field accuracy and parameter count against the cap. Track tokens per image and test higher `max_image_tokens`.

## 8. Open items and unverified

- Exact size and architecture of the hosted d1 API model, and which benchmark "first to outperform Jev" refers to: [UNVERIFIED].
- d1 release dates differ across sources (Sep 22 / Sep 29 / Sep 30 / Oct 1 / Oct 5). The open-weights date of Oct 7 comes from the HF org listing.
- Whether the d1 training recipe or code is released: [UNVERIFIED].
- Whether LEAP Finetune lists LFM2.5-VL-3B as supported: [UNVERIFIED]. TRL and Unsloth notebooks exist for LFM2.5-VL-3B per the card.
- MMMU (val), MMBench and OCRBench v1 numbers for LFM2.5-VL-3B come from secondary summaries only [UNVERIFIED].
- A secondary page (pasqualepillitteri / search snippet) cited a "58.9" self-reported d1 skill score on Decision Index 0.2.1. It conflicts with the card's 48.57, so I used the card and the live board.
- Liquid medical or biomedical-specific variants: none found.
- Vision-training data provenance and any third-party-teacher distillation: not published [UNVERIFIED].
