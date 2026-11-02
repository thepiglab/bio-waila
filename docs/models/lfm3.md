# LFM3 (Liquid AI, 2026-10-27): what it changes for us

Sources:
- Blog: [Introducing LFM3](https://www.liquid.ai/blog/lfm3), [Open d1](https://www.liquid.ai/blog/open-d1).
- Model cards and `config.json`: [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), [d1-2B](https://huggingface.co/LiquidAI/d1-2B), [d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B).
- The d1-2B remote code (`attention.py`, `pointer.py`, `slots.py`, `prompt.py`, `runner.py`), read on 2026-10-27.
- Parameter counts come from the HF safetensors metadata.

The blog page was truncated in fetching, so every number below is from the model cards and configs.

## 0. Summary

| Role | Before | Now |
|---|---|---|
| **Track-3 student** | Gemma-4-E2B vs LFM2.5-VL-3B | **LFM3-VL-2B first**, Gemma-4-E2B as the challenger. LFM2.5-VL-3B drops out: LFM3-VL-2B beats it on every category in Liquid's table, with 1.2B fewer parameters. |
| **Free decision voter** | d1-3B | **d1-2B.** It is better on every decision suite on its card and much better on images (ImajevBench 75.5 vs 64.0). |
| **Jev teacher** | Gemma-4-E4B LoRA (F2/F3/F4) | Unchanged, plus a **d1-2B fine-tune arm**. d1-2B already has the F4 layout trained in, multi-label `choice` and more than 26 options per question. It wins if its held-out macro accuracy is within noise of E4B: it is about 4x cheaper, and it shares the student's architecture. |
| **Track-2 model** | Gemma-4-26B-A4B family | Unchanged. LFM3 has nothing larger than 2B. |

## 1. The models

| | LFM3-VL-2B | d1-2B | d1-Omni-2B |
|---|---|---|---|
| Params (safetensors) | **1,918,181,888**: LM 1.73B + vision 164M + projector 25M | 1,918,181,888 (same backbone) | 2,166,928,384 |
| Built on | LFM3-1.7B-Base (22T tokens) + new in-house vision encoder | LFM3-VL-2B | LFM3-Omni-2B |
| LM | 24 layers: 13 gated short-conv, 5 sliding-window (256), 6 global; hidden 2048, 16 heads / 4 KV, vocab 128k, 128k context | same | same |
| Vision encoder | **In-house**, 14 layers: 8 gated depthwise 2D conv + 6 bidirectional attention, 16×16 patches, 2D RoPE, hidden 1024. **Not SigLIP.** | same | same, plus audio |
| Image tokens | Native resolution; each 2×2 patch block is one token, so a 512² image is 256 tokens. The processor allows 256²–4096² px. | Capped at 1,638,400 px (about 1,600 tokens; `d1_vision_max_pixels`) | same |
| Output | Text (thinking on by default; `enable_thinking=False` turns it off) | Typed answers, no generated tokens | same |
| License | LFM Open License v1.0, ungated | same | same |
| Code | Remote code (`lfm3_vl` is not in transformers 5.19) | Remote code, `transformers>=5.19` | same |

**Origin check (no Chinese-origin bases):** passes on everything published.
- The LM is Liquid's own pretraining.
- The vision encoder is Liquid's own and is no longer SigLIP2.
- Post-training distils from Liquid's own RL teachers (MOPD).
- Qwen, InternVL and MiniCPM appear only as comparison baselines.
- Residual unknowns are the same as for LFM2.5: third-party synthetic data and Liquid's vision training data.

## 2. LFM3-VL-2B vs Gemma 4 (Liquid's table, vLLM 0.27.1, 62 benchmarks)

| Category | LFM3-VL-2B think | LFM3-VL-2B no think | LFM2.5-VL-3B | Gemma-4-E2B (think) | Gemma-4-12B (think) |
|---|---|---|---|---|---|
| General | 66.0 | 56.4 | 57.6 | 56.9 | 74.0 |
| Perception | 75.6 | 68.9 | 67.9 | 59.1 | 73.5 |
| Doc / OCR / chart | 83.9 | 71.6 | 71.2 | 72.3 | 85.7 |
| **Visual grounding** | **77.3** | 72.6 | 68.3 | 41.2 | 74.1 |
| Average (62) | 71.4 | 62.1 | 57.6 | 53.6 | 71.7 |

Selected rows:

| Benchmark | LFM3-VL-2B think | no think | Gemma-4-E2B |
|---|---|---|---|
| RefCOCO-avg | 90.5 | 87.4 | 68.0 |
| HR-Bench 4K | 82.3 | 71.0 | 65.5 |
| V* | 86.4 | 80.6 | 44.5 |
| MMMU val | 55.7 | 48.6 | 57.3 |

**Reading it for MMBU:**
- **Detection.** MMBU scores detection as pixel boxes at IoU ≥ 0.5. Gemma-4-E2B is weak at grounding (RefCOCO 68.0, Point-Bench 14.3). LFM3-VL-2B is strong even without thinking (RefCOCO 87.4). This alone can decide the student.
- **Resolution.** Native resolution plus strong high-resolution results (HR-Bench, V*) suit pathology, microscopy and fundus images, where we already planned a 1,120-token run for Gemma.
- **Knowledge.** MMMU is the one place E2B is level or ahead. Medical knowledge questions in MMBU may favour Gemma. Liquid publishes **no medical benchmark** for LFM3, so this must be measured (§4).
- **Thinking.** Thinking is a big lift (62.1 → 71.4 average). The rules allow one end-to-end pass, no iterative prompting. Whether a thinking trace counts as one pass is a question for the organizers (added to `training_plan.md` §5). Until they rule, plan on no-think. It still beats E2B-with-thinking on average (62.1 vs 53.6).
- **Not in the table:** Gemma-4-E4B. It would not be Track-3-eligible anyway.

**Track 3 arithmetic:**
- Multiplier (4B/P)^0.1: **1.076** for LFM3-VL-2B (1.92B), versus 1.026 for LFM2.5-VL-3B (3.1B).
- Gemma-4-E2B gets 1.05 if 2.4B effective parameters count, and is ineligible if its 5.1B stored parameters count.
- LFM3-VL-2B also leaves 2B of headroom, for example for a larger vision tower if a medical tower ever needs it.

## 3. d1-2B for the teacher stack

| | d1-2B | d1-3B (current voter) | d1-Omni-2B |
|---|---|---|---|
| Decision Index public suite v0.3 | **51.1** | 49.0 | 49.0 |
| JevBench v1.4.1 | **77.9** | 76.2 | 75.3 |
| ImajevBench v2.0-lite (images) | **75.5** | 64.0 | 72.3 |
| Image average (18 benchmarks) | **70.2** | 65.4 | 67.7 |
| H100, one 384 px picture | 12.3 ms | — | 12.1 ms |

**What its code shows, and why it matters for us:**
- **F4 is native.** `d1_attention: "set"` with `d1_set_size: 16` packs up to 16 questions after one state: `[state][q1]…[qN]`, each ending at its own answer slot.
  - In global layers, the state and each question are read both ways; questions never attend to each other.
  - Conv and sliding layers stay causal, so a question does see earlier ones through them. Training packs questions in random order so that serving matches training.
  - This is the mammoth-1 mask, which our `jev_packed.py` re-implements for Gemma. Here it is already trained.
- **More than 26 options.** Two-letter codes (`d1_codes: 2`, AA, AB, …) make body part (99 classes) a single question, with no tournament.
- **Multi-label.** `choice` with `multiple: true` returns each option's own P(yes). That is the right semantics for body part and domain, which our letter readout fakes with a softmax.
- **Calibration.** One temperature (1.337) plus per-type tables in `calibration.json`.
- **Abstention.** d1-Omni-2B's card reports a "none of the options applies" abstention test (78.5 written), so abstention is trained in.
- **Boxes.** `box` / `boxes` questions return normalised boxes: a cheap pseudo-labeller for any MMBU detection-style data.
- **No training code.** The readout is the LM head over option-code tokens at the answer slots (`slots.py`). The set attention has a differentiable varlen-flash backward (`attention.py`). So fine-tuning with our own loss (CE + Brier + abstain, `jev_gemma.letter_loss`) over their prompt builder is feasible. It is work, not a config change.
- **Weakness.** "Knowledge-heavy questions favor larger models." The card's MMMU-Pro is 26.4 as a decision. Medical knowledge is the risk again.

**Shared backbone.** A d1-2B teacher fine-tuned on Pass-2 has exactly the student's architecture. That opens two cheap transfers:
1. Swap its adapted 164M vision encoder into the LFM3-VL-2B student. This replaces the EmbeddingGemma-2 bridge-tower plan, which only applied to E2B.
2. Weight-average teacher and student backbones (Liquid builds d1 itself by merging checkpoints).

Both need checking on held-out datasets.

## 4. What to run (added to the stage plan)

1. **Stage 0, zero-shot, now:** `waila-eval` / zoo on mmbu-public:
   - `dec_liquid_d1_2b` on its own (d1-3B is not rerun: d1-2B replaces it), read against the generative Gemma-4-E2B and LFM numbers;
   - `gen_lfm3_vl_2b` against `gen_lfm25_vl_3b` and stock Gemma-4-E2B.

   These are entries in `configs/zoo.yaml`, added 2026-10-27. One GPU, under an hour each (estimate).
2. **Stage 1a:** add the LFM3 vision encoder (164M, frozen features) to the probe bake-off. It is the student's own tower, so its gap to SigLIP2 / MedSigLIP on our slices is what a tower swap would have to close.
3. **Stage 2:** run a d1-2B fine-tune arm next to the E4B Jev teacher, on the same folds and the same loss. It costs about 4x less per token than E4B.
4. **Stage 3:** run the student head-to-head T0/T1 on **LFM3-VL-2B** and Gemma-4-E2B, no thinking. LFM2.5-VL-3B only as a fallback.

**Decision rule:**
- If zero-shot d1-2B is within ~3 points of zero-shot Gemma letter readout on mmbu-public fields, the d1-2B fine-tune arm gets priority over the E4B LoRA.
- If LFM3-VL-2B no-think is at or above E2B on mmbu-public T, it becomes the default student.
- Detection is weighted separately, because the IoU scoring is where the gap should show.

**Zero-shot results, 2026-11-01** (`experiments/zoo.md`, Results). These are noted here and not yet adopted:
- **d1-2B vs Gemma letter readout:** d1-2B scored C 0.514 against Gemma-4-E4B's 0.590, 7.6 points behind. By the first rule, the E4B LoRA keeps priority.
- **LFM3-VL-2B vs Gemma-4-E2B:** 0.387 against 0.328, so by the second rule LFM3-VL-2B becomes the default student. LFM2.5-VL-3B scored 0.410. LFM3 returned valid JSON for only 18% of images.
- **d1-3B, which this note replaced:** 0.641, the best of all zoo models, and 13 points above d1-2B.
