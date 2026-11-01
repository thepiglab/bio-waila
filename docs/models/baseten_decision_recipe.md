# Baseten mammoth-1 / elephant-1: what we take for the Pass-2 Jev teacher

Date: 2026-10-21. Source: [Baseten Labs, "mammoth-1 and elephant-1: frontier decision models"](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models). Weights: [baselabs/mammoth-1](https://huggingface.co/baselabs/mammoth-1), [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) (MIT).

## 0. What was actually released

- **Released:** weights only, plus the recipe as prose and tables.
- **Not released:** the training data and the code. The article says the data-loop system and the Jev benchmark will be published "shortly".
- **Not usable as bases:** both models are full fine-tunes of Qwen3.5 (27B and 9B), so they are banned on HiPerGator (`no-chinese-models-hipergator`). They are also **text-only**; multimodal is listed as future work. So nothing here replaces our Gemma-4 teacher. What we take is the recipe.

## 1. The recipe in one table

| Item | mammoth-1 / elephant-1 | Evidence in the article |
|---|---|---|
| Readout | Softmax over the option-letter logits at the answer position. No text generation. | Same family as ours (`jev_gemma.py` letter readout). |
| Multi-question | Shared prefix (system + state) computed once and cached; one suffix per question. Full-attention layers use a **block-bidirectional** mask: prefix never sees suffixes, so the cache is shared. Linear (GDN) layers stay causal. | One ablation run on an early mix: causal 44.8 → block-bidirectional 48.6 on the Jev Decision Index. |
| Loss | CE + λ·Brier, λ = 1 | Not ablated in the article. |
| Abstention | An "I don't know" option. Its target is 1 when the correct answer is removed from the options, or when a reference model's probability of the correct answer is below τ ≈ 0.45 ("a little less than 0.5"). | Mix 5 added it: +5.5 points with only ~4% more rows. **Second-largest single gain.** |
| Data scale | Mix 1 → Mix 4: 254k → 4.19M rows. | 45.5 → 66.0. **Largest gain.** |
| Data repair | All 253 sources re-audited (Mix 7). | +2.2 on top of Mix 6; the article credits repair with 7.9 points from Mix 4 to Mix 7, a span that also contains the abstention gain. |
| Labelling | Five-judge panel with shuffled options, kept only if ≥ 4 of 5 agree; or deterministic labels. Then: clean ambiguous / multi-answer items, decontaminate, have an independent agent re-derive labels and look for answer giveaways. | Process description. |
| Source pilot | Every new source gets a 240-step training pilot before entering the mix. | Process description. |
| Loop | Evaluate → find the weakest areas → feed sources back into the next mix. "Largely agentic." | Process description. |
| Augmentation | Randomised serialisation of the state; distractor options added to entity-list questions with p = 0.5. | Not ablated. |
| Optimisation | One epoch, full FT, vision encoder frozen, AdamW β = (0.9, 0.95), no weight decay, peak LR 3.5e-6 (9B) / 2.7e-6 (27B), 3% warm-up then cosine, clip 1.0, bf16. | 9B: 5,546 steps on 128 H100s. |

**Order of value:** scale > data repair ≈ abstention > attention mask. The two architecture-free items (scale, repair) gave most of the points.

## 2. Mapping to Pass-2

| Their concept | Ours |
|---|---|
| State (JSON) | The image (plus F3 context: earlier fields' answers). |
| Questions | The six Pass-2 fields. Each is a `choice` question; body part and domain are multi-label. |
| Shared prefix | The ~280 image tokens. Today we re-encode the image once per (image, field) prompt. |
| "I don't know" | **"none of these options"**. It has three bio-specific uses: (a) body part has 99 classes and is shown in chunks of 20, so most chunks really contain no right answer; (b) metadata labels are often incomplete or wrong, so a teacher that can abstain does not have to fit noise; (c) the abstention mass marks images whose pseudo-label should stay masked (§2.4 of the training plan). |
| Answer giveaways | Burned-in text that names the modality or stain ("H&E 40x", "T2"), scale bars and borders unique to one dataset, DICOM overlays, JPEG-grid or resampling artefacts, filename leaks. Our JPEG and resampling augmentation already targets the last two. |
| Five-judge panel | Five teachers we already plan: SigLIP2-So400m FT, MedSigLIP-448 FT, Jev-E4B, d1-3B and Gemma-4-26B-A4B. |
| Decontamination | Already done: dHash near-duplicate filter against mmbu-public, which stays eval-only. |
| Source pilot | Per *group* of datasets (a backfill batch, the nc layer, a pseudo-label layer), not per dataset: 420 pilots is too many. A frozen-feature probe (minutes) first, then a 240-step Jev pilot on a fixed val subset. |

**Not adopted:**
- The `score` (ordinal) question type: no Pass-2 field is ordinal.
- Synthetic data from DeepSeek V4-Pro or GLM-5.3: Chinese-origin, and our labels come from dataset metadata, not from generation.
- Abstention in the **student**: Pass 2 always needs an answer. Abstention lives in the teachers only.

## 3. What changed in the code (2026-10-21)

All in `models/jev_gemma.py` and `train/train_jev.py`, switched on in `configs/jev_gemma4_e4b.yaml`, with off-switches in `configs/ablations.yaml` (`jev:` block):

| Config key | Default | What it does |
|---|---|---|
| `train.brier` | 1.0 | Adds λ·Σ(p − y)² to the CE. For candidate rows the term is (1 − p(candidate set))². |
| `train.abstain` | true | Adds "none of these options" as the last letter, in training and in evaluation. |
| `train.absent_p` | 0.15 | Share of training prompts whose correct options are removed, so the target is the abstain letter. |
| `train.paraphrase` | true | Picks a random question wording per training prompt (`PARAPHRASES`). Evaluation keeps the canonical one. |
| `train.label_smoothing` | 0.0 (was 0.05) | Teacher logits feed the ensemble, so no smoothing; Brier calibrates. |
| `model.full_ft` + `train.mixed_precision`, `train.betas` | off | The elephant-1 arm: every language-model weight trains, the tower stays frozen, LR 3e-6, β2 0.95. It needs fp32 master weights (`model.dtype: float32`, `mixed_precision: bf16`): a bf16 weight cannot take a 3e-6 step. |

**Tournament evaluation with abstention:** class scores keep the mass the abstain letter took, without renormalising. So a chunk of 20 body parts that holds no right answer scores lower across the board, and non-finalists from different chunks become comparable.

**Not yet implemented:**
- **τ-rule abstention targets** (abstain when a reference model gives the correct answer < τ). This needs the CLM teacher's probabilities per row, so it comes after stage 1b. Implement it as `train.abstain_ref: <path to teacher probs>`, `train.abstain_tau: 0.45`.
- **Distractors with p = 0.5:** `choose_options(prefer=…, prefer_p=…)` exists, but no CLM top-k is wired in yet. Wire it once a CLM teacher exists.

## 4. F4: shared image prefix and block mask (implemented 2026-10-21)

Code: `models/jev_packed.py` (packing, masks, readout), `GemmaBackend.packed_letter_logits`, `packed_fields_logprobs` (evaluation), `train.formulation: F4` in `train_jev.py`. Ablations: `jev.f4_packed`, `jev.f4_bidir`.

**Layout.**
- One sequence per image: `[prefix] | q_1 | q_2 | … | q_K`, where q_1 … q_K are the image's labelled fields (all fields at evaluation).
- `prefix` is the longest common token prefix of the image's unpacked F2 prompts: the chat header, the image soft tokens and the shared `Field: ` text. `pack` refuses a prefix that ends before the image.
- Every block restarts its position ids at len(prefix), so each block has exactly the positions of its unpacked sequence.
- The answer letter is read at the last token of each block.

**Masks.** Built per layer type and passed to `Gemma4Model` as a `{"full_attention", "sliding_attention"}` dict, which it hands straight to the decoder layers. No transformers patch is needed. Dense 4D bool masks, with sdpa or eager attention:
- **Structure:** a token sees its own block and the prefix. The prefix never sees a block, and blocks never see each other.
- **Causal:** a key's position is at most the query's position, using the per-block positions.
- **Sliding:** query position − key position < the window (512 on E4B, 1024 on 26B-A4B), also in per-block positions. So a late block keeps the image in view exactly as its unpacked sequence would. This was the risk named in `jev_methods.md` §1.3.
- **Vision:** where `use_bidirectional_attention == "vision"` (26B-A4B; E4B has none), image tokens see each other in sliding layers, as Gemma 4's own `create_masks_for_vision_model` does.
- **`train.bidirectional`:** the trained mammoth-1 variant. Global layers attend both ways inside the prefix and inside each block; sliding layers stay causal.

**Equivalence (bidirectional off).**

| Check | Max difference |
|---|---|
| Tiny random Gemma 4, causal (E4B-like), sdpa | 3e-7 |
| Tiny random Gemma 4, vision-bidirectional (26B-like), sdpa | 2e-7 |
| Tiny random Gemma 4, causal, eager | 3e-7 |
| Real processor, chat template and left padding (`scripts/check_f4_equivalence.py --tiny`), 18 questions | 1.4e-6 in letter log-prob; 18/18 argmax |

- The tiny-model tests are in `tests/test_jev_packed.py`. Their sliding window is 8 tokens against ~65 image tokens, so the sliding path is exercised.
- Controls: flat positions (no restart) move the logits by 0.34, and the bidirectional mask also changes them. So the test can fail.
- **Still to run on HiPerGator with the real weights:** `uv run python scripts/check_f4_equivalence.py --model google/gemma-4-E4B-it --dtype float32 --images <dir>`. It should print OK.

**Token saving.** The real processor on six questions per image gives 2.2× fewer tokens. These were small synthetic images; at a 280-token budget the saving is larger.

**Bidirectionality is weaker on E4B than on mammoth-1.** E4B's last 18 of 42 layers reuse the K/V of earlier layers (`num_kv_shared_layers`). The bidirectional mask therefore acts only through the 4 global layers among the first 24. In the shared layers the readout token sees K/V computed under the earlier layers' masks. On the tiny model, with all global layers shared, bidirectional attention left the readout unchanged; it needed a model without sharing to show an effect. So the +3.8 from mammoth-1 should be expected to shrink. Treat `f4_bidir` as an experiment, and `f4_packed` as the compute win.

**Also fixed on the way:** the letter readout's token table stopped at the largest letter token id (236,954), not at the vocabulary size (262,144). As a result `letter_logits` would have failed on the first real Gemma-4 forward. It is now padded to the logit width (regression test in `tests/test_jev_packed.py`).

**Limits.**
- F4 has no F3 context: blocks cannot see each other's answers.
- Flash-attention kernels take no 4D masks, so F4 uses sdpa. Its masked path is the memory-efficient kernel, not flash.
- The masks are dense [B, 1, T, T]: about 8 × 1.5k² bools per layer type for a batch of 8, which is negligible.

## 5. Scale: our biggest lever

`examples_per_epoch: 200000` is about 6% of one epoch over our labelled (image, field) pairs: about 873k images × about 4 labelled fields ≈ 3.5M (estimate). elephant-1 went from 45.5 to 66.0 by growing 254k → 4.19M rows.

One full epoch of E4B LoRA costs about:
- **~11 B200-h** with unpacked F2: ~1.5B tokens at ~4N FLOPs/token plus checkpoint recompute, at 0.8 PFLOP/s;
- **~6 B200-h** with F4;
- **~17 B200-h** for full FT unpacked.

These are FLOP estimates, not measurements. The prototype keeps 200k for the formulation comparisons; the **winner is trained on one full epoch**.
