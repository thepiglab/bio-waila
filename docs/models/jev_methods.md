# Turning an image-capable VLM into a Jev-style decision model for the Pass-2 metadata task

Date: 2026-10-06. Scope: METHODS (architectures, objectives, multimodal handling, recipe, and a concrete design for our six-field task). The Decision Index model catalogue is another agent's job.

Conventions:
- **[V]** means I read it in a primary source (paper abstract, model card, repo doc, vendor blog) during this session.
- **[UNVERIFIED]** means a secondary or social-media claim, or an inference I could not confirm.
- **[MY INFERENCE]** marks my own reasoning, which is not in any source.
- Hard constraint: no Chinese-origin bases. Strands Decider, Clef, pplx-decider and CLM-8B are all Qwen-based, so I use them only as method references and never as weights.
- Search budget: 7 WebSearch calls were used (of the shared 25). Everything else came from direct fetches.

## 0. What the open Jev-style models actually do

This is the ground truth from primary sources (see the Sources list). Everything later builds on it.

| Model | Backbone and what is trained | Option encoding and readout | Multi-question | Losses / calibration | Images |
|---|---|---|---|---|---|
| **Strands Decider 2B** (AWS) [V: repo docs, blog] | Qwen3.5-2B-Base torso (a hybrid with 18 Gated-DeltaNet and 6 full-attention layers). The LM head is discarded and replaced by a ~1M-param pointer head (dim 256, fp32). Rank-16 LoRA on the torso projections. The checkpoint is ~88 MB. | Layout: `<state>…</state><question type=…><options>1. …\n2. …</options></question><answer>`. Pointer logit: `logit_k = <q(h_answer), k(h_last_token_of_option_k)>/sqrt(d)`, followed by a softmax over the K options. No per-option learnable parameters, so there is no positional or ID bias baked into weights. Noul = 2 options, Score = L ordered options (2-10), all in one masked softmax. | State KV cache computed once and forked per question (shared-prefix cache). Each question is a short suffix. One 640-token state with 6 questions takes 217 ms, versus 219 ms for one question. | Loss = CE with ordinal smoothing (10% mass to adjacent levels for Score) + 0.3·KL to the frozen torso's own readout (adapter off, option-number tokens after `<answer>`) + 1.0·KL to a frozen Qwen3.5-4B teacher (multi-step rows only). Option order is **re-shuffled every example, every epoch**; Score slots are reversed, not permuted. Post-hoc per-primitive temperature (v19: noul 0.91, choice 0.77, score 0.96). ECE ≈ 0.05-0.064, Brier 0.34-0.39. A slot head performed "significantly worse" than the pointer head. Recipe: ~100k rows from 21 public tasks plus ~13k multi-step rows, ~11 h on one RTX 3090. | **Yes, frozen vision tower.** Qwen3.5's vision tower, images capped at 448 px, ~196 tokens per square image, placed inside the state and shared across questions via the prefix cache. "The vision tower is frozen; nothing in it is adapted." Reported NaturalBench accuracy 78.4%, ECE 0.014, with no vision-specific training. Removing the image dropped it to chance. |
| **K2-Type-0.9B** (IFM) [V: HF card] | K2-Horizon-0.9B (origin unverified), **full fine-tune** on ~354k decision records. The unused LM head is still in the checkpoint. | `<state> … \| <q> question <opt> option </opt> … <decide> \| <q> …`, with 5 reserved tokens. The pointer head scores each option's `</opt>` hidden state against the question's `<decide>` hidden state. Softmax at fitted temperature 1.478. | **All questions in ONE sequence, block-causal mask.** Questions share the state but cannot see each other, so adding a question never changes another's answer. | Targets mixed with soft labels from a 7B teacher decision model. 50 PPO iterations on Snake. Temperature refit on held-out data. Brier 0.328, ECE 0.065. Weak on long documents (>8k tokens). | Text only. |
| **Clef / Clef-flash** (Cloudflare) [V: blog + HF card; they conflict on LoRA] | Qwen3.8-27B / Qwen3.5-9B. The blog says frozen backbone + rank-256 LoRA trained jointly with a **routing head**. The HF card never mentions LoRA. | A small transformer head reads the final hidden states. It runs a "two-stage attention routing": each valid choice extracts context from the prompt, and fields cross-attend to each other and back to the payload, then everything is scored. Output: one logit per option per question, softmax per question. | Joint schema head, all questions in one pass. Training data permutes field orders, prompts and schema structures (synthetic, size not disclosed). | Label-smoothed CE + Brier loss, then **RLCD**, which gives partial credit to adjacent ordinal choices. | **Yes**, via the inherited Qwen vision encoder (PIL images and video frames). Token count is not stated. Whether the tower is tuned is not disclosed. |
| **pplx-decider-v1-27b** (Perplexity) [V: HF card, secondary coverage] | Qwen3.8-27B fine-tune, called a "generative discriminator". | Head type, training objective and image handling are **not disclosed** in anything I could fetch. The API takes `images=[…]`. | n/a | "Calibrated probabilities" are claimed, with no method given. | Yes. |
| **CLM-v0.1-8B** (Stanford/NVIDIA) [V: HF card] | A frozen Qwen3-8B with 2 small projection heads (state, action), last-token pooling. | Dual encoder. Bidirectional InfoNCE. Action embeddings are cacheable. | n/a | Stages: 60M Q&A pairs, 30M synthetic hard negatives, 1M agent trajectories. Calibration is not discussed. | Multimodal CLM-35B is "forthcoming". I found no details. |
| **Cygnet** (blockbrain) [V: benchlm snippet only] | **Frozen Gemma-4-12B-it, no training.** A shim presents options as letters, reads the letter probabilities and applies one calibration temperature. | The LM head itself, with ≤26 options. | n/a | One temperature. JevBench 73.7 vs Jev 72.1, so a training-free generative-letter readout is a strong baseline. Details come from a search-result snippet [UNVERIFIED beyond that]. | n/a |
| **DiffusionGemma-as-Jev** (Prem) [V: Prem blog] | DiffusionGemma (Gemma-4 26B-A4B MoE, an encoder-decoder with a bidirectional 256-token "canvas"). **Not fine-tuned.** | "Structured read": an answer template with blanks is held in place and scores are read at those positions in one denoising step. | Up to 16 independent questions per request. | Probabilities are "normalized model scores, not calibrated estimates", and option order matters. JevBench 83.5% (193/231). | Yes, one image (base64) or a ≤30 s MP4 (up to 4 sampled frames). |
| **Jev** (TypeSafe) | Closed. A pointer head and RLCD (proper scoring rules) per the vendor and press. Text only. | | | | |

**Take-aways from the table.**
1. The consensus architecture is a **causal LLM + LoRA + tiny pointer head**. The query is a dedicated answer token. The keys are the last-token hidden states of each option. No new per-option parameters.
2. Multi-question handling comes in two flavours: a shared-prefix KV cache with separate suffixes (Strands) or one sequence with a block-causal mask (K2-Type). Both give the same independence property.
3. Images are plain token inputs to a frozen or near-frozen VLM. **Nobody in this ecosystem has published a recipe that adapts the vision tower to the decision task.** That is exactly our novel part.
4. Calibration comes from the loss (smoothed CE + Brier) plus a post-hoc temperature fitted on held-out data. Cygnet shows that the temperature alone gets you far.

## 1. Architectures

### 1.1 Option-readout families

| Family | How an option gets a score | Key references |
|---|---|---|
| **A. Pointer head over option positions** | `<q(h_answer), k(h_opt_k)>/√d`, softmax over K | Strands Decider [V], K2-Type [V]. Pointer Networks (Vinyals 2015) is the ancestor [not fetched]. |
| **B. Routing / cross-attention head** | A small transformer lets option vectors cross-attend to the state (and to other fields) before scoring | Clef [V]. Visual analogues: Query2Label (labels as queries cross-attending to image features) [V], ML-Decoder (query-based head scaling to thousands of classes, word-based queries for zero-shot) [V], CAML label-wise attention [V]. |
| **C. Answer-ID readout through the LM head** | Options labelled A, B, C… (or IDs), read the next-token probabilities. | MCSB (Robinson et al., ICLR 2023) [V]: letters beat cloze scoring once the model can bind symbols. Cygnet [V snippet]. Known failure: selection bias attached to ID tokens (Zheng et al., ICLR 2024) [V]; order sensitivity of 13-75% (Pezeshkpour & Hruschka 2023) [V]. FIRST (single-token listwise reranking) is the reranking analogue [V]. |
| **D. Generative log-likelihood of the option text** | Σ log p(option tokens), optionally length-normalised, one pass per option | Cloze scoring (the MCSB paper's baseline) [V]. The noisy-channel variant (Min et al. 2021) was not fetched. Costs K passes unless the prefix is cached. Biased by option length and token priors. |
| **E. Dual encoder / contrastive** | cos(state embedding, option embedding), InfoNCE | CLM-8B [V]. In our setting this is SigLIP2/BiomedCLIP-style image-text matching. Fast (option embeddings are cacheable, and CLM mentions caching ~1k candidates) but no cross-attention between option and image. |
| **F. Per-option yes/no ("Noul per option")** | One binary question per label, reading P(yes) | "Dichotomic prompting" (Langner et al., 2025) [V]: multi-label is decomposed into independent yes/no queries with prefix caching. It matched generative multi-label accuracy while allowing new labels. |
| **G. Structured-read diffusion** | Fill blanks in a template in one denoising step, read the logits | DiffusionGemma (Prem) [V]. Bidirectional attention inside the canvas gives true option-option interaction, but it needs a 26B MoE. |
| **H. Plain classification head** | A linear layer on a pooled hidden state or on `<answer>`, with a fixed label set | LS-LLaMA / LS-unLLaMA (LoRA + a projection to label space, outperforming 10x larger LLMs on classification) [V]. The ViT-style multi-head baseline. |

### 1.2 Pointer head vs generative scoring: trade-offs

- **Accuracy.** There is no clean head-to-head in the sources. Two weak signals:
  - Strands found a "slot" head (a per-slot classifier) clearly worse than the pointer head [V].
  - Cygnet shows a training-free letter readout on a 12B model already competes with Jev [V snippet].
  - So with ≥5-10k examples per class, a trained head should beat the zero-shot letters. How much it beats a trained linear head on the same hidden states is an open question I would test (see 5.6).
- **Calibration.**
  - Pointer logits are a softmax over a handful of real vectors, so there is no vocabulary-wide mass leakage to non-option tokens.
  - Strands keeps the head in fp32 because "a low-precision classifier is a needless source of calibration error" [V].
  - Generative letter readouts need renormalising over the option tokens, and the ID-token prior biases them (Zheng et al.) [V]. PriDe-style debiasing or permutation averaging helps.
- **Latency.** All are single-pass. Strands reports ~115 ms on an RTX 3090 and ~153 ms on an M3 Pro for short states [V]. The cost is dominated by the number of option tokens, not the head.
- **Number of options.**
  - Pointer: unlimited in principle. The cost is the tokens of the option text (~5 tokens per short label is typical [MY INFERENCE]). Jev advertises ≤255 options. Our 102 body parts, with short names, is about 500 tokens.
  - Letter or ID readout: limited to the number of single-token IDs. Cygnet caps at 26. Digits are not single tokens in Gemma-style tokenizers, so you would need reserved ID tokens [MY INFERENCE].
  - Dual encoder: effectively unlimited (cache embeddings).
- **Order and position bias.**
  - Under causal attention, option k only sees options < k. Strands reports that reversing the list moves the distribution only ~0.016 [V]. The cause is that the readout lives on content tokens, not on ID tokens.
  - Fix: shuffle options per example per epoch. Optionally average 2-4 permutations at inference using the cached prefix.
- **Interpretable per-option independence.** A softmax over options is a *competition*. That is right for single-label fields and wrong for multi-label. See 2.3.

### 1.3 Single-pass multi-question encoding

Two proven patterns:
1. **Shared prefix + separate suffix forwards** (Strands) [V]. Encode state (image + text) once, fork the KV cache per question, forward only the question tokens. Hybrid-attention torsos need their recurrent and convolution states copied, which is a Gated-DeltaNet implementation detail that does not apply to Gemma.
2. **One packed sequence with a block-causal mask** (K2-Type) [V]: `state | q1 | q2 | …`. Each question attends to the state and to itself, never to sibling questions. This is the same family as Parallel Context Windows (block attention masks plus reused position ids) [V] and Set-Encoder (permutation-invariant inter-passage attention) [V].
   - **Position ids:** give every question block the same starting position id (as in PCW), so there is no question-order dependence.
   - **Should questions see each other?** K2 chooses independence by design. Clef lets fields cross-attend [V]. For us, field dependence (modality → submodality → stain) is better handled in the loss and the decoding rules (2.5), not by letting blocks see each other.
- Empirical comfort: Son et al. (MTI Bench) found that asking LLMs several questions in one prompt did not hurt and often helped, and cut total time ~1.46x [V].
- **Implementation risk for Gemma 4** [MY INFERENCE]: Gemma 4's E2B/E4B interleave sliding-window (512 tokens) and global layers [V]. A custom block mask must compose with the sliding-window mask, and the window must be measured in per-block position ids, not in flat indices. Otherwise option tokens late in a long packed sequence lose sight of the image tokens in local layers. FlexAttention is the usual route. Verify this before the first long run. A fallback is training one question group per sequence.

### 1.4 Diffusion / structured-read decoding

DiffusionGemma is encoder-decoder, with a bidirectional 256-token canvas that cross-attends to the cached context [V]. The Prem API reads scores at blank positions in one step and reports 83.5% on JevBench with no fine-tuning, ~440 ms p95 on a 96 GB Blackwell [V]. Option-option bidirectionality is attractive for set-style decisions (multi-label body parts). But it is a 25B MoE with an unusual serving stack, so it is a reference point, not a base we can train at 0.5-1M images. I rank it below the pointer head on a Gemma-E4B for our budget.

## 2. Training objectives

### 2.1 Per-question losses

| Objective | Evidence | Use for us |
|---|---|---|
| **CE over options** (softmax) | Strands [V], and every single-label baseline | Hard single-label fields (modality, submodality, specimen, stain). |
| **Label smoothing** | Clef uses it [V]. Müller et al.: it improves calibration but hurts distillation [V]. | Use light smoothing (ε ≈ 0.02-0.05) only on the student-facing run. Avoid it in a run whose logits will serve as a **teacher**, because smoothing erases inter-class similarity information. |
| **Brier loss** | Clef [V] pairs label-smoothed CE with Brier. RLCR (Damani et al., 2025) proves that bounded proper scoring rules give both accurate and calibrated predictions [V]. | Add λ·Brier (λ ≈ 0.25-1) to CE. It is a proper score, bounded, and robust to label noise, which helps because our labels come from metadata. |
| **Ordinal / hierarchical smoothing** | Strands: 10% of Score mass to adjacent levels [V]. Bertinetto et al.: hierarchy-aware losses, so mistakes inside the right parent are cheaper [V]. | Soft targets that put a little mass on siblings under the same parent (conventional vs spiral CT, brightfield vs darkfield). |
| **KL to a teacher** | Strands: 0.3·KL(frozen torso) and 1.0·KL(teacher), applied only where the teacher was measured to help [V]. K2: soft labels from a 7B teacher [V]. | Candidate-set rows (see 2.4) could get soft targets from a baseline classifier (SigLIP2 or BiomedCLIP, restricted to the candidates). |
| **Post-hoc temperature scaling** | Guo et al. 2017: a single-parameter fit is surprisingly effective [V]. Strands uses one per primitive [V]. K2 uses one at 1.478 [V]. | Per-field temperatures fitted on the **dataset-disjoint validation split** (53 datasets), not on a random split. Strands notes that "one calibration set cannot serve two distributions" [V]. |
| **RL with calibrated reward (RLCD / RLCR)** | Clef's RLCD gives partial credit to adjacent ordinal choices [V]. K2 ran PPO on a game [V]. RLCR adds a Brier term to a correctness reward [V]. | **[MY INFERENCE]** For a categorical head that outputs the distribution directly, the expected Brier reward is differentiable in the logits, so supervised Brier + partial-credit soft targets give the same expected gradient as RL with no sampling variance. RL only buys something when the reward is non-differentiable (multi-step state, set-level F-measure, partial credit after a generation step). Skip RL in v1. |

### 2.2 Partial-label training (our `candidates` mode)
- The loss `−log Σ_{c∈C} p(c)` is the classifier-consistent form studied by Feng et al. (NeurIPS 2020), where the candidate set contains the true label [V]. PRODEN adds progressive disambiguation of the true label inside the candidate set [V].
- **Practical points** [MY INFERENCE]:
  - Candidate rows can dominate the gradient early, because a flat distribution over C already scores well. Warm up on `hard` rows first, or weight candidate rows by 1/|C|^0.5.
  - After ~1 epoch, optionally apply PRODEN-style sharpening: re-weight within C by the model's own current posterior (detached), so the model commits to one label. Never do this for rows where the candidate set encodes true ambiguity (e.g. a mixed fundus + OCT dataset). There the unresolved set should stay.
  - Submodality candidate sets are cheap to generate from modality: if only modality is known, the candidates are that modality's submodalities.

### 2.3 Multi-label decisions (body part, domain)

| Option | Mechanism | Pros | Cons |
|---|---|---|---|
| **Independent sigmoid per option** from the pointer logit (shared learned scale and bias) | BCE, or ASL (Ben-Baruch et al.: asymmetric loss that down-weights easy negatives and hard-thresholds likely mislabeled ones) [V] | Marginal probability per label, directly calibratable. Handles a variable number of positives. Matches the "unlisted does not mean absent" noise in our metadata labels (ASL's negative handling). | No competition between options, so near-synonyms can all fire. Needs a threshold. |
| **Threshold option** | Add a learned `<none/threshold>` pseudo-option. Labels with a logit above it are positive. This is the ATLOP adaptive-thresholding idea (learnable, instance-dependent threshold) [V]. | A per-example threshold with no global cutoff. Still a single softmax-style readout. | Slightly awkward semantics when the set is empty. |
| **"Select-all" question** | Pose "which of these apply?" and train softmax with a uniform target over the positives (soft multi-target CE) | Reuses the Choice machinery. Gives a ranking. | Probabilities are not marginals. Hard to compare across images with 1 vs 3 positives. |
| **Per-option yes/no (Noul)** | Dichotomic prompting [V] | Clean semantics, cacheable | 102 queries per image. Too many tokens. |
| **Thresholded ranking** | Rank by pointer logit, choose top-k by a rule | Simplest | Needs a k and a calibration rule. |

Recommendation: **sigmoid-per-option with ASL-style negative handling and a shared scalar bias for body part and domain**, with an optional threshold pseudo-option as an ablation. Calibrate by Platt or temperature on the validation split. For the competition's JSON output, take every option above the tuned threshold, always keeping at least the argmax.

Two data-specific points:
- **Hierarchical closure for body parts.** The label space has nested terms (abdomen / abdominal organs / liver ...). Propagate a positive to its ancestors when the ancestor is in the vocabulary, or mask the ancestors' negatives for that row. Otherwise "liver" images teach "abdomen = negative".
- **Weak negatives.** A `multi` row lists only what metadata named. Down-weight its negatives (w_neg ≈ 0.3) or use ASL's negative clipping.

### 2.4 Candidate-set multi-label rows
If a row says "at least one of C applies" use a noisy-OR margin: `−log(1 − Π_{c∈C}(1 − p_c))`, which is the sigmoid counterpart of the softmax candidates loss.

### 2.5 Hierarchical / conditional decisions

Modality → submodality → (stain, specimen) is a taxonomy. Three ways to encode it:
1. **Flat with parent-aware option text** (recommended). One submodality question over all 37 options, each rendered as `computed tomography › conventional ct`. The pointer sees the parent in the text.
2. **Derive modality by marginalisation.** If every submodality has exactly one parent modality (a property to check in `vocab.py`), then `P(modality=m) = Σ_{s∈m} P(s)`. Train the modality question too, and add a coherence term `KL(P_mod ‖ marginal of P_sub)`.
3. **Conditional / masked decoding.** At inference, renormalise the submodality softmax inside the predicted modality, or take the joint argmax of `P(m)·P(s|m)`.

Rows that know only the modality supply submodality candidates (2.2). **Gate stain and specimen by modality**: the label space has no "not applicable" class (stain has "unstained" but nothing for CT). So add a synthetic `n/a` option to stain and specimen. Assign it as a hard label when the modality is certain and non-microscopy and the dataset is hard-labelled. Leave it `masked` where unknown. I could not check how the MMBU scorer treats an empty stain for a CT, so confirm that against `src/bio_waila/data/eval_set.py` and the public split before fixing the N/A convention.

## 3. Multimodal decision models

What the sources say:
- **Strands** (the only one that documents it) [V]: the frozen Qwen3.5 vision tower, ≤448 px, ~196 tokens, image tokens placed in the state through the VLM's placeholder mechanism, encoded once and shared across questions by the KV prefix cache. The LoRA was **trained text-only** and remapped to the multimodal decoder. Result: 78.4% NaturalBench, ECE 0.014. Without the image, performance collapsed to chance, so the answers really depend on the image. The lesson is that a text-trained pointer LoRA transfers surprisingly well to image states, and that an image-trained competitor was much worse calibrated (ECE 0.080).
- **Clef** [V]: inherits the Qwen vision encoder; the routing head sits on the final hidden states, which already include image information. No image-token budget and no tower-tuning disclosure.
- **pplx-decider** [V]: accepts `images=`. Nothing else is disclosed.
- **Prem/DiffusionGemma** [V]: one image or ≤4 video frames, zero training.
- **Jev 1.13**: text-only (prior notes).
- **CLM-35B**: multimodal and announced, but I found no document.

**Is the vision tower tuned anywhere?** In the Jev-style family, no source says so. Outside it, there is direct evidence that tuning matters for medical images:
- MedGemma's MedSigLIP, a medically tuned SigLIP, gives +15.5-18.1% on chest-X-ray finding classification over the base models [V].
- Zhang et al. (NeurIPS 2024) find that VLM image classification is limited by training-data exposure per class and becomes competitive once enough classification data is given [V]. We have ~763k labelled images across 393 datasets.

So the design bet is: **Stage 1** freezes the tower (the Strands recipe), **Stage 2** unfreezes the tower (see 4.1).

**Image-token budgets.** Gemma 4 offers 70 / 140 / 280 / 560 / 1120 soft tokens per image (default 280); Google and third-party notes say lower budgets suit classification [V]. For modality, submodality, stain and body-part cues, 280 is a sensible default. Run 140 and 560 as ablations: histology stains and small-structure body parts may want 560, while chest X-rays and fundus images probably do not. Image tokens cost almost nothing against the ~1.3k option tokens in our sequence.

**Causal-order consequence** [MY INFERENCE]. The image tokens must precede the option text, otherwise the option hidden states cannot see the image. Under causal masking, the image tokens themselves never see the options. The "option asks the image a question" step therefore happens only inside the option tokens' own attention, through all LoRA'd layers. A small cross-attention routing head on top (Clef / Query2Label / ML-Decoder style, with the option vectors as queries and the image-token hidden states as keys) is a cheap supplement worth a single ablation.

## 4. Practical recipe

All numbers marked [calc] are my back-of-envelope arithmetic.

### 4.1 What to train

| Component | Stage 1 | Stage 2 |
|---|---|---|
| LLM LoRA | r = 16-64 on q,k,v,o,gate,up,down of all decoder layers, **excluding the LM head** | same, continued |
| Pointer head | ~1M-param fp32 q/k projections (d = 256), as in Strands | same |
| Vision tower | frozen | LoRA r = 32 on tower attention/MLP, or unfreeze the last 4-6 blocks at 10-20x lower LR (1e-5). The projector is always trainable. |
| LM head | frozen | frozen |

- **LoRA rank.** Strands gets 88 MB checkpoints and strong text results at rank 16 on a 2B model [V]. Clef uses rank 256 on 9B-27B [V]. Our domain shift is visual, medical, and fine-grained (stains, 100 body parts), so I would start at r = 64 for the LLM and run r = 16 as a cheap check.
- **Which layer to read.** The sources read the final hidden states (Strands, K2, Clef). Alternatives worth one ablation each:
  - the final layer (baseline);
  - a learned scalar mix of the last 4 layers (ELMo-style);
  - a mid-late layer (about 2/3 depth). "Label Words are Anchors" shows that label-word representations aggregate their information in shallow-to-middle layers [V], so a mid-depth readout is plausible. It also leaves the top layers free to keep doing their LM job (see 5.6).
- **fp32 head, bf16 torso** (Strands) [V].
- **Option text.** Keep it short (≈3-6 tokens). Optionally append a one-line gloss ("x-ray imaging (radiography)") and randomly drop it (~40%) so the model works with both.
- **Optimiser (starting point, no source for these numbers):** AdamW, LoRA LR 1e-4, head LR 1e-3, cosine, 3% warmup, effective batch ~128 images, 2-3 epochs, bf16.

### 4.2 Option ordering, negatives, augmentation
- **Shuffle options per example per epoch and remap the label** [V Strands]. Reverse, not shuffle, ordinal options.
- **Randomly drop questions** (train with 1-6 of the 6 fields per image) and randomise block order. With shared position ids this makes inference independent of which fields are asked.
- **Hard-option sampling.**
  - For 11-39 option fields, always show all options.
  - For body part (102), train mostly with all options. Occasionally train with a sampled K (gold + 20-40 confusable negatives from the confusion matrix of the previous checkpoint or a SigLIP2 baseline + random fill). This is a speed-up and lets the model handle any K at test.
  - The softmax-normalised fields should also see the full list regularly, since sampled-K softmax changes the normalisation.
- **Permutation TTA** at inference: 2-4 shuffled lists averaged. It is almost free with the cached image prefix.
- **Remove dataset shortcuts** (Strands: "Remove the shortcuts the data allows" [V]). Most datasets here have one modality, one stain and one species for every image. A model can learn "this scanner's noise = label". Mitigate with per-dataset caps (already in the plan), group-balanced sampling, strong augmentation, dataset-disjoint validation, and a held-out-dataset ECE report.
- **Class balance.** Sample submodality and stain with a square-root class-frequency rule. Rare stains and specimens may have only hundreds of images, so keep option text informative (so the pointer can generalise semantically).

### 4.3 Data scale
- References: Strands 100k rows + 13k multi-step rows, ~11 h on one RTX 3090 (2B torso, short sequences) [V]. K2-Type full fine-tune on ~354k records [V]. A social-media claim of a 60K-decision model trained for $5.40 in 2h10m on one A100 [UNVERIFIED, via prior notes].
- Ours: ~763k images × up to 6 questions ≈ 4.6M decisions, which is more than enough for the closed vocabulary. The limiting factor is **per-class diversity across datasets** (Zhang et al.: exposure per class drives VLM classification accuracy [V]), not volume.

### 4.4 Compute for E2B / E4B on ~0.76M images [calc]
- Tokens per sample: ~280 image + ~250 options × ~5 tokens + ~100 overhead ≈ 1.6k. One epoch ≈ 1.2B tokens.
- LoRA with frozen base weights costs ~4·N·T FLOPs (forward + activation-gradient backward). With N ≈ 4.5B (E4B effective) that is ≈ 2.2e19 FLOPs per epoch. For E2B (2.3B effective) it is about half.
- At realistic utilisation (B200 ≈ 0.4 PFLOP/s, H100 ≈ 0.3, A100 ≈ 0.1) this is ≈ 15 / 20 / 56 GPU-hours per epoch for E4B.
- **Budget: ~50-150 GPU-hours for a 3-epoch E4B run, so ~6-20 wall-clock hours on 8 GPUs.** Cutting options per sample (sampled negatives) or going to E2B reduces it 2-3x. The vision tower (~150M) is negligible in FLOPs, but activations at 560-1120 tokens are not.
- The PLE embeddings in the E-series (5.1B and 8B "with embeddings" vs 2.3B and 4.5B effective [V]) are lookups, not FLOPs, but they count toward memory and (possibly) toward Track-3 parameter totals. Check the challenge rule's wording on total vs active parameters before choosing E4B over E2B (E4B at 8B total is under the 12B total cap, and 4.5B effective is above the 4B active cap if "effective" is what is counted [UNVERIFIED]).
- **Moondream 3.1** (9B total / 2B active MoE) [V: HF page]. The checkpoint is Photon-format, and fine-tuning is documented only through Moondream's API [V]. Training a pointer head and LoRA there would need custom kestrel/Photon training code (see memory note: not loadable by vLLM or standard HF). It is a poor first base for this experiment. Gemma 4 E2B/E4B has standard HF support, QLoRA, and Apache-2.0 [V].

## 5. Concrete design for the Pass-2 task

### 5.1 Backbone and layout

Base: **Gemma-4-E4B-it** (E2B for the Track-3 pilot). A fallback for a cleaner vision story: Gemma 3-based MedGemma-4B, which already has MedSigLIP [V]. Check its Track-3 eligibility and contamination rules separately.

Sequence (all questions in one packed sequence, block-causal, shared position ids):

```
<bos><start_of_turn>user
[IMAGE: 280 soft tokens]                      <- the "state"
Classify this biomedical image.
<q:modality>    <opt>computed tomography</opt> <opt>digital pathology</opt> ... <decide>
<q:submodality> <opt>computed tomography › conventional ct</opt> ...            <decide>
<q:body_part>   <opt>abdomen</opt> <opt>abdominal organs › liver</opt> ...        <decide>
<q:stain>       <opt>n/a</opt> <opt>h&e</opt> <opt>dapi</opt> ...                 <decide>
<q:specimen>    <opt>n/a</opt> <opt>human</opt> <opt>mouse</opt> ...              <decide>
<q:domain>      <opt>cardiology</opt> ...                                         <decide>
```

- The five new special tokens (`<q:…>`, `<opt>`, `</opt>`, `<decide>`) can reuse reserved tokens (K2-Type reused 5 reserved tokens [V]).
- Mask: image and prompt tokens follow Gemma's native image mask. Each question block attends to the image and prompt prefix and to itself (causal), never to another block. Position ids restart at the end of the prefix for each block.
- Option order is shuffled each epoch. Roughly 250 options at ≈5 tokens ≈ 1.3k tokens, plus 280 image tokens.
- Serving: encode the image prefix once into the KV cache. Only the six question blocks are forwarded. If the blocks are identical across images except for the shuffle, the option text could in principle be KV-cached too, but the option hidden states depend on the image, so they cannot.

### 5.2 Heads

- **Query:** the hidden state at each block's `<decide>` token. **Keys:** the hidden state at each option's `</opt>` token (the last token, as in Strands/K2) [V]. Optionally mean-pool the option tokens as an ablation.
- **Pointer:** `logit_k = <Wq h_decide, Wk h_opt_k>/√256`, fp32, ~1M parameters, plus per-field learned temperature and (for multi-label fields) a shared learned bias.
- Optional v1: a 1-2 layer routing head in which option vectors cross-attend to image-token hidden states (Clef / Query2Label style [V]), testable as a single ablation.
- **Also keep an auxiliary LM-head readout.** At the same `<decide>` position, ask the frozen LM head for the next-token distribution over reserved ID tokens (or the option's first token). Train it with a small weight (see 5.6).

### 5.3 Fields as questions

| Field | Question type | #options | Readout | Notes |
|---|---|---|---|---|
| modality | Choice | 11 | softmax | Coherence term with the submodality marginal. |
| submodality | Choice | 37 | softmax | Parent-aware option text, modality-derived candidate sets, post-hoc renormalisation inside the predicted modality. |
| body_part | multi-select | 102 (+ optional threshold option) | sigmoid-per-option, ASL, ancestor closure | Output = everything above the tuned threshold, at least the argmax. |
| stain | Choice | 39 + `n/a` | softmax | `n/a` gated by modality (2.5). |
| specimen | Choice | 38 + `n/a` | softmax | Same. |
| domain | multi-select | 23 | sigmoid-per-option | |

### 5.4 Losses per label mode

For field f with option set O_f and row label `<f>_mode`:

| Mode | Single-label softmax field | Multi-label sigmoid field (body part, domain) |
|---|---|---|
| `hard` | CE (ε ≈ 0.02) + λ·Brier + sibling-soft targets | Not applicable (`hard` implies one positive). Use BCE with that one positive, ancestors closed, and down-weighted negatives. |
| `multi` | Rare for single-label fields. Treat as `candidates`. | ASL/BCE with all listed values positive (ancestors closed), other options as weak negatives (w ≈ 0.3). |
| `candidates` | `−log Σ_{c∈C} p(c)`. Warm-up on `hard` first. Optional PRODEN-style sharpening on rows that are not truly ambiguous. | Noisy-OR margin `−log(1 − Π(1 − p_c))`. |
| `masked` | No loss. | No loss. |

- **Derived constraints.** A hard modality supplies submodality candidates. A non-microscopy modality supplies a hard `n/a` for stain and specimen where that convention is confirmed.
- **Row weights.** Use `<field>_conf` as the weight, down-weight rows flagged with `<field>_conflict`, and cap each dataset's share per batch.
- **Total loss:** `Σ_fields w_f·L_f + 0.3·KL(LM-head readout ‖ frozen base readout) + μ·L_gen`. The first KL term is the Strands anchor idea [V]. `L_gen` is the optional JSON-generation loss (5.6).
- **Validation is dataset-disjoint** (the 53 validation datasets), plus **mmbu-public** for calibration and evaluation only, never for training (per the README).

### 5.5 Calibration

1. Train with CE + Brier (no heavy label smoothing in the teacher run).
2. Fit **one temperature per field** (and a Platt scale for sigmoid fields) on the validation datasets (dataset-disjoint), as Strands and K2 do [V].
3. Report ECE and Brier per field and per held-out dataset. Strands' own note says that one calibration set cannot serve two distributions [V], so expect miscalibration to differ across modality groups. If it matters, fit temperature per modality group.
4. Optional second-stage fine-tune with a Brier-only reward (the RLCD/RLCR idea [V]). As argued in 2.1, supervised Brier is already the expected gradient, so skip RL.

### 5.6 Transfer to the final generative VLM

The competition needs one end-to-end VLM that emits JSON, with no auxiliary models. Four transfer routes, combinable:

1. **Share the vision tower and projector.** Stage 2 gives a vision tower adapted on 0.76M labelled medical images with the whole decision objective. Export the tuned tower and projector (plus a tower LoRA) and plug them into the final VLM, as long as it is the same base. This route is the most robust. It is also the main reason to tune the tower here, because no Jev-style system publishes this.
2. **Keep the generative path alive while training the decision path.** The pointer LoRA sits on the same decoder layers that the LM head reads.
   - Leave the LM head frozen. Include (a) the 0.3·KL anchor to the frozen base's own readout (Strands) [V] and (b) a modest fraction (say 20-30%) of the batch in **generate mode**: the same image with the JSON target `{"modality": …, "submodality": …, …}` and the usual next-token loss. A mode token in the prompt selects the path.
   - This is **[MY INFERENCE]**, not demonstrated by any Jev-style source (none ships a generative head). The risk is representational interference: the pointer reads hidden states that are being bent to separate options, which may degrade fluent generation. Mitigation: read the pointer at a mid-late layer, so the top layers stay LM-shaped (testable, per 4.1). A fallback is the two-adapter solution: train separate LoRAs for the decision and generative modes, and merge or distil afterwards.
3. **Teacher / distillation.** Use the decision model as a calibrated labeller for unlabeled or `candidates` images, with soft targets, and train the final VLM's generative JSON on them. Calibration matters here: low-confidence rows can be dropped or down-weighted.
4. **Constrained generation at test.** Because the final VLM generates JSON values from closed sets, decode with a trie or grammar-constrained decoding so every emitted value is valid. Order the fields modality → submodality → body part → stain → specimen → domain, so each field conditions on the earlier ones (a free form of the hierarchical conditioning in 2.5). The decision model's pointer path is then the "non-autoregressive teacher" for this autoregressive student.

A cheap tier-0 baseline (no new parameters): the Cygnet-style letter readout [V snippet]. It needs ≤26 options per question, so body part would need chunking.

### 5.7 Ablation plan (smallest set that settles the design)

| # | Question | Variants |
|---|---|---|
| 1 | Is a pointer head worth it on a closed vocabulary? | (a) linear head on `<decide>` hidden state with no option text, (b) pointer over option text, (c) LM-head ID readout, (d) SigLIP2/DINOv3 multi-head baseline from the earlier notes |
| 2 | Vision tower | frozen vs LoRA vs last-N-blocks unfrozen |
| 3 | Readout layer | final vs scalar-mix last 4 vs ~2/3 depth |
| 4 | Multi-label | sigmoid + ASL vs threshold option vs select-all |
| 5 | Image tokens | 140 / 280 / 560 |
| 6 | Order robustness | shuffled vs fixed training order, TTA over 1/2/4 permutations |
| 7 | Generative coexistence | decision-only vs decision + 25% JSON generation vs anchor KL only |

Metrics: per-field accuracy or F1 on dataset-disjoint validation and mmbu-public, ECE, Brier, plus a 100-image JSON-quality check from the decision-and-generation variant.

## 6. Pros/cons table

| Approach | Accuracy potential | Calibration | Latency | Options scale | Transfer to a generative VLM | Verdict for Pass-2 |
|---|---|---|---|---|---|---|
| **Pointer head over option positions** (Strands/K2) | High with enough data. A slot head did worse [V]. | Good: fp32 head, softmax over real options, temperature scaling. | One pass, ~115-150 ms [V]. | Large (≤255 in Jev). The cost is option tokens. | Medium. Same decoder, but the LM head must be kept alive deliberately. | **Primary candidate** |
| **Routing / cross-attention head** (Clef, Query2Label, ML-Decoder) | Highest on paper. Option-image interaction beyond the LLM's own. | Same. | Small extra. | Large. | Medium. | Ablation on top of the pointer |
| **ID or letter readout via LM head** (MCSB, Cygnet) | Good zero-shot. Selection bias [V]. | OK with temperature and permutation averaging. | One pass. | ≤26 single-token IDs. | **Highest**, since it is the native generative path. | Baseline and the auxiliary path |
| **Log-likelihood of option text** | Biased by length and priors. | Mediocre. | K passes unless prefix-cached. | Large but slow. | High. | Not recommended |
| **Dual encoder / contrastive** (CLM, SigLIP2) | Strong for closed labels with good image-text pretraining. No cross-attention. | Needs fitted temperature. | Fastest (cached embeddings). | Effectively unlimited. | Low. | Complementary baseline |
| **Per-option yes/no** (dichotomic prompting) | Matches generative accuracy [V]. | Clean per-label probabilities. | ~100 suffixes per image. | Linear in labels. | Medium. | Only for the few ambiguous labels |
| **Structured-read diffusion** (DiffusionGemma/Prem) | 83.5% JevBench zero-shot [V]. | "Not calibrated" [V]. | ~440 ms. | 16 questions per request. | Low (26B MoE). | Reference only |
| **Plain linear multi-head** | Strong for closed sets. | Needs temperature. | Fastest. | Fixed vocab only. | Low. | Mandatory control |
| **Constrained JSON generation from the final VLM** | Whatever the VLM learns. | Poor unless logits are scored. | Autoregressive. | Unlimited. | The final answer format itself. | The end target |

## 7. Honest caveats

- **No published head-to-head** of pointer vs linear vs letter readout on images exists in what I fetched. The recommendation to try the pointer first rests on the Jev-ecosystem convergence and on transfer reasoning, which is not evidence for our task. Ablation 1 is the arbiter.
- On a **closed** vocabulary with abundant labels, the practical advantage of the pointer over a linear head may be mostly semantic sharing for rare classes and the interface match with the generative model. If ablation 1 shows no gain, use the linear head and keep the pointer as an option.
- **Label noise is the real ceiling.** Our labels come from metadata and `candidates`. Calibrated probabilities against noisy labels will look under-confident on clean datasets and over-confident on shifted ones. Per-dataset ECE reporting is essential.
- The `multi` sigmoid, `n/a`, and joint-generation designs are **[MY INFERENCE]** and untested by any source.
- Gemma-4 specifics I could not verify: how the image-token mask interacts with a custom block mask, and whether the sliding-window mask composes correctly (see 1.3).
- Details missing from the sources: pplx-decider's head/training/image handling; Clef's tower tuning and image-token count; the Clef LoRA discrepancy between its blog and its model card; the CLM-35B multimodal design. I did not find the full text of "Jev in the Wild" (arXiv 2609.30216) with technical content: the page gave only the abstract (a GitHub usage study).

## Sources

Jev-style models (primary, fetched 2026-10-06):
- Strands Decider blog: https://strandsagents.com/blog/introducing-strands-decider/
- Strands Decider repo: https://github.com/strands-labs/strands-decider (docs/architecture.md, docs/inference.md, docs/vision.md, training/recipe.sh, evaluation/results.md under raw.githubusercontent.com/strands-labs/strands-decider/main/)
- Cloudflare Clef blog: https://blog.cloudflare.com/clef-decision-models/
- Cloudflare Clef model card: https://huggingface.co/cloudflare/clef
- K2-Type-0.9B: https://huggingface.co/IFM/K2-Type-0.9B
- pplx-decider-v1-27b: https://huggingface.co/perplexity-ai/pplx-decider-v1-27b and https://aiweekly.co/alerts/perplexity-open-sources-27b-decider-edges-jev-on-11-test-panel
- CLM-v0.1-8B: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- GLiDE: https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model
- Prem DiffusionGemma Jev API: https://www.premai.io/blog/a-jev-compatible-api-for-text-images-and-video/
- DiffusionGemma (Transformers docs): https://huggingface.co/docs/transformers/v5.14.0/model_doc/diffusion_gemma.md
- Decision-model listings (Cygnet, Winnow, Jev): https://benchlm.ai/md/decision-models.md, https://benchlm.ai/models/cygnet
- Jev in the Wild: https://arxiv.org/abs/2609.30216 (abstract only)
- Prior notes: data_discovery/research/jev_clm_research.md

Base models:
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma 4 token budgets (search summary): https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know
- Moondream 3.1: https://huggingface.co/moondream/moondream3.1-9B-A2B
- MedGemma technical report: https://arxiv.org/abs/2507.05201

Precursor and method papers:
- Robinson, Rytting, Wingate, Leveraging LLMs for Multiple Choice QA (MCSB): https://arxiv.org/abs/2210.12353
- Zheng et al., LLMs Are Not Robust Multiple Choice Selectors (PriDe): https://arxiv.org/abs/2309.03882
- Pezeshkpour & Hruschka, Sensitivity to the Order of Options: https://arxiv.org/abs/2308.11483
- FIRST (single-token listwise reranking): https://arxiv.org/abs/2406.15657
- Zhang et al., Generative Verifiers: https://arxiv.org/abs/2408.15240
- Damani et al., RLCR, Beyond Binary Rewards: https://arxiv.org/abs/2507.16806
- Zhang et al., Why are VLMs bad at image classification?: https://arxiv.org/abs/2405.18415
- LS-LLaMA / LS-unLLaMA: https://arxiv.org/abs/2310.01208
- Feng et al., Provably Consistent Partial-Label Learning: https://arxiv.org/abs/2007.08929
- Lv et al., PRODEN: https://arxiv.org/abs/2002.08053
- ML-Decoder: https://arxiv.org/abs/2111.12933
- Query2Label: https://arxiv.org/abs/2107.10834
- Asymmetric Loss: https://arxiv.org/abs/2009.14119
- ATLOP adaptive thresholding: https://arxiv.org/abs/2010.11304
- CAML (label-wise attention): https://arxiv.org/abs/1802.05695
- Multi-Task Inference (MTI Bench): https://arxiv.org/abs/2402.11597
- Set-Encoder: https://arxiv.org/abs/2404.06912
- Parallel Context Windows: https://arxiv.org/abs/2212.10947
- Bertinetto et al., Making Better Mistakes: https://arxiv.org/abs/1912.09393
- Guo et al., On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Müller et al., When Does Label Smoothing Help?: https://arxiv.org/abs/1906.02629
- Wang et al., Label Words are Anchors: https://arxiv.org/abs/2305.14160
- Kadavath et al., Language Models (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
- VLM2Vec: https://arxiv.org/abs/2410.05160
- LLM2Vec: https://arxiv.org/abs/2404.05961
- Langner et al., Dichotomic Prompting (multi-label with cached yes/no): https://arxiv.org/abs/2511.03830
- Kementchedjhieva & Chalkidis, multi-label encoder-decoder approaches (LWAN etc.): https://arxiv.org/abs/2305.05627
