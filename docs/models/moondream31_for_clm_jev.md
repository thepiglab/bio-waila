# Moondream 3.1 as a backbone for CLM-vision, CLM-LLM and Jev

Date: 2026-10-08. Scope: `moondream/moondream3.1-9B-A2B` (M87 Labs, US). Background row: `backbones.md`, the Moondream line.

Evidence used:
- the safetensors header read with HTTP range requests (no 9B download);
- the `kestrel==0.9.1` and `moondream==2.6.1` wheels, unzipped and read only;
- the GitHub `m87-labs/moondream` torch code;
- the license text fetched from moondream.ai;
- one numerical experiment on ~0.9 GB of the vision tensors.

Items marked [UNVERIFIED] were not confirmed.

## 0. Verdict

| Teacher | Feasibility | Engineering | Expected benefit | Recommendation |
|---|---|---|---|---|
| (a) CLM-vision on Moondream's tower | **Easy to load, medium to make useful** | 0.5 day to load. About 3 days with label re-alignment and a bake-off | **Low.** The tower is a heavily fine-tuned SigLIP2 whose features no longer match SigLIP2's text tower (section 2.4). Only ensemble diversity | Do not make it primary. At most a 1-day frozen-probe check next to SigLIP2 |
| (b) CLM-LLM on Moondream LM hidden states | **Hard** (own torch forward plus FP8 dequant; no kestrel hook) | 5 to 7 days | **Low.** The LM is a drop-upcycled Moondream-2 / Phi-1.5 lineage MoE with no medical language prior. Gemma 4 E4B is the better frozen-LLM arm | Skip |
| (c) Jev with Moondream 3.1 | Zero-shot letter readout: **hard** (kestrel exposes no distribution). LoRA: **hard** | Zero-shot via a patched kestrel: 3 days. LoRA via torch: 8 to 12 days | **Low.** Not trained for lettered MCQ. The vendor's own glaucoma page reports a 17.6% base score on a 3-way task | Skip. Spend the days on Gemma 4 E2B/E4B Jev |

**Net: do not build on Moondream 3.1.** The one useful discovery is mechanical. Its vision tower is a drop-in, one-to-one match for `SiglipVisionModel` (strict load, no shape changes), and the Photon tensor names are the ones the GitHub torch loader expects. The license is permissive about fine-tuning and about using outputs to train a Gemma student.

## 1. Checkpoint layout (header read by range request)

`model.safetensors` header: 107,584 bytes, 870 tensors, no `__metadata__`. `config.json` is `{"tokenizer": "moondream/starmie-v1"}`. The repo holds four files: `.gitattributes`, `README.md`, `config.json`, `model.safetensors`.

| Group | Tensors | dtype | Notes |
|---|---|---|---|
| `vision_encoder.encoder.model.visual.*` | 333 (27 blocks x 12, plus patch_embed x2, pos_embed, norm x2) | **all BF16** | |
| `vision_encoder.projection.mlp.fc1/fc2` | 4 | BF16 | fc1 `[8192, 2304]` (input is 2 x 1152, concat of global and local features), fc2 `[2048, 8192]` |
| `text_model.*` dense / attention / embeddings | | BF16 | wte `[51200, 2048]`; lm_head `[51200, 2048]`; Wqkv `[6144, 2048]`; layers 0-3 dense MLP `[8192, 2048]` |
| `text_model.transformer.h.{4..23}.mlp.experts.weight` | 20 | **F8_E4M3** `[64, 2048, 2048]` | up-proj, rows are 2x1024 (gate half, then up half) |
| `...mlp.output_experts.weight` | 20 | **F8_E4M3** `[64, 2048, 1024]` | down-proj |
| `...moe_quant.up_scale / down_scale` | 40 | F32 `[64, 2048]` | per-expert, per-output-row scales |
| `...kv_quantizer.*` | 168 | BF16 / BOOL scalars | FP8-KV-cache calibration; irrelevant to us |
| `...tau_wq, tau_wv, tau_alpha`, `gate.*` | | BF16 | |
| `region_model.*` | 12 | BF16 | box / point head |

**The vision weights are BF16, not FP8.** Only the 20 MoE layers (about 8.05 B parameters) are FP8.

The 3.1 tensor names are not the HF `moondream3-preview` safetensors names. That repo uses `model.text.blocks.*` and `model.vision.*` (index shows 662 names, 0 shared with 3.1). But **3.1 does match the naming that the GitHub `moondream/torch/weights.py::_load_weights` expects.** I built that loader's full expected key set (660 keys) and compared it with the 3.1 header:
- all 660 are present;
- the only unmatched 3.1 tensors are 168 `kv_quantizer` scalars, 40 `moe_quant` scales and 2 `region_model.ln.*`.

So the "no shared names" statement holds for the HF preview shards, not for the GitHub torch loader or the preview's `model_fp8.pt` [UNVERIFIED for `model_fp8.pt`].

### 1.1 FP8 dequantisation (verified numerically)

I range-fetched layer 4, expert 0 (4 MB).
- Every FP8 row has max |value| = 448 (E4M3 max), so the scale is per-row amax / 448.
- `bf16_weight = fp8.float() * scale[expert, row][:, None]`, for both `experts.weight` (scale `[64, 2048]`) and `output_experts.weight` (scale `[64, 2048]`).
- Dequantised std is 0.036 (up) and 0.048 (down). Dense layer-0 fc1 std is 0.032, so the magnitudes are sane.
- Scale range is 4.5e-7 to 1.7e-3. Some rows are near-dead.
- Dequantised BF16 total is about 18.5 GB (kestrel's loader also keeps the up weight in an 8-row gate/up interleave; the checkpoint itself is half-split).

## 2. Vision tower versus `SiglipVisionModel`

### 2.1 Config match

`google/siglip-so400m-patch14-384` and `google/siglip2-so400m-patch14-384` have identical vision configs (checked on the HF API): 27 layers, hidden 1152, MLP 4304, 16 heads, patch 14, image_size 384.

Moondream's `VisionConfig` (from kestrel): `enc_dim 1152, enc_ff_dim 4304, enc_n_layers 27, enc_n_heads 16, patch 14, crop_size 378, max_crops 12, overlap_margin 4`.

### 2.2 Name and shape mapping (strict load verified)

I built a `SiglipVisionModel` and loaded the Moondream tower with `strict=True`: all keys matched, no shape changes. Prefix `P = vision_encoder.encoder.model.visual.`

| Moondream 3.1 tensor | HF `SiglipVisionModel` key | Transform |
|---|---|---|
| `P+patch_embed.linear.weight` `[1152, 588]` | `embeddings.patch_embedding.weight` `[1152, 3, 14, 14]` | `.reshape(1152, 3, 14, 14)` (patch order is C, p1, p2, same as Conv2d flattening) |
| `P+patch_embed.linear.bias` | `embeddings.patch_embedding.bias` | none |
| `P+pos_embed` `[1, 729, 1152]` | `embeddings.position_embedding.weight` `[729, 1152]` | `[0]` |
| `P+blocks.N.norm1 / norm2 .{weight,bias}` | `encoder.layers.N.layer_norm1 / layer_norm2` | none |
| `P+blocks.N.attn.qkv.{weight,bias}` `[3456, 1152]` | `self_attn.q_proj / k_proj / v_proj` | split rows 0:1152, 1152:2304, 2304:3456 (q, k, v) |
| `P+blocks.N.attn.proj` | `self_attn.out_proj` | none |
| `P+blocks.N.mlp.fc1 / fc2` | `mlp.fc1 / fc2` | none |
| `P+norm.{weight,bias}` | `post_layernorm` | none |

Differences and cautions:
- **Position embedding:** 729 = 27x27 for both 378/14 and 384/14. In HF the stride-14 conv on 384 px also yields 27x27 (floor), so no interpolation is needed. Feed 378 px (Moondream's crop size) or 384 px (drops 6 px); features differed negligibly in my test (pooled cosine 0.667 versus 0.666 against SigLIP2).
- **No MAP head.** Moondream drops the pooling head. The tokens go through `post_layernorm` ("norm") into the projector. Our `SiglipCLM` copies `trunk.head` into the per-field heads, so the head must be supplied separately (section 4a).
- **No text tower.**
- **Projection MLP** (`vision_encoder.projection.mlp`, 2304 to 8192 to 2048) maps into the LM. Not needed for CLM-vision.
- **Normalisation:** `(x/255 - 0.5)/0.5`, same as SigLIP. GELU is the tanh variant, same as HF `gelu_pytorch_tanh`.
- **LayerNorm eps:** kestrel's `nn.LayerNorm` default is 1e-5, HF SigLIP uses 1e-6. Use 1e-5 for the Moondream tower [harmless on fine-tuning].
- **Multi-crop scheme:** the model uses a global 378 crop plus up to 12 overlapping local crops (`overlap_margin 4`), reconstructed, adaptively pooled back to 27x27, then concatenated with the global features (hence 2304 to 8192). A single-crop CLM-vision uses only the global-crop tokens, which is the Moondream "global" branch. That discards the high-resolution pathway, which is Moondream's one architectural edge.
- **transformers 5.19.0** (the bio-waila `.venv`): `SiglipVisionModel.state_dict()` keys have no `vision_model.` prefix (flattened). `SiglipModel` still uses `vision_model.*`.

### 2.3 Lineage: the tower is SigLIP2, not SigLIP v1

I range-fetched all `vision_model.*` tensors of both `google/siglip-so400m-patch14-384` and `google/siglip2-so400m-patch14-384` (1.7 GB for SigLIP2) and compared them with Moondream's.

| Tensor | cos vs SigLIP v1 | cos vs SigLIP2 |
|---|---|---|
| pos_embed | 0.008 | **0.999996** |
| patch_embed | -0.002 | **0.9996** |
| post_layernorm weight | 0.968 | **0.999997** |
| L0 q / k / fc1 | about 0.001 | 0.968 / 0.972 / 0.984 |
| L13 q / fc1 | about 0 | 0.854 / 0.907 |
| L26 q / fc1 | about 0 | 0.756 / 0.883 |

So the tower was initialised from SigLIP2-So400m-patch14-384 and then fine-tuned in the VLM, with more drift in deeper layers. (Cosine on LayerNorm gains is uninformative: the vectors are near-constant. Judge by the patch, position and block weights.)

### 2.4 How much has it drifted (the key number for CLM-vision)

I fed 32 random MMBU-public images through both towers. The Moondream tower was given SigLIP2's own MAP head (`head.*`), since it has none.

- Hidden-state cosine versus SigLIP2, layers 0/1/3/6/9/12/15/18/21/24/27: 1.00, 0.97, 0.86, 0.74, 0.69, 0.63, 0.53, 0.48, 0.46, 0.48, **0.23**.
- Post-LN token cosine: 0.135 on average. Mean-pooled tokens: 0.064.
- **SigLIP2 MAP-head pooled embedding, Moondream tower versus SigLIP2 tower: cosine 0.667 on average (minimum 0.599).** Nearest-neighbour agreement is 0.28 (chance is 1/31). Similarity-structure correlation (RSA) is 0.72.

Reading: the Moondream tower's features still carry SigLIP2's coarse structure, but they are no longer in the space of SigLIP2's text tower. Initialising label vectors from the SigLIP2 text tower and copying the SigLIP2 MAP head would **not** reproduce the "model starts as SigLIP2 zero-shot" property that `clm_siglip.py` relies on (and that makes WiSE-FT exact).

Frozen-probe sanity check (small and noisy; 1,025 MMBU-public images from the 13 datasets with at least 40 images; features = SigLIP2 head pooled or token mean; ridge-regularised softmax regression):

| Test | SigLIP2 head | SigLIP2 mean | Moondream tower + SigLIP2 head | Moondream tower mean |
|---|---|---|---|---|
| Fine-grained answer, within-dataset 5-fold, 4 datasets with 40 to 80 images (mean acc.) | 0.496 | 0.481 | 0.470 | 0.449 |
| Modality, leave-datasets-out, 5 classes, 1,025 images | 0.907 | 0.873 | 0.922 | 0.929 |

Conclusion: no measurable advantage either way. n is tiny and the within-dataset test has only 4 datasets (about 0.03 acc. per-dataset noise). This does not support, or rule out, a gain from the Moondream tower. Scratch code is not kept. The recipe is in section 2.2 plus a logistic probe.

## 3. Engine API: kestrel 0.9.1 and `moondream` 2.6.1

Both wheels were unzipped, not installed. Neither runs without CUDA kernels (`kestrel-kernels==0.7.4`, `kestrel-native==0.1.8` are compiled and closed). Kestrel's README says Apple-silicon Metal kernels also exist.

- **Public API:** `InferenceEngine.create(RuntimeConfig(...))`, `engine.model()` returns a `ModelHandle` with `query / caption / detect / point / segment / run / stream`. `moondream.photon()` / `PhotonClient` is a sync wrapper over the same engine.
- **Image embeddings:** not exposed by the engine. A skill returns `output["answer"]` (query), plus optional `reasoning`. The vision path is a plain Python function, `kestrel.models.moondream.vision.encode_image(image_np, module, config, device=, dtype=)`, which returns the `[729, 2048]` projected tokens. You could call it, with the loaded module, on a CUDA or Metal device. It is internal API, and it uses the closed-source kernels.
- **Logits / probabilities: no.**
  - The only per-token number is a private setting `settings["_logprobs"] = True`, returned as `EngineResult.logprobs`. It is the logprob of the *sampled* token only, and for greedy requests (`temperature <= 0`) it is just 0 or -inf (`_selected_logprobs_from_logits`).
  - With sampling on, you get a single sampled token's log-softmax, not the distribution over the letters.
  - `settings["_suppress_next_token_ids"]` masks the first generated token, and `_generated_prefix` forces a prefix (the latter needs caller-supplied logprobs).
  - So a letter readout over A..K would need K+ passes with suppression and sampling noise, or a patch to the runtime's `SamplingHooks.process_logits` (a documented hook) to copy `logits[:, letter_ids]` out [UNVERIFIED: not tried; the kernels are closed and need a CUDA run on our GPU cluster].
- **Batch scoring:** the scheduler batches concurrent requests (`max_batch_size`; `mmbu-eval` sets 32) with paged KV and a radix prefix cache keyed on the image hash. This is good for repeated prompts on one image (several fields per image share the image prefill).
- **Training / LoRA hooks: none for local training.**
  - `moondream/finetune.py` is a client for the hosted "Lens" tuning API (`api.moondream.ai/v1/tuning`, needs an API key). Our data would leave the cluster, and the license restricts hosting a fine-tuning *service* for others, not our own use.
  - Kestrel only *serves* LoRA adapters through an `AdapterProvider`. The shipped provider downloads from the Moondream API; `max_lora_rank` is 32. A local provider could be written against the protocol [UNVERIFIED].
  - LoRA targets in kestrel: attention, dense MLP and MoE experts (rank split across active experts); also a vision `proj_mlp.fc2` LoRA.
- **Prompt construction (`QuerySkill.build_prompt_tokens`):** `[BOS=0] + image(729 tokens, inserted by the runtime right after the first prompt token) + [1, 15381, 2] + tokenizer.encode(question) + [3]` (answer opener; `[4]` for reasoning mode).
  - Token 15381 decodes to "query"; 1, 2, 3, 4 are `<|md_reserved_*|>` tokens.
  - The first 730 positions (BOS plus image) are fully bidirectional; text after them is causal (`prefix_attn: 730`).
  - Tokenizer: `moondream/starmie-v1`, a 51,200-token BPE. Letters are single tokens (`A`=53, `B`=54; `ĠA`=339, `ĠB`=384, and so on), so a Jev-style multi-token readout is possible *if* logits were accessible.
- **Gemma 4 also runs in kestrel** (its README lists E2B / E4B / 26B-A4B / 31B). This is only relevant as a possible fast server for the Gemma students; not investigated.

## 4. Torch path (`m87-labs/moondream`, `moondream/torch`)

The repo is Apache-2.0. Its last push was 2026-04-20, so it predates 3.1 (released 2026-07-07). It contains `moondream.py, text.py, vision.py, layers.py, region.py, rope.py, weights.py, lora.py, config.py, image_crops.py, sample.py, hf_moondream.py`.

- **Loader:** `weights._load_weights` already uses the Photon-style names (section 1). To load 3.1 you must (i) skip `kv_quantizer.*` and `moe_quant.*`; (ii) dequantise the 40 expert tensors with the per-row scales (section 1.1); and (iii) skip `region_model.ln.*`, or add it (kestrel's region has an LN, the preview torch code does not). Estimated effort 0.5 to 1 day for the loader. Total ~18.5 GB BF16.
- **Architecture differences between the torch code (preview) and kestrel (3.1) that must be reconciled by a logit-parity test against a CUDA kestrel run [UNVERIFIED].**
  - `max_context` 4096 versus 32768.
  - Tau gating uses exact `F.gelu` in torch and `F.gelu(..., approximate="tanh")` in kestrel.
  - Expert GeGLU is `gelu(h) * (g + 1)` in torch; kestrel's fused kernel is closed, so I cannot confirm 3.1 uses the same form.
  - The torch MoE forward is a Python loop over 64 experts per layer (differentiable, slow).
- **Data flow worth knowing:** `[BOS] + 729 image tokens` are encoded once with a bidirectional mask. Image-token hidden states therefore cannot see the prompt.
  - For CLM-LLM `image_mean` pooling this means the image states are prompt-independent, so one embedding per image serves all six fields.
  - The last-token state does depend on the prompt.
- **LoRA support:** `torch/text.py` accepts a `lora` dict with A/B pairs for `qkv`, `proj`, and the dense `fc1/fc2` MLP, which exist only in layers 0 to 3.
  - There is no LoRA on the MoE experts, no PEFT integration, and no training loop. `lora.py` only downloads *variants* from the Moondream API.
  - Layers are not `nn.Module` classes with standard names, so PEFT `target_modules` would need adapter shims.
  - The forward functions are autograd-compatible if you set `block.kv_cache = None`, use a plain boolean mask (not flex attention), and avoid `torch.inference_mode`. Treat this as [UNVERIFIED: not tried].
- **Training cost for Jev-LoRA** (my estimate): about 780 positions per sample (730 image prefix + 30 to 80 prompt), 2.2 B active params, so about 3.4 TFLOP forward, about 10 TFLOP forward + backward. `backbones.md` (Moondream row of the training table) already estimated about 3.3 h (1 crop) to 4.2 h (4 crops) per 1 M-image epoch in its first GPU column; the Python expert loop would run several times slower than that.

## 5. Teacher-by-teacher feasibility

### (a) CLM-vision with Moondream's tower — easy to load, medium to make useful, low benefit

Loading: the mapping in section 2.2, 0.5 day including a unit test that fails if `strict=True` does not match.

Missing pieces and how to fill them:
- **No MAP head.** Options: (i) copy SigLIP2's head and train it (and the label vectors) briefly on the frozen Moondream trunk; (ii) fresh-init a head (a small MAP block with a learned probe, or attention pooling).
- **No aligned text tower.**
  - Option 1, align SigLIP2's text vectors to the new tower. Freeze the Moondream trunk and the SigLIP2 text embeddings, and train a MAP head to reproduce SigLIP2's pooled image embeddings on unlabeled images (a feature-distillation stage, about 1 to 2 GPU-hours on 1 M images). Afterwards the head output lives in SigLIP2's text space and the existing prompt-embedding machinery works. The stage costs the zero-shot-exactness of WiSE-FT unless the aligned head becomes the new "initial" weights.
  - Option 2, prototype init. Compute `w_c` as the normalised class mean of head outputs on the training images and drop the text prior (rare classes with 1 to 3 images are the loss).
  - Option 3, a linear map. Fit a ridge map from Moondream pooled features to SigLIP2 pooled features on unlabeled images, then use SigLIP2 text vectors through that map. This is cheapest, but the 0.667 cosine suggests it will be lossy.
  - From scratch is the fallback; there is no reason to prefer it.

Expected benefit: low. The tower is SigLIP2 plus VLM fine-tuning for detection, pointing, counting, charts and documents (release notes). The medical part of MMBU is out of that distribution. My small probe shows no gain. Fine-tuning SigLIP2 directly already uses the aligned head and text tower. Possible value: a diverse second tower for the ensemble, or higher-resolution crops (the multi-crop pathway, which the 3-day estimate does not include).

Engineering: 0.5 day loading, 1 to 2 days head and label alignment, 1 day bake-off; about 3 days.

### (b) CLM-LLM with Moondream's LM hidden states — hard, low benefit

- Kestrel cannot return hidden states. The torch path is required: Photon-to-torch loader (1 day), a faithful forward including tau, prefix mask, multi-crop reconstruction via `image_crops.py` (1 to 2 days), logit-parity check against kestrel on CUDA (1 day), batching and throughput work (1 day), embedding extraction at `layer: 0.6` (24 layers: layer about 14) with `image_mean+last` pooling (0.5 day). About 5 to 7 days.
- The LM backbone is Moondream-2-lineage (kestrel's own config comment says Moondream 2 uses a Phi-1.5 text backbone; 3.x is drop-upcycled from it, so parent LM [UNVERIFIED]). The vocabulary is 51 k BPE. Medical word knowledge for label texts such as "mitotic figure" or "diabetic retinopathy grade 2" is weak compared with Gemma 4, so the action-side (label-text) embeddings are the likely bottleneck.
- Expected benefit: low; the arm already is expected to trail SigLIP2 (the module docstring cites a 10-point MIEB linear-probe gap for LLM-embedders).

### (c) Jev with Moondream 3.1 — hard, low benefit

- **Zero-shot readout via kestrel.** Prompt `image + options + "Answer with the letter"` is easy to build. A *probability* over letters is not available (section 3). Options: (1) patch `SamplingHooks.process_logits` to capture letter logits, about 2 to 3 days with a CUDA node and risk from closed kernels; (2) use sampled/greedy answers only (a hard label, no calibrated probabilities), which is useless as a soft-label teacher. Zero-shot quality is a concern: Moondream's query skill is trained on short free-form answers and counting/detection; its own marketing page on glaucoma staging reports 17.6% base accuracy on a 3-class task (below chance) rising to 69.2% only after 100 RL fine-tuning steps on their hosted service (vendor figure, not independently checked).
- **LoRA via the torch path** (section 4): 8 to 12 days (loader 1, differentiable forward and parity 3, LoRA wiring on attention and dense MLPs only 1 to 2, memory and throughput work 2, training/eval 2). Experts stay frozen in BF16 (16 GB), so LoRA only touches attention in 24 layers and the four dense MLPs. That is a weak adaptation surface compared with Gemma 4 E2B, where plain PEFT works.
- Expected benefit: low relative to Gemma 4 E2B/E4B Jev, which also matches the student family.

## 6. License: Moondream Model License 1.0 (effective 2026-07-07, M87 Labs)

Fetched from `https://moondream.ai/licenses/model/1.0`.

- **Grant (section 2):** perpetual, worldwide, royalty-free, non-exclusive, including commercial use. It explicitly covers fine-tuning, adapting, quantizing, pruning, converting, compressing and merging the weights, creating and distributing Derivative Models, running on owned or leased cloud/GPU infrastructure, and using Model Outputs for any purpose.
- **The single restriction (section 3):** you may not provide a general-purpose *hosted* inference or fine-tuning service for the model or a derivative to third parties. A domain-specific application, a customer-controlled deployment, or a free non-commercial evaluation interface is allowed. Our use (internal training and evaluation on our own cluster) is not hosting.
- **Derivative Model (section 17):** includes fine-tunes, LoRAs, adapters, weight deltas, quantisations, merges and conversions, and any artifact "initialized from" or "fine-tuned from" the weights. A fine-tuned CLM-vision tower is therefore a Derivative Model. A CLM-LLM that only reads frozen hidden states and trains separate heads does not modify the weights, so its heads are not derivatives (my reading, not legal advice). **Redistribution (section 7)** of a derivative needs a copy of the license, the notices, and a note of the modifications. We would not redistribute, so this mostly does not bite.
- **Model Outputs (sections 6 and 17):** outputs explicitly include embeddings, logits, scores, labels and text. They are not Model Materials; the licensor claims no ownership; outputs may be used for any purpose subject to law and third-party rights.
- **Training another model on outputs:** the definitions state that "A model trained solely on Model Outputs is not a Derivative Model..." unless it contains, is initialised from, fine-tuned from or merged with the weights. A Gemma student trained on Moondream labels or logits therefore is not a Moondream derivative and carries no Moondream license obligation. There is no no-train-competitor clause and no attribution requirement for outputs.
- **Other terms:** no use of the "Moondream" name or branding to imply endorsement (section 8, except truthful statements such as "built with Moondream"). The patent license terminates if you sue over patents on the model (section 9). Termination is on breach with a 30-day cure (section 11). Delaware law (section 15). Warranty disclaimed, and liability capped at the greater of fees paid in the last 12 months or USD 100 (section 13).
- **"The card excludes medical data":** I found no such statement in the HF README, the release notes, or the license text (grep for medical / health / clinical returned nothing in the license and release notes). [UNVERIFIED] The user-supplied claim may come from a different page or a prior version.
- **Not covered by the license:** the kestrel and `moondream` packages are separate software under their own terms. Kestrel's PyPI metadata gives no license field, its GPU kernels are closed binaries, and the GitHub torch repo is Apache-2.0.
- **MMBU rules:** keep the disclosure rule (rule 6 in `backbones.md`) in mind. The release notes say the base weights were distilled from per-benchmark LoRAs trained on public benchmark splits (COCO, SKU-110K, DOTA, etc.). Any contamination with MMBU data is unknown, and Moondream's pretraining data is undisclosed. Moondream is US-based; whether any of its training used Chinese-model outputs is [UNVERIFIED].

## 7. Evidence that it helps

- **Medical or fine-grained classification results for Moondream 3 / 3.1 or its encoder versus SigLIP: none found.** Searches (web, HF, the release notes) returned no published classification or medical benchmark for Moondream 3/3.1. The release notes report detection (COCO 81.5 F1@0.5, ODinW-13 93.9, DOTA-v2 59.6), counting (PixMo-Count 88.7, CountBench 90.4) and documents (ChartQA 86.0, DocVQA 88.6), plus throughput (34 req/s). No classification, no medical.
- Only vendor data point: the glaucoma-staging page (moondream.ai/ft/glaucoma-detection) reports 17.6% base accuracy, 69.2% after 100 RL steps on their hosted fine-tuning service (47 min, USD 15.68), and compares it with GPT-5.4 at 33.2%. The page does not say whether it is Moondream 3 or 3.1.
- **Encoder versus SigLIP:** no published comparison. My own small test (section 2.4) found no gain from the Moondream tower on a 1,025-image frozen probe, and a clear loss of SigLIP2 text-space alignment.
- Cross-check with `backbones.md`: it concluded "the useful part is available as plain SigLIP2 at lower cost". This investigation supports that statement and adds that the Moondream tower is *derived from* SigLIP2 (section 2.3).

## 8. Open items

1. [UNVERIFIED] Logit parity between the torch implementation (preview) and kestrel (3.1): tau GELU variant, GeGLU `(g+1)` form, any other 3.1 changes. Needs a CUDA run.
2. [UNVERIFIED] Whether `SamplingHooks.process_logits` can be patched from user code without the closed kernels interfering.
3. [UNVERIFIED] Whether kestrel can load a local (non-API) LoRA adapter.
4. [UNVERIFIED] The medical-data exclusion; Moondream's parent LM; any use of Chinese-model outputs in training.
5. [UNVERIFIED] Whether `model_fp8.pt` in the preview repo uses the Photon naming.
6. The frozen probe (section 2.4) is small. A proper grouped-by-dataset bake-off over the registry would take about a day if anyone still wants the tower as an ensemble member.

## 9. Sources

- Moondream 3.1 card and header: https://huggingface.co/moondream/moondream3.1-9B-A2B
- License: https://moondream.ai/licenses/model/1.0
- Release notes (2026-07-07): https://moondream.ai/blog/moondream-3-1-beyond-benchmarks
- Glaucoma fine-tune page: https://moondream.ai/ft/glaucoma-detection
- Torch code: https://github.com/m87-labs/moondream/tree/main/moondream/torch
- kestrel 0.9.1 and moondream 2.6.1 wheels: https://pypi.org/project/kestrel/0.9.1/ and https://pypi.org/project/moondream/2.6.1/
- SigLIP2 so400m patch14-384: https://huggingface.co/google/siglip2-so400m-patch14-384
- SigLIP so400m patch14-384: https://huggingface.co/google/siglip-so400m-patch14-384
- Moondream tokenizer: https://huggingface.co/moondream/starmie-v1
- Local: `mmbu-eval/servers/moondream/server.py`, `backbones.md`, `src/bio_waila/models/{clm_siglip,clm_llm,jev_gemma}.py`
