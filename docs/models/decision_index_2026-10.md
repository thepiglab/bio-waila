# Jev Decision Index and JevBench: fresh look, 2026-10-07, and what it means for the Pass-2 teacher

Date: 2026-10-07. Method: raw data files from the Space (`data/index.json`, `v03.json`, `vision.json`, snapshot **2026-10-07T18:25:58Z**, vision board 16:59Z, Space HEAD commit 2026-10-07T18:48Z), the JevBench site (benchmarkheaven.com, board v1.7.18 / release v1.6.1, page text parsed from the server-rendered HTML), the JevBench GitHub repo, and HF model cards read via the HF API. Anything not confirmed is marked [UNVERIFIED]. Builds on `jev_methods.md`, `jev_recipes_index.md` and `backbones.md` (all 2026-10-06); recipe cards for models already covered there are not repeated.

"Jev" here is TypeSafe AI's typed-decision model (context + options in, calibrated probabilities out). It is not JEPA.

## 0. Bottom line

1. **Two separate leaderboards exist, and the earlier notes' JevBench ranking is now stale.** (a) The HF **Jev Decision Index 0.3** (115 entries; Full score = 20% public + 50% private same-skill + 30% private new-domain). (b) **JevBench** (Benchmark Heaven, Florian Standhartinger; github.com/fstandhartinger/jevbench), now **v1.6.1** with a new headline "Capability Score" and a rotated item pool. The earlier "Cygnet = No. 1 of 106 on JevBench v1.5.4 (73.70)" is **obsolete**: on v1.6.1 Cygnet (frozen Gemma-4-12B-it) is Capability rank 9 (70.9, official composite #6). Rank 1 is now **Quyet-1.0-Large** (Gemma-4-31B-it, merged LoRA r16, letter readout), 81.7.
2. **Changes on the Decision Index since 2026-10-06:** exactly three rows added and no score changed: **Liquid AI d1-3B** (rank 51, Full 40.08, public 48.99), **Strands Decider 2B** (rank 82, 21.75) and **Liquid AI d1-omni-600M** (rank 97, 9.45). Ranks below 50 shifted by one. Calibration (ECE/Brier) is now shown for every entrant. The vision board gained **JEV-27B-VL** (Qwen3.8-27B, rank 1, 69.82). Top of the board is unchanged: Perplexity Decider v1.1 62.75 (Qwen3.8-27B), Fastino GLiDE 60.21, Jev 60.11 (tied), Torchcast-27B 59.91 (Qwen), deck31b 59.03 (frozen Gemma-4-31B-it).
3. **The best non-Chinese entries are all Gemma 4**: frozen or lightly tuned 31B (rank 5-9), 26B-A4B (rank 8-12; rank 8 Blink v0.3, rank 10 Rune, rank 11 GEV, rank 12 Xor) and 12B (rank 25-44). Everything at <=10B that is high on the board is Qwen3.5/3.6 (42 of 51 entries at 1-10.5B are Qwen-family, all banned for us).
4. **Small non-Chinese entries are weak but one is new and interesting:** the best <=4B total non-Chinese entry is now **Liquid d1-3B** (3.12B including its SigLIP2 vision tower; LFM2.5-VL-3B base; image-capable; Full 40.08, ECE 0.047). It beats Winnow-E4B (38.73), Lev1-3.7B (38.63), Jevstral-8B (35.93), Gevva-E4B (33.02). It is ~10 points below the best Gemma-4-12B LoRA (AJev 50.02) and ~9 below frozen 12B (47.47).
5. **No medical image content anywhere** in either board. Image benchmarks are UI screenshots, charts, receipts, spatial, counting, everyday photos. The only medical-adjacent items are NLI4CT (clinical-trial text NLI) and tiny-weight R-Bench-M / MMMU-Pro (some medicine questions). Medical Jev-style models exist but are **text only**: Docto Decision (French, MedGemma-1.5-4B / Gemma-4-E4B / Qwen), BioDecision-4B, BioJev, MedJev (mostly Qwen).
6. **Implication:** keep **Gemma-4-12B-it as the ≤12B image-capable teacher base** (best-evidenced non-Chinese option, within ~1 point of fine-tunes when frozen on JevBench), **Gemma-4-E4B for the prototype** (the only controlled head-to-head, Docto, has E4B ahead of Qwen3.5-4B, MedGemma-1.5-4B and E2B as a decision-model base), and **Gemma-4-26B-A4B for scale** (not 12B, if a >12B teacher is acceptable): it has the only strong image-decision evidence on a non-Chinese base (Image JevBench #2; Rune's own image test +5.3 over stock). **Liquid d1-3B/LFM2.5-VL-3B is the only credible alternative** and only for the Track-3-sized student role, not for the teacher. Ministral 3, Phi, Llama and SmolVLM have no competitive entries.

## 1. What changed since the earlier notes

| Item | Earlier notes (2026-10-06) | Now (2026-10-07) |
|---|---|---|
| Decision Index entries | 112 (111 + Jev, per our table) | 115 (114 + Jev); the +3 are d1-3B, d1-omni-600M, Strands Decider 2B |
| Any existing Decision Index score | n/a | **unchanged** (diff of all 111 old rows: zero changes) |
| Calibration columns | blank for many entrants | filled for all (commit "Calibration for all entrants"; 32 benchmarks, 1-in-6 request sample) |
| Vision board | Solomon v1.1 (Qwen3.8-27B) 67.66 was #1 | JEV-27B-VL (LoRA on Qwen3.8-27B) added at #1 (69.82); Solomon is now #2; 20 entrants |
| JevBench | "v1.5.4, 112 systems, 106 ranked, Cygnet No. 1 at 73.70" | **v1.6.1 (board v1.7.18, 2026-10-08 revision)**, 1,500 decisions per system (1,200 sealed + 300 public), 123 ranked open-weights systems; open-weights and API boards split (v1.7.0); headline is now **Capability Score = mean(Intelligence, Calibration)** among "Jev-class" systems (cost and median latency each <=2x Jev's) |
| JevBench leader | Cygnet (frozen Gemma-4-12B-it) | Quyet-1.0-Large (Gemma-4-31B) 81.7; deck-31B (frozen) 77.6; H2O-Lightning-4B (Qwen3.5-4B) 75.0; Rune 26B-A4B 73.5; Xor 26B-A4B 72.4; torchcast-decision-12b 71.7; Cygnet drops to 70.9 |
| Image benchmark | not covered | **Image JevBench v0.3.0** (2,441 items on 2,160 images; 54 systems measured); leader Imajev-4B (Qwen3.5-4B) 68.0 |
| New Liquid AI models | not covered | d1-3B (open, Oct 5), d1-omni-600M (open, Oct 5), a closed hosted "d1" on the JevBench API board |

## 2. Decision Index 0.3: current top 32 (Full score)

Source: `data/index.json` + `v03.json` (snapshot 2026-10-07T18:25Z). Rank is on Full; Jev, GLiDE and pplx-decider v1.1 are in a statistical tie (0.9-point band). Pub = public-benchmark skill, Priv-same = private same-skill tests, Priv-new = private new-domain tests. ECE/Brier = calibration block (32 benchmarks, confidence = probability on the chosen option). Params = `served_params` (including vision tower and, for Gemma E-models, embeddings). "Space label" is the Space's technique label, not a verified readout. V = also on the vision board. Origin in brackets: CN = Chinese-origin base, **ineligible for us**.

| Rank | Model | Full | Pub | Priv-same | Priv-new | ECE | Brier | Params | Base (origin) | Space label | V |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|:-:|
| 1 | Perplexity Decider v1.1 (27B) | 62.75 | 62.25 | 61.1 | 55.57 | 0.0713 | 0.3354 | 27.8B | Qwen/Qwen3.8-27B (CN) | full fine-tune | V |
| 2 | Fastino GLiDE no-thinking (28B) | 60.21 | 59.06 | 59.49 | 52.87 | 0.0398 | 0.34 | 27.8B | None (?) | full fine-tune |  |
| 3 | Jev 1.13.0 (TypeSafe, closed) | 60.11 | 57.96 | 58.04 | 54.98 | 0.074 | 0.3558 | ? | undisclosed (other) | closed API |  |
| 4 | Torchcast Decision 27B | 59.91 | 65.1 | 58.13 | 50.83 | 0.0235 | 0.2841 | 27.8B | Qwen/Qwen3.8-27B (CN) | LoRA |  |
| 5 | deck31b (Gemma 4 31B) | 59.03 | 58.19 | 58.6 | 51.48 | 0.2043 | 0.4691 | 32.7B | google/gemma-4-31B-it (US) | inference technique |  |
| 6 | Kev 27B | 58.78 | 56.69 | 55.51 | 55.49 | 0.0218 | 0.3591 | 27.8B | Qwen/Qwen3.8-27B (CN) | full fine-tune |  |
| 7 | Quyet-1.0-Large | 58.71 | 60.79 | 58.83 | 49.12 | 0.0778 | 0.3748 | 32.7B | google/gemma-4-31B-it (US) | LoRA |  |
| 8 | Blink v0.3 26B-A4B (NVFP4) | 57.76 | 58.21 | 57.21 | 50.08 | 0.0404 | 0.3391 | 25.8B | google/gemma-4-26B-A4B-it (US) | full fine-tune |  |
| 9 | Decider chat · Gemma-4-31B | 57.62 | 57.79 | 56.88 | 50.35 | 0.0471 | 0.3928 | 32.7B | google/gemma-4-31B (US) | inference technique |  |
| 10 | Surogate Rune 26B-A4B v3 | 57.43 | 58.28 | 57.09 | 49.38 | 0.1196 | 0.3704 | 25.8B | google/gemma-4-26B-A4B-it (US) | full fine-tune | V |
| 11 | GEV-26B-Decide | 56.17 | 58.03 | 54.99 | 49.02 | 0.0721 | 0.3527 | 25.8B | google/gemma-4-26B-A4B-it (US) | LoRA |  |
| 12 | Xor 26B-A4B | 55.95 | 56.3 | 56.54 | 47.37 | 0.0396 | 0.3678 | 25.8B | google/gemma-4-26B-A4B-it (US) | LoRA |  |
| 13 | vLLM-SR Decision 2.0 Vega 27B | 55.88 | 56.97 | 55.71 | 47.91 | 0.0861 | 0.3758 | 27.8B | Qwen/Qwen3.8-27B (CN) | LoRA + head |  |
| 14 | JEV-27B | 55.64 | 53.64 | 54.22 | 50.84 | 0.0759 | 0.3826 | 27.8B | Qwen/Qwen3.8-27B (CN) | full fine-tune |  |
| 15 | Jebadiah 27B | 55.63 | 55.11 | 55.58 | 48.36 | 0.014 | 0.3653 | 27.8B | Qwen/Qwen3.8-27B (CN) | LoRA |  |
| 16 | JADE | 55.56 | 53.42 | 54.29 | 50.66 | 0.2379 | 0.5098 | 27.8B | Qwen/Qwen3.8-27B (CN) | LoRA |  |
| 17 | Eikos-27B | 55.34 | 52.85 | 55.92 | 48.36 | 0.0574 | 0.4009 | 27.8B | Qwen/Qwen3.8-27B (CN) | LoRA |  |
| 18 | Bespoke Nimble 9B v3 | 54.67 | 57.19 | 52.34 | 49.05 | 0.1229 | 0.3705 | 9.7B | Qwen/Qwen3.5-9B (CN) | LoRA |  |
| 19 | Cloudflare clef | 53.08 | 61.71 | 53.11 | 41.88 | 0.0395 | 0.306 | 27.8B | Qwen/Qwen3.8-27B (CN) | full fine-tune |  |
| 20 | simple-jev · Qwen3.8-27B (featherless) | 51.79 | 56.05 | 55.46 | 38.56 | 0.113 | 0.4001 | 27.8B | Qwen/Qwen3.8-27B (CN) | inference technique |  |
| 21 | JPT-35B-A3B | 51.41 | 52.68 | 52.25 | 43.3 | 0.0491 | 0.3696 | 36.0B~ | Qwen/Qwen3.5-35B-A3B (CN) | LoRA |  |
| 22 | reflex 27B | 51.4 | 52.27 | 53.25 | 42.22 | 0.0242 | 0.3972 | 27.8B | Qwen/Qwen3.8-27B (CN) | inference technique |  |
| 23 | Decider chat · Qwen3.6-27B | 51.25 | 51.57 | 52.15 | 43.57 | 0.0209 | 0.3998 | 27.8B | Qwen/Qwen3.6-27B (CN) | inference technique |  |
| 24 | Xor 1.2 | 51.15 | 52.28 | 53.6 | 41.17 | 0.0236 | 0.3829 | 36.0B | Qwen/Qwen3.6-35B-A3B (CN) | LoRA | V |
| 25 | AJev lora5 (Gemma 4 12B) | 50.02 | 51.66 | 51.35 | 41.51 | 0.0932 | 0.4102 | 12.0B | google/gemma-4-12B-it (US) | LoRA |  |
| 26 | Winnow-12B | 49.27 | 50.66 | 48.14 | 44.16 | 0.1679 | 0.4862 | 12.0B | google/gemma-4-12B (US) | LoRA | V |
| 27 | Decider 35B-A3B | 48.53 | 47.7 | 45.12 | 47.57 | 0.0226 | 0.3878 | 36.0B | Qwen/Qwen3.5-35B-A3B-Base (CN) | full fine-tune |  |
| 28 | JevK5 9B v0.3.3 | 48.35 | 47.62 | 48.19 | 43.33 | 0.0598 | 0.414 | 9.7B | Qwen/Qwen3.5-9B (CN) | LoRA |  |
| 29 | vLLM-SR Decision 2.0 Lux 9B | 48.21 | 46.76 | 47.35 | 44.45 | 0.056 | 0.4216 | 9.7B | Qwen/Qwen3.5-9B (CN) | full fine-tune |  |
| 30 | decisio 0.4.0 (Qwen3.6-35B-A3B) | 47.72 | 48.43 | 49.15 | 40.14 | 0.0554 | 0.4192 | 36.0B | Qwen/Qwen3.6-35B-A3B (CN) | inference technique |  |
| 31 | Cloudflare clef-flash | 47.61 | 56.15 | 44.01 | 42.5 | 0.0249 | 0.3333 | 9.7B | Qwen/Qwen3.5-9B (CN) | full fine-tune |  |
| 32 | decisio 0.4.0 (Gemma 4 12B) | 47.47 | 49.72 | 48.44 | 39.79 | 0.1031 | 0.4554 | 12.0B | google/gemma-4-12B-it (US) | inference technique |  |

Reading the table:

- **Eligible (non-Chinese) entries in the top 32:** Gemma-4-based ranks 5, 7, 8, 9, 10, 11, 12, 25, 26, 32; and GLiDE (rank 2, closed API, base undisclosed, so not usable regardless). The first eligible *open weights* entry is deck31b at rank 5 (a recipe on stock `gemma-4-31B-it`, no new weights).
- Largest private/public gaps (overfitting to public benchmarks): Clef (public 61.71, private-new 41.88), Torchcast-27B (public 65.10, Full 59.91), Quyet-1.0-Large (60.79 vs 49.12 private-new). Gemma stock-readout entries have small gaps (deck31b 58.19 / 51.48).
- Calibration: best ECE among eligible top entries is Xor 26B-A4B (0.040), Blink v0.3 (0.040); deck31b has ECE 0.204 as shipped (T=3.4 taken from the Cygnet recipe is not tuned for this suite), so the **stock readout is accurate but needs its own temperature fit** (the Decider-chat entry on the same model has ECE 0.047 with option-count-dependent T).

## 3. Vision board (image-capable entrants)

Source: `data/vision.json` (2026-10-07T16:59Z). Full = 0.5 x public vision benchmarks + 0.5 x private tests of the same skills. 11 benchmarks: CV-Bench, BLINK, RealWorldQA, CharXiv, InfographicVQA, Mind2Web, Winoground, KIE (CORD+FUNSD), Moderation (Hateful Memes), R-Bench-M (0.5 weight), MMMU-Pro vision (0.25 weight). Stock references on the board: Qwen3.5-397B 63.44, Qwen3-VL-8B 53.66, SigLIP-2 6.17 (contrastive floor).

| V-rank | Model | Full | Pub | Priv | ECE | Params | Stock base (pub/priv) | Base origin | Caveat |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | JEV-27B-VL | 69.82 | 72.78 | 66.86 | 0.0187 | 27.8B | Qwen/Qwen3.8-27B (61.3/51.68) | CN | Ran through the author's own server (vLLM + POST /v1/decide, System 1 only) with the context cap lifted to 256 |
| 2 | Solomon v1.1 (27B) | 67.66 | 69.8 | 65.52 | 0.0758 | 27.8B | - | ? | Answers up to 8 options: MMMU-Pro rows with more options count as failures. |
| 3 | Xor 1.2 (35B-A3B) | 64.3 | 70.32 | 58.27 | 0.0084 | 36.0B | Qwen/Qwen3.6-35B-A3B (69.49/56.53) | CN | Environment rebuilt from the reference image (no lane recipe existed); same model code. |
| 4 | Perplexity Decider v1.1 (27B) | 63.2 | 67.0 | 59.4 | 0.0496 | 27.8B | Qwen/Qwen3.8-27B (61.3/51.68) | CN | Ran through the checkpoint's own decision model, which accepts images. |
| 5 | JPT-9B | 61.87 | 65.5 | 58.24 | 0.0749 | 9.7B | Qwen/Qwen3.5-9B-Base (63.47/54.09) | CN |  |
| 6 | Winnow-12B | 59.84 | 64.38 | 55.29 | 0.0788 | 12.0B | google/gemma-4-12B (18.73/11.9) | US | Its base answers at chance through the converted projector, so the base comparison is not meaningful. |
| 7 | JPT-4B | 59.65 | 64.83 | 54.47 | 0.0584 | 4.7B | Qwen/Qwen3.5-4B-Base (59.56/46.8) | CN |  |
| 8 | Surogate Rune 26B-A4B v3 | 58.63 | 65.27 | 51.99 | 0.0797 | 25.8B | google/gemma-4-26B-A4B-it (58.25/47.34) | US |  |
| 9 | Intern-Decision-4B | 56.89 | 61.83 | 51.95 | 0.0134 | 4.7B | Qwen/Qwen3.5-4B-Base (56.89/41.26) | CN | The vision patch embed was swapped for the identical linear map to run at normal speed; outputs agree to bf16  |
| 10 | RSI-Jev v5.0 VL 3B | 56.54 | 61.85 | 51.22 | 0.0252 | 4.7B | Qwen/Qwen3.5-4B-Base (29.06/23.1) | CN | Its base was read through the plain vision-language path, where a pretrained base answers letters poorly, so t |
| 11 | NeoHorse-Jev-4B | 50.73 | 57.21 | 44.25 | 0.0245 | 4.7B | - | ? | Takes a single image only: multi-image rows (BLINK, Winoground) count as failures. |
| 12 | Intern-Decision-2B | 48.36 | 53.29 | 43.42 | 0.0339 | 2.3B | Qwen/Qwen3.5-2B-Base (43.77/34.45) | CN | The vision patch embed was swapped for the identical linear map to run at normal speed; outputs agree to bf16  |
| 13 | Jev-Omni (12B) | 48.32 | 51.9 | 44.73 | 0.1279 | 12.0B | - | ? | Takes a single image only: multi-image rows (BLINK, Winoground) count as failures. |
| 14 | Winnow-E4B | 43.41 | 48.46 | 38.37 | 0.0478 | 8.0B | google/gemma-4-E4B (45.56/39.79) | US |  |
| 15 | JPT-0.8B | 42.85 | 44.72 | 40.98 | 0.0636 | 0.9B | Qwen/Qwen3.5-0.8B-Base (34.51/30.92) | CN |  |
| 16 | OneJev 0.8B | 41.61 | 46.84 | 36.39 | 0.0166 | 0.9B | Qwen/Qwen3.5-0.8B (35.08/29.74) | CN |  |
| 17 | djev (26B-A4B) | 36.86 | 44.51 | 29.21 | 0.1581 | 25.8B | - | ? | Ran with its published canvas 128 and 32K context, since the long-context engine takes no images. |
| 18 | Intern-Decision-0.8B | 36.86 | 38.78 | 34.95 | 0.0595 | 0.9B | Qwen/Qwen3.5-0.8B-Base (29.58/26.81) | CN | The vision patch embed was swapped for the identical linear map to run at normal speed; outputs agree to bf16  |
| 19 | razorback16 diffgemma (26B-A4B) | 36.36 | 43.91 | 28.8 | 0.1808 | 25.8B | - | ? |  |
| 20 | mmastrac diffgemma (26B-A4B) | 30.28 | 36.15 | 24.41 | 0.1611 | 25.8B | - | ? |  |

Observations (all computed from the file):

- **Eligible entries on the vision board:** Winnow-12B (59.84, rank 6; but its stock-base comparison is meaningless because the base "answers at chance through the converted projector"), Surogate Rune 26B-A4B v3 (58.63, rank 8; stock `gemma-4-26B-A4B-it` scores 58.25 public / 47.34 private, i.e. Full 52.8, so **the Rune fine-tune adds ~5.8 Full points**, mostly on the private half: 51.99 vs 47.34), Winnow-E4B (43.41, rank 14; stock `gemma-4-E4B` 45.56 / 39.79 = Full 42.7, so **LoRA adds only ~0.7**), Jev-Omni 12B (48.32, rank 13, single-image only, ECE 0.128), the three DiffusionGemma "inference technique" entries (30-37).
- Gevva-E4B and Gevva-E2B are **excluded** from the vision board ("official path takes no images"), although JevBench's Image board scored a separate "text checkpoint, image input" Gevva E4B (49.6).
- Fine-tuning a vision-capable Gemma-4 for letter readout without touching the tower yields small and uneven gains on images. Winnow, Rune, Xor, shisa-de-1 all say "vision tower untouched / frozen" (see section 7).

## 4. JevBench v1.6.1 (open-weights board) and Image JevBench v0.3.0

Source: benchmarkheaven.com/jev-models (parsed from the page HTML), plus the repo `fstandhartinger/jevbench`. JevBench measures 1,500 decisions per self-hosted system (1,200 sealed, 300 public); the board "ranks open-weights systems we ran ourselves on hardware we rent". Capability Score = mean(Intelligence [chance-corrected accuracy], Calibration). The "official" composite is a weighted harmonic mean of Intelligence, Calibration, Speed, Cost (equal weights, gates below 50); both ranks shown. "Jev-class" = <=2x Jev's cost and median latency; 88 systems qualify, 123 are ranked in total, and the 27B-class Qwen systems are listed as carried "v1.5 reference" rows outside the caps. **JevBench does not report parameter counts**, so size must come from the base name.

### 4.1 Open-weights Jev-class ranking, top 48 by Capability Score

| Cap rank | System | Capability | Intelligence | Calibration | Official # | Base | Origin |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | Quyet-1.0-Large | 81.7 | 73.4 | 90.0 | #2 | google/gemma-4-31B-it | US |
| 2 | deck-31B | 77.6 | 73.0 | 82.2 | #5 | google/gemma-4-31B-it | US |
| 3 | H2O-Lightning-4B v1.1 | 75.0 | 60.0 | 90.0 | #1 | Qwen/Qwen3.5-4B | CN |
| 4 | Surogate Rune 26B-A4B v3 | 73.5 | 56.1 | 90.9 | #13 | google/gemma-4-26B-A4B-it | US |
| 5 | Xor 26B-A4B | 72.4 | 58.7 | 86.1 | #10 | google/gemma-4-26B-A4B-it | US |
| 6 | torchcast-decision-12b | 71.7 | 60.5 | 82.9 | #3 | google/gemma-4-12B-it | US |
| 7 | Jev-Omni | 71.3 | 55.5 | 87.0 | #9 | google/gemma-4-12B-it | US |
| 8 | Winnow-12B Q8 | 71.2 | 59.5 | 83.0 | #4 | google/gemma-4-12B-it | US |
| 9 | Cygnet | 70.9 | 54.8 | 87.0 | #6 | google/gemma-4-12B-it | US |
| 10 | decisio v0.8.0 on gemma-4-12B-it | 70.7 | 52.3 | 89.2 | #7 | google/gemma-4-12B-it | US |
| 11 | decider chat on Gemma-4-31B-it | 70.6 | 58.5 | 82.6 | #11 | google/gemma-4-31B-it | US |
| 12 | GEV-26B-Decide | 70.1 | 49.8 | 90.3 | #14 | google/gemma-4-26B-A4B-it | US |
| 13 | decisio v0.9.0 on gemma-4-12B-it | 70.0 | 52.3 | 87.6 | #8 | google/gemma-4-12B-it | US |
| 14 | Hopper 12B trained | 68.1 | 51.7 | 84.6 | #12 | google/gemma-4-12B-it | US |
| 15 | SPX-CD-Omni | 68.1 | 48.8 | 87.5 | #15 | google/gemma-4-12B-it | US |
| 16 | Aplomb 1 | 67.5 | 45.8 | 89.2 | #17 | Qwen/Qwen3.5-4B | CN |
| 17 | Bespoke Nimble 9B v3 | 66.8 | 56.5 | 77.0 | #18 | Qwen/Qwen3.5-9B | CN |
| 18 | Plumb-4B | 65.4 | 43.0 | 87.9 | #20 | alibiserikbay/JevK5 | other |
| 19 | decider-4b v2 | 64.9 | 40.1 | 89.6 | #28 | Qwen/Qwen3.5-4B-Base | CN |
| 20 | JevK5 v0.3 | 64.2 | 38.5 | 89.9 | #30 | Qwen/Qwen3.5-4B | CN |
| 21 | Clef-Flash | 63.8 | 42.2 | 85.5 | #33 | Qwen/Qwen3.5-9B | CN |
| 22 | deck-4B v1.0 | 63.6 | 38.6 | 88.6 | #32 | alibiserikbay/JevK5 (built on Qwen/Qwen3.5-4B) | CN |
| 23 | janus 4B | 62.9 | 37.6 | 88.1 | #35 | Qwen/Qwen3.5-4B | CN |
| 24 | JevK5 v0.2.0 | 62.8 | 36.3 | 89.4 | #37 | Qwen3.5-4B | CN |
| 25 | Hopper | 62.4 | 34.7 | 90.1 | #41 | Qwen/Qwen3.5-4B | CN |
| 26 | Imajev-4B | 61.9 | 34.6 | 89.2 | #42 | Qwen/Qwen3.5-4B | CN |
| 27 | jqv | 61.9 | 33.9 | 89.9 | #44 | Qwen3-32B | CN |
| 28 | metask-jev-4b | 61.8 | 39.2 | 84.4 | #31 | Qwen3.5-4B | CN |
| 29 | Kahn1 4B | 61.7 | 39.7 | 83.8 | #39 | Qwen/Qwen3.5-4B | CN |
| 30 | lev | 61.7 | 41.1 | 82.3 | #25 | Qwen/Qwen3.5-4B | CN |
| 31 | Malkuth-4B | 61.1 | 33.8 | 88.4 | #43 | Qwen/Qwen3.5-4B-Base | CN |
| 32 | Decision 4B v1.2 | 60.4 | 32.4 | 88.4 | #49 | Qwen/Qwen3.5-4B | CN |
| 33 | Quyet-1.0-Medium | 59.2 | 41.7 | 76.8 | #24 | Qwen/Qwen3.5-4B | CN |
| 34 | Decision-4B | 59.1 | 37.9 | 80.3 | #34 | Qwen/Qwen3.5-4B | CN |
| 35 | Manchego v2.1 | 58.3 | 32.5 | 84.0 | #48 | Qwen/Qwen3.5-4B | CN |
| 36 | Messier One v0.2 | 58.1 | 42.4 | 73.9 | #21 | Qwen/Qwen3.5-4B | CN |
| 37 | Diffusion Jev | 58.1 | 54.9 | 61.3 | #16 | google/diffusiongemma-26B-A4B-it | US |
| 38 | jev-local | 57.5 | 41.9 | 73.1 | #27 | Qwen/Qwen3.5-9B | CN |
| 39 | Decision 4B v1.1 | 57.2 | 29.1 | 85.2 | #60 | Qwen/Qwen3.5-4B | CN |
| 40 | JEV Qwen3.5-9B Base NVFP4 | 57.1 | 29.3 | 85.0 | #68 | ig1/Qwen3.5-9B-NVFP4 | CN |
| 41 | CoCo-Decision-4B | 56.9 | 32.7 | 81.2 | #47 | Qwen/Qwen3.5-4B | CN |
| 42 | APUS-OpenJev-v1-9B | 56.9 | 47.2 | 66.6 | #19 | Qwen/Qwen3.5-9B | CN |
| 43 | typecastlm | 56.5 | 29.8 | 83.3 | #57 | Qwen3.5-4B | CN |
| 44 | TypeCastLM 1.4.0 | 56.4 | 35.4 | 77.4 | #40 | Qwen/Qwen3.5-4B | CN |
| 45 | SemIf | 55.4 | 27.1 | 83.6 | #69 | Qwen/Qwen3.5-4B | CN |
| 46 | local-jev Qwen3.5-4B | 55.0 | 24.1 | 85.8 | #72 | Qwen/Qwen3.5-4B | CN |
| 47 | Decision 2B | 54.8 | 22.9 | 86.8 | #75 | openbmb/MiniCPM5-2B | CN |
| 48 | spark-s1-4b-v6 | 54.0 | 43.2 | 64.8 | #22 | Qwen3.5-4B | CN |

Not on this board at all: Winnow-E4B, Gevva-E4B/E2B, Liquid d1-3B/d1-omni-600M (the open d1-3B has not been measured; the **hosted, closed "d1" (Liquid AI) is on the API board at Capability 74.4, Intelligence 62.1, Calibration 86.8, official Composite #2, $0.017/1k decisions; weights not public, and whether it is d1-3B is [UNVERIFIED]**). The API board also lists **"system-one-open" (Gemma 4 E2B LoRA on an L4): Capability 51.9, Intelligence 27.9, Calibration 75.9, latency 1.02 s**, and ClassOne Gemma 4 E2B (heads, no fine-tune of the LM) at 15.2. Controls: "Raw Phi-4 mini direct logits" Capability 35.3 (Intelligence 13.7), raw Qwen3-4B-Instruct 34.6.

What it says about our bases:

- Frozen Gemma-4-12B-it with a letter readout is within ~1 point of the 12B fine-tunes on JevBench (Cygnet 70.9, decisio v0.8.0 70.7 vs Winnow-12B 71.2, Jev-Omni 71.3, torchcast-12b 71.7). Intelligence is where fine-tuning helps (torchcast 60.5, Winnow 59.5, decisio frozen 52.3, Cygnet 54.8); calibration of the frozen ones is as good or better (89.2 for decisio).
- The 31B-to-12B gap is ~+10 Capability (Quyet 81.7 vs best 12B 71.7; frozen deck-31B 77.6 vs frozen Cygnet 70.9).
- 26B-A4B (4B active) lands between: Rune 73.5, Xor 72.4, GEV 70.1.
- A Qwen3.5-4B fine-tune (H2O-Lightning-4B, 75.0) beats every 12B Gemma on Capability: that is the "small Qwen" effect we cannot use.

### 4.2 Image JevBench v0.3.0 (2,441 items on 2,160 images; 300 public / 1,200 sealed draw; built 2026-10-07)

Tracks: UI screenshots, small text, spatial, diagrams, receipts, document photos, misleading charts, multi-panel comparison, counting, dense UIs, agent safety, chart reasoning, multilingual. Use cases: document checks, diagrams and planning, chart decisions, UI agent actions, counting and inspection. **No medical imaging category.** Jev-class ranking (28 systems):

| Rank | System | Capability | Intelligence | Calibration | Base | CN base? |
|---:|---|---:|---:|---:|---|:-:|
| 1 | Imajev-4B | 68.0 | 49.9 | 86.1 | Qwen/Qwen3.5-4B | CN |
| 2 | Surogate Rune 26B-A4B v3 | 64.8 | 44.8 | 84.8 | google/gemma-4-26B-A4B-it |  |
| 3 | AutoJev-27B | 62.1 | 37.6 | 86.7 | Qwen/Qwen3.8-27B | CN |
| 4 | Jevify Gemma 4 26B-A4B | 58.7 | 38.5 | 79.0 | google/gemma-4-26B-A4B-it |  |
| 5 | NeoHorse Jev 4B | 57.6 | 36.8 | 78.4 | NeoHorse-1-4B | CN |
| 6 | vjev-vision | 55.7 | 21.7 | 89.6 | Qwen/Qwen3.5-4B | CN |
| 7 | JPT-4B | 55.6 | 38.3 | 73.0 | Qwen3.5-4B | CN |
| 8 | Jev-Omni | 54.3 | 37.5 | 71.0 | Gemma 4 12B IT |  |
| 9 | Reflex 4B | 53.3 | 27.4 | 79.3 | Qwen/Qwen3.5-4B | CN |
| 10 | shisa-de-1 | 51.5 | 34.9 | 68.0 | Gemma 4 26B-A4B |  |
| 11 | Gevva E4B | 49.6 | 17.4 | 81.8 | Gemma 4 E4B |  |
| 12 | Gevva E2B multimodal | 47.3 | 12.8 | 81.9 | Gemma 4 E2B |  |
| 13 | OmniJev 4B | 47.2 | 28.0 | 66.4 | Qwen/Qwen3.5-4B | CN |
| 14 | imajev 2B | 46.4 | 19.5 | 73.3 | Qwen/Qwen3.5-2B | CN |
| 15 | JPT-0.8B | 46.1 | 20.4 | 71.7 | Qwen3.5-0.8B | CN |
| 16 | Jevify Qwen3-VL-2B T2 | 43.9 | 15.1 | 72.8 | Qwen/Qwen3-VL-2B-Instruct | CN |
| 17 | Jevify Gemma 4 E4B | 43.7 | 20.4 | 67.0 | google/gemma-4-E4B-it |  |
| 18 | Mapika decider-2b-vision BF16 | 42.8 | 23.1 | 62.6 | Qwen/Qwen3.5-2B-Base | CN |
| 19 | OmniJev 0.8B | 38.6 | 11.4 | 65.8 | Qwen/Qwen3.5-0.8B | CN |
| 20 | OmniJev 2B | 37.9 | 15.0 | 60.9 | Qwen/Qwen3.5-2B | CN |
| 21 | Jev-Vision 8B | 37.8 | 39.0 | 36.5 | Qwen/Qwen3-VL-8B-Instruct | CN |
| 22 | Visual-Jev 4B Answer-SFT | 35.9 | 24.0 | 47.7 | Qwen/Qwen3-VL-4B-Instruct | CN |
| 23 | OmniJev-Qwen3.5-9B-v4 | 33.1 | 29.0 | 37.3 | Qwen/Qwen3.5-9B | CN |
| 24 | Glance | 28.4 | 21.3 | 35.6 | Qwen3-VL-4B | CN |
| 25 | jev-spatial | 27.5 | 14.0 | 41.0 | allenai/Molmo2-ER |  |
| 26 | PlayJev 0.8B | 25.6 | 7.6 | 43.6 | Qwen/Qwen3.5-0.8B-Base | CN |
| 27 | djev-spark NVFP4 | 24.5 | 1.5 | 47.5 | google/diffusiongemma-26B-A4B-it |  |
| 28 | Qevi-2B | 11.1 | 2.7 | 19.6 | Qwen/Qwen3-VL-2B-Instruct | CN |

Not Jev-class or unranked but eligible and informative: Autoloops Gemma-4-31B-it (closed API) 68.3; **Winnow-12B (Q8 + F16 vision projector) 50.0** (Intelligence 38.6, Calibration 61.4, outside the cost cap); **Standard One 8B (Ministral-3-8B-Instruct, adapter) 51.5 and Standard One 3B (Ministral-3-3B) 49.6**, both "measured but unrankable" (no GPU-seconds from runtime). Reading: the best non-Chinese image decision models are Gemma-4-26B-A4B fine-tunes (64.8 / 58.7 / 51.5, spread caused by training recipe); at 12B Jev-Omni 54.3 (calibration 71.0 only); at E4B 43.7-49.6.

## 5. Small models

Counting convention: Params are the Space's `served_params` (vision tower and, for Gemma E-models, per-layer embeddings included). Gemma-4-E4B shows 8.0B total because of per-layer embeddings (4.5B effective); Gemma-4-E2B shows 5.1B total (about 2.4B compute-active per `backbones.md`).

### 5.1 Eligible (non-Chinese-origin) entries with 1-10.5B total parameters (all 9 on the Decision Index)

| Rank | Model | Full | Pub | ECE | Brier | Params | Base | Space label | V |
|---:|---|---:|---:|---:|---:|---:|---|---|:-:|
| 51 | LiquidAI d1-3B | 40.08 | 48.99 | 0.0468 | 0.3902 | 3.1B | LiquidAI/LFM2.5-VL-3B | full fine-tune |  |
| 55 | Winnow-E4B | 38.73 | 40.86 | 0.058 | 0.4613 | 8.0B | google/gemma-4-E4B | LoRA | V |
| 56 | Lev1 3.7B | 38.63 | 42.68 | 0.0578 | 0.4474 | 3.7B~ | IFM/K2-Horizon-3.7B | LoRA |  |
| 64 | Jevstral 8B | 35.93 | 34.01 | 0.0472 | 0.4948 | 8.0B~ | mistralai/Ministral-3-8B-Base-2512 | LoRA |  |
| 68 | Gevva E4B | 33.02 | 32.44 | 0.1451 | 0.5678 | 8.0B | google/gemma-4-E4B-it | full fine-tune |  |
| 74 | Gevva E2B | 28.18 | 28.53 | 0.0902 | 0.6003 | 5.1B | google/gemma-4-E2B-it | full fine-tune |  |
| 80 | EXAONE-4.0-1.2B-JEV v0.3 | 23.31 | 30.68 | 0.0794 | 0.5485 | 1.3B~ | LGAI-EXAONE/EXAONE-4.0-1.2B | full fine-tune |  |
| 95 | Kodiak v0.2 1B (accuracy) | 11.8 | 19.05 | 0.0773 | 0.606 | 3.0B | jhu-clsp/ettin-encoder-1b | full fine-tune |  |
| 105 | LFM2.5-2.6B-RLCD | 6.13 | 6.78 | 0.2545 | 0.7921 | 2.7B | LiquidAI/LFM2.5-2.6B-Base | full fine-tune |  |

Notes:

- **Liquid d1-3B** is the only <=4B-total entry that is both image-capable and non-Chinese. The model card (LiquidAI/d1-3B) reports Decision Index **0.2.1** = 48.57 (self-scored with the official scorer, "not a leaderboard submission"); the **0.3 Full on the Space is 40.08** (public 48.99, so the private halves drop it ~9 points). Its own 0.2.1 table puts Winnow-12B at 50.02 and Winnow-E4B at 39.89.
- Lev1-3.7B (K2-Horizon base, UAE) and EXAONE-1.2B (KR; the earlier notes record an NC-research-only licence) are text-only.
- Kodiak-1B is a text encoder (JHU Ettin); LFM2.5-2.6B-RLCD is *inference-only* parallel constrained decoding on unchanged Liquid weights (6.13), not a trained Jev model.
- Gemma-4-E4B/E2B entries are weaker than the Qwen3.5-4B ones: **Winnow-E4B 38.73 (LoRA r32/a64, letter) vs ezjev-4B 46.95 / vLLM-SR Nox 44.95 (Qwen3.5-4B)**. Gevva (full fine-tune with a cross-encoder classification head) is worse still (33.02 / 28.18).
- Nothing on the Decision Index is built on Phi, Llama, SmolVLM, Granite or Ministral-3-3B. Ministral appears only as Jevstral 8B (Ministral-3-8B-Base, LoRA, 35.93, text-only) and, on JevBench, Standard One 3B/8B.

### 5.2 Family dominance among 1-10.5B Decision Index entries (51 entries)

- Qwen family (Qwen3/3.5, incl. derived bases such as Mapika decider-2b): 42 of 51 (82%); other CN (MiniCPM, etc.) adds a few on JevBench.
- Gemma 4 (E2B, E4B): 3 (Winnow-E4B, Gevva-E4B, Gevva-E2B).
- Liquid: 2 trained (d1-3B, d1-omni-600M) + 2 inference-only/other (LFM2.5-2.6B-RLCD; lev-350m on JevBench).
- Mistral: 1 (Jevstral 8B). IFM K2: 1. EXAONE: 1. Encoders (JHU Ettin, ModernBERT, DeBERTa, mmBERT, GLiNER): the rest.
- Phi, Llama, SmolVLM: 0 trained entries.

### 5.3 Best Chinese-origin small entries (ineligible; listed for transparency)

Best <=10B overall are Qwen: Bespoke Nimble 9B v3 (54.67, rank 18; Qwen3.5-9B LoRA), JevK5 9B (48.35), vLLM-SR Lux 9B (48.21), Clef-flash (47.61), ezjev-4B (46.95, rank 35). On JevBench the top small system is H2O-Lightning-4B v1.1 (Qwen3.5-4B). First 25 Qwen/CN rows at <=10.5B:

| Rank | Model | Full | Pub | ECE | Brier | Params | Base | Space label | V |
|---:|---|---:|---:|---:|---:|---:|---|---|:-:|
| 18 | Bespoke Nimble 9B v3 | 54.67 | 57.19 | 0.1229 | 0.3705 | 9.7B | Qwen/Qwen3.5-9B | LoRA |  |
| 28 | JevK5 9B v0.3.3 | 48.35 | 47.62 | 0.0598 | 0.414 | 9.7B | Qwen/Qwen3.5-9B | LoRA |  |
| 29 | vLLM-SR Decision 2.0 Lux 9B | 48.21 | 46.76 | 0.056 | 0.4216 | 9.7B | Qwen/Qwen3.5-9B | full fine-tune |  |
| 31 | Cloudflare clef-flash | 47.61 | 56.15 | 0.0249 | 0.3333 | 9.7B | Qwen/Qwen3.5-9B | full fine-tune |  |
| 35 | ezjev 4B s2 | 46.95 | 50.82 | 0.063 | 0.4031 | 4.7B | Qwen/Qwen3.5-4B | LoRA |  |
| 37 | JEV-9B | 45.21 | 43.28 | 0.0726 | 0.4462 | 9.7B | Qwen/Qwen3.5-9B | full fine-tune |  |
| 38 | vLLM-SR Decision 2.0 Nox 4B | 44.95 | 44.21 | 0.0298 | 0.4358 | 4.2B | Qwen/Qwen3.5-4B-Base | LoRA + head |  |
| 39 | Sieve 9B | 44.59 | 41.88 | 0.0433 | 0.4409 | 9.7B | Qwen/Qwen3.5-9B | LoRA + head |  |
| 40 | JPT-9B | 44.45 | 46.96 | 0.0773 | 0.4346 | 9.7B | Qwen/Qwen3.5-9B-Base | LoRA | V |
| 41 | Kev 9B v2 | 43.27 | 41.54 | 0.0453 | 0.4506 | 9.7B | Qwen/Qwen3.5-9B-Base | LoRA + head |  |
| 42 | jiwo 4B | 42.86 | 45.76 | 0.0741 | 0.4331 | 4.7B | Qwen/Qwen3.5-4B | full fine-tune |  |
| 43 | Hopper (G) 1.2 | 42.61 | 41.15 | 0.0927 | 0.4672 | 4.7B | Qwen/Qwen3.5-4B-Base | LoRA |  |
| 46 | JPT-4B | 41.63 | 42.82 | 0.0793 | 0.4478 | 4.7B | Qwen/Qwen3.5-4B-Base | LoRA | V |
| 47 | JevK5 v0.3 | 41.54 | 41.14 | 0.0605 | 0.4587 | 4.7B | Qwen/Qwen3.5-4B | LoRA |  |
| 48 | Bespoke Nimble 9B v2 | 40.95 | 39.9 | 0.024 | 0.4671 | 9.7B | Qwen/Qwen3.5-9B-Base | LoRA |  |
| 50 | Decider 4B | 40.33 | 41.19 | 0.0837 | 0.4542 | 4.7B | Qwen/Qwen3.5-4B-Base | full fine-tune |  |
| 53 | Jet v6.2 | 40.01 | 42.17 | 0.1411 | 0.4793 | 4.7B | Qwen/Qwen3.5-4B-Base | LoRA |  |
| 54 | Kev 4B r10 | 39.5 | 37.95 | 0.032 | 0.4847 | 4.7B | Qwen/Qwen3.5-4B-Base | LoRA + head |  |
| 57 | RSI-Jev v5.0 VL 3B | 38.44 | 38.11 | 0.0478 | 0.4664 | 4.7B | Qwen/Qwen3.5-4B-Base | full fine-tune | V |
| 58 | Intern-Decision-4B | 38.21 | 38 | 0.0278 | 0.4737 | 4.7B | Qwen/Qwen3.5-4B-Base | full fine-tune | V |
| 59 | NeoHorse-Jev-4B | 37.97 | 37 | 0.1039 | 0.5134 | 4.7B | Qwen/Qwen3.5-4B-Base | head / adapter | V |
| 60 | Ryotide Qwen v0.3.1 | 36.99 | 37.28 | 0.0916 | 0.4852 | 4.0B | Qwen/Qwen3.5-4B | inference technique |  |
| 61 | Kas 4B | 36.7 | 38.35 | 0.0984 | 0.4632 | 4.0B | Qwen/Qwen3-4B | LoRA |  |
| 62 | lev | 36.53 | 39.2 | 0.0638 | 0.4905 | 4.7B | Qwen/Qwen3.5-4B-Base | LoRA + head |  |
| 65 | Intelif Qwen3-4B | 34.25 | 32.47 | 0.0708 | 0.5064 | 4.0B | Qwen/Qwen3-4B | LoRA + head |  |

## 6. Multimodal and medical content

**Multimodal tasks in the Index:** yes, but only as the separate **Vision board** (section 3). The main 0.3 index is 37 text benchmarks in five areas (Knowledge & Reasoning, Language, Retrieval & Classification, Tools & Automation, Arts & Human Taste), 110,201 requests. The vision board: CV-Bench, BLINK, RealWorldQA, CharXiv, InfographicVQA, Mind2Web, Winoground, KIE (CORD+FUNSD), Hateful Memes moderation, R-Bench-M, MMMU-Pro vision. Entries that use vision inputs on that board: 20 (list in section 3); 13 have Qwen-family bases and 7 are Gemma-4-based (Winnow-12B, Winnow-E4B, Jev-Omni, Rune, and three DiffusionGemma inference-only entries: djev, razorback16, mmastrac). Liquid d1-3B reports 11 public image benchmarks on its card (mean 74.1 vs base LFM2.5-VL-3B 73.9; without images the same questions score 45.1) but is not yet on the vision board [UNVERIFIED whether it will be added].

**Medical content:**

- Decision Index: NLI4CT (clinical-trial report entailment, text, 5,500 requests) and GPQA Diamond (bio/chem/physics). Vision board: R-Bench-M (medicine is one of several disciplines) and MMMU-Pro vision (includes Health & Medicine subjects), at 0.5 and 0.25 weight. **No radiology, pathology, dermatology, fundus or microscopy images.**
- Image JevBench: no medical track.
- Medical Jev-style models on the Hub (found by HF search, none on the Decision Index or JevBench boards):
  - **Docto Decision** (bofenghuang; French clinical text; LoRA r16/a32 on `google/medgemma-1.5-4b-it`, 943,013 decision records, target = 50% gold + 50% Qwen3.8-27B distribution). Text only, French only, "not clinically validated".
  - BioDecision-4B v2 (lighteternal; Qwen3.5-4B, research-only licence), BioJev-9B (Qwen3.5-9B-Base), MedJev (Qwen3.5-0.8B pointer head; Qwen3.5-4B LoRA), clinical-jev-qwen3.5-2b. All Qwen, all text.
- GEV-26B-Decide card (Gemma-4-26B-A4B LoRA) reports a MedMCQA slice (System 1 66.7%; with adaptive thinking 71.7%): text-only medical knowledge.
- **The Docto table is the only controlled base-model comparison for a medical decision task** (same data and recipe; mean accuracy over 12 tasks, then ECE):

| Base (after identical Jev-style LoRA) | Accuracy | ECE | Zero-shot (same base) |
|---|---:|---:|---:|
| Gemma-4-E4B | 90.13 | 7.79 | 80.74 (ECE 14.2) |
| Qwen3.5-4B | 88.94 | 7.91 | 77.79 |
| MedGemma-1.5-4B | 86.25 | 7.78 | 59.00 (ECE 23.5) |
| Gemma-4-E2B | 84.94 | 7.07 | 76.49 |
| Qwen3.5-2B | 84.91 | 7.25 | 60.12 |
| Qwen3.5-0.8B | 81.85 | 7.75 | 47.80 |

  The silver labels come from Qwen models, which may favour Qwen conventions; E4B still wins. Both zero-shot ECEs show why calibration (temperature or training) is required. **MedGemma's medical pretraining did not make it a better decision base than Gemma-4-E4B**, and its zero-shot letter-readout is poor (59.0) until tuned.

## 7. Recipes used by the top open entries (eligible bases), with links

Only new facts are given here; see `jev_recipes_index.md` for the rest. "Tower" = the vision tower is untouched unless stated. Temperature values are per-type unless stated.

| Entry (Full rank / JevBench cap rank) | Base | Training | Readout | Calibration | Image handling | Link |
|---|---|---|---|---|---|---|
| deck31b / deck-31B (5 / 2) | gemma-4-31B-it | none (frozen, FP8) | letter logits | T=3.4 (Cygnet recipe) | prompt-only | github.com/krishna-gogineni-765/deck31b |
| Quyet-1.0-Large (7 / 1) | gemma-4-31B-it | merged LoRA rank 16 | next-token letter logits, <=10 options, fixed prompt v2 | calibrated temperatures in `quyet_config.json` | not described | huggingface.co/chinhnc/Quyet-1.0-Large |
| Blink v0.3 26B-A4B NVFP4 (8 / n.m.) | gemma-4-26B-A4B-it | full fine-tune (v0.1-v0.3 iterations) | letter logits, one question per prompt, <=26 options, system prompt "reply with exactly one option letter" | T=1.3 (v0.2 wanted 1.9); ECE 0.016 over 216,942 decisions (author) | card tag "vision" | huggingface.co/PixilabAI/Blink-v0.3-26B-A4B-NVFP4 |
| Decider chat Gemma-4-31B (9 / 11) | gemma-4-31B | none (frozen) | letter | T(n)=max(Tmin, a+b ln n) | n/a | github.com/Mapika/decider |
| Surogate Rune v3 (10 / 4) | gemma-4-26B-A4B-it | full fine-tune (surogate engine) | letter logits ("surogate decisions v1") | T=2 recommended (ECE at T=1 is worse for text; images ECE 5.1-6.9% at T=1) | **vision tower kept; Image JevBench-style own test: 83.4% with 1,120 image tokens, 77.1% at default 280, stock Gemma-4-26B-A4B 71.8%; ECE 22.9% stock vs 5.6-6.9% Rune**; raising the image token budget gave +6.3 points | huggingface.co/surogate/rune-26b-a4b-GGUF |
| GEV-26B-Decide (11 / 12) | gemma-4-26B-A4B-it | LoRA (+ head per JevBench label) | System 1 letter distribution; optional adaptive thinking (below 0.8 confidence, think, then fold the reasoning-answer distribution in) | ECE 0.035 (author) | text and images (screenshots, robot-arm camera) | huggingface.co/autotrust/GEV-26B-Decide |
| Xor 26B-A4B (12 / 5) | gemma-4-26B-A4B-it | two LoRA lineages (attention/dense + router, then all 128 experts with tiered ranks 16/8/4), merged and weight-averaged "pulled back towards the base" | letter logprobs, thinking off | T=1.95 (choice), 1.0 (noul, score) | **rejects requests with images (HTTP 400)** | huggingface.co/juspay/Xor-26B-A4B |
| AJev lora5 (25 / n.m.) | gemma-4-12B-it | LoRA r32/a64, 131M trainable params; newer lora7 (12B, 55.06 on 0.2.1) and a 26B-A4B version (57.42 on 0.2.1) exist but are not on the 0.3 board | letter, choice up to 255 options | per-type temperatures in `ajev_lm_config.json`; ECE on typed decisions 0.280 (zero-shot) -> 0.142 | text | huggingface.co/andyzhang232/ajev-gemma4-12b-lora5 |
| torchcast-decision-12b (n.m. / 6, official #3) | gemma-4-12B-it | LoRA, **50/50 gold labels and a larger instruction model's option distributions**, procedurally generated decisions + Open-Jev train + public train sets; CC-BY-NC-4.0 weights | letter logprob at the answer position | per-type T fitted on public non-JevBench splits | text only | huggingface.co/torchcast-ai/torchcast-decision-12b |
| Winnow-12B / Winnow-E4B (26 / 8; 55 / n.m.) | gemma-4-12B-it / gemma-4-E4B-it | LoRA r32/a64, language tensors only, merged in FP32 to GGUF | letter via llama.cpp `/v1/systemone` | E4B Q8 T=1.257 (778 calibration questions), BF16 T=1.333; adaptive reasoning improved Brier 0.525 -> 0.481 | vision/audio modules frozen; F16 mmproj | huggingface.co/EldanRing/Winnow-E4B , /Winnow-12B |
| Jevify Gemma-4-E4B (n.m. / Image 17) | gemma-4-E4B-it | LoRA r64 on attention projections, 2 epochs, ~47k items from 16 sources, **loss = KL(target distribution || predicted label distribution)**, no teacher | next-token over labels | no post-hoc scaling: OOD ECE 0.043 on 307 items (author) | text-trained, image-scored on Image JevBench (43.7) | huggingface.co/kushalpatil/jevify-gemma4-e4b |
| Shisa DE-1 (n.m. / Image 10) | gemma-4-26B-A4B-it | softmax CE over the restricted option-letter distribution (the loss equals the deployment readout), 3 epochs | letter | not stated here | vision encoder untouched | huggingface.co/shisa-ai/shisa-de-1 |
| Jev-Omni (44 / 7) | gemma-4-12B | LoRA + classification head over hidden states, 30,000-question run, FP32 head | head, <=20 options advised | n/a | original Gemma 4 multimodal components (image, audio <=30 s, video 16 frames); ECE 0.161 | huggingface.co/akhilaaa3/Jev-Omni |
| Hopper 12B (34 / 14) | gemma-4-12B-it | LoRA r32, served unmerged | letter logits | one temperature per answer type; ablation of a richer calibration map: +0.1 | text | huggingface.co/HopitAI/hopper (branch `hopper-12b`) |
| **Liquid d1-3B** (51 / not measured) | LFM2.5-VL-3B (text backbone averaged with LFM2.5-2.6B) | **full fine-tune, multiple seeds and data mixtures then weight-merged; "long inputs, shuffled answer options, fixing shortcuts in the data mattered more than fancier techniques"** (Liquid blog); no LoRA stated | custom `system_one` head via `trust_remote_code`, zero output tokens, one forward pass, `noul`/`choice`/`score` types | not disclosed (ECE 0.047 on the Space) | SigLIP2 NaFlex 400M; base-parity on 11 image benchmarks (74.1 vs 73.9); images removed -> 45.1 | huggingface.co/LiquidAI/d1-3B |
| Liquid d1-omni-600M (97 / not measured) | LFM2.5-Encoder-350M (bidirectional) | full fine-tune on decision tasks; vision: adapter + LoRA on the backbone **active only with image input**, vision encoder (from LFM2.5-VL-450M) frozen; audio: FastConformer trained with frozen text backbone; final LoRA merge, weight-average with previous checkpoint | custom head | not disclosed | vision + audio | huggingface.co/LiquidAI/d1-omni-600M |
| Docto Decision MedGemma-1.5-4B / Gemma-4-E4B (not on the boards) | medgemma-1.5-4b-it / gemma-4-E4B | LoRA r16/a32, 1 epoch at 1e-4, 943,013 records, **target = half gold, half Qwen3.8-27B distribution** | single token letter / oui-non | ECE ~7-8% after tuning; chat template has `<bos>`, tokenise with `add_special_tokens=False` | text | huggingface.co/bofenghuang/docto-decision-medgemma-1.5-4b-fr-v0.1 |

Cross-cutting patterns (verified in cards above):

- **Frozen vs LoRA vs full FT:** on the stock Gemma family a *frozen* model + letter readout + temperature is already near the top (deck31b 59.03, decider-chat 57.62, decisio 47.47 at 12B). tuning adds little on the 0.3 Full scale: at 12B the best LoRA (AJev lora5 50.02) is +2.6 over the frozen decisio 47.47 and Hopper 12B (LoRA) equals it; at 31B the Quyet LoRA (58.71) is level with frozen deck31b (59.03); at 26B-A4B the full fine-tunes (Blink 57.76, Rune 57.43) are only ~1-2 above the LoRAs (GEV 56.17, Xor 55.95). On the vision board Rune is +5.8 over stock 26B-A4B but Winnow-E4B only +0.7 over stock E4B. Weight-averaging of separately tuned adapters or checkpoints (Xor, Liquid d1) is the newest trick.
- **Readout:** letter logits dominate all Gemma entries; Gevva (cross-encoder head, full FT) and Jev-Omni (classification head) are the weakest Gemma entries (33.0 / 42.4). Pointer heads and "readout heads" dominate the best Qwen 4B entries on JevBench (Aplomb, janus 4B pointer head r64, lev label-token + candidate path head).
- **Calibration:** per-type temperature is the universal trick (T≈1.3-4 depending on the model and tuning: Blink 1.3, Winnow-E4B 1.26, Xor 1.95, Rune 2, deck31b 3.4). Training with soft targets (gold + a teacher's distribution; KL/Brier) lowers the needed T. A bounded linear calibration map was worth only +0.1 over temperature alone (Hopper).
- **Teachers:** the best cheap recipe is half gold labels, half a larger model's option distribution (torchcast-12b, Docto, Perplexity). This is directly our Jev-teacher distillation plan.

## 8. Implications for the Pass-2 teacher on biomedical images

Evidence quality warning: **neither leaderboard measures biomedical image classification**. They rank bases on text decisions and on UI/chart/document images, so everything below is transfer evidence for (i) the readout and calibration mechanism and (ii) which Gemma/Liquid model tolerates the recipe, not for medical perception. The `backbones.md` conclusion (a frozen-feature bake-off on grouped CV before committing, and adapting the tower to medical modalities) is untouched by this new data.

1. **Best <=12B, non-Chinese, image-capable base for a Jev-style teacher: Gemma-4-12B-it** (not the pretrain `gemma-4-12B`, because every high 12B entry uses `-it`).
   - Best eligible 12B results: AJev lora5 50.02, Winnow-12B 49.27, decisio frozen 47.47, Hopper 47.27 on the Decision Index; JevBench Capability 70.9-71.7 for Cygnet/decisio/Winnow/torchcast/Jev-Omni; Image JevBench Jev-Omni 54.3.
   - Caveats: the 12B is the "encoder-free" Gemma 4 (48x48x3 patches through one 35M matmul; per `backbones.md`), so there is no tower to adapt separately; Winnow-12B's own vision comparison is invalid (base at chance through the converted projector); Jev-Omni's head recipe is weak at calibration (ECE 0.16).
   - Do **not** expect the leaderboard gap between 12B and E4B (Winnow 49.27 vs 38.73 = +10.5 Full) to carry over fully: on the *vision board* the Winnow-E4B LoRA added only ~0.7 over stock E4B (43.41 vs 42.7).
2. **Does the evidence still favour Gemma-4-E4B for the prototype and 12B / 26B-A4B for scale? Yes, with two refinements.**
   - E4B: the Docto controlled comparison puts Gemma-4-E4B above Qwen3.5-4B, MedGemma-1.5-4B and Gemma-4-E2B as a Jev-style LoRA base (90.1 / 88.9 / 86.3 / 84.9), E4B is ~2.2x faster than 12B at 64 questions per request on one RTX 5070 Ti (147.8 vs 67.2 decisions/s, author-measured for Winnow), and stock Gemma-4 benefits from a letter readout without any training. Its Decision Index entries are mediocre (38.7 Winnow-E4B, 33.0 Gevva), but those recipes are generic and our task is a narrow supervised one.
   - E4B is **8.0B total, 4.5B effective**, so it is a teacher/prototype base only; it would not fit Track 3 (E2B, 2.4B compute-active, would; see `backbones.md`).
   - 26B-A4B beats 12B at similar active compute: Decision Index 56-57.8 (Blink, Rune, GEV, Xor) vs 47-50 for 12B; JevBench 70.1-73.5 vs 70.7-71.7; it has the **only strong non-Chinese image-decision evidence** (Rune: Image JevBench #2; own 1,196-item image test, stock 71.8% -> Rune 77.1% at default tokens and 83.4% at 4x image tokens, and ECE from 22.9% stock to 5.6-6.9%). If the teacher may exceed 12B total parameters (the 26B-A4B is 25.8B total, 3.8B active), prefer it for scale; at <=12B stay with Gemma-4-12B-it. Raising the image token budget to 1,120 matters for dense or small-detail images (our microscopy and pathology cases) and costs 0.46 s vs 0.18 s per decision.
3. **Is there a stronger choice (Liquid, Ministral 3, Phi)? Not for the teacher.**
   - **Liquid d1-3B / LFM2.5-VL-3B:** the only <=4B-total image-capable Jev-style model (3.12B, SigLIP2 NaFlex 400M tower, 8 ms/decision on an RTX 4090). Decision Index Full 40.08 beats every E4B/E2B entry and all non-Chinese <=10B entries; 11-image-benchmark parity with its base (74.1 vs 73.9). But: (a) it is ~10 Full points below the best 12B Gemma; (b) the training recipe is only described qualitatively (full FT + weight merging), no code; (c) it uses a custom `system_one` head via `trust_remote_code` (transformers >=5.14); (d) the **LFM Open License v1.0** allows commercial use only below USD 10M annual revenue and exempts "Qualified Non-Profit" research use; whether UF qualifies is [UNVERIFIED] and needs a licence check before any submission; (e) it is not on the vision board or JevBench open board yet [UNVERIFIED if/when added]. It is the best candidate for a **Track-3-sized student**, not a teacher; run it as a baseline arm.
   - **Ministral 3:** Jevstral 8B (Ministral-3-8B-Base LoRA, Kev recipe, text-only) 35.93 at 8B is below 4B-class Winnow-E4B (38.73). Standard One 8B/3B (adapter on Ministral-3 Instruct) are on JevBench (8B Capability 58.4, outside the cost cap) and Image JevBench (51.5 / 49.6, unrankable). Ministral-3 has a Pixtral-style frozen tower and native-resolution images (per `backbones.md`), but no evidence beats Gemma. Keep as an ablation only.
   - **Phi, Llama, SmolVLM, Granite:** no trained entries. The only Phi data point is a raw Phi-4-mini logit control (Capability 35.3, Intelligence 13.7).
   - **MedGemma-1.5-4B:** its Gemma-3 base with medical pretraining lost to Gemma-4-E4B and Qwen3.5-4B after identical Jev tuning (Docto, text only) and starts at 59.0 zero-shot, so it is not a better Jev base on this evidence; it may still help for medical *perception* (MedSigLIP tower), which the Index does not measure.
4. **What to copy:** Cygnet/deck31b letter readout with sum over all vocabulary tokens that decode to each letter, thinking off, option-count-dependent temperature fitted on our own held-out data; add a light LoRA (r16-r32, language tensors only, tower frozen at first); train on 50% gold, 50% teacher distribution with KL or Brier; apply Mapika-style option-count temperature because body-part has ~100 options (letters cap at 26 without two-letter codes; JevBench evidence for >26 options is thin).
5. **What not to take from the boards:** do not rank bases for medical images on Decision Index Full, since the private halves reward text skills; the public/private gap is large for several top entries (Clef, Torchcast-27B, Quyet). Treat 1-point differences as noise (the Space's tie band is 0.9; JevBench itself reports wide intervals for small samples, e.g. 51.6-64.6 for one API row on 600 items).

## 9. Unverified and open items

- Whether Liquid's hosted "d1" on the JevBench API board is d1-3B: [UNVERIFIED]. The card does not say.
- Whether d1-3B / d1-omni-600M will be added to the Space's vision board or JevBench's open board: [UNVERIFIED].
- UF eligibility under the LFM Open License v1.0 non-profit exemption: [UNVERIFIED].
- Readout details for Perplexity, Kev-27B, Torchcast-27B, GLiDE: from earlier notes and the Space's labels; not re-read today.
- GEV-26B-Decide's "head" component: from the JevBench label, not from its card [UNVERIFIED].
- JevBench numbers were parsed from the rendered page of the site; the repo's JSON for v1.6.1 was not downloaded, so decimals could differ from the aggregate JSON (repo has results only up to v1.4.2.2 in the visible tree).
- The Decision Index README says the private tests are "never published"; reproduction of private numbers is impossible, and several authors self-report 0.2.1 scores that are not on the 0.3 scale.
- Spanish/French/Japanese multilingual performance (Image JevBench multilingual track, Docto French) was not examined.

## Sources

- Space: https://huggingface.co/spaces/multimodalart/jev-decision-index (data/index.json, v03.json, vision.json; README; commit log via `https://huggingface.co/api/spaces/multimodalart/jev-decision-index/commits/main`)
- JevBench: https://benchmarkheaven.com/jev-models , https://benchmarkheaven.com/jev-models/api , https://benchmarkheaven.com/image-jev-bench , https://github.com/fstandhartinger/jevbench
- Model cards: https://huggingface.co/LiquidAI/d1-3B , https://huggingface.co/LiquidAI/d1-omni-600M , https://www.liquid.ai/blog/d1-open , https://huggingface.co/LiquidAI/LFM2.5-VL-3B (LFM Open License v1.0 text at /LiquidAI/d1-3B/blob/main/LICENSE), https://huggingface.co/EldanRing/Winnow-E4B , https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora5 , https://huggingface.co/torchcast-ai/torchcast-decision-12b , https://huggingface.co/chinhnc/Quyet-1.0-Large , https://huggingface.co/PixilabAI/Blink-v0.3-26B-A4B-NVFP4 , https://huggingface.co/surogate/rune-26b-a4b-GGUF , https://huggingface.co/juspay/Xor-26B-A4B , https://huggingface.co/autotrust/GEV-26B-Decide , https://huggingface.co/shisa-ai/shisa-de-1 , https://huggingface.co/kushalpatil/jevify-gemma4-e4b , https://huggingface.co/akhilaaa3/Jev-Omni , https://huggingface.co/HopitAI/hopper , https://huggingface.co/StandardThinking/StandardOne-3B , https://huggingface.co/bofenghuang/docto-decision-medgemma-1.5-4b-fr-v0.1 , https://huggingface.co/lighteternal/biodecision-v2-4b , https://huggingface.co/Gabriel382/BioJev-9B , https://huggingface.co/junma/MedJev-Qwen3.5-0.8B , https://huggingface.co/helloimsaif/medjev-4b-lora
