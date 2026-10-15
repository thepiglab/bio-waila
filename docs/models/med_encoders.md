# Medically specialised vision encoders for the Pass-2 labeler (CLM-style) or a Jev-style vision tower

*Research date: 2026-10-07. Builds on `backbones.md` (general SigLIP2 / DINOv3 / Gemma 4 / Ministral) and `clm_methods.md` (cosine-classifier recipe). I do not repeat those. Verified from primary sources today: HF Hub API (license, gating, safetensors totals, last-modified), raw model cards and `config.json` where public, arXiv PDFs read as text (MedGemma report 2507.05201, MedGemma 1.5 report 2604.05081, Gemma 4 report 2607.02770, MedImageInsight 2410.06542, BIOMEDICA 2501.07171, MedPMC 2607.07673, ConceptCLIP 2501.15579, MMKD-CLIP 2506.22567), and the HAI-DEF terms page. Gated HF cards (UNI2, Virchow2, H-optimus-1, GigaPath, Hibou, CONCH, MedSigLIP's `config.json`) cannot be read without login; for those, license and gating come from the API and architecture/data facts from the papers, search summaries or memory, and are marked. `[UNVERIFIED]` = no primary source read. `[ESTIMATE]` = my arithmetic. Hard constraint applied throughout: no Chinese-origin models and no models built on them; origin is flagged on every row.*

## 0. Bottom line

1. **No single medical encoder covers our label space.** The best-documented broad ones were trained on a radiology / dermatology / ophthalmology / histopathology mix and say nothing about endoscopy, EM, fluorescence or cell microscopy:
   - MedSigLIP: 98.1% of its medical pairs are histopathology patches (32.55M of 33.2M), and its test set is CXR, derm, fundus and histopathology only.
   - MedImageInsight: 14 domains, X-ray, CT, MRI, mammography, ultrasound, derm, clinical photos, histology, OCT, fundus. Nothing in endoscopy or microscopy.
   - UniMed-CLIP: six modalities (X-ray, CT, MRI, ultrasound, pathology, fundus).
   - Only the PubMed Central (PMC) figure-trained CLIPs (BiomedCLIP, BMC-CLIP, MedPMC-CLIP, Open-PMC) see endoscopy, EM and fluorescence, but they are at 224 px and are CLIP-class, not SigLIP-class.
2. **Evidence that medical specialisation beats a strong general encoder after fine-tuning is thin.** What exists is mixed:
   - u-Bench (Stanford, microscopy): biomedical CLIPs are often worse than generalists.
   - ConceptCLIP (a Chinese-origin SigLIP-400M continued-pretrain, so excluded; its numbers are evidence only): general SigLIP-400M is within 0.1-0.6 AUC of it after full fine-tuning on endoscopy, derm and pathology.
   - Pelvic-MRI series benchmark (MICCAI 2026): after end-to-end training all four models (PubMedCLIP, BiomedCLIP, MedSigLIP, SigLIP-2) reach at least 93% sequence accuracy; MedSigLIP 95.1% sequence accuracy, SigLIP-2 90.5% orientation accuracy (different metrics, so no direct ranking).
   - The MedSigLIP advantage is largest frozen or few-shot, and for rare classes. That is where our tail is.
3. **MedSigLIP cannot be dropped into Gemma 4.** Gemma 4's tower is a different design (RMSNorm, no attention bias, 2-D RoPE plus an absolute position table, patch 16). Layer shapes match the 550M tower (27 layers, 1152-d, 4304 MLP, 16 heads) but weights do not transfer. MedSigLIP is a **teacher only**, or a tower swap that makes the whole student an HAI-DEF Model Derivative (section 3).
4. **Pathology specialists are narrow (H&E/IHC patches).** They are worth testing only as an add-on expert for stain/specimen, not as a backbone.
5. **Recommendation (detail in section 7).** Keep **SigLIP2-So400m as the primary teacher backbone**. Bake off **MedSigLIP-448, BMC-CLIP-CF (ViT-L/14), MedPMC-CLIP (ViT-L/14), MedImageInsight, and UNI2-h (or Virchow2) as a pathology add-on** on grouped-by-dataset CV, with per-slice reporting. Expect a mixture (SigLIP2 plus MedSigLIP, optionally plus a PMC-CLIP for microscopy/EM/endoscopy) to beat any single encoder. A single medical encoder is unlikely to win on every slice.

## 1. Broad (multi-modality) medical encoders

Columns: org / country, params and architecture, input resolution, training data, text tower (needed for label-text-initialised classifiers), license and gating, HF availability, best evidence, fit.

| Model | Org / origin | Params / arch | Res | Training data (modalities) | Text tower | License / gating | Availability | Best evidence | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **MedSigLIP-448** | Google, US | 878.3M total (HF safetensors); vision about 400M (SigLIP-400M-class, derived from the Gemma 3 tower, which the MedGemma report calls "SigLIP-400M", **not SigLIP 2**) plus about 400M text | 448 px, patch 14 (1,024 tokens); text 64 tokens | 33M+ medical pairs: 635k non-histopathology (MIMIC-CXR 231k, CT slices 60k, MRI slices 47.6k, knee X-ray 1.5k, EyePACS 199k fundus, internal derm 51k, PAD-UFES-20 2k, PMC single-panel 42k) plus **32.55M internal histopathology patch-caption pairs**. Medical data mixed at **2% weight** with the original WebLI to keep general ability | **Yes** (SigLIP text tower, jointly trained) | HAI-DEF terms (`other`), **auto-gated** (instant approval after accepting terms) | `transformers` `SiglipModel`; no open_clip | Zero-shot AUC: CXR 13 findings avg 0.844 (+2.0 over ELIXR/CXR Foundation); derm 79-class 0.851 (LP 0.881); DR 5-class 0.759 (LP 0.857); path avg of 10 tasks 0.870 (LP 0.878). Frozen MedSigLIP beats MedGemma-4B on derm by 10.26 pp (2609.36557) | **Best medical SigLIP.** Same code path as SigLIP2. Coverage gap on endoscopy, US, EM, fluorescence is the open risk |
| **MedImageInsight (MI2)** | Microsoft Health & Life Sciences, US (with U. Wisconsin, U. Washington) | DaViT image encoder 360M + UniCL language encoder 252M; Florence-derived | 480 px (training images stored at 512; report-generation setting uses 480) | 14 domains: CXR (MIMIC-CXR 203k, NIH-CXR-LT 68k), IRMA X-ray, bone age, mammography, chest CT, MR/CT exam parameters, breast US, echo parameters, dermoscopy (ISIC2019), clinical photos (SD198), histopathology (PatchCamelyon 262k), OCT, fundus. About 1,000 text labels | **Yes** (classifier head generated from text) | Paper calls it open-source; community mirror tagged MIT. Official weights via Azure AI Foundry catalog. Exact license text [UNVERIFIED] | Official: Azure only. Community mirror `lion-ai/MedImageInsights` (MIT, ungated, custom code, not a `transformers`/open_clip model) | SOTA or equivalent on CT retrieval, CXR, derm, OCT; mAUC above 0.9 on most tasks; beats BiomedCLIP, Med-Flamingo, RAD-DINO on cross-model retrieval | Good on radiology/derm/OCT/fundus/breast-US. **Nothing on endoscopy, cell microscopy, EM**. Non-standard model code. Second-tier |
| **UniMed-CLIP** | MBZUAI, UAE (with EPFL, ANU, Linkoping) | ViT-B/16 (MetaCLIP init) + BiomedBERT; also ViT-L/14 at 336 | 224 / 336 | UniMed 5.3M pairs, 6 modalities (X-ray, CT, MRI, US, pathology, fundus). Classification datasets turned into captions with an LLM from (category, anatomy, modality) triplets | Yes | Repo license NOASSERTION; HF upload `UzairK/unimed-clip-vit-b16` tagged **CC-BY-NC-4.0**, ungated. Larger variants on OneDrive only | open_clip (HF `.pt`) | +12.61 absolute over BiomedCLIP, averaged over 21 zero-shot datasets (avg 61.63 vs 49.02). L/14-336 62.09 to 64.84 | Method is exactly our situation (labels to captions). Six modalities only. Cheap control |
| **BiomedCLIP** | Microsoft Research, US | ViT-B/16 (timm) + PubMedBERT-256, about 150M total | 224 | PMC-15M figure-caption pairs (4.4M papers; radiology, pathology, microscopy, charts) | Yes | **MIT**, ungated | **open_clip**, HF native | 21-dataset zero-shot avg 49.02 (UniMed table); over-predicts labels on imbalanced data (clm_methods) | Cheap baseline. Weak features at 224 px |
| **BMC-CLIP (BIOMEDICA, "CF" concept-filtered 6M)** | Stanford (Yeung-Levy lab), US | OpenCLIP ViT-L/14 (CommonPool-XL init) continued-pretrained, context 77 | 224 | BIOMEDICA: 24M PMC image-caption pairs from 6M articles; concept-filtered 6M subset keeps clinical + scientific imaging (48% clinical imaging and microscopy). 12 global, 170 local concepts incl. EM/fluorescence/endoscopy | Yes | **MIT**, ungated | HF `BIOMEDICA/BMC_CLIP_CF` (`.pt`, open_clip-compatible); GitHub `minwoosun/biomedica-etl` (MIT); BMC-LongCLIP (512-token text) | 40 tasks, zero-shot +6.56% avg over prior SOTA (CVPR 2025 paper); vs BiomedCLIP: derm +29.8, ophthalmology +17.5, breast US +8.0, non-neoplastic histopath +7.0; **microscopy composition (light / fluorescence / EM) marginally better, +8.16 with WiSE-FT** | **Best-documented PMC CLIP for our long-tail microscopy / EM / endoscopy.** Check the unusual HF packaging |
| **MedPMC-CLIP** | Yale (BIDS Chen lab), US | OpenCLIP ViT-L/14, same init and recipe as BMC-CLIP | 224 | MedPMC 11M PMC pairs after screening and panel separation (95.3% medically relevant). Microscopy non-pathology 31.9% of the mix, endoscopy 1.4%, pathology 24.8%, clinical photos 5.6%, derm, EEG/ECG | Yes | **CC BY-NC-SA 4.0**, ungated | HF `Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1` (open_clip, safetensors) | Self-reported: +7.1 pp AUC over BMC-CLIP on 26 benchmarks / 11 specialties; baselines also include MedSigLIP and BiomedCLIP. Swapped into LLaVA-Med it lifts OmniMedVQA modality recognition 46.9 to 75.2 | Newest (July 2026), unreplicated. Non-commercial is fine for a teacher. Worth a slot |
| **Open-PMC-18M CLIP** | Vector Institute, Canada | ViT-B/16 + PubMedBERT-256 (BiomedCLIP architecture), 512-d | 224 | Open-PMC-18M (18M PMC pairs, panel-split) | Yes | **Apache-2.0**, ungated | open_clip (`hf-hub:vector-institute/open-pmc-18m-clip`) | MICCAI 2026 paper (arXiv 2506.02738); numbers not read | Alternative to BiomedCLIP; same size class. Low priority |
| **MMKD-CLIP** | Emory, US | MetaCLIP ViT-B/16 + BiomedBERT, 224 | 224 | 2.9M PMC-OA pairs (26 modalities) then feature distillation from nine CLIP teachers (BMCA, GenMedCLIP, MedCLIP, BiomedCLIP, UniMed, PLIP, PubMedCLIP, QuiltNet, **PMC-CLIP**) | Yes | Weights at GitHub `wangshansong1/MMKD-CLIP`, license [UNVERIFIED] | GitHub | 58 datasets, outperforms its teachers | **Exclude.** PMC-CLIP is a Shanghai-origin teacher [origin from memory, UNVERIFIED], so the student is "built on" a banned lineage |
| **MerMED-FM** | A*STAR / SERI (Ting lab), Singapore | self-supervised ViT with memory module [arch UNVERIFIED] | [UNVERIFIED] | 3.3M images, 10+ specialties, 7 modalities: CT, CXR, US, pathology patches, CFP, OCT, derm | No (SSL only) | HF `youngzhou12/MerMED` gated auto; license [UNVERIFIED] | HF | Preprint only; I did not read the tables | Image-only; no text prior. Watch, do not bake off |
| **LLaVA-Med v1.5 tower** | Microsoft, US (LM: Mistral-7B, France) | OpenAI CLIP ViT-L/14-336 (`config.json` `mm_vision_tower`), general, not medically retrained | 336 | n/a (tower unchanged; LM tuned on PMC-15M captions) | (CLIP) | Apache-2.0 (card says Mistral license); research use only | HF | n/a | **Not a medical encoder.** Included only because it was listed. MedPMC-CLIP swapped in for this tower is the real result |
| Excluded, wrong origin (evidence only) | **ConceptCLIP**: HKUST + Tencent YouTu (China). **PMC-CLIP**: Shanghai Jiao Tong / Shanghai AI Lab [origin from memory, UNVERIFIED]. **EndoCLIP**, **EchoCare**: China-led. **VisionFM**: CUHK + Beijing Tongren (China). **NV-Reason-CXR-3B** (NVIDIA Clara): fine-tuned from **Qwen2.5-VL-3B**, NVIDIA OneWay Non-Commercial. **Qwen/InternVL-based** medical VLMs generally | | | | | | | | |

**Resolution note.** Every PMC-based CLIP above is 224 px. EM, fluorescence and fine histology texture lose information at 224, but modality, stain and body part rarely need it. The bake-off should include an up-resolution check for BMC/MedPMC (interpolate position embeddings to 336) because the MedSigLIP-vs-CLIP comparison otherwise confounds resolution with domain.

## 2. Domain specialists

### 2.1 Radiology

| Model | Org / origin | Params / arch | Res | Data | Text tower | License / gating | Availability | Evidence | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **RAD-DINO** | Microsoft Health Futures, US/UK | ViT-B/14 (DINOv2-base init), 86.6M | 518 shorter-side | 882,775 chest X-rays (MIMIC-CXR 369k, CheXpert 224k, NIH 112k, PadChest 137k, BRAX 41k) | No | **MIT**, ungated. (`rad-dino-maira-2`: MSRLA, other) | `transformers` | Matches or beats text-supervised CXR encoders (Nature Mach. Intell. 2025); small-lesion signal is partly lost in frozen CLS (2606.11606) | CXR only. Narrow. Useful only as a probe for the CXR slice |
| **CXR Foundation (ELIXR)** | Google, US | [UNVERIFIED] | 1280 | CXR | text variant exists | HAI-DEF, auto-gated | `cxr-foundation` (TF/keras) | MedSigLIP zero-shot is 2.0 AUC higher (MedGemma report) | Narrow, superseded by MedSigLIP |
| **Merlin** | Stanford MIMI, US | 3D I3D-ResNet152 + Clinical Longformer | 3D CT | Abdominal CT with EHR + reports (Nature 2026) | Yes | **MIT**, ungated | `merlin-vlm` | n/a | **Mismatch**: needs 3D volumes. We have 2-D images |
| **CT-FM** | AIM-Harvard / Lighter, US | 3D SegResNet, 77.8M | 3D | 148k CT scans (IDC) | No | **Apache-2.0**, ungated | HF `project-lighter/ct_fm_feature_extractor` | n/a | Mismatch: 3D |
| **CT-CLIP / CT-RATE** | Zurich / Bern / Istanbul, Switzerland-Turkey | 3D CT-ViT | 3D | CT-RATE | Yes | CC-BY-NC-SA-4.0, auto-gated | HF dataset repo | n/a | Mismatch: 3D |
| **MedDINOv3** | academic [origin UNVERIFIED] | DINOv3 continued on CT-3M (about 4M axial slices) | n/a | CT slices | No | [UNVERIFIED] | [UNVERIFIED] | segmentation-focused | CT only, segmentation-oriented |
| **BiomedParse** | Microsoft + U. Washington + Providence, US | SEEM/Focal-style segmentation model | n/a | 9 modalities, segmentation/detection/recognition | text prompts | HF weights **CC-BY-NC-SA-4.0**, auto-gated; GitHub repo Apache-2.0 | HF | Segmentation model, not a classifier backbone | Not a backbone |
| **EchoCLIP** | Cedars-Sinai, US | ViT + text | n/a | echo frames/reports | Yes | MIT, ungated | open_clip | echo only | Narrow |

### 2.2 Pathology (all narrow: H&E patches, some IHC)

None of these has seen non-histology images. They would add value only on pathology slices (H&E vs IHC vs special stain, tissue/specimen type, organ).

| Model | Org / origin | Params / arch | Res | Data | Text | License / gating | Availability | Notes |
|---|---|---|---|---|---|---|---|---|
| **UNI2-h** | Mahmood Lab, Harvard/MGB, US | ViT-H/14-reg8, 681M | 224 | 200M+ tiles, 350k+ **H&E and IHC** slides (MGB) | No | **CC-BY-NC-ND-4.0**, auto-gated | timm | Strongest "includes IHC" public option. Non-commercial, no-derivatives: usable as a teacher feature extractor with disclosure |
| UNI (v1) | same | ViT-L/16, 100M patches, 100k H&E WSIs | 224 | | No | CC-BY-NC-ND-4.0, auto-gated | timm | Superseded by UNI2-h |
| **CONCH** | same | ViT-B/16 + text, 1.17M pathology image-caption pairs | [UNVERIFIED, 448 from memory] | PMC + educational + in-house | **Yes** | CC-BY-NC-ND-4.0, auto-gated | timm / `conch` package | The only pathology model with a text tower of this quality |
| **Virchow2** | Paige (US) with Microsoft Research | ViT-H/14 632M (Virchow2G 1.9B, Mini 22M) | 224 | 3.1M H&E and **IHC** WSIs, mixed magnification | No | **CC-BY-NC-ND-4.0**, auto-gated | timm | Competitor to UNI2-h. Virchow v1 is Apache-2.0 gated, 632M, 1.5M H&E slides |
| **Prov-GigaPath** (tile encoder) | Microsoft Research + Providence + UW, US | ViT-g/14 (1.13B) DINOv2 tile encoder + LongNet slide encoder | 224 | 1.3B tiles from 171k slides (Providence) | No | **Apache-2.0** (API), auto-gated | timm | License is clean; gating is a click-through |
| **H-optimus-0** | Bioptimus, France | ViT-g/14, 1.1B | 224 | 500k+ H&E slides [from memory] | No | **Apache-2.0**, auto-gated | timm | Cleanest strong licence among large models |
| H-optimus-1 | Bioptimus, France | ViT-g/14, 1.13B | 224 | over 1M H&E slides, 50+ organs, 4,000+ centres | No | **CC-BY-NC-ND-4.0**, **manual** gate | timm | Top PathBench claim, vendor-reported |
| **Midnight-12k** | Kaiko, Netherlands | ViT-g/14, 1.14B | 224 | TCGA only (12k FFPE H&E WSIs) | No | **MIT, ungated** | HF | Fully open fallback. Trained only on public TCGA |
| Phikon-v2 | Owkin, France | ViT-L/16 (DINOv2), 303M | 224 | PANCAN-XL 456M tiles from 60k public WSIs (TCGA, CPTAC, GTEx) | No | Owkin non-commercial licence (`other`), **ungated** | transformers | Easiest to obtain. Phikon v1 is ViT-B (86M), iBOT, TCGA |
| Hibou-L / Hibou-b | HistAI, US (San Francisco per search result; team origin [UNVERIFIED]) | ViT-L (303M) / ViT-B (86M), DINOv2 | 224 | 1M+ proprietary H&E/IHC slides | No | **Apache-2.0**, auto-gated | transformers | Permissive and small |
| **PLIP** | Stanford (Zou lab), US | CLIP ViT-B/32 fine-tuned, 224, 768-d | 224 | OpenPath 208k Twitter pairs | Yes | license not stated on HF [UNVERIFIED]; ungated | transformers | Old, weak |
| **QuiltNet-B-32 / B-16** | UW + Stanford, US | CLIP ViT-B/32 (B/16, B/16-PMB variants) | 224 | Quilt-1M (YouTube + PMC + Twitter, about 1M pairs) | Yes | **MIT**, ungated | open_clip | Cheap histology CLIP with text tower |
| TITAN / MUSK | Mahmood Lab (CC-BY-NC-ND, gated) / Stanford | slide-level / 675M multimodal | n/a | | | | | Slide-level or text-heavy. Not a patch backbone |
| Path Foundation | Google, US | ViT-S MSN [UNVERIFIED] | 224 | histopathology patches | No | HAI-DEF, auto-gated | TF-keras | MedSigLIP is within 1.9% of it on histopathology linear probe (MedGemma report) |
| Excluded | **KEEP**, **PathGen-CLIP** and similar | China-led [origin UNVERIFIED] | | | | | | not examined |

**Caveat for every pathology model:** their pretraining slides include TCGA / CPTAC / Camelyon-derived content, and several of our 420 registry datasets (e.g. PatchCamelyon, TCGA-derived tiles) overlap. SSL pretraining does not use labels, so this is mild, but report pathology-slice numbers with that caveat. The same applies to MedSigLIP's training (it names TCGA, CAMELYON, PAD-UFES-20, MIMIC-CXR, EyePACS).

### 2.3 Ophthalmology, dermatology, endoscopy, ultrasound

| Model | Org / origin | Params / arch | Data | Text | License / gating | Fit |
|---|---|---|---|---|---|---|
| **RETFound** (MAE and DINOv2; `natureCFP`, `natureOCT`, `meh`, `shanghai`) | UCL / Moorfields Eye Hospital, UK | ViT-L (about 303M) | 1.6M retinal images (CFP and OCT). `shanghai` variants pretrain on Shanghai-hospital data; weights are UK-built but the data are Chinese, so prefer `meh` / `natureCFP` / `natureOCT` and flag it | No | CC-BY-NC-4.0 (mirror API), auto-gated | Fundus + OCT only. A 2026 calibration benchmark (known only from a search-result summary, [UNVERIFIED]) reports frozen RETFound falling to AUC 0.697 on external DR validation while frozen MedSigLIP keeps 0.915. General DINOv3 is competitive with it (2509.03421). Narrow, low value for metadata |
| **VisionFM** | CUHK (Hong Kong) + Beijing Tongren Eye Center, **China** | ViT-B, separate encoder per modality | 3.4M images, 8 ophthalmic modalities (fundus, OCT, FFA, slit lamp, B-scan, external eye, MRI, UBM) | No | [UNVERIFIED] | **Excluded: Chinese origin** |
| **Derm Foundation** | Google, US | ResNet-style, 448 [UNVERIFIED] | derm photos | No | HAI-DEF, auto-gated | Narrow, image-only. MedSigLIP beats it on 79-class |
| **PanDerm / DermLIP** | Monash University, Australia (some Chinese-named co-authors; institution is non-Chinese, flag as [origin partly UNVERIFIED]) | ViT (MAE-style) / CLIP ViT-B/16 + PanDerm text | 2M+ skin images, 4 modalities (dermoscopy, clinical, TBP, pathology) | DermLIP yes | **CC-BY-NC-4.0**, auto-gated | Derm slice only |
| **EndoViT** | TU Munich, Germany | MAE ViT-B/16 | Endo700k (laparoscopic / surgical) | No | Apache-2.0, ungated | Narrow (surgical). No broad GI-endoscopy coverage |
| EndoFM / EndoCLIP | CUHK / Zhejiang, **China** | | | | | **Excluded** |
| EchoCare / USFM-type | China-led (EchoCare: CASIA co-author [UNVERIFIED]) | | | | | **Excluded** |
| ZEN (surgical video) | AITRICS/Korea [affiliation UNVERIFIED] | multi-teacher distilled | 4M frames, 21 procedures | No | [UNVERIFIED] | Surgical only, not examined |

### 2.4 Microscopy and cell biology

| Model | Org / origin | Arch / data | Text | License | Fit |
|---|---|---|---|---|---|
| **OpenPhenom** | Recursion, US | CA-MAE ViT-S/16, **6-channel** CellPainting (RxRx3, JUMP-CP) | No | **Non-commercial EULA**, ungated | Cannot take RGB images; the card calls brightfield and H&E out of scope. Not a backbone for us |
| **SubCell** | KTH + Stanford + CZI Biohub (Lundberg lab), Sweden/US | ViT-B family, Human Protein Atlas fluorescence, **single-cell, 1-4 channel** | No | code MIT (`SubCellPortable`); weights licence [UNVERIFIED] | Narrow (HPA-type IF). Fine only for fluorescence-cell slice |
| **CellCLIP** | Su-In Lee lab (UW), US | Cell Painting contrastive, 450M safetensors | Yes (perturbation text) | repo "Other" | Cell Painting only |
| **DINO4Cells** | Broad Institute, US | DINO on Cell Painting / HPA | No | none stated | Code only; narrow |
| **BMC-CLIP / MedPMC-CLIP** (from table 1) | | PMC figures: light / fluorescence / EM micrographs | Yes | MIT / CC-BY-NC-SA | **The only open options with EM, fluorescence and brightfield figure coverage** |

There is no open, broad, RGB cell / EM microscopy foundation model in the set I found. `u-Bench` (Stanford, 22 microscopy tasks) reports specialist biomedical CLIPs (BiomedCLIP, PLIP) underperforming generalist VLMs, and that fine-tuning causes forgetting that weight interpolation partly repairs. This supports using WiSE-FT in our recipe and not trusting a microscopy-specialist label.

## 3. MedSigLIP in depth

**What it is.** `google/medsiglip-448`: two-tower SigLIP, 878.3M parameters total, about 400M vision and about 400M text. Image 448 px, patch 14, text 64 tokens. Released July 2025 with MedGemma; the HF weights have not been updated since. License is the Health AI Developer Foundations terms (`other`), gated `auto` (accept terms, immediate approval).

**Is it literally MedGemma's tower? Yes.** The MedGemma report (2507.05201, section 2.2.2): the vision encoder of Gemma 3 (SigLIP-400M) was fine-tuned on 33M+ medical pairs (635k from several modalities plus 32.6M histopathology patches), then the released 448 px version "shares the same model weights as the 896x896 encoder" with down-sampled position embeddings. MedGemma 4B uses the 896 variant. MedGemma 1.5 (2604.05081, Jan 2026) keeps this encoder **frozen** and only extends the LM (3D CT/MRI slicing, WSI), and the report states MedSigLIP "remains available for medical image classification and retrieval". There is no newer MedSigLIP. Its base is **SigLIP-400M as used in Gemma 3, not SigLIP 2** (a few secondary sources claim SigLIP 2; the primary report does not).

**Data mix and what it implies for us.**
- Non-histology pairs: CXR 231k + knee X-ray 1.5k + CT slices 60k + MRI slices 47.6k + fundus 199k + derm 53k + PMC single-panel figures 41.9k. The PMC part is the only source of microscopy / endoscopy / EM content, and it is 0.13% of the medical mix.
- 98.1% of medical pairs are internal histopathology patches. Expect strong H&E and good generic tissue features, but no extra knowledge of ultrasound, endoscopy, EM, fluorescence or clinical photographs.
- The 2% weight on medical data inside the original WebLI mix means it should retain general-image ability, but nobody has published a MedSigLIP-vs-SigLIP2 head-to-head on non-medical-looking microscopy. [UNVERIFIED: no such comparison found.]

**Published numbers (MedGemma report, section 7; model card).** Zero-shot AUC: CXR 13 findings 0.844 vs ELIXR 0.824 (+7.1 on fractures); derm 79 conditions 0.851 (LP 0.881, beats Derm Foundation LP); DR 5-class 0.759 (LP 0.857); histopathology 10 tasks 0.870 (LP 0.878, within 1.9% of Path Foundation). Linear probe wins over zero-shot, which matches the "fine-grained zero-shot is weak" pattern: a pelvic-MRI series benchmark found every model's zero-shot (14.5-29.4%) below the 36.2% majority-class baseline, and fine-tuning closed the gap for MedSigLIP and SigLIP-2 alike. Community fine-tunes on HF (kvasir endoscopy, CRC100k histology, SCIN skin, ROCOv2 retrieval, VinDr CXR) show it is used across domains, but I read none as evidence.

**Can it be used in a Gemma-4 student?** Three routes, only one clean:

| Route | Feasible? | Why |
|---|---|---|
| (A) **Teacher only**: MedSigLIP-based classifier produces calibrated labels and soft targets for the Pass-2 SFT set; the student keeps its own tower | **Yes. Recommended.** | No tower mismatch. HAI-DEF: "Outputs are not deemed Model Derivatives" |
| (B) Feature distillation from MedSigLIP into Gemma 4's tower (auxiliary loss on pooled features) | Technically fine | The HAI-DEF "Model Derivatives" definition (iii) explicitly covers a model made by transfer of the weights, operations or Output "to cause that model to perform similarly to HAI-DEF, including distillation methods that use intermediate data representations or ... generation of synthetic data Outputs". The same clause then says "Outputs are not deemed Model Derivatives". The two sentences are in tension, so (B) risks the student becoming a Model Derivative (Section 3.1 flow-down: carry the use restrictions, ship the agreement and a Notice file). This also narrows the claim in `backbones.md` section 5 item 3. **Ask the organizers or UF counsel before relying on it.** Hard-label / probability SFT targets are the safer reading |
| (C) **Swap MedSigLIP in as the student's tower** (Gemma-4 LM + `SiglipVisionModel` + new projector, as MedGemma did with Gemma 3) | Only with custom code; **not drop-in** | Gemma 4's `gemma4_vision` config: RMSNorm (`rms_norm_eps`), `attention_bias: false`, axial 2-D RoPE (`rope_theta` 100) plus a 10,240-entry position table, patch 16, 3x3 pooling, `standardize`, `head_dim` 72. SigLIP uses LayerNorm, biases, learned absolute positions, patch 14. The 550M tower (27 layers, 1152, 4304, 16 heads) has the **same layer shapes** as SigLIP So400m but the weights are not functionally compatible, so loading MedSigLIP weights is only a warm start. E2B/E4B towers (16 layers, 768-d) do not match at all. MedGemma needed a full LM re-adaptation pretraining stage after the tower change. The whole student then becomes an HAI-DEF derivative, and gated weights conflict with the "ungated submission" rule (see `backbones.md` section 5). Parameter cost on Track 3: about +0.25B over E2B's 150M tower, ~2.7B total [ESTIMATE], which is within the cap |

The ungated way to get a "medically adapted" tower in the student is to adapt a **SigLIP2-So400m (Apache-2.0)** tower with our own data and swap that in (alt row in `backbones.md` section 4c). That keeps the benefit and the licence clean, and it is the same code path as the teacher.

## 4. Evidence: specialised vs general

| Source | Finding | Reading for Pass-2 |
|---|---|---|
| MedGemma report, MedSigLIP section | LP beats zero-shot by up to 9.8 AUC points (DR); derm LP 0.881 | Train a head, don't rely on label-text zero-shot |
| ConceptCLIP (2501.15579, excluded model) | SigLIP-400M beats or ties ConceptCLIP in full fine-tuning on endoscopy, derm, pathology (0.08, 0.59, 0.12 AUC); ConceptCLIP wins 7 of 10 modalities | After fine-tuning, a SigLIP-class general encoder is within a point of a medical continued-pretrain |
| u-Bench (2407.01791) | Specialist biomedical CLIPs often worse than generalists on microscopy; fine-tuning forgets; weight interpolation repairs | Use WiSE-FT with the base; do not expect medical CLIPs to win microscopy |
| BIOMEDICA / BMC-CLIP | Microscopy composition (light vs fluorescence vs EM): marginal gain over BiomedCLIP; +8.16 with WiSE-FT | PMC data does carry EM/fluorescence knowledge |
| MedPMC | +7.1 AUC over BMC-CLIP; swapping into LLaVA-Med lifts modality recognition 46.9 to 75.2 | A better PMC CLIP helps modality tagging in VLMs, self-reported |
| Pelvic MRI (MICCAI 2026) | Zero-shot all below majority baseline (14.5-29.4% vs 36.2%); after fine-tuning all four models at least 93% sequence accuracy (MedSigLIP 95.1; SigLIP-2's reported 90.5 is orientation accuracy, a different metric) | Series-type tagging (our "submodality") needs training; medical vs general pretraining is not the deciding factor once fine-tuned |
| Retinal benchmarks (2509.03421, 2511.22001) | General DINOv2/v3 competitive with RETFound; compact ImageNet models often enough | Skip ophthalmology specialists |
| DINOv3 medical (2509.06467, via `backbones.md`) | Strong but degrades on WSI, EM, PET | Why a PMC CLIP / pathology add-on may still matter |
| Frozen CXR probe (2606.11606) | Frozen CLS features lose small-lesion signal | Irrelevant for metadata |
| OpenMedQ (Stanford, 2606.12953) | ViT-B encoder 0.757 vs BiomedCLIP 0.745 macro-F1 on 8 benchmarks; narrow-modality encoders still edge it on breast US; **weights not released** | Gains between medical CLIPs are small |

**No head-to-head exists** between MedSigLIP and SigLIP2-So400m (or DINOv3) on modality + submodality + body-part tagging across all our modalities. This is the main question the bake-off answers. Treat the ranking below as priors, not results.

## 5. Mixture vs single encoder

- **Coverage argues for a mixture.** MedSigLIP is strong on radiology, derm, fundus, histology; the PMC CLIPs see EM, fluorescence, endoscopy and clinical photos; SigLIP2 is the strongest generic perceiver and has the best resolution handling. Their errors differ by slice.
- **Cost is small.** Frozen features for 880k images are dominated by JPEG decoding (3-6 h per tower, `clm_methods.md` §4.7). Concatenating pooled embeddings, or averaging per-tower cosine logits, keeps a single head. FLOPs per image: MedSigLIP-448 about 0.8 TFLOP, ViT-L/14 at 224 about 0.16, ViT-H/14 pathology at 224 about 0.32 [ESTIMATE].
- **Pathology specialists are a gated add-on, not a peer.** They are trained on H&E/IHC patches only. Feed them the whole dataset and let a linear head learn to ignore them off-domain, or route only images that SigLIP2/MedSigLIP assign to a histology submodality. Expect gains on stain and specimen, not on modality or body part.
- **A single medical encoder is unlikely to win everywhere.** MedSigLIP might win radiology / derm / fundus / histology slices and lose endoscopy / EM / fluorescence to SigLIP2 or a PMC CLIP. The deciding view is per-slice (section 6).
- The mixture does not carry into the student for free: the student has one tower. That is why the CLM teacher produces labels and soft targets (route A) instead of features.

## 6. Ranked shortlist for the bake-off against SigLIP2-So400m

Use the grouped-by-dataset CV, the cosine-classifier recipe and the LP, LP-FT, WiSE-FT ladder from `clm_methods.md` §4. For every arm report macro-F1 and rare-class recall per field **and per slice** (radiology, ophthalmology, derm, endoscopy, histology, cell / fluorescence, EM, clinical photo, ultrasound). Add a resolution control for the 224 px CLIPs (336 px with interpolated position embeddings).

| Rank | Encoder | Origin / licence / gating | Why it is in | What would eliminate it |
|---|---|---|---|---|
| 1 | **MedSigLIP-448** | Google US; HAI-DEF; auto-gated | Same architecture family and code path as SigLIP2; the best-documented medical tower; text tower; 448 px; frozen probes strong | No gain over SigLIP2 on radiology/derm/fundus/histology slices, or a clear loss on endoscopy / EM / fluorescence that a concatenation does not repair |
| 2 | **BMC-CLIP-CF (ViT-L/14)** | Stanford US; MIT; ungated | Only ungated, permissively licensed encoder with explicit light / fluorescence / EM and endoscopy figure coverage; has text tower; reported large derm/ophtho gains over BiomedCLIP | Loses to SigLIP2 even on the microscopy slices at 336 px |
| 3 | **MedPMC-CLIP (ViT-L/14)** | Yale US; CC-BY-NC-SA; ungated | Newest PMC CLIP; claims +7.1 AUC over BMC-CLIP and a lead over MedSigLIP on 26 benchmarks (self-reported); 32% non-pathology microscopy | Not better than BMC-CLIP on grouped CV. Then drop one of them |
| 4 | **MedImageInsight** | Microsoft US; MIT (mirror), official via Azure [license text UNVERIFIED]; custom code | Different architecture (DaViT, 480 px, UniCL) gives diversity for radiology / derm / OCT / fundus / breast US; text-conditioned classifier | Packaging effort exceeds the gain, or no gain over MedSigLIP on its covered slices |
| 5 | **UNI2-h** (alt. **Virchow2**; clean-licence fallbacks **H-optimus-0** Apache-2.0, **Midnight-12k** MIT ungated) as a **pathology add-on arm** | Harvard US / Paige US / Bioptimus France / Kaiko NL; CC-BY-NC-ND, auto-gated (UNI2-h, Virchow2) | The only arm that can add real information on H&E vs IHC vs special stains and tissue type | No gain on stain/specimen macro-F1 over the best general or medical tower, then drop pathology specialists |

Free controls (run frozen only): **BiomedCLIP** (MIT), **UniMed-CLIP ViT-B/16** (CC-BY-NC), and **RAD-DINO** on the CXR slice. Skip RETFound, VisionFM (excluded), Merlin / CT-FM / CT-CLIP (3-D), OpenPhenom / SubCell (channel-specific), MMKD-CLIP (Chinese teacher lineage), MerMED-FM (image-only, unclear license).

## 7. Recommendation for the CLM-style teacher backbone

1. **Default**: the teacher is **SigLIP2-So400m fine-tuned** (Apache-2.0, ungated, text tower, same family as the Gemma students) per `backbones.md` and `clm_methods.md`. Nothing in the medical literature justifies replacing it blindly: the post-fine-tuning gap between general and medical SigLIP-class encoders is about a point or less (ConceptCLIP, pelvic MRI), and no head-to-head on our task exists.
2. **Add MedSigLIP-448 as a second tower** (concatenate pooled embeddings, or average per-tower cosine logits). Both use the same sigmoid / cosine-head code, so this costs hours. Adopt MedSigLIP as primary only if it beats SigLIP2 by at least 1 macro-F1 on grouped-CV modality + submodality **and** does not lose on the endoscopy / EM / fluorescence slices. These thresholds are my proposal, not from a source.
3. **If microscopy / EM / endoscopy slices lag**, add **BMC-CLIP-CF** (MIT) as a third view, or MedPMC-CLIP, and keep WiSE-FT against the base for any tower you tune.
4. **Add a pathology expert** (UNI2-h or Virchow2; H-optimus-0 or Midnight-12k if the NC-ND terms are a problem) only if the stain/specimen slice improves.
5. **Teacher outputs, not features, go to the student.** Hard / soft labels from the ensemble feed the Gemma-4 SFT set (route A). Keep MedSigLIP out of any feature-level distillation or tower swap unless the organizers clarify the HAI-DEF Model Derivative clause and the ungated rule. Use a tuned SigLIP2 tower, not MedSigLIP, if a tower swap into the student is attempted.
6. **Disclosure.** Gated or non-commercial items in the teacher, flagged for the disclosure rule: MedSigLIP (HAI-DEF, gated), UNI2-h / Virchow2 / H-optimus-1 / CONCH (CC-BY-NC-ND, gated), MedPMC-CLIP (CC-BY-NC-SA), UniMed-CLIP (CC-BY-NC), RETFound (CC-BY-NC, gated), OpenPhenom (non-commercial EULA), Phikon (Owkin non-commercial). Ungated permissive: BMC-CLIP (MIT), BiomedCLIP (MIT), RAD-DINO (MIT), Midnight-12k (MIT), Open-PMC (Apache), QuiltNet (MIT). Gated but permissive: H-optimus-0, Hibou, Prov-GigaPath, Virchow v1 (Apache-2.0).

## 8. Open questions and risks

- **HAI-DEF wording.** The Model Derivatives definition and the "Outputs are not deemed Model Derivatives" sentence conflict on synthetic-data / distillation uses. Get a ruling before using MedSigLIP in the student pipeline.
- **MedSigLIP position embeddings at non-448 px.** Not tested; run the bake-off at 448 and one lower setting.
- **Pretraining overlap.** MedSigLIP (MIMIC-CXR, TCGA, CAMELYON, PAD-UFES-20, EyePACS), PMC CLIPs (figures from papers that use public datasets) and pathology FMs (TCGA / CPTAC) may have seen images from our 420 datasets. Grouped CV does not remove this. Report it.
- **Unverified items.** Weights and licence for MMKD-CLIP, MedDINOv3, MerMED-FM; PLIP licence; MedImageInsight's official license text; Path / Derm / CXR Foundation architectures; CONCH input size; PMC-CLIP and several others' origin; whether Gemma 4's tower was initialised from any SigLIP checkpoint (its config says not weight-compatible).
- **Chinese-origin lineage inside Western releases.** MMKD-CLIP distills from PMC-CLIP. Check teacher lists of any further "generalist" medical CLIP before adding it. LLaVA-Med's tower is OpenAI CLIP (US), not Chinese.
- **Unreplicated claims.** MedPMC-CLIP > MedSigLIP and BMC-CLIP > BiomedCLIP are paper-reported by the same groups that built the models.

## 9. Sources

- MedGemma technical report (MedSigLIP data, Table 1, sections 2.2.2, 6, 7): https://arxiv.org/abs/2507.05201
- MedGemma 1.5 technical report (MedSigLIP frozen): https://arxiv.org/abs/2604.05081
- MedSigLIP card (HF, public part): https://huggingface.co/google/medsiglip-448
- HAI-DEF terms: https://developers.google.com/health-ai-developer-foundations/terms
- Gemma 4 technical report (vision section, Table 10): https://arxiv.org/abs/2607.02770 ; Gemma 4 `config.json` (vision_config): https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json
- SigLIP2 So400m p14/384 config (27 layers, 1152, 4304): https://huggingface.co/google/siglip2-so400m-patch14-384
- MedImageInsight: https://arxiv.org/abs/2410.06542 ; community mirror: https://huggingface.co/lion-ai/MedImageInsights
- UniMed-CLIP: https://arxiv.org/abs/2412.10372 ; repo: https://github.com/mbzuai-oryx/UniMed-CLIP ; weights: https://huggingface.co/UzairK/unimed-clip-vit-b16
- BiomedCLIP card: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- BIOMEDICA / BMC-CLIP: https://arxiv.org/abs/2501.07171 ; https://huggingface.co/BIOMEDICA ; https://github.com/minwoosun/biomedica-etl
- MedPMC / MedPMC-CLIP: https://arxiv.org/abs/2607.07673 ; https://huggingface.co/Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1
- Open-PMC-18M CLIP: https://huggingface.co/vector-institute/open-pmc-18m-clip
- MMKD-CLIP (teacher list, Emory): https://arxiv.org/abs/2506.22567
- ConceptCLIP (evidence only, excluded): https://arxiv.org/abs/2501.15579
- OpenMedQ: https://arxiv.org/abs/2606.12953
- u-Bench (microscopy): https://arxiv.org/abs/2407.01791
- RAD-DINO: https://arxiv.org/abs/2401.10815 ; https://huggingface.co/microsoft/rad-dino
- Merlin: https://huggingface.co/stanfordmimi/Merlin ; CT-FM: https://huggingface.co/project-lighter/ct_fm_feature_extractor
- UNI: https://arxiv.org/abs/2308.15474 ; UNI2-h (HF API, gated); CONCH: https://arxiv.org/abs/2307.12914
- Virchow: https://arxiv.org/abs/2309.07778 ; Virchow2: https://arxiv.org/abs/2408.00738
- Prov-GigaPath: https://huggingface.co/prov-gigapath/prov-gigapath ; H-optimus: https://www.bioptimus.com/h-optimus
- Phikon-v2: https://arxiv.org/abs/2409.09173 ; https://huggingface.co/owkin/phikon-v2
- Hibou: https://arxiv.org/abs/2406.05074 ; Midnight-12k: https://huggingface.co/kaiko-ai/midnight
- Quilt-1M / QuiltNet: https://arxiv.org/abs/2306.11207
- RETFound repo: https://github.com/rmaphoh/RETFound ; generalist vs specialist retinal: https://arxiv.org/abs/2509.03421 ; https://arxiv.org/abs/2511.22001
- VisionFM (CUHK / Beijing Tongren): https://arxiv.org/abs/2310.04992 ; https://github.com/ABILab-CUHK/VisionFM
- PanDerm: https://arxiv.org/abs/2410.15038 ; DermLIP: https://huggingface.co/redlessone/DermLIP_ViT-B-16
- OpenPhenom: https://huggingface.co/recursionpharma/OpenPhenom ; SubCell: https://www.biorxiv.org/content/10.1101/2024.12.06.627299 ; https://github.com/CellProfiling/SubCellPortable
- NV-Reason-CXR-3B (Qwen2.5-VL base): https://huggingface.co/nvidia/NV-Reason-CXR-3B
- MedSigLIP dermatology study: https://arxiv.org/abs/2609.36557 ; Pelvis-SigLIP benchmark: https://papers.miccai.org/miccai-2026-sat/CAPI_WOMEN_019.html
- Frozen CXR embeddings: https://arxiv.org/abs/2606.11606 ; DINOv3 medical benchmark: https://arxiv.org/abs/2509.06467
- LLaVA-Med v1.5 card and config: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b
- HF Hub API for license / gating / safetensors totals, 2026-10-07: https://huggingface.co/api/models/<repo>
