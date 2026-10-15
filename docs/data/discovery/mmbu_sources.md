# MMBU public rows -> likely source datasets -> our registry and Pass-2 plan

Written 2026-10-07 by the `waila-mmbu-sources` task. Metadata-only analysis of `mmbu-public/metadata.jsonl` (3,147 rows). **No public image was used for training**; two montages (about 40 thumbnails) of public images were viewed to identify sources. Nothing in `pass2/src` or `pass2/plan` was edited.

## Method

1. Grouped the 3,147 rows by (`metadata_submodality`, `metadata_topic`, `metadata_body_part`, `metadata_specimen`, `metadata_stain`, `task`): **295 groups** (one-off groups of 1-3 rows are usually distractor-vocabulary or sub-sampled cells of the same source). Then merged groups into **120 source clusters** by question wording, option vocabulary, image size/aspect (the public images are down-scaled to <=512 px, which preserves aspect ratio, e.g. 4288x2848 -> 512x340 = IDRiD) and mpp.
2. Identified the likeliest source from class names (e.g. JSIEC's 39 fundus classes, RadImageNet's `ankle or foot plantar plate tear` strings, MIAS's 7 mammography classes, TBX11K's `Sick but not Tuberculosis`), then looked each candidate up in `data_discovery/build/registry.json` (id / name / aliases / homepage / mirrors / class names) and checked `eligible()` exactly as `pass2/src/bio_waila/data/select.py` defines it (tier A license, open/registration access, scriptable route, not dropped in `registry_fixes.json`/audits, <=2 modalities, >=30 images...), plus membership in `plan/selection.jsonl`.
3. Confidence: **H** = class vocabulary/format is specific to that dataset; **M** = family/edition ambiguous (e.g. HeiChole vs CholecInstanceSeg); **L** = a guess among several candidates, or no match.
4. WebSearch: 2 calls used (of 30), WebFetch: 2; the rest was registry + Zenodo/Kaggle/DataCite/HF API lookups. The MMBU site, HF card and paper do not list source datasets, so every attribution is an inference, not a confirmed fact.

Cluster ids refer to groups of the same grouping, numbered by descending size (group 0 = laparoscopic instrument-type masks, 170 rows). Statuses in the table: `[SEL]` already in selection.jsonl, `[ELIG]` passes eligible() but is not selected, `[no]` fails (reason in the action column).

## Summary

* 3,147 public rows -> 120 clusters. Rows attributed with **H confidence: 1,836 (58%)**, **M: 661 (21%)**, **L: 411 (13%)** (candidate named, low confidence); **239 rows (7.6%)** have no source identified.
* Of the H+M rows (2,497): **already selected 805 rows (25.6% of all rows; 25 sources)**; **eligible but not selected 301 rows (+19 rows of low-confidence candidates) -> 15 force_include ids**; **blocked 1,391 rows in 41 clusters**: unknown license (tier D) 512 rows / 18 clusters, research-only/request/challenge (tier C) 489 / 7, non-commercial (tier B) 355 / 12, and 35 rows (4 clusters) that are tier A but lack a route or were dropped. A further 411 rows have only low-confidence candidates (141 of them point at already-selected datasets, 107 at eligible-but-optional ones, 144 at blocked ones, 19 at force_include ones).
* Biggest blocked sources by public rows: RadImageNet (~255 rows, tier C, request), HeiChole (184, tier C), JSIEC (182, Zenodo `other-open`, registry tier D), CholecInstanceSeg (89, tier B), ISIC 2019 (91, tier B), MHSMA (68, tier B), LYSTO (54, license conflict CC-BY vs CC-BY-NC), EndoVis 2018 (46), PitVis (36), PSI-AVA/GraSP (34), LC25000 (27, unspecified license).
* **Usable: 25 sources already selected + 15 force_include proposals (40). Not usable under the tier-A rule: ~41 H/M clusters (plus most L-confidence candidates).**
* Under-weighting warning: the plan caps every dataset at ~2,000 images, so MMBU's mix is only approximated. Public-row share vs plan-image share for sources already selected: ChestX-Det 9.4% of public rows vs 0.46% of planned images, IDRiD 2.5% vs 0.07%, NIH CXR-bbox 1.7% vs 0.13%, HuSHeM 1.3% vs 0.03%, Chen knee-OA 1.1% vs 0.13%, while DSAD / PBC / bone-marrow get 1.5% of images each for 0.6-0.8% of rows. If proportions should follow MMBU, raise per-dataset caps for the high-row sources (ChestX-Det, IDRiD, HuSHeM, TBX11K) rather than adding datasets.

## Table: public row cluster -> source -> registry -> eligible? -> selected? -> action

Sorted by number of public rows. `n` = public rows.

| cluster | n | likely source | conf | registry id [status] | eligible? | selected? | action |
|---|---:|---|---|---|---|---|---|
| X1 | 295 | ChestX-Det (13 thoracic findings; masks+boxes) | H | `chestx-det` [SEL], `chestx-det10` [no] | yes | yes (chestx-det) | already selected; keep |
| R1 | 193 | RadImageNet MRI (224x224 slices; 'ankle or foot ...', 'shoulder ...' labels) | H | `radimagenet` [no] | no | no | ineligible: tierC/request; multi-source (label strings are RadImageNet's) |
| L1 | 184 | HeiChole-family cholecystectomy instrument-type/part masks (960x540 frames; 7 instrument categories) | M | `heichole` [no] | no | no | ineligible: tierC/registration; no route (classes grasper/clipper/coagulation/scissors/suction-irrigation/specimen bag/stapler match HeiChole categories) |
| O1 | 182 | JSIEC 1000 fundus (39 classes) | H | `jsiec-1000` [no] | no | no | ineligible: tierD/open (class names (Bietti, laser spots, ...) are JSIEC's) |
| D1 | 91 | ISIC 2019 (8-class dermoscopy) | H | `isic-2019` [no] | no | no | ineligible: tierB/open |
| L2 | 89 | CholecInstanceSeg / Cholec80-derived instrument masks (854x480 frames) | M | `cholecinstanceseg` [no] | no | no | ineligible: tierB/registration (instrument vocab grasper/bipolar/hook/clipper/irrigator/scissors) |
| P1 | 81 | NCT-CRC-HE-100K / CRC-VAL-HE-7K | H | `nct-crc-he-100k` [ELIG], `crc-val-he-7k` [ELIG] | yes | no | FORCE_INCLUDE nct-crc-he-100k, crc-val-he-7k (9 tissue classes, 224x224, 0.5 mpp) |
| F6 | 81 | synthetic + calibration-bead/organelle fluorescence 'what structure' set | L | `bbbc005` [SEL], `bbbc031` [SEL] | yes | yes (bbbc005, bbbc031) | candidate already selected (low confidence) (source unidentified) |
| O2 | 80 | IDRiD | H | `idrid` [SEL] | yes | yes (idrid) | already selected; keep (4288x2848 -> 512x340) |
| H2 | 68 | MHSMA (128x128 sperm acrosome/head/vacuole/tail) | H | `mhsma` [no] | no | no | ineligible: tierB/open |
| O3 | 55 | DR grading level 0-4 fundus (DeepDRiD/EyePACS/APTOS-like) | L | `deepdrid` [no], `aptos-2019` [no], `eyepacs-kaggle-dr-2015` [no] | no | no | ineligible: no route; tierC/registration |
| P3 | 54 | LYSTO lymphocyte counts (IHC, 299x299) | H | `lysto` [no] | no | no | ineligible: tierD/open |
| X2 | 52 | NIH ChestX-ray8 bbox subset | H | `nih-chestxray14-bbox` [SEL] | yes | yes (nih-chestxray14-bbox) | already selected; keep |
| L4 | 46 | EndoVis 2018 robotic scene segmentation (+2017 kidney boundary) | H | `endovis-2018-rss` [no], `endovis-2017-kidney-boundary` [no] | no | no | ineligible: tierD/registration; tierD/registration; no route (covered kidney / clamps / thread / suturing needle / ultrasound probe) |
| G2 | 43 | EDD2020 (BE, suspicious, HGD, cancer, polyp; boxes+masks) | H | `edd2020` [ELIG] | yes | no | FORCE_INCLUDE edd2020 (5-class GI disease detection) |
| M4 | 42 | axial brain MRI lesion/tumor boxes (positive/negative) | L | `pkdarabi-brain-tumor-detection` [ELIG], `sorour-brain-tumor-bboxes` [ELIG], `kaggle-brain-tumor-object-detection-mgmt` [ELIG] | yes | no | eligible but low confidence; optional |
| R2 | 40 | RadImageNet CT (abdomen/lung labels) | H | `radimagenet` [no] | no | no | ineligible: tierC/request; multi-source |
| H1 | 40 | HuSHeM | H | `hushem` [SEL] | yes | yes (hushem) | already selected; keep |
| O4 | 40 | cataract/glaucoma/retina-disease/normal fundus (Kaggle) | M | `eye-diseases-classification-gunavenkat` [no], `cataract-retina-kaggle` [no] | no | no | ineligible: tierD/registration |
| L8 | 36 | PitVis 2023 (endoscopic pituitary surgery phases) | H | `pitvis-2023` [no] | no | no | ineligible: tierB/open (sellotomy / durotomy / gasket seal construct) |
| K1 | 35 | Knee OA KL grading (Chen 2018, 224x224) | H | `knee-oa-severity-chen-2018` [SEL] | yes | yes (knee-oa-severity-chen-2018) | already selected; keep |
| K2 | 35 | MIAS mammography (7 classes) | H | `mias` [ELIG] | yes | no | FORCE_INCLUDE mias (calcification/circumscribed/spiculated/ill-defined/architectural distortion/asymmetry/normal) |
| L7 | 34 | PSI-AVA / GraSP (robotic prostatectomy atomic actions, boxes) | M | `psi-ava` [no], `grasp` [no] | no | no | ineligible: tierD/open; tierD/open; no route (action vocabulary pulling prostate / clipping vas deferens) |
| G1 | 34 | Kvasir v2 / HyperKvasir labelled images (8 anatomical/pathological classes) | H | `hyperkvasir` [ELIG], `kvasir-v2` [no] | yes | no | FORCE_INCLUDE hyperkvasir (normal-cecum, z-line, pylorus, dyed lifted polyps, esophagitis, ulcerative colitis) |
| U2 | 34 | Hitachi transvaginal US early-pregnancy gestational sac/embryo boxes | L | - | - | - | unidentified (image viewed: Hitachi gynecology scans with box on sac/embryo; source not found) |
| P2 | 34 | Kather 2016 colorectal texture (8 classes, 0.495 mpp) | H | `kather-crc-texture-2016` [ELIG] | yes | no | FORCE_INCLUDE kather-crc-texture-2016 |
| L9 | 31 | surgical-instrument tray photos with boxes (scalpel, Mayo scissors, clamps) | L | `surgical-instruments-labelme-383` [ELIG] | yes | no | eligible but low confidence; optional (image viewed: instruments on steel tray; exact source unconfirmed) |
| X3 | 30 | Kermany pediatric pneumonia CXR | H | `kermany-pediatric-pneumonia-cxr` [SEL] | yes | yes (kermany-pediatric-pneumonia-cxr) | already selected; keep |
| P19 | 27 | LC25000 | H | `lc25000` [no] | no | no | ineligible: tierD/open |
| H3 | 27 | BCCD / blood-cell detection (RBC/WBC/platelet) | M | `bccd` [ELIG], `txl-pbc-2025` [ELIG] | yes | no | FORCE_INCLUDE bccd |
| O6 | 27 | glaucoma fundus sets | L | `refuge` [no], `g1020` [no], `rim-one-dl` [no], `origa` [no] | no | no | ineligible: tierC/open; tierD/open; tierD/registration; tierD/request |
| L5 | 26 | Dresden Surgical Anatomy Dataset (DSAD) | H | `dsad` [SEL] | yes | yes (dsad) | already selected; keep (abdominal organ names (vesicular glands, IMA, ureter...)) |
| P4 | 25 | PBC (Acevedo) peripheral blood cells | H | `pbc-acevedo-2020` [SEL] | yes | yes (pbc-acevedo-2020) | already selected; keep |
| O14 | 24 | corneal confocal nerve (CORN) | M | `corn-3` [no], `corn-1500` [no], `corn-complex` [no] | no | no | ineligible: tierC/request |
| P8 | 23 | amyloid-beta IHC plaque morphology (diffuse/cored/CAA) | L | - | - | - | unidentified (not in registry; source not located) |
| P14 | 23 | Mendeley LBC Pap smear (NILM/LSIL/HSIL/SCC) | H | `mendeley-lbc-hussain-2019` [ELIG] | yes | no | FORCE_INCLUDE mendeley-lbc-hussain-2019 (963 imgs 2048x1536 -> 512x384) |
| F2 | 23 | BBBC002 (Kc167 counts) / BBBC007 (Kc167 outlines) | M | `bbbc002` [no], `bbbc007` [SEL] | yes | yes (bbbc007) | already selected; keep (BBBC002 (likely source, license unknown) is not eligible; selected bbbc007 is the same cell line but only 16 images) |
| O8 | 23 | fundus laterality / quality (DeepDRiD / EyeQ-like) | L | `deepdrid` [no], `fiqs` [ELIG] | yes | no | eligible but low confidence; optional |
| L6 | 22 | synthetic domain-randomised dVRK suturing masks (Potts scissors, Cadiere forceps, needle) | L | `sims-2023` [no], `syn-iss-2023` [no], `sisvse` [no] | no | no | ineligible: tierC/challenge-only; no route; tierD/challenge-only; synthetic; no route; tierD/open; synthetic (looks like a surgical-instrument sim-to-real set; exact source unconfirmed) |
| R3 | 22 | RadImageNet ultrasound | H | `radimagenet` [no] | no | no | ineligible: tierC/request; multi-source |
| X9 | 21 | misc CXR one-offs (lung laterality boxes, orientation, rib fracture, aneurysm, cardiomegaly) | L | - | - | - | unidentified (unidentified) |
| P21 | 21 | misc H&E one-offs (colon benign/malignant, gland, necrosis, metastasis) | L | - | - | - | unidentified (unidentified) |
| F7 | 21 | HEp-2 I3A / ICPR cell patterns | H | `hep2-icip2013-i3a` [no] | no | no | ineligible: tierD/registration (78x78 patches) |
| C1 | 20 | lung nodule CT (LUNA16/LIDC) | M | `luna16` [ELIG], `lidc-idri` [SEL] | yes | yes (lidc-idri) | already selected; keep |
| H6 | 20 | TB bacilli (ZN sputum) boxes | L | `tb-wellgen-mscan-2024` [ELIG], `tb-sputum-ufam-figshare-plus-2025` [SEL] | yes | yes (tb-sputum-ufam-figshare-plus-2025) | candidate already selected (low confidence) |
| L3 | 18 | CholecSeg8k (organ/tissue masks on Cholec80 frames) | M | `cholecseg8k` [no] | no | no | ineligible: tierB/open (liver/fat/gallbladder/GI tract classes, 854x480 aspect) |
| D3 | 18 | PAD-UFES-20 | H | `pad-ufes-20` [SEL] | yes | yes (pad-ufes-20) | already selected; keep (six clinical-photo classes) |
| P5 | 18 | Matek bone-marrow cytomorphology | M | `bone-marrow-cytomorphology` [SEL] | yes | yes (bone-marrow-cytomorphology) | already selected; keep |
| P6 | 18 | BACH / BreakHis breast histology | M | `bach-iciar2018` [no], `breakhis` [no] | no | no | ineligible: tierC/open; tierD/open |
| U1 | 17 | breast ultrasound (BUSI and others) | M | `busi` [SEL], `busi-whu` [ELIG], `bus-uc` [ELIG] | yes | yes (busi) | already selected; keep |
| P15 | 17 | BioMediTech hPSC-RPE phase contrast | H | `biomeditech-rpe-maturation` [SEL] | yes | yes (biomeditech-rpe-maturation) | already selected; keep |
| F4 | 17 | BBBC003 mouse embryo DIC | H | `bbbc003-mouse-embryos-dic` [no] | no | no | ineligible: tierD/open; <30 imgs; no route |
| O9 | 17 | optic disc/cup + vessel masks (REFUGE/DRIVE/STARE...) | L | `refuge` [no], `drive` [no], `stare` [no] | no | no | ineligible: tierD/open; tierD/registration |
| M2 | 16 | Brain MRI tumor yes/no (Navoneel / Br35H style) | M | `kaggle-brain-mri-yes-no-navoneel` [no], `br35h` [no] | no | no | ineligible: tierD/open |
| H7 | 15 | Gram-stained clinical bacteria (cocci/bacilli) | H | `clinical-bacteria-gram-mrose-2024` [SEL] | yes | yes (clinical-bacteria-gram-mrose-2024) | already selected; keep |
| F1 | 15 | BBBC010 C. elegans live/dead | H | `bbbc010-celegans-live-dead` [no] | no | no | ineligible: no route |
| E1 | 15 | cross-species EM 'what structure' (CEM500K-like) | L | - | - | - | unidentified (unidentified) |
| E2 | 15 | SBF-SEM cultured cell types | L | - | - | - | unidentified (unidentified) |
| C4 | 14 | CT cardiovascular / fracture one-offs | L | - | - | - | unidentified |
| D2 | 13 | ISIC 2018 task 2 dermoscopic attributes | H | `isic-2018` [no] | no | no | ineligible: tierB/open (globules/streaks/negative network/pigment network/milia-like cyst) |
| U3 | 13 | carotid ultrasound B-mode/Color Doppler (Kaggle) | M | `carotid-us-color-doppler-kaggle` [SEL] | yes | yes (carotid-us-color-doppler-kaggle) | already selected; keep |
| O7 | 13 | PALM pathologic myopia | H | `palm` [SEL] | yes | yes (palm) | already selected; keep |
| P7 | 12 | BRACS | H | `bracs` [SEL] | yes | yes (bracs) | already selected; keep (7-class breast lesion subtypes) |
| P12 | 12 | PanNuke cell-type labels | M | `pannuke` [no] | no | no | ineligible: tierB/open |
| O10 | 12 | Kermany OCT / OCTID | M | `oct2017-kermany` [SEL], `octid` [ELIG] | yes | yes (oct2017-kermany) | already selected; keep |
| O12 | 12 | RAVIR NIR retinal A/V | H | `ravir` [no] | no | no | ineligible: tierD/challenge-only |
| G4 | 11 | colonoscopy polyp segmentation set(s) (Kvasir-SEG / CVC-ClinicDB / PolypGen) | L | `polypgen` [ELIG], `kvasir-seg` [no], `cvc-clinicdb` [no] | yes | no | FORCE_INCLUDE polypgen (cannot tell which) |
| P9 | 11 | breast IHC HER2/PR/ER (BCI / others) | L | `bci-her2` [no], `her2-ihc-40x` [ELIG] | yes | no | eligible but low confidence; optional |
| F5 | 11 | C. elegans Oil Red O fat | M | `bbbc012-celegans-infection-marker` [no] | no | no | ineligible: tierD/open; no route (BBBC012 is clec-60::GFP; ORO set unidentified) |
| F10 | 11 | zebrafish larva adipose confocal | L | - | - | - | unidentified (unidentified) |
| G3 | 10 | Kvasir-Capsule (336x336 capsule frames) | H | `kvasir-capsule` [SEL] | yes | yes (kvasir-capsule) | already selected; keep |
| X4 | 10 | TBX11K | H | `tbx11k` [SEL] | yes | yes (tbx11k) | already selected; keep (Healthy / Sick-but-not-TB / TB with boxes) |
| D4 | 10 | Fitzpatrick17k (3-/9-partition labels) | H | `fitzpatrick17k` [no] | no | no | ineligible: tierB/open |
| D7 | 10 | iToBoS 2024 total-body-photography tile detection | M | `itobos-2024` [ELIG] | yes | no | FORCE_INCLUDE itobos-2024 (mole/lesion/scar/hair/specular highlight boxes) |
| D8 | 10 | dermoscopy lesion+artifact segmentation (hair, specular highlight, reflection) | L | `skin-hair-mask-dataset` [SEL] | yes | yes (skin-hair-mask-dataset) | candidate already selected (low confidence) |
| F8 | 10 | BBBC fluorescent vs brightfield modality identification | L | - | - | - | unidentified |
| O5 | 10 | ODIR-5K 8-class | H | `odir-5k` [no] | no | no | ineligible: tierD/open |
| O11 | 10 | OCT retinal layer segmentation | L | - | - | - | unidentified |
| G7 | 9 | esophageal WLE bleeding/tumor/ulcer/inflammation/instrument boxes | L | - | - | - | unidentified (unidentified) |
| X5 | 9 | COVID/viral pneumonia/TB/normal CXR compilation | L | `covid19-radiography-database` [no], `covid-qu-ex` [SEL] | yes | yes (covid-qu-ex) | candidate already selected (low confidence) |
| C3 | 9 | Chest CT-Scan images (adeno/large cell/squamous w/ TNM names) | H | `chest-ctscan-images` [no] | no | no | ineligible: tierD/open (class names carry TNM strings) |
| H8 | 9 | darkfield / DIC red blood cells (BBBC009-like) | L | `bbbc009-rbc-dic` [no] | no | no | ineligible: <30 imgs |
| O13 | 9 | MRL Eye infrared eyeglasses/open-closed | H | `mrl-eye` [no] | no | no | ineligible: tierD/open (92x92) |
| X7 | 8 | CXR heart/clavicle/lung masks (JSRT-SCR / Montgomery) | M | `scr-jsrt-masks` [no], `montgomery-heart-clavicle-masks` [no], `shenzhen-tb-cxr` [no] | no | no | ineligible: dropped:masks only; tierC/open; tierD/open; no route |
| D5 | 8 | MED-NODE / PH2 / ISIC-2020-style binary lesion sets | L | `med-node` [no], `ph2` [no] | no | no | ineligible: tierC/registration; no route; tierD/open |
| P10 | 8 | BCNB ALN status | M | `bcnb` [no] | no | no | ineligible: tierC/registration; no route |
| P11 | 8 | prostate Gleason (SICAPv2 / Gleason2019) | L | `sicapv2` [ELIG], `gleason2019` [no], `arvaniti-gleason-tma-2018` [ELIG] | yes | no | FORCE_INCLUDE sicapv2 |
| E4 | 8 | cryo-ET neuronal mitochondria | L | - | - | - | unidentified (unidentified) |
| O16 | 8 | OCTA DR lesions / systemic | M | `drac-2022` [SEL] | yes | yes (drac-2022) | already selected; keep |
| D10 | 7 | UV DNA damage (CPD/6-4PP) IHC skin | M | `uv-dna-damage-cpd-wsi` [no] | no | no | ineligible: tierD/open |
| U5 | 7 | mouse embryo high-frequency US | M | `mouse-embryo-hfu` [no] | no | no | ineligible: no route |
| P13 | 7 | Herlev Pap smear (7 classes, 93x105) | H | `herlev-pap-smear` [no] | no | no | ineligible: tierD/open |
| P16 | 7 | breast histologic grade 1-3 patches | L | - | - | - | unidentified (unidentified) |
| H4 | 7 | malaria (BBBC041 stages incl. leukocyte/RBC; NIH cell crops) | M | `bbbc041-p-vivax-malaria` [no], `nih-malaria-cell-images` [no] | no | no | ineligible: tierB/open; tierD/open |
| H5 | 7 | ALL blood smear normal/ALL | L | `c-nmc-2019` [SEL], `all-idb1-multiclass-seg` [ELIG] | yes | yes (c-nmc-2019) | candidate already selected (low confidence) |
| F9 | 7 | Fluorescent Neuronal Cells v2 | M | `fluorescent-neuronal-cells-v2` [SEL] | yes | yes (fluorescent-neuronal-cells-v2) | already selected; keep |
| B1 | 7 | rodent lung vessel H&E / sheep vascular graft H&E | L | - | - | - | unidentified (unidentified) |
| G5 | 6 | EAD2019/2020 endoscopic artefact detection | H | `ead2020` [no], `ead2019` [no] | no | no | ineligible: tierB/open (blur/bubbles/saturation/specularity/blood/instrument/low contrast) |
| U4 | 6 | intussusception abdominal US | L | - | - | - | unidentified (no registry or public hit found) |
| O15 | 6 | FA diabetic macular edema leakage | L | - | - | - | unidentified |
| K3 | 6 | hand/wrist X-ray joints | L | - | - | - | unidentified |
| M5 | 6 | abdominal MRI organ masks (CHAOS/AMOS-like, 266x266) | L | - | - | - | unidentified (unidentified) |
| L10 | 5 | gastrectomy / ICG / robotic scene one-offs | L | `laparoscopic-gastrectomy-annotated-images` [SEL] | yes | yes (laparoscopic-gastrectomy-annotated-images) | candidate already selected (low confidence) (low confidence) |
| G6 | 5 | ISBI AIDA-E confocal laser endomicroscopy (celiac, Barrett) | M | `isbi-aida-e-cle-celiac` [no], `isbi-aida-e-cle-barrett` [no] | no | no | ineligible: tierD/request; no route |
| X6 | 5 | pneumothorax CXR (SIIM-ACR / PTX-498) | L | `ptx-498` [SEL], `siim-acr-pneumothorax` [no] | yes | yes (ptx-498) | candidate already selected (low confidence) |
| D6 | 5 | mpox 6-class skin photos | M | `mpox-hsam-6class` [ELIG], `msid` [ELIG], `mpox-vision` [ELIG] | yes | no | FORCE_INCLUDE mpox-hsam-6class (monkeypox/cowpox/chickenpox/measles/smallpox/healthy) |
| U6 | 5 | brachial plexus / liver US | L | `brachial-plexus-regional-us` [no], `ubpd` [no] | no | no | ineligible: tierB/open; tierD/unavailable; no route |
| P17 | 5 | Nirschl cardiac H&E heart failure | H | `nirschl-heart-failure-2018` [SEL] | yes | yes (nirschl-heart-failure-2018) | already selected; keep |
| P18 | 5 | malignant lymphoma CLL/FL/MCL (Orlov) | H | `malignant-lymphoma-orlov` [no], `iicbu-2008` [no] | no | no | ineligible: tierD/open; tierD/unknown; no route |
| P20 | 5 | osteosarcoma viable/necrotic/non-tumor (UTSW) | H | `osteosarcoma-utsw` [no] | no | no | ineligible: no route |
| E3 | 5 | SBF-SEM Vero cell SARS-CoV-2 organelles | H | `sbfsem-vero-sars-cov-2-mitochondria` [ELIG] | yes | no | FORCE_INCLUDE sbfsem-vero-sars-cov-2-mitochondria |
| G8 | 4 | colposcopy transformation-zone types (Intel/MobileODT-like) | L | `intel-mobileodt-refined-766` [SEL], `intelkamat-colposcopy` [SEL] | yes | yes (intel-mobileodt-refined-766, intelkamat-colposcopy) | candidate already selected (low confidence) |
| M3 | 4 | 3-class brain tumor MRI (glioma/meningioma/pituitary) | M | `kaggle-brain-tumor-mri-nickparvar` [ELIG], `kaggle-brain-tumor-classification-sartaj` [ELIG] | yes | no | FORCE_INCLUDE kaggle-brain-tumor-mri-nickparvar |
| O17 | 4 | AV nicking / specular microscopy one-offs | L | - | - | - | unidentified |
| K4 | 4 | osteoporosis X-ray healthy/osteoporotic | M | `knee-xray-osteoporosis-wani` [SEL], `multi-class-knee-osteoporosis-xray` [ELIG] | yes | yes (knee-xray-osteoporosis-wani) | already selected; keep |
| P22 | 4 | Pap-smear nucleus/cytoplasm masks | L | - | - | - | unidentified (unidentified) |
| F3 | 3 | BBBC001/008 HT29 cell counts | M | `bbbc001` [no], `bbbc008` [no] | no | no | ineligible: tierB/open; <30 imgs |
| X8 | 2 | SIIM-FISABIO-RSNA COVID-19 CXR | M | `siim-covid19-detection` [no] | no | no | ineligible: tierB/challenge-only (negative/typical/indeterminate/atypical) |
| F11 | 2 | multiphoton axonal bouton | L | - | - | - | unidentified (unidentified) |
| D9 | 1 | foot-ulcer segmentation photos (FUSeg / AZH) | L | `fuseg-2021` [no], `azh-foot-ulcer-segmentation` [no] | no | no | ineligible: tierB/open; no route; tierD/open; no route |

## force_include proposal

Paste into `plan/registry_fixes.json` (`force_include` is read by `select.py`; ids that fail `eligible()` or lose the route-dedup are ignored silently). All entries are eligible per the table above and not yet selected.

```json
{
 "force_include": {
  "hyperkvasir": "Likely source of 34 public rows (Kvasir-v2 style GI classes: z-line, pylorus, cecum, dyed lifted polyps, esophagitis, UC); CC-BY-4.0, HF mirror; not yet selected (only kvasir-capsule is). Also helps under-filled gastroscopy/colonoscopy.",
  "edd2020": "Likely source of 43 public rows (BE / suspicious / HGD / cancer / polyp, boxes+masks); CC-BY-4.0, Kaggle mirror route; not selected.",
  "nct-crc-he-100k": "Source of ~81 public rows (9-class colorectal texture, 224x224, 0.5 mpp, H&E); CC-BY-4.0; not selected.",
  "crc-val-he-7k": "Companion validation set of NCT-CRC-HE (shares the Zenodo 1214456 route; pool dedup keeps whichever scores higher).",
  "kather-crc-texture-2016": "Source of 34 public rows (8 classes, 0.495 mpp, 150px tiles); CC-BY-4.0; not selected.",
  "mendeley-lbc-hussain-2019": "Source of 23 public rows (LBC Pap smear NILM/LSIL/HSIL/SCC, 2048x1536 -> 512x384, Papanicolaou); CC-BY-4.0; not selected.",
  "mias": "Source of 35 public rows (7 MIAS classes exactly: calcification, circumscribed, spiculated, ill-defined, architectural distortion, asymmetry, normal); CC-BY-2.0; mammography has only 5 selected sets.",
  "itobos-2024": "Likely source of 10 public rows (total-body-photography lesion boxes: mole/lesion/scar/hair/specular highlight); CC-BY-4.0, figshare.",
  "sbfsem-vero-sars-cov-2-mitochondria": "Likely source of 5 public rows (SBF-SEM Vero cell, SARS-CoV-2 organelles); MIT; HF; adds rare SEM class data (label 'Vero cells').",
  "kaggle-brain-tumor-mri-nickparvar": "Plausible source of 4 public rows (glioma/meningioma/pituitary); CC-BY-4.0.",
  "bccd": "Plausible source of ~23 public rows (RBC/WBC/platelet boxes, 416x416 YOLO-style; Giemsa); MIT; not selected. Low-medium confidence.",
  "octid": "Plausible source of 4 public rows (OCT with macular hole / DR / AMD classes); CC0; not selected.",
  "sicapv2": "Plausible source of 8 public rows (Gleason level 1-5 patches, 512x512); CC-BY-4.0; not selected.",
  "polypgen": "Colonoscopy is badly under-filled (645 of 8,000 planned images); MMBU polyp segmentation rows (10) come from a Kvasir-SEG/CVC/PolypGen-type set; PolypGen is the only tier-A one; CC-BY-4.0, registration route.",
  "mpox-hsam-6class": "Plausible source of 5 public rows (monkeypox/cowpox/chickenpox/measles/smallpox/healthy skin photos); CC-BY-4.0; low weight."
 }
}
```

## Route / eligibility fixes worth a human look (not in the json block)

* `bbbc010-celegans-live-dead` (15 public rows, CC0, tier A) is ineligible only because the registry has no scriptable route. A direct file exists: `https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip` (HTTP 200, 68 MB; the v1 images URL returns 403). Adding it as a `route: direct` entry would make the cluster usable.
* `osteosarcoma-utsw` (5 rows, CC-BY-3.0, TCIA) and `mouse-embryo-hfu` (7 rows, IEEE DataPort) have no route; the former needs a TCIA-specific route, the latter needs a login.
* `deepdrid` (CC-BY-SA-4.0, tier A) has no route; Kaggle copies `yoctoman/deepdrid` (CC BY 4.0, 1.4 GB) exist and could serve as a route if the DR-grading/laterality/quality clusters (O3, O8, ~78 rows, L confidence) are DeepDRiD.
* **License policy decision for the user**: with the organizers' permission for source datasets, the largest gap is tier-C/D data. The two cheapest wins are `jsiec-1000` (182 rows; Zenodo license id `other-open`, Kaggle "Open Database", copyright note says JSIEC) and `lc25000` (27 rows, paper says "freely available", no license file). The registry marks both tier D (unverified/custom). Whether to relax tier for them is a policy call; I did not change any registry value.
* Note on `chest-ctscan-images` (9 rows, ODbL, tier D): the exclusion list says Lingshu trained on it, which would contradict it being an MMBU source; the 9 rows may come from another aggregator.

## New TOML ids

**None added.** Every identified source that is public already exists in the registry (5,099 entries). The unidentified sources are listed as leads in the table with no registry match:

* amyloid-beta IHC plaque morphology (diffuse/cored/CAA/negative, 256x256, 0.5027 mpp): 23 public rows; Zenodo candidates found (Scalco et al. 131 WSIs, zenodo 10668642, CC-BY-4.0, but 133 GB of .svs and not patch-labelled) do not clearly match, so no TOML was written.
* Hitachi transvaginal US early-pregnancy boxes (34 rows), intussusception abdominal US (6), AV-nicking fundus (2), glaucoma-severity fundus 240x240 (10), 'what structure' fluorescence/calibration-bead set (about 75 rows), cross-species EM structure set (15), SBF-SEM cultured cell types (15), cryo-ET mitochondria (8), zebrafish adipose confocal (11), breast histologic grade patches (7), synthetic eccentricity phenotype set (38 rows: partly `bbbc005`/`bbbc031`-like, unconfirmed). Zenodo/Kaggle/DataCite/HF API queries for these returned nothing that matches with evidence, so they stay leads.
