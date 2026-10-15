# Pass-2 gap discovery (2026-10-07, agent tag `waila-gaps`)

Target: classes in `plan/label_space.json` with 0-2 `train_datasets`, restricted to MMBU's public vocabulary. Method: Zenodo, figshare, HF, Kaggle, BioImage Archive (BioStudies), DataCite, RODARE APIs plus 8 WebSearch calls (budget 30) and 3 WebFetch of BBBC pages. Metadata only, no data downloaded. 21 new TOMLs written, all tier A, validator: 0 errors in `path-he`, `fluor`, `cyto-heme`, `lm-cellbio`, `bio-organisms`, `em`. Slice `_progress.md` files updated.

## Main finding: most "0 dataset" classes are tagging gaps, not data gaps

The registry (5,099 TOMLs) already holds tier-A open datasets for nearly every class below, but they carry coarser tags than MMBU's public vocabulary (`blood`, `lung`, `colon`, `brain`, `drosophila`) so the plan counted them as 0. The public eval parquet shows what each class really means:

| Class | What the 8-26 public items are | Existing registry source |
|---|---|---|
| lung tissue / colon tissue | LC25000-type H&E tiles (digital pathology, brightfield) | `lc25000` is tier D (no license); new CC BY copy below |
| lymphocytes, neutrophils, eosinophils, monocytes, white blood cells | brightfield blood-smear WBC sets (20 items) | ~36 tier-A WBC sets in `cyto-heme`, tagged `blood` |
| oil red o | 11 items, all C. elegans whole-body brightfield | `bbbc011-celegans-metabolism` (tier D) |
| kc167 cells | epifluorescence Hoechst/phalloidin (23 items) | `bbbc007` (A), `bbbc002` (D) |
| mouse embryo cells | DIC "contrast microscopy" (17 items) | `bbbc003-mouse-embryos-dic` (D) |
| leishmania haptomonad, guinea pig | SEM, uranyl acetate (5 and 2 items) | `empiar-leishmania-haptomonad-sbfsem` (A); guinea pig new below |
| cellmask, lysotracker | epifluorescence, human/vero (5 and 4 items) | `bbbc034`, `spacr-cross-channel-*` (CellMask only) |
| thoracoabdominal region | 19 conventional-CT items | TotalSegmentator, CT-ORG, SAROS etc. |
| abdominal organs, retroperitoneum, genitourinary | laparoscopy (26, 15, 23 items) | ~44 tier-A endo-surgical sets, only 17 selected |
| upper GI tract, esophagus | white-light endoscopy | HyperKvasir, GastroVision, GIED, Renji-ESD (unselected) |
| glandular epithelium | 8 WSI items | `crc-ihc-cish-he-epithelium`, `gzmh-gland`, `gagl-gastric-glands` |
| reproductive system | 15 brightfield items | sperm sets (`shseg-sperm-head-2026`, `sperm-scd-instance-seg-2026`, `smids`) |
| cerebrum | 2 multiphoton items | `bass-2017-axonal-synapses-2p`, `bass-axon-segmentation-2p-labels` |

Recommendation: fix tags/overrides for the existing ids in the "Retag" section first (cheapest, zero download risk), then add the new ids below.

## New datasets by class (all tier A; "count" = what the source states, unknown where archives were not opened)

| Class (field) | New id | Count | License | Route |
|---|---|---|---|---|
| lung tissue, colon tissue (body_part) | `lc25000-zenodo-cc-by` | 25,000 H&E tiles (paper; zip 0.93 GB) | CC BY 4.0 (author's Zenodo, verified) | Zenodo 14998042, direct file |
| colon tissue | `colopola-polarimetric-biad1989` | 20,592 TIFF (572 slices x 36 polarization states) | CC0 | BioImage Archive S-BIAD1989 |
| lung tissue | `tem-postmortem-covid-biopsies-biad3706` | ~22 TEM images (lung/liver/kidney/spleen) | CC BY 4.0 | BIA S-BIAD3706 |
| lung tissue | `rat-lung-stereology-lm-tem-zenodo-3738928` | unknown (LM zip 0.34 GB, EM zip 1.5 GB) | CC BY 4.0 | Zenodo 3738928 |
| white blood cells, lymphocytes, monocytes | `blood-cell-painting-pbmc-biad2344` | 577,256 files (PNG) | CC0 | BIA S-BIAD2344 |
| white blood cells (+neutrophil/eosinophil/lymphocyte/monocyte text) | `wbcmor-vqa-figshare-2026` | unknown QA (images.zip 0.39 GB) | CC BY 4.0 (upstream WBCAtt license unclear) | figshare 32727159 |
| leukocytes | `leukocyte-tem-yolo-zenodo-5024988` | unknown frames (4 train zips ~50 MB) | CC BY 4.0 | Zenodo 5024988 |
| lymphocytes, glandular epithelium, specimen cat | `feline-intestinal-lymphocytes-he-biad1129` | 10,994 files (json/rds/png overlays; raw WSIs unverified) | CC BY 4.0 | BIA S-BIAD1129 |
| reproductive system (body_part) | `rat-estrous-cytology-he-biad647` | 127 JPEG, H&E | CC0 | BIA S-BIAD647 |
| specimen red blood cells | `pig-rbc-deformability-biad1458` | 2,009 JPEG, 3 deformability grades | CC0 | BIA S-BIAD1458 |
| specimen guinea pig | `em-metabolic-tissues-biad3923` | 351 EM (311 SEM + 40 TEM; mouse/human/guinea pig) | CC BY 4.0 | BIA S-BIAD3923 |
| specimen guinea pig | `guinea-pig-skin-tick-bite-he-zenodo-7884760` | 4 H&E CZI images (8 GB) | CC BY 4.0 | Zenodo 7884760 |
| specimen vero cells | `vero-e6-culture-images-zenodo-16619228` | unknown (4 zips, ~6 GB; LM + TEM) | CC BY 4.0 | Zenodo 16619228 |
| stain oil red o | `oro-liver-hfd-mouse-biad1212` | 8 TIFF | CC0 | BIA S-BIAD1212 |
| stain oil red o | `oro-syzygium-atherosclerosis-plos-2026` | unknown (3 zips, 21-64 MB) | CC BY 4.0 | figshare 32179589, 32179577, 32179571 |
| stain lysotracker (+hoechst) | `lysotracker-autophagy-fadu-hgfb-zenodo-7116549` | 702 CZI z-stacks (12 GB) | CC BY 4.0 | Zenodo 7116549 |
| stain cellmask (+dapi) | `hcc1143-paclitaxel-cellmask-dapi-zenodo-11237850` | unknown (6 zips ~2 GB, InCell) | CC BY 4.0 | Zenodo 11237850 |
| stain cellmask (+dapi) | `tnbc-sirna-paclitaxel-cellmask-zenodo-11238552` | unknown (6 fixed-plate zips, 27 GB) | CC BY 4.0 | Zenodo 11238552 |
| stain fluorescent beads | `alveolar-macrophage-beads-phagocytosis-biad1871` | 612 TIFF | CC0 | BIA S-BIAD1871 |
| stain dapi/phalloidin (cell seg) | `helacytonuc-rodare-3001` | 2,676 images + nuclei/cytoplasm masks, official split | CC BY 4.0 | RODARE record 3001 (direct URLs) |
| (body_part multiple, mouse histology) | `impc-histopathology-biad3040` | 13,298 files (JPEG/TIFF) | CC BY 4.0 | BIA S-BIAD3040 |

Overlap risk: `lc25000-zenodo-cc-by` = high (the MMBU public pooled label `colon tissue|lung tissue` matches it; fine for Pass-2 training, never submitted). `helacytonuc-rodare-3001` = medium. Everything else low (recent deposits, raw archives, derived QA).

Strongest finds: `lc25000-zenodo-cc-by` (explicit CC BY from the author; the existing `lc25000` entry has no license and was unselectable), `em-metabolic-tissues-biad3923` (per-image guinea pig SEM), `lysotracker-...-7116549` and the two Heiser-lab CellMask sets (first named LysoTracker/CellMask epifluorescence supply in the registry), `blood-cell-painting-pbmc-biad2344` (577k CC0 WBC fluorescence images).

## Classes with no honest new supply
- tobacco: only `empiar-11831-tobacco-chloroplast-sbfsem` (existing) plus CVPPP (class C). The BY-2 microtubule set found (figshare 27683220) ships model weights only.
- kc167 cells, mouse embryo cells: only the classic BBBC sets, whose licensing is unstated (see rejected).
- retroperitoneum, abdominal organs, genitourinary (laparoscopy), thoracoabdominal CT, upper GI endoscopy, musculoskeletal system MRI: registry already saturated (CT/GI) or surgical video is not on scriptable open platforms; resolve by retagging.
- specular microscopy, tetraspeck beads, alexafluor-tubulin: not addressed (earlier batch A found no supply for specular microscopy).

## Retag / re-select suggestions (existing registry ids, tier A, open route)
- lung tissue / colon tissue: `nct-crc-he-100k` (100k), `huncrc`, `crc-val-he-7k`, `bari-crc-tiles-2022`, `kather-gi-tumor-detection-tiles`, `mihic` (310k lung), `lunghist700`, `histoset-5x14`.
- WBC classes (white blood cells, lymphocytes, neutrophils, eosinophils, monocytes): `raabin-wbc`, `multifocus-wbc-park-2024`, `wbc-hr-bodzas-2023`, `univali-leukocyte-14class`, `blood5-wbc-2026`, `chula-wbc-2025`, `naturalize-2k-pbc-2023`, `leukocyte-classification-ieee-song-2025`, `wbcatt-plus-2026`, `mll23-peripheral-blood`.
- lower respiratory tract: `zn-sputum-its-indonesia-2026`, `afb-sputum-imelda-2026`, `ai-tb-zn-uganda-2026`, `tb-bacilli-cusat-2026` (plus selected `tb-sputum-ufam-figshare-plus-2025`, `clinical-bacteria-gram-mrose-2024`). These also lift `ziehl-neelsen` and `gram` (1 dataset each).
- reproductive system: `shseg-sperm-head-2026`, `sperm-scd-instance-seg-2026`, `smids`, `evisan`.
- glandular epithelium: `crc-ihc-cish-he-epithelium`, `gzmh-gland`, `gagl-gastric-glands`.
- upper GI tract / esophagus: `hyperkvasir`, `gastrovision`, `gied`, `renji-esd-esophagus`, `renji-esd-egc`, `gastric-lesion-labeled-images-zeng`.
- thoracoabdominal region: `totalsegmentator-v1`, `ct-org`, `saros`, `totalsegmentator-ct-idc-segmentations`, `medmnist-ct-subsets`, `tcga-esca-ct`, `tcga-coad-ct`.
- laparoscopy abdominal organs / genitourinary: `simuscope`, `t-dvc-cholect44`, `lapgc-kvad-30`, `raar-imac-30`, `ud-ureter-uterine-artery-nerve`, `enid` (need per-dataset organ check).
- cerebrum: `bass-2017-axonal-synapses-2p`, `bass-axon-segmentation-2p-labels`.
- specimen kc167 cells: `bbbc007`; leishmania haptomonad: `empiar-leishmania-haptomonad-sbfsem`; tobacco: `empiar-11831-tobacco-chloroplast-sbfsem`; vero cells: `sbfsem-vero-sars-cov-2-mitochondria`, `rki-sars-cov-2-variants-thin-section-tem`, `rki-sars-cov-particles-tem-2020`, `covid-if-training-data`; P. falciparum/RBC: `malaria-sulawesi-yolov8-2026`, `nlm-malaria-screener-sudan`, `malaria-coco-nih-thin-2025`, `bbbc041seg-depto-2021`, `nih-nlm-thin-smears-pf`.
- stains: oil red o `multiplex-special-stains-shen`; cellmask `bbbc034`, `spacr-cross-channel-nuclei-from-cellmask`, `spacr-cross-channel-toxoplasma-from-cellmask`; masson's trichrome `masson-saphenous-vein-wall`, `wound-healing-collagen-mt-shg-2026`.

## Rejected or not added, with reasons
- BBBC002 (Kc167), BBBC003 (mouse embryo DIC), BBBC011 (C. elegans Oil Red O): BBBC pages state no usable license (BBBC011 "Copyright Broad Institute"; BBBC003 text cites CC BY-NC-SA for its images). Stay tier D. BBBC007 copies on Cell Image Library are described as public domain, but it is already in the registry (tier A).
- Zenodo 5814298 (WBC + cytoplasm, StarDist): CC BY on Zenodo, but it re-hosts a Mendeley dataset (10.17632/w7cvnmn4c5) whose license is CC BY-NC 3.0. License conflict.
- Zenodo 44884 (WBC + MCF7 confocal): LSM-only, 278 GB.
- BioImage Archive S-BSST567, S-BIAD43, S-BIAD44 (label-free P. falciparum RBC, UV and Leica): no license field in the records (0.7-1.4 M files).
- S-BIAD2983 / S-BIAD3460 (iPSC organelles multispectral): CC BY but CZI 0.7-2.4 GB each and organelle markers unnamed (no stain evidence). S-BIAD1702 Cytoland (Hoechst + CellMask test sets): OME-Zarr zips, 100+ GB.
- figshare 7492664 (sperm acridine orange): the file is only a Dropbox link, not scriptable. figshare 14289268 (guinea pig cochlea SEM): 5 figure panels. figshare 27683220 (BY-2 microtubule models): model weights.
- Zenodo 17964511 (Leishmania giemsa skin smears): no license in record (existing `leishmania-cl-libya-2026`, tier D).
- Kaggle `saife245/malaria-plasmodium-falciparum-species` (CC0, 127 MB) and `tmtrnhelloworld/lungcoloncancersplitted` (CC0): re-uploads of third-party sets (Gutierrez malaria; LC25000), not original; not added.
- S-BIAD1657 (C. elegans SFP-1 fat metabolism, CC0): unclear whether the images are Oil Red O; S-BIAD300 (C. elegans lifespan, 36k files) has no license field; S-BIAD2282 (Vero phospholipidosis) is CC BY-NC.
- Beads sets (Zenodo 15149351 "Beads - Operetta", 3632331 OPT beads, S-BSST622 HyLFM beads): calibration data, not processed. MMBU's tetraspeck/bead items are too rare to justify.
- Zenodo 20149726 (Pap smear, Tello-Mijares): likely the same group as existing `pap-nuclei-seg-tello-2026`; Pap is already covered by many registry sets.
- HF laparoscopy/surgical repos found (`ali-sh07/endoscopes-laparoscopic-cholecystectomy`, `introvoyz041/cholec80` etc.): mirrors of datasets already in the registry or without a license.
- RODARE 3900 (virus-infection reporter virtual staining, CC-BY): numpy arrays, 10-35 GB per virus, cell types and stains not read; noted as a lead in `fluor/_progress.md`.

## Caveats
- Image counts marked "unknown" need a first-pull enumeration; zips were never opened.
- Two entries are low-confidence content: `feline-intestinal-lymphocytes-he-biad1129` (may hold only overlays/ROIs) and `impc-histopathology-biad3040` (stain not stated in the record, left empty).
- `wbcmor-vqa-figshare-2026` inherits upstream WBCAtt image-license uncertainty.
