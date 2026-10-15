# Pass-2 labelling audit, part A (selection.jsonl lines 1-197)

Scope: metadata-only audit of 197 selected datasets (file listings, zip central directories via remotezip, first 64 KB of csv/tsv/json/xlsx, HF datasets-server `/info` + `/first-rows`, OpenNeuro S3 listings). No images or archives were downloaded. Per-dataset records are in `audit_A.jsonl` (one JSON object per dataset: `access_status`, `metadata_sources`, `per_field`, `problems`, `notes`). Heads are first-50-row samples, so a column that looks constant in the head (for example `cell_line`) can vary later in the table.

Method per dataset: lister from `src/bio_waila/data/download.py` -> structure summary (extensions, folder levels, mask-like count) -> zip member listing for the 6 largest zips -> table heads -> `match_value` on every folder level / table column / HF ClassLabel -> manual review against the registry label.

## 1. Access status (197 datasets)

| status | n | datasets |
|---|---|---|
| ok (listable, images present; caveats may still apply, see `problems`) | 154 | |
| ok, but single `.rar` / `.tar` / `.tar.gz` / split archive: no remote listing, `src/bio_waila/data/download.py` cannot range-sample | 11 | wbcatt-plus-2026, protlocnet-2026, tb-sputum-ufam-figshare-plus-2025, mslesseg-2024, exodeepfinder-lysosomal-exocytosis-tirf, plasmodium-4species-rtdetr-france-2023, cross-choleact, nec-xpert, modus-urine-sediment-2026, angiocad, hcfmrp-covid19-lid |
| ok, but pixels only as zarr / npz / h5 / ims arrays | 3 | harris-2015-hippocampal-neuropil, ispy2, 2p-ivm-leukocyte-cell-death |
| ok, but nested zips (cannot be sampled) | 1 | intrapartum-us-video-dataset |
| ok with caveats that change what is downloaded | 4 | deepspot-smfish-simulated, mitoem, harvard-fairfedmed-oph, mammo-mx |
| **record has no usable pixels** (figure/PDF/xlsx/manifest/code only, or images hosted elsewhere) | 13 | rodent-estrous-cytology-compendium-hf, embryonet-zebrafish, popsicle-cryoet-benchmark, biotisr, tryptag-tbrucei-localisation, lapgc-kvad-30, behsof-nafld-us, grape, drosophila-wing-database-gigadb, laparoscopic-gastrectomy-annotated-images, cervical-tct-heilongjiang-2025, cholec80-cvs, sonocorpus |
| **masks / labels only** | 2 | vertebral-bodies-ct-labels, crlm-ct-seg-refined-flr |
| **wrong source** (galar: figshare record is split CSVs; real data are 7z archives on figshare.plus 25304616, up to 202 GB each) | 1 | galar |
| **moved** (mitoem-2: HF repo is README-only; data is Zenodo 20417683, 16 zips with a metadata.csv) | 1 | mitoem-2 |
| restricted (Zenodo access request) | 3 | smart-om, liver-mets-longitudinal-ct-isfahan, dermlens-5k |
| embargoed | 1 | ptc-thyroid-fnac-elazig-2025 |
| needs Kaggle credentials | 3 | gbm-multi-institutional-ds007045 (also on OpenNeuro ds007045: verified, 1,452 T1w niftis, no auth -> switch method), rdbmc-romanian-dense-breast, lymphus-cervical-ln-ptc (original site qamebi.com) |
| 404 / login walls | 0 | |

24 datasets (12%) cannot be used as selected (no pixels, masks only, wrong/moved source, restricted, embargoed, credentials); 15 more only work with new archive handlers (rar, tar.gz, split zip, nested zip, zarr/h5/npz).

## 2. Which fields become hard through metadata

Registry modes in this half: modality 197 hard; submodality 196 hard + 1 masked; body_part 70 hard / 114 candidates / 13 masked; stain 29 hard / 6 candidates / 162 masked; specimen 164 hard / 26 candidates / 7 masked; domain 195 multi / 2 masked.

| field | non-hard in registry | resolved per sample by metadata | resolvable only by a dataset-level decision (registry edit) | unresolved | notes |
|---|---|---|---|---|---|
| submodality | 1 | 0 | 0 | 1 (harris-2015: TEM, no class) | 5 more **hard** labels are wrong for part of the images and need a per-sample split: cystods (BLC), mcoa-corneal-opacity (ASP photos), woives-uwf-octa (SLO/CFP/en-face), biotisr (SIM variants), sechenov-cystoscopy (light mode) |
| body_part | 127 | 13 | 85 | 29 | per-sample: openorganelle, cellmap-challenge-2025, mitoem-2, openswisshcc-2026 (BIDS sidecar), pan-multiplex-nimbus, totalsegmentator-v3, tissue-fold-teaching-slides, magic-ct, rcc-aid, lnq2023 (DICOM), recist-ct-hcuch, pediurf, mcoa-corneal-opacity |
| stain | 168 | 2 | 1 | 40 (125 are non-microscopy: not applicable) | per-sample: rodent-estrous (Shorr etc.), spinedl (by zip name); biotisr registry hard=phalloidin is wrong (mEmerald/Lifeact/Halo fusions) |
| specimen | 33 | 13 | 6 | 14 | per-sample: openorganelle, cellmap, mitoem-2, spherohq, spheromix, vetxray, rodent-estrous, embryonet, popsicle, mitoem, uaal, leishmania-canine, bacteria-mother-machine-7 |

The 85 dataset-level body_part cases mean the registry candidate set is a topic-level ambiguity that no per-image metadata resolves. Recommended registry edits (priors checked against `plan/eval_public.parquet`, e.g. fundus: eye 407 / retina 40; chest x-ray: chest+lung 271 / chest 126):
- brain 30 (all brain MRI / head CT/CTA / brain 2-photon sets in this half), eye 7 (fundus, orbital MRI), retina 4 (OCT/OCTA), multi [chest, lung] 4 (chest radiographs), liver 3, prostate 3, lung 2 (LIDC, NSCLC CT), spine 2, coronary arteries 2, pelvis 2, cervix 2, whole body 2, [abdomen, pancreas] 2, elbow 2, abdomen 2; 1 each: thorax, ovary+pelvis, oral cavity, cornea, anterior segment, gallbladder, head and neck, skeletal system, eye+retina, bladder+epithelial cells, sperm cells, eyelid, fetus+pelvis, gallbladder+liver.
- DICOM datasets (nsclc-radiogenomics, hcc-tace-seg, qin-prostate, nci-isbi, lnq2023, head-cta, vetxray): BodyPartExamined / Modality / SeriesDescription can only be read after download (header-only); they are flagged in each record.

Net effect: in this half metadata resolves specimen for 13 datasets, stain for 2 and body_part for 13; body_part is mostly a registry (dataset-level) decision, not a metadata one.

## 3. Most common metadata patterns

1. **Class/diagnosis folders or columns** (the majority of datasets): folder names or ClassLabel names are diagnoses, cell types or taxa (Normal/Glioma, Bethesda classes, plankton taxa, BA/non-BA). These never carry Pass-2 fields; `SKIP_COLUMN` handles most columns but not path levels (see matcher fixes e, i).
2. **Mask/label siblings next to images** (88 datasets list mask-like members): `masks/`, `labels/`, `annotation_N`, `seg/`, `GT/`. Current `is_mask` catches most but has false positives/negatives (section 5).
3. **NIfTI/BIDS** (45 datasets): sidecar JSON with `Modality`, `BodyPartExamined`, `SeriesDescription` exists for openswisshcc, BIDS suffix gives T1w/T2w/FLAIR. Suffixes used by BraTS-style sets (`t1c`, `t1n`, `t2f`, `t1gd`, `T1wCE`, `MRA`) are not in `BIDS`.
4. **HF parquet with image bytes** (40 datasets): only 6 expose ClassLabel; 61 HF datasets have plain string/int columns. Most HF image columns sit next to mask Image columns (`mask`, `seg`, `overlay`, `cup_exp1`...).
5. **Patient/series tables** (csv/xlsx in 90 datasets): clinical data only; Pass-2-relevant columns are rare.
6. **MedOtter DICOM/NIfTI mirrors** (`MedOtter/*` repos, 29 in this half): `series_to_patient.json` + preview parquet; no Pass-2 columns.
7. **Zip with images/ + masks/ + annotations** (118 datasets with at least one listable zip): central directory gives path levels but nothing else.
8. **Per-sample Pass-2-style metadata is the exception**: nirschl-heart-failure (modality/submodality/stain columns), cystods (`modality` WLC/BLC), mcoa (`AS-OCT|ASP` path level), woives (`OCTA|SLO|CFP`), recist-ct-hcuch (`abdomen|thorax` path), tissue-fold (`organ` path), magic-ct (zip name), mitoem-2 (`metadata.csv` organism/tissue), rcc-aid (`BodyPartExamined`), spherohq/spheromix (`cell_line`), vetxray (`specie`), uaal (zip `clinical|phantom`), nih-chestxray14 and tompei-cmmd/mammo-mx (`View Position`, `view`, `View`).

## 4. Matcher fixes and new mappings for `src/bio_waila/data/rules.py` (with evidence)

Verified by running `match_value` / `bids_hits` on the values seen.

Wrong mappings (add tests):
- a. `angiogra\w*` -> x-ray: `'ct angiography abdomen-pelvis-'` (totalsegmentator-v3 `meta.csv` study_type) returns modality x-ray + submodality fluoroscopy (both strong). CT/MR angiography must stay CT/MR; apply the angiography rule only when no `ct|cta|mr|mri` token is present.
- b. `mammo\w*` is too greedy: `'Mammosomatotroph'` (pituitary cell type, pituitary-tumor-carotid-mri `cell_type`) -> x-ray + mammography. Use `mammogra\w*|\bmammography\b|\bmammo\b`.
- c. `PATH_CODES['ct']`: a folder named `CT` (= control) in zebrafish-embryo-phenotypes-jeanray2015 -> computed tomography. Ignore path-code hits when the registry modality is hard and another path level/sidecar disagrees, or drop `ct` from PATH_CODES (keep `mr`/`mri`).
- d. `colposcop\w*` -> endoscopy: MMBU public and the registry both put colposcopy under **optical imaging** (4/4 in eval_public). Map to optical imaging; also `'colposcopist_notes_v1'` (intelkamat `source` column) triggers it: require a word boundary / modality-like column.
- e. Weak `us` token: `'ASC-US'` (Bethesda class folder in bicervi-2026) -> ultrasound (weak). Add a stoplist (`asc-us`, `asc-h`).
- f. Stain hits on non-microscopy data: `'ZN_PAIR_0011'` (mri-pairedfield sample folders) -> ziehl-neelsen (strong). Restrict stain (and specimen) hits to `STAINED_MODALITIES` / microscopy registry modalities.
- g. `FIELD_COLUMNS['body_part']` contains `primary`: beamster `Origin`/`primary_origin` ('Non-Small Cell Lung Cancer', 'Gastrointestinal tract' = primary tumor site of brain metastases) -> lung / GI tract. Remove `primary`, add `origin|primary|metast` to `SKIP_COLUMN`.
- h. Exact class names are trusted in any column: `'Tobacco'` (habit column, code-oral-mucosa) -> specimen `tobacco`; `'Homo sapiens (Jurkat cell line)'` -> body_part `cell` instead of specimen cell line. Restrict specimen exact-name matches to specimen-like columns or add `homo sapiens ... cell line`, `jurkat|hela|cos-?7|u2os|l929|mcf-?7|sum-?159` -> specimen cell line.
- i. Disease/cell-type terms mapped to body parts: `'Optic Disc Disorder'` -> optic disc, `'Retinal Hemorrhage'` -> retina (fd3611 findings), `'Segmented neutrophils'` -> neutrophils, `'Neutrophils_crops'` (2p-ivm). Add a diagnosis-term stoplist (disorder, hemorrhage, occlusion, retinopathy, edema, carcinoma, tumor, ...) for body_part, or skip folders whose siblings are classes.
- j. Multi-region values give nothing: `'ct thorax-abdomen-pelvis'` has 3 strong body hits and returns none. Return a multi-label list (`mode=multi`) when >=2 strong body hits appear in one value.

New value -> class mappings to add:
- submodality CODES: `wlc -> white-light endoscopy`; `blc` (blue-light cystoscopy) -> no class: exclude. `ssSEM`, `FIB-SEM`, `SBF-SEM`, `SEM` -> `scanning electron microscopy` (FIB-SEM/SBF-SEM already give modality electron microscopy but no submodality; ssSEM gives nothing). `ssTEM` -> electron microscopy, no submodality. `TIRF-SIM` already -> TIRF; `GI-SIM`, `3D-SIM`, `NL-SIM` -> fluorescence microscopy, no submodality. `AS-OCT` already -> OCT; `ASP` (anterior segment photograph) and `SLO` -> unmapped (mask). `Enface_*` -> OCT.
- specimen: `mus`/`jrc_mus-*` -> mouse; `jrc_fly-*` -> drosophila; `jrc_zf-*` -> zebrafish; `specie` (column name; FIELD_COLUMNS has only `species`); `dogs` -> dog (plural); `Sprague Dawley|Wistar` -> rat; `C57BL/6J|BALB/c` -> mouse; `ecol` -> e. coli (other 4-letter species codes -> bacteria); `coco_ins_phantom` / config `phantom` -> phantom.
- body_part: zip/folder organ tokens already work (`Small_Intestine`, `kidney_cyst_scans4`); `Testis`, `Reproductive (epididymis)` have no class; `Opaque/Normal Cornea` -> cornea is correct but is a condition label.
- BIDS suffix table: add `t1, t1c, t1ce, t1gd, t1n, t2, t2f, t1wce` -> structural mri; `mra, dwi, adc, trace, swi, cbf, cbv, ttp, tmax, asl` -> modality MRI only (not structural mri). Currently `UCSF-PDGM-*_T1c.nii.gz`, `*_brain_t1n.nii.gz`, `*_t2f.nii.gz`, `W1_t1gd.nii.gz`, `sub-064_T1wCE.nii.gz` all return `{}`.
- `PATH_CODES`: allow trailing sequence tokens (`Meningioma T1C+`, `Glioma T1`): currently only whole-folder `t1/t2` match.
- `.tab` (Dataverse tab-separated, bosque-test-set `BOSQUE_metadata.tab`) is missing from `META_EXT`.

## 5. Pipeline (src/bio_waila/data/download.py / files.py) problems found

1. **No `__MACOSX` / `._*` filter** (10+ datasets: vetxray-canine-feline-thorax, nih-chestxray14, deepcell-brightfield-single-cells, multifocus-wbc-park-2024, bosque-test-set, pancreas-pda-segmentations-mayo, mfxr-monteggia, rescale4dl-u2os-nuclei, ...): resource-fork entries have image/volume extensions and are picked as media (up to 50% of members).
2. **`is_mask` false positives/negatives**: `manual` token marks every file of delta-ecoli-mother-machine-fluor (folder `from_manual_mixed_200x800`) -> 0 images; `gt` token excludes the high-SNR image half of care-restoration-datasets; `labelsBraTioUS-public-dataset` (glued token) and `*_od_K.png` (fovea-intraoperative-fundus) are not caught.
3. **Extension-less DICOM** (nsclc-radiogenomics 593/749, qin-prostate 330, nci-isbi, mammo-mx all members): `kind()` returns `other`.
4. **Unsupported containers**: `.rar` (5), multi-volume zip/rar (plasmodium-4species, angiocad, pan-multiplex codex_colon), nested zips (deepspot data.zip, intrapartum-us), single `tar.gz` (tb-sputum 127 GB, cross-choleact, protlocnet, hcfmrp rar), zarr-in-zip (harris-2015), h5/ims/npz (2p-ivm, m4raw k-space, ispy2).
5. **Archive ordering**: archives are sorted biggest-first and filtered only by `mask`/`label` in the name: picks `models.zip`/`.ckpt` (cardiag 17 GB, plasmodium), `5-fold.zip` derived plots (biliary-atresia), a code+detections zip (pediashf-dx), cumulative duplicate versions (code-oral-mucosa V1-V5, tryp v1/v2/v3, dr2 by-lesions/by-referral, spheromix vs spherohq, fiqs full-size vs resized, ispy1 two image versions, cisd stack/edf/center_slice, rvo-me-oct 3 copies).
6. **`fetch_hf` fallback**: falls back to file listing only when `load_dataset` raises. harvard-fairfedmed-oph loads the Chest CSV config (no Image column) -> "no image column found"; rodent-estrous has `image_url` strings only; mitoem-2 repo has no data.
7. **Zip metadata pick**: `meta[:50]` takes the first 50 table-like members (per-image json/csv), which can crowd out the one useful table (deepspot csv/N.csv, SENORA).
8. **Records whose pixels are elsewhere**: see section 1 (13 datasets).

## 6. Registry errors to fix (labels contradicted by the files)

| dataset | field | registry | evidence | fix |
|---|---|---|---|---|
| cystods | submodality | white-light endoscopy (hard) | `modality` column WLC/BLC (450 of 8067 are blue-light) | per-sample WLC -> WLE, drop BLC |
| mcoa-corneal-opacity | submodality | OCT (hard) | path d4: AS-OCT 9264 / ASP 579 anterior segment photographs | per-sample split |
| woives-uwf-octa | submodality | OCTA (hard) | path d3: OCTA 994 / SLO 128 / CFP 63 + OCT en-face layers | per-sample split (PATH_CODES handle octa/cfp) |
| biotisr | stain, submodality | phalloidin, TIRF (hard) | imaging_conditions.xlsx: Label = mEmerald/Lifeact/Halo fusions; Imaging method GI-SIM/3D-SIM/TIRF-SIM/NL-SIM | stain -> gfp (or mask), submodality only for TIRF-SIM; also no images in the record |
| pan-multiplex-nimbus | modality | fluorescence (hard) | mibi_decidua is MIBI (mass spec ion beam) | split MIBI |
| synthstrip-dataset, soop-stroke, senora-mri, isles-2015 | submodality | structural mri (hard) | contain EPI/MRA/DWI/perfusion (CBF, CBV, TTP, Tmax) volumes; SynthStrip upstream also CT/PET | filter by suffix or relabel |
| nsclc-radiogenomics | modality | CT (hard) | TCIA series include PET | filter series_to_patient.json Modality |
| wormswin-celegans | submodality, specimen | brightfield, c. elegans (hard) | synthetic_images_dataset.zip (10,000 of ~15,000 images) is generated | exclude zip |
| deepspot-smfish-simulated | submodality, specimen | synthetic (hard) | data.zip/wound_healing_assay.zip is a real assay | exclude |
| drosophila-wing-database-gigadb | body_part | upper limbs (hard) | insect wing, no class | mask; also no data |
| sechenov-cystoscopy | submodality | WLE (hard) | clinical_info.xlsx Light mode: CLARA+CHROMA (28) / white light (2) | verify rest of column |
| bosque-test-set | submodality | dermoscopy (hard) | metadata (skin_phototype, evolution, size buckets, clinical_label) reads like clinical photography | verify images |
| feltrin-brain-tumor-mri-44c | id/name | 44 classes | HF repo has 14 classes / 3,827 images; overlaps feltrin 30c | rename / dedupe |
| mitoem | body/specimen | candidates human, rat | file names EM30-H / EM30-R -> human / rat; zip tifs are `seg*.tif` labels; only rat image volume visible | restrict |
| spherohq / spheromix | duplicates | both selected | identical file names in both zips | keep one |
| mitoem-2 | method | hf | HF is README-only | zenodo:20417683 |
| gbm-multi-institutional-ds007045 | method | kaggle | same data on OpenNeuro ds007045 | method=openneuro |
| galar | method/ref | figshare 31065262 | CSV splits only | figshare.plus 25304616 (7z, huge) or drop |

Specimen/stain notes: registry specimen candidates for openorganelle/cellmap omit zebrafish (`jrc_zf-cardiac`); rodent-estrous stains (Shorr) have no class; vocab has no TEM, plankton, trypanosoma, planaria, Tribolium or slit-lamp/SLO classes.

## 7. Datasets to drop or re-point (action list)

Drop or re-point: galar, mitoem-2 (-> zenodo), gbm-multi-institutional-ds007045 (-> openneuro), lapgc-kvad-30, behsof-nafld-us, grape, drosophila-wing-database-gigadb, laparoscopic-gastrectomy-annotated-images (ref `058882` looks mistyped), cervical-tct-heilongjiang-2025, cholec80-cvs, sonocorpus, vertebral-bodies-ct-labels, crlm-ct-seg-refined-flr, popsicle-cryoet-benchmark, rodent-estrous-cytology-compendium-hf, biotisr, tryptag-tbrucei-localisation, embryonet-zebrafish (code + 725 demo jpgs only), smart-om, liver-mets-longitudinal-ct-isfahan, dermlens-5k, ptc-thyroid-fnac-elazig-2025, rdbmc-romanian-dense-breast and lymphus-cervical-ln-ptc (manual download).

Total useful images per selected dataset can fall well below `plan_images` where the record is small (embryonet 725 vs 2110 planned, celegans-l1-nuclei-3d 28 volumes, camg-meibomian 100 base images, mm-odir-129 129, bosque 151).

## Files

- `audit_A.jsonl` - one record per dataset (197 lines).
- `_audit_A_src/` - `audit_collect.py` (metadata-only collector, writes heads to a temp dir), `audit_digest.py` (compact digest + auto-mapping), hand-written per-dataset overrides `ov*.py`, and `assemble.py` that regenerates the jsonl. The raw heads were deleted after the audit; re-run `audit_collect.py` to rebuild them (set `TMP` inside the scripts).
