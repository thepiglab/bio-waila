# Pass-2 labeling audit, second half of selection.jsonl (lines 198-393, 196 datasets)

Machine-readable records: `audit_B.jsonl` (one object per dataset: id, access_status, metadata_sources, per_field, problems, flags, notes). Only metadata was fetched (record file listings, zip central directories via HTTP Range, first 32-64 KB of csv/json heads, small xls/xlsx tables, HF datasets-server schemas/first rows, public JSON APIs). No images or archives were downloaded; the temp directory was deleted.

Caveats: folder-level evidence comes from the first <=4,000 members of each zip (zip order), not the full tree. RAR/7z/split-zip/tar archives cannot be listed over HTTP, so 14 datasets rely on the registry description and README text only. Mendeley and Kaggle were not reachable (no auth), IEEE DataPort needs a login. For those, per_field entries are marked low confidence and based on registry text.

## 1. Access status (196 datasets)

| status | n | meaning |
|---|---|---|
| ok | 115 | Zenodo/Figshare/Dataverse/direct listing worked, files readable |
| ok_hf | 12 | HF repo + datasets-server worked (5 of 12 have no datasets-server config: script/zip repos) |
| manual_mendeley | 17 | HTTP 403 from the public API; browser download needed |
| needs_credentials_kaggle | 16 | Kaggle API credentials needed; contents unverified |
| ok_rar_unlistable | 11 | listing works but data is in .rar archives (cannot be range-sampled or listed) |
| unsupported_host_ieee_dataport_login | 8 | IEEE DataPort page, login required |
| unsupported_host_scidb | 3 | ScienceDB (scidb.cn), no handler |
| unsupported_host_tcia | 2 | TCIA collection page, no handler |
| ok_via_figshare_api_after_doi_dispatch | 2 | UCL RDR is figshare-backed; resolve_doi lacks the rule |
| ok_via_opara_dspace_api | 2 | OPARA DSpace API is open (files listed in problems) |
| ok_via_isic_api | 1 | public ISIC API v2 |
| ok_physionet_open_listing | 1 | PhysioNet open directory listing |
| ok_via_biostudies_api | 1 | BioStudies API open |
| ok_via_idr_api | 1 | IDR API open |
| ok_dataverse_conditions_of_use | 1 | DANS Dataverse, additional conditions-of-use PDF |
| ok_7z_unlistable | 1 | 7z archives |
| ok_but_no_images_in_record | 1 | record exists but contains only code |
| no_public_files | 1 | figshare shows no public files |

IDs per non-trivial status:

- **manual_mendeley** (17): fetal-pwd-doppler-cagliari, mcfi-2026, rsua-cxr-airlangga, skin-hair-mask-dataset, corneal-curvature-fundus-yoo, shallow-acd-fundus-yoo, ju-tbv1, octa-macula-coronal-views, rajbari-multiclass-fundus-2026, uwf-to-fp-style-transfer-yoo, dermocc-gan-dataset, rao-fundus-yoo, bajwa-multi-eye-disease, pk-covid-cxr-3568, ropdb-sfax-mendeley-2024, actinic-keratosis-hfus, oct-fundus-amd-pairs-yoo
- **needs_credentials_kaggle** (16): vip20k, nih-chestxray14-bbox, mstt-199, vindr-mammo-density-masks-uef, oai-tissue-segmentations, shseg-sperm-head-2026, shoulder-xray-classification-sysu, drimdb, colposcopy-images-mit-kaggle, amdnet23-fundus, gb-wce-bleeding, cellscribe-synthetic-fluorescence, malhari-cervical-dataset, blood-cell-images-leukemia-sumith-2025, roft-ocular-disease-ieee-2024, mcellseg-dic-hek293t-huvec
- **ok_rar_unlistable** (11): rarpl-workflow, rarpl6-workflow, sted-microtubules-diffusion-restoration, colorectal-polyps-gaze, sperm-dna-fragmentation-scd-2025, nantex-smlm-shareloc-subset, polyp-size, cas-colon, egid-gastritis, gim-endo, renji-esd-esophagus
- **unsupported_host_ieee_dataport_login** (8): ibn-al-haitham-glaucoma, flori21, fundus-photo-enhancement-kangnam, ieee-cervigram-sequences-yue, eecp-vascular-doppler-ieee, stir-orig, endo-ttapc5, vessel-tracking-rarp-lnd
- **unsupported_host_scidb** (3): raar-imac-30, enhance-pet-1k6, fus-awake-mouse-mechanical-stimulation
- **unsupported_host_tcia** (2): codex-imaging-of-hcc, cptac-aml-slides
- **no_public_files** (1): mhh-covid19-image-repository
- **ok_but_no_images_in_record** (1): hancock
- **ok_7z_unlistable** (1): deepcytometer-klf14-mouse-wat
## 2. Datasets that should not (or cannot yet) feed training

Counts are from the `flags` field of `audit_B.jsonl`; ids are in the records.

**Not usable as image data (23): drop or replace.** Masks/labels/manifests/model files only, or no images in the record:
dermartifactdb (attention heatmaps only, original dermoscopy not included), camelyon-plus (labels, WSIs not redistributed), nih-cxr14-ptx-device-annotations (manifest), fitzpatrick17k-c (CSV of original URLs; the only pixels are DermaMNIST dermoscopy npz), inbreast-breast-pectoral-masks, mini-mias-breast-pectoral-masks, aptos-vascular-arcade-onh-masks, montgomery-heart-clavicle-masks (masks only), irfundusset (catalogue CSV), oai-tissue-segmentations (masks), rosmat24 (YOLO weights), glaucoma-od-oc-boundaries-guo-2019 (.mat boundaries), mura-shoulder-quality-labels (JSON + weights), rat-brain-3d-ulm-chavignon (microbubble tracks), open-3dulm-awake-mouse (raw IQ .mat), hancock (record = GitHub code zip), dridb-blood-vessels (50 tiny pngs that are extracted vessel maps), steatotic-liver-3d-mpus-rat (MATLAB code), weiss-laparoscope-calibration (calibration-target images, no anatomy), tcga-diagnostic-slides / nlst-pathology / cgci-blgsp-slides (IDC s5cmd manifests only), real-colon-temporal (frame-label CSVs; useful as labels for `real-colon`).

**Images exist only as arrays/pickles (8):** zerocostdl4mic-deepstorm-data (3.7 GB TIFF stack), pcle-regression-prd (pkl frames), fucci-rpe1-cellcycle-naef-2025 (npy tracks), intel-mobileodt-refined-766 (X_*.npy), eye-laterality-cfp-liu-2019 (pkl/h5), small-bowel-retraction-force (npy frames + mp4; one corrupt zip), paviaatn-hacat-4channel (2 giant TIFF stacks), spheroid-cell-death-calcein-pi-cnn (pickles).

**Video only, frame extraction needed (16):** arthrophase (and its status file says "will be made available upon publication"), gauze-mis-segmentation, drends, surgpose (also green-screen subset), rectal-cancer-surgery-ochra (765 GB), ir-pupil-spectral-exposure, rosma, cystofold, polyp-size, wl-me-gastric-paired-videos-2026, appendix300, stir-orig, d4d-dresden-4d-reconstruction, open-h-embodiment (LeRobot mp4, 4.5 TB), endo-ttapc5, vessel-tracking-rarp-lnd.

**Archive types src/bio_waila/data/download.py cannot sample (15):** RAR (rarpl-workflow, rarpl6-workflow, sted-microtubules..., colorectal-polyps-gaze, sperm-dna-fragmentation-scd-2025, nantex-smlm..., polyp-size, cas-colon, egid-gastritis, gim-endo, renji-esd-esophagus), 7z (deepcytometer-klf14-mouse-wat), split zips (cpath-e2e-preprocessed-patches, cov-caldas XNAT parts, snubh MIAS-DB .z01), nested zips (rhode-island-vce: Study NNN-MMM.zip contains sNNN.zip).

**Duplicate / derivative risk:** rarpl-workflow, rarpl6-workflow, polimi-stereo-rarp, vessel-tracking-rarp-lnd share the same Milan prostatectomy series (keep in one split group); brachiolab-chestx (224 px NIH CXR14), asce-enhanced-cxr (pseudo-coloured derivative), amdnet23-fundus and rao-fundus-yoo (re-compilations of ODIR/RFMiD/JSIEC), ju-tbv1 (aggregate of other TB sets), colposcopy-images-mit-kaggle (probably Malhari), tevg (6 archives with the same 1,401 patches), egid-gastritis (3 RARs with the same 5,883 images), nachos-kidney-oct-cxr (same images repeated under several partition schemes), hrf-registration-thin-features (deformed copies), plism (13 renditions of each tile), skin-lesion-yolo-* (Roboflow/augmented copies), blaves (augmented folder), spheroid pickles (augmented).

## 3. Registry errors found (fix in data_discovery / src/bio_waila/data/select.py)

**Reference bug.** `bio_waila.data.select._figshare_id` regex `figshare\.com/(?:articles/(?:dataset/)?[^/]*/?)?(\d{6,})` lets `[^/]*` swallow the id when the URL has no title slug, so an 8-digit id is truncated to its last 6 digits. In this half: multi-disease-fundus-structseg (646378 should be 32646378; the wrong id is a protein-search figure), iscrc-immunoscore (007515 should be 24007515), cystoscopy-atlas-stanford (899997 should be 21899997). The same bug hits at least four first-half entries (selection lines 99 behsof-nafld-us 389069 -> 26389069, 113 grape 575926 -> 23575926, 164 laparoscopic-gastrectomy-annotated-images 058882 -> 30058882; 115 drosophila-wing-database-gigadb 437616 may be a truncation of 1437616). Fix: `(\d{6,})` should be anchored with `/(\d{6,})(?:[/?#]|$)` after the optional slug, or take the longest digit run. Also cecleanliness-wce has a figshare ref but a Mendeley homepage.

**Registry labels that are wrong for part or all of the files** (22): international-tmtv-benchmark (half the DICOM is PET), medmnist-ct-subsets (repo has all 18 MedMNIST npz; only adrenal/fracture/nodule/organ* are CT), fitzpatrick17k-c (pixels are DermaMNIST dermoscopy, not clinical photos), nachos-kidney-oct-cxr-partitioning (OCT_data.zip is porcine kidney OCT, not chest x-ray), pcle-pto (body part heart is wrong for 255 lens-paper videos), segthy (MRI configs inside an "ultrasound" entry), pix2pix-huvec-turku-virtual-staining (BF folder is brightfield), optical-microscopy-simulator-2026 (experimental and endoscope subfolders are not synthetic), quadrant-darkfield-qdf-zangle (Beads.zip and EmptyDarkfieldImages.zip are not cells), spcyclegan-synthetic-nuclei-3d-purdue (real volumes vs synthetic ones; and masked specimen/body part are resolvable from folder names), zerocostdl4mic-deepstorm-data (simulated stacks vs real rat glia), drends (abdomen vs colon/intestine/stomach/pancreas folders; specimen phantom vs ex-vivo pig by zip), dridb-blood-vessels (vessel maps), small-bowel-retraction-force (benchtop robot force experiment, laparoscopy doubtful), enhance-pet-1k6 (PET volumes), codex-imaging-of-hcc (H&E and CODEX in one collection), cptac-aml-slides (CT/US component), actinic-keratosis-hfus (HFUS is ultrasound, not dermoscopy), and three spectral/pulsed-wave Doppler entries labelled color doppler (fetal-pwd-doppler-cagliari, ninfea-fetal-ecg-pwd, eecp-vascular-doppler-ieee; the vocabulary has no spectral-Doppler class, consider dropping).

**Candidate sets that conflict with MMBU-public conventions:** laparoscopy vs white-light endoscopy (mmbu-public: 405 laparoscopy rows, 162 white-light endoscopy, 5 colonoscopy: use laparoscopy for abdominal surgery video and one consistent value for colonoscopy-type videos); fundus body part (407 eye vs 40 retina rows, default eye); cell-line microscopy specimen (mmbu-public mostly `human`, 167 of 199 cell rows; decide human vs cell line once); sperm datasets (mmbu-public body part `sperm cells`, 93 rows, so drop `male genitourinary system` candidate); RARP series (mmbu-public has `male genitourinary system` for 23 laparoscopy rows; the registry candidates lymph nodes/pelvis/prostate are not what MMBU uses); synthetic fluorescence rows have empty body part (keep masked).

## 4. Fields that become hard through metadata

Dataset-level knowledge (README/description) resolves many candidates; the list below is only where a **per-sample** column, folder, file-name token or archive name decides. All are in `per_field` with mapping examples.

| field | datasets and the deciding metadata |
|---|---|
| body_part | stimage-1k4m (`meta/meta_all_gene.csv` tissue), cov-caldas (`bp-chest` in file names), viralskin-4 (folder: Hands/Feet/Face/Mouth/Arms/Legs/Torso), drends (Vid00_Colon, Vid03_Intestine, Vid06_Stomach, Vid09_Pancreas), spcyclegan-synthetic-nuclei (Kidney_*/Rat_liver_1/mouse_intestine), polimi-stereo-rarp and rarpl* (phase folder: prostate removal / lymphadenectomy), thrombus-plus-dvt (Anatomical Site CFV/FV/PV/GS), cpath-e2e (archive brca vs panda), medmnist-ct (npz name), rhode-island-vce (per-frame organ h5), cas-colon (10 region labels), real-colon-temporal (per-frame segment CSV), pcle-pto (hearts vs lens), nachos (cortex/medulla/pelvis = kidney), tcga/nlst (IDC collection id), DermaMNIST-C.csv `localization` (fitzpatrick17k-c), han-severance (biopsy site, refinement only) |
| specimen | stimage-1k4m (species column), oriented-instrument-hysterectomy (dataset_phantom vs dataset_inhouse), drends (Phantom vs ExVivo), pcle-pto (lens=phantom, hearts=cow), nachos (OCT=pig, X-ray=human), spcyclegan (human/rat/mouse in folder names), bbbc026 (should be multi human+mouse), optical-microscopy-simulator (experimental vs simulated folders), oi-dic-smlm (BioStudies Biosample cell line) |
| stain | breast-cyto-histo-ukraine-2023 (histology = H&E; ER/PR/KI67/HER2NEU folders = IHC), bbbc050 (train = H2B-mRFP1, test = H2B-mCherry), bbbc013/014/015/016 (Channel1/Channel2 or d0/d1 suffix), codex-imaging-of-hcc (CODEX vs H&E file type), cgci-blgsp (IDC slide staining tags) |
| modality | international-tmtv (CT vs PET folder / DICOM Modality), nachos (OCT_data vs Xray_data), mcr-sl (clinical vs dermoscopic folder), cti-tongren (FA/ICGA vs US folder), segthy (`modality` column MRI/US), pix2pix-huvec (BF vs nuclei folder), actinic-keratosis-hfus (HFUS vs dermatoscopic), cptac-aml (SVS vs CT/US) |
| submodality | topcon-mm-fddm (ClassLabel fundus_images / oct_images), olives (every row is OCT), schistoscope (`_0` = brightfield, `_2` = darkfield, to verify in the paper), wl-me-gastric (`.WL.mp4` vs `.ME.mp4`), cti-tongren (FA/US), mcr-sl, bbbc050 (train widefield vs test confocal), segthy (config), prostate-ppat (`sequence` AX T2 / DWI), pix2pix-huvec (BF folder) |

Net effect on registry modes in this half: 183 candidate fields exist (108 body_part, 28 submodality, 21 specimen, 16 modality, 10 stain). After this audit, 44 of them resolve with high confidence (20 body_part, 15 specimen, 4 submodality, 3 modality, 2 stain), 74 with medium confidence (mostly dataset-level knowledge from the README or description, or mmbu-public frequencies), 56 stay low confidence and 9 are not applicable (no images in the record). Counting only true per-sample resolvers (table above) it is roughly 20 body_part, 9 specimen, 5 stain, 8 modality and 9 submodality cases. Low-confidence cases are concentrated in Kaggle, Mendeley, IEEE and RAR datasets where no listing was reachable; the partial-label loss remains the right treatment there.

## 5. Most common metadata patterns

1. **Dataset-level only** (roughly 110 datasets): no per-sample column; all six fields come from the registry or README (typical for single-class Zenodo drops, BBBC plates, fundus sets).
2. **Folder level = diagnosis/class** (roughly 45): NORMAL/PNEUMONIA, Cancer/Stones, ARMD/BRVO, HFMD/Mpox. These are not Pass-2 fields and the metalabel 60% column rule correctly ignores them, but they mean `match_value` often returns weak hits (ER, PR, CT) from class folders.
3. **Folder level = image type or body region** (roughly 20): clinical/dermoscopic, FA/ICGA/US, BF/nuclei, CT/PET, Vid00_Colon, hands/feet/face. These are the high-value resolvers.
4. **Paired image/mask trees** (roughly 40): image and mask side by side; correct mask filtering decides whether any usable image remains (see section 7).
5. **Per-patient tables only** (xlsx/csv/xls, about 30): age, sex, scores, diagnosis; no Pass-2 field.
6. **File-name tokens** (roughly 10): `bp-chest_vp-ap_cr`, `.WL.`/`.ME.`, `_0`/`_2`, `Channel1`/`Channel2`, `d0`/`d1`, `_FPL/_FPR`, `Human_Breast_*`, `_CT.jpg` (tumour centre, not CT).
7. **Containers that hide metadata:** RAR/7z/split zip/nested zip (15), numpy/pickle/h5 (8+), video (16).

## 6. Matcher additions for `src/bio_waila/data/rules.py` (with evidence)

New value -> class rules and fixes:

- **specimen**: add `phantom` as strong keyword (oriented-instrument `dataset_phantom`, drends `Phantom`, rosma, rosmat24); `synthetic|simulated|simcep` -> synthetic (bbbc005, bbbc031, micsim, lmrg, cellscribe); `ex ?vivo` + porcine/pig (drends `ExVivo`, d4d, gauze); cell-line tokens `u2os|u-2 os|hela|hek ?293t?|mcf-?7|a549|mda-?mb-?231|bt-?20|ht-?1080|rpe-?1|hacat|huvec|vero` -> cell line (human origin: choose human or cell line once, mmbu-public uses human for most), `cos-?7` -> non-human primate, `eph4` -> mouse (bbbc013/14/15/16, stardist-mda, ht1080, fucci, sted, eph4 dataset); `lens` folder -> phantom only as a per-dataset override (pcle-pto).
- **body_part**: plural/inflection (`hearts`, `hands`), camelCase splitting before matching (`OlfactoryBulb`, `LymphNode`, `LargeIntestine` from stimage), `cecum|caecum` and `hepatic flexure|splenic flexure|sigmoid|anal canal` -> colon/rectum (cas-colon, real-colon-temporal, captiv8), `ileum|terminal ileum|small bowel|intestine` -> small intestine (rhode-island-vce, drends, real-colon-temporal), `scalp|trunk|back|upper extremity|lower extremity|arms|legs|torso|mouth` (DermaMNIST-C localization, viralskin-4), `lymphadenectomy` -> lymph nodes, `peritoneum` -> abdomen, `cfv|fv|pv|gsv` in an "anatomical site" column -> blood vessels (thrombus-plus-dvt), `Optic_Disc` style folders must not set body_part when the folder holds masks.
- **BIDS-like entities**: `path_columns` should also parse `bp-<x>` (body part) and `vp-<x>` (view position) and treat the trailing `_cr`/`_dx` suffix as a DICOM modality code (cov-caldas MIDS file names).
- **submodality**: `\bwl\b` -> white-light endoscopy and `\bme\b|magnif\w*` -> gastroscopy for the wl-me dataset; PATH_CODES `fa` -> fluorescein angiography, `icga` -> none; `dwi|adc|swi` -> MRI modality and structural mri (only MRI submodality; prostate-ppat); `clinical` / `clinical images` folder -> clinical photography (mcr-sl); folder token `BF`/`brightfield` -> brightfield microscopy when it appears as a path component (pix2pix-huvec `230412_cc_BF`); `nbi|narrow band` -> no class (gim-endo, endowli-nbi), do not map to white-light.
- **stain**: add `her2neu|her2 neu` (breast-cyto-histo), make `ER|PR|KI67|HER2` strong when they are sibling folders of each other, `h2b-mrfp1|h2b-mcherry` tokens, `channel ?1/2` is not resolvable without a dataset table (use overrides).
- **False positives to suppress** (all were found by running `match_value` on real values): `device`/`model` columns (Topcon `3D OCT-1000` -> optical coherence tomography in rav-rotterdam; scope names in endowli-nbi); `dx_type=confocal` -> confocal microscopy (fitzpatrick17k-c: add `dx_type` to SKIP_COLUMN); `hepatic flexure` -> liver (captiv8, cas-colon); `pelvis` for renal pelvis (nachos OCT folders cortex/medulla/pelvis); `CT` as tumour-centre in IHC file names `1_CD3_CT.jpg` (iscrc-immunoscore; harmless because registry hard wins, but it spams conflicts); `Scan (n/49)` numeric column (olives) picked up as a submodality column by FIELD_COLUMNS.
- **metalabel I/O gaps**: xls needs `xlrd` (han-severance, esophagus clinical sheet); xlsx/xls without a header row (SeveranceA.xls takes the first record as header); `.h5` label files (rhode-island-vce) and `.docx` READMEs are not read; json nested dicts keyed by id (china-fundus-cimt `data_info.json`).

## 7. Fixes for `src/bio_waila/data/files.py` and `src/bio_waila/data/download.py`

**`is_mask` false positives (every image dropped):** the regex is applied to the whole member path, so a dataset-root folder name triggers it: `lesions-seg/` (skin-lesion-yolo-seg-4class: 3,995 of 3,995 jpg flagged), `MemBrain_seg_training_data/` (490 of 490), `ground_truth_and_synthetic/` (761 of 761), `Masks/.../Frames/` (gauze-mis: real frames inside a folder called Masks). Fix: ignore the first path component(s) that equal the archive stem or dataset root, and match on the last two components only.
**False negatives (masks kept as images):** `BinaryMask`, `Annotated` (far-polyp-seg), `Training_Target` (stardist-mda), `SEGMENTED` (trend-portable-fundus), `Folder_1_Optic_Disc/Vessels/Alpha_Beta_Zones` (hrf-seg-plus, 180 of 225 kept files are masks), `weight_maps` (partaker), `*.curvature.tiff` (micsim-fluocurv: 1,000 target maps per zip), `overlay`-like copies in NETRA-ANNOTATED and `visualized_masks`. Add `binarymask|binary_mask|target|annotated|segmented|curvature|weight_?maps?` and compare camelCase.
**Other handlers needed:** `resolve_doi` rule `rdr.ucl.ac.uk/articles/.../<id>` -> figshare (rectal-cancer-surgery-ochra 76 mp4; weiss 3.4 GB zip) and DSpace (OPARA: appendix300 has a 234 GB Appendix300_Images.zip, d4d has tar.gz), PhysioNet (`pwd_images/*.bmp`), ISIC API, BioStudies, IDR, IDC s5cmd manifests (tcga/nlst/cgci records contain only manifests), TCIA, ScienceDB; RAR/7z/split-zip/nested-zip extraction; HF repos that are scripts or plain zips (medmnist-v2, skin-lesion-yolo-ham, topcon-mm, SegThy raw nii.gz, CPath lmdb); figshare collection ids resolve fine (cov-caldas, rhode-island-vce).
**Stereo/time-series duplicates:** polimi-stereo-rarp (L_/R_), BBBC channel pairs, schistoscope BF/DF pairs, real-colon prev/curr/next triplets (use rgb_curr only), HF olives (49 B-scans per eye) should be sampled across eyes/patients.

## 8. Notes on specific decisions

- `plan/hf_remote_profile.json` entries were re-derived for the 12 HF datasets here: stimage-1k4m, medmnist-ct-subsets, skin-lesion-yolo-ham, cpath-e2e and open-h-embodiment have no datasets-server config; topcon-mm-fddm (ClassLabel fundus_images/oct_images), olives (OCT only), plism (scanner/stainer columns), segthy (modality column MRI/US, classes_present) give real per-row resolvers.
- Candidate resolution by dataset-level knowledge is marked `confidence: medium/low` and names its source (registry description, README, or mmbu-public frequency). Nothing was guessed where no source existed: those fields carry `confidence: low` and say why.
- First-half/second-half overlap: nih-chestxray14-bbox, nih-cxr14-ptx and brachiolab-chestx all derive from NIH ChestX-ray14.
