# Pass-2 labelling audit, part C3 (31 datasets from `audit_todo_C3.json`)

Machine-readable records: `audit_C3.jsonl` (one object per dataset: id, access_status, access_detail, method, ref, metadata_sources, per_field, problems, notes; same schema as `audit_A.jsonl`).

Scope and method: metadata only. Used: file listings (Zenodo, figshare, HF repo siblings, Kaggle API with the authorised token), zip central directories via remotezip, first <=64 KB of small tables, HF datasets-server `/info` and `/first-rows`, record descriptions, `match_value` / `text_hits` / `is_mask` / `kind` on the folder, column and file names seen, and the MMBU public split (`plan/eval_public.parquet`) as a label prior. No images or archives were downloaded. The Kaggle API rate-limits hard (HTTP 429 after 4-6 consecutive listing pages), so Kaggle listings are partial (marked in the records). The temp dir was deleted.

## 1. Access status (31 datasets)

| status | n | datasets |
|---|---|---|
| ok (pixels listable and sampleable; caveats in `problems`) | 23 | brats-2021, tn5000, fives, hvsmr-2-0, hrf, octdl, verse-2019, weighted-ensemble-bio-em-ulm, behsof-nafld-us, chestx-det, msd-task07-pancreas, endoscopic-bladder-tissue-classification, knee-oa-severity-chen-2018, carotid-us-color-doppler-kaggle, toxofundus, humanembryo2-simula, paraguay-dr, vce-anomalynet, bc-multiset, gastrohun, puma-2024, qpi-labelfree-segmentation-review-vicar, virtual-staining-caco2-viability |
| ok_array_format (pixels are npy video arrays inside 16 GB zips) | 1 | cardiosyntax |
| ok_stack_tif (multi-GB multipage TIFF stacks; src/bio_waila/data/prepare.py reads frame 0 only) | 2 | three-photon-cerebral-organoids-rett, deepphd-two-photon |
| ok_multiconfig_hf (data fine, but `fetch_hf` cannot load it) | 1 | arcade |
| ok_ref_broken (selection ref is the string `https:/`) | 1 | see-ai (correct slug `capsuleyolo/kyucapsule`) |
| wrong_source | 2 | visem-tracking (HF repo = YOLO training plots; real data on Zenodo 7293726), airc-labden (Zenodo = analysis code and weights; images on Mendeley) |
| masks_only | 1 | multi-organ-abdominal-ct-reference |
| restricted / embargoed / needs credentials | 0 | (Kaggle worked with the token; Mendeley only appears as the real source of airc-labden) |

`behsof-nafld-us` was in the `reaudit` list (truncated figshare id). With the full id 26389069 it is **ok**: image_Data.zip (795 MB) holds 1,598 jpg + 52 avi in 113 patient folders, so the earlier status `no_images_in_record` was an artefact of the id bug. Remove it from `reaudit`.

## 2. DROP list

Hard drops (nothing usable can be downloaded as selected):

| id | status | reason |
|---|---|---|
| multi-organ-abdominal-ct-reference | masks_only | Zenodo 1169361 has two label tarballs (8.5 MB, 3.5 MB), `cropping.csv` and `index.htm`; the record says the CT images are not included (they belong to TCIA Pancreas-CT and BTCV). |
| airc-labden | wrong_source | Zenodo 23138199 is `zenodo_paczka.zip` (1.2 GB: code, derived tables, 15 `.pt` weights, `.npz`/`.npy`) and states "The photographs and label files ... are not redistributed here". The photographs are in Mendeley g8yhdvgjy2/3 and xjs4bfgzj5/3 (script access blocked, manual browser download). As selected the sampler would download the 1.2 GB zip "whole" (it is under budget and the quota is credited without looking inside) and find 0 images. |

Soft drops (reachable, but not worth it with the current pipeline):

| id | reason |
|---|---|
| deepphd-two-photon | 30 loose TIFF movies of 3-5.3 GB (112.7 GB) plus 6 zips with one more TIFF each; every movie exists as highSNR and lowSNR. `prepare.read_image` does `Image.open` + `seek(0)`, so one frame per file: about 15 images for about 50 GB of download against a plan of 1000. |
| three-photon-cerebral-organoids-rett | three z-stack TIFFs (1.4 GB), three organoids: 3 images with the current reader against a plan of 1500. |
| vce-anomalynet | aggregate of KID, Kvasir-Capsule, SEE-AI and WCEBleedGen frames. Kvasir-Capsule, WCEBleedGen, SEE-AI and capsule-vision-2024 are selected separately (same split group, so no leakage, but duplicate pixels). Also hit by the zip-member exclude bug below. Drop if the source datasets are kept. |

Re-routes (keep, but the selected source is wrong or inferior) are in the JSON below: visem-tracking, see-ai, tn5000, octdl, hvsmr-2-0.

## 3. LABEL FIXES (exact format of `plan/registry_fixes.json`)

```json
{
  "drop": {
    "multi-organ-abdominal-ct-reference": "Zenodo 1169361 holds reference segmentation tarballs only; the CTs are in TCIA Pancreas-CT / BTCV",
    "airc-labden": "Zenodo 23138199 is code + weights of a secondary analysis; photographs are only on Mendeley (g8yhdvgjy2/3, xjs4bfgzj5/3), script access blocked",
    "deepphd-two-photon": "112 GB of near-duplicate multipage TIFF movies; src/bio_waila/data/prepare.py reads one frame per file (about 15 usable images)",
    "three-photon-cerebral-organoids-rett": "three multipage z-stack TIFFs (3 organoids); src/bio_waila/data/prepare.py reads one frame per file"
  },
  "route": {
    "visem-tracking": {"method": "zenodo", "ref": "7293726"},
    "see-ai": {"method": "kaggle", "ref": "capsuleyolo/kyucapsule"},
    "tn5000": {"method": "figshare", "ref": "28455641"},
    "octdl": {"method": "zenodo", "ref": "10839556"},
    "hvsmr-2-0": {"method": "figshare", "ref": "25226360"}
  },
  "labels": {
    "weighted-ensemble-bio-em-ulm": {"submodality": {"mode": "masked", "values": []}},
    "endoscopic-bladder-tissue-classification": {"submodality": {"mode": "masked", "values": []}},
    "tn5000": {"body_part": {"mode": "hard", "values": ["neck"]}},
    "chestx-det": {"body_part": {"mode": "multi", "values": ["chest", "lung"]}},
    "gastrohun": {"submodality": {"mode": "candidates", "values": ["gastroscopy", "white-light endoscopy"]}},
    "virtual-staining-caco2-viability": {"body_part": {"mode": "hard", "values": ["colon"]}},
    "knee-oa-severity-chen-2018": {"body_part": {"mode": "hard", "values": ["knee"]}},
    "arcade": {"body_part": {"mode": "hard", "values": ["coronary arteries"]}},
    "cardiosyntax": {"body_part": {"mode": "hard", "values": ["coronary arteries"]}},
    "hvsmr-2-0": {"body_part": {"mode": "hard", "values": ["heart"]}},
    "msd-task07-pancreas": {"body_part": {"mode": "multi", "values": ["abdomen", "pancreas"]}},
    "carotid-us-color-doppler-kaggle": {"body_part": {"mode": "hard", "values": ["carotid artery"]}}
  },
  "exclude_paths": {
    "weighted-ensemble-bio-em-ulm": "Pretrained|Codes\\.zip",
    "vce-anomalynet": "Anomalies\\(Combined\\)|Source_Dataset",
    "qpi-labelfree-segmentation-review-vicar": "(^|/)reconstructions/",
    "visem-tracking": "visem-extracted-30s",
    "tn5000": "benchmark methods"
  }
}
```

Evidence and confidence per fix:

| id | field | evidence | confidence |
|---|---|---|---|
| weighted-ensemble-bio-em-ulm | submodality masked | Datasets.zip = 7 EM sets: Dataset 1 FIB-SEM (323 of 1,053 images), Dataset 2-3 serial-section TEM (250), Dataset 4-7 TEM (480); registry hard "scanning electron microscopy" is wrong for 730 images (69%). TEM has no class. Alternative that keeps hard SEM: exclude_paths `Dataset [2-7]/` (323 images left). | high |
| endoscopic-bladder-tissue-classification | submodality masked | `annotations.csv` column `imaging type`: WLI 1,433 / NBI 321 (18%). With mode `masked`, the existing rule (WLI -> white-light endoscopy, verified) makes WLI hard per sample and NBI stays masked; with hard, metadata can never override. | high |
| tn5000 | body_part hard neck | MMBU public thyroid ultrasounds: body_part neck 6/6 (b-mode). Candidate set [neck, thyroid] is also safe. | medium |
| chestx-det | body_part multi [chest, lung] | eval_public chest x-ray: [chest, lung] 271, chest 126, thorax 20, lung 10 (same fix audit A gave for other CXR sets). | medium |
| gastrohun | submodality candidates | selected_for lists both gastroscopy and white-light endoscopy; stomach screening = gastroscopy (paper-only class, 0 public examples); public white-light endoscopy includes stomach/esophagus/colon. | medium |
| virtual-staining-caco2-viability | body_part hard colon | MMBU public differential-fluorescent images: all 10 are fluorescence microscopy / human / colon / stain masked; Caco-2 = human colon adenocarcinoma. Modality and submodality stay candidates (they need the file-suffix rule in section 4). | medium |
| knee-oa-severity-chen-2018 | body_part hard knee | every file is a knee radiograph; public digital-radiography knee 35 of 45. | high |
| arcade, cardiosyntax | body_part hard coronary arteries | both are coronary angiography; cardiosyntax `videos[].artery` is LCA or RCA for every video. | medium |
| hvsmr-2-0 | body_part hard heart | whole-heart CMR (8 chambers/great vessels), README. | medium |
| msd-task07-pancreas | body_part multi [abdomen, pancreas] | all volumes are abdominal CT of pancreatic lesions; public conventional CT abdomen is the largest group. | low-medium |
| carotid-us-color-doppler-kaggle | body_part hard carotid artery | LC/RC/EC tokens in every file name (left, right, external carotid). | medium |

Registry facts that are fine as they are: brats-2021, fives, hrf, octdl (route only), verse-2019, behsof-nafld-us, toxofundus, paraguay-dr, humanembryo2-simula, bc-multiset, puma-2024, see-ai, three-photon, qpi (candidates are right; per-sample resolution below).

## 4. Matcher additions and per-sample rules (with evidence)

All checked with `match_value` / `text_hits` on the real values.

Wrong match:
- `fundus` is strong in `_R["modality"]` and `_R["submodality"]`. GastroHUN folder `Fundus_Cardia` returns modality optical imaging + submodality fundus photography, both strong; `gastric fundus`, `fundus of stomach` and `uterine fundus` do the same. Under a registry hard modality this is only logged as a conflict, but it pollutes review.csv and the column rule. Fix: require `fundus (photo|image|camera)|retinal fundus|colou?r fundus|\bcfp\b` for strong; treat bare `fundus` as weak, and drop it when `gastric|stomach|cardia|uter` is present.

Missing, resolves samples when added (all inside the registry candidate set, so safe):
- submodality, "no class" sentinel: `nbi|narrow.?band` (bladder annotations.csv `imaging type` = NBI, 321 images) and `blc` (cystods, audit A) should map to an explicit "no class" so masked-mode datasets stay masked and hard-mode datasets log a conflict. Today NBI returns `{}`.
- submodality, light-microscopy contrast techniques (qpi-labelfree: folders `DIC microscopy`, `HMC microscopy`, `PC microscopy`, `QPI microscopy`, files `DIC_01_raw.tif`): `hmc|hoffman modulation contrast`, `dic|differential (interference|image) contrast` -> contrast microscopy (strong); `pc` as a whole folder name or file prefix -> phase contrast microscopy; `qpi|quantitative phase` -> no class. Today `phase contrast` works, `DIC` is weak only, `PC`, `HMC`, `QPI`, `Hoffman modulation contrast` return `{}`.
- body_part, `artery` column (cardiosyntax all.json `videos[].artery` = LCA / RCA): `lca|rca|lad|lcx` -> coronary arteries, and add `artery` to FIELD_COLUMNS["body_part"]. all.json is a list of studies with nested `videos[].path`, so metalabel's table join by file name would still need a nested-JSON flattener.
- The `bids_hits` suffix table never fires on BraTS-style names (`BraTS-GLI-00000-000-t2f.nii.gz`, dash separators), but `text_hits` already returns structural mri there, so no change is needed.

Dataset-specific values that need `plan/overrides.csv` rows (column names as `path_columns()` produces them: last 4 non-generic folder names, `zip_members/<zip stem>` counts as a level; verify with the metalabel profile before applying):

```
dataset_id,column,value,field,class
carotid-us-color-doppler-kaggle,path:d3,UL,submodality,b-mode ultrasound
carotid-us-color-doppler-kaggle,path:d3,CD,submodality,color doppler
puma-2024,hf:cohort,primary,body_part,skin
qpi-labelfree-segmentation-review-vicar,path:d3,DIC microscopy,submodality,contrast microscopy
qpi-labelfree-segmentation-review-vicar,path:d3,HMC microscopy,submodality,contrast microscopy
qpi-labelfree-segmentation-review-vicar,path:d3,PC microscopy,submodality,phase contrast microscopy
```

(puma `metastatic` stays a candidate: lymph nodes or other sites.)

Not expressible today (needs a per-dataset file-name rule, e.g. `name_rules: [{regex, labels}]` in registry_fixes):
- virtual-staining-caco2-viability: file suffix `<field>.0.jpg` = phase contrast (light microscopy), `.1/.2` = the two differential fluorescent channels (record text, order not verified on pixels). Folder names carry only set tokens (`Set_1_Cam_AO_PI`, `Set_2_Ctrl_Dio_PI`: AO / DIO + PI), which narrow stain candidates (acridine orange or dioc6, plus propidium iodide) but not modality. Cheap alternative after checking 3 fields on pixels: exclude_paths `\.0\.jpg$` and hard modality/submodality fluorescence / differential fluorescent (the class has no other eligible dataset).
- deepphd-two-photon (soft drop): `Zebrafish-*` -> zebrafish already works (text_hits strong); `Neuropil`, `Spine` (dendritic spine, must not map to spine), `Neutrophil` have no rule.

Per-sample metadata that resolves fields today (no change needed): bc-multiset `types.csv` `type`=Breast -> breast; endoscopic-bladder `imaging type` WLI; qpi `DIC microscopy` (weak).

## 5. Pipeline bugs (src/bio_waila/data/download.py / src/bio_waila/data/select.py / src/bio_waila/data/prepare.py / files.py)

1. **src/bio_waila/data/select.py line 73-75, Kaggle DOI refs.** `see-ai` registry entry is `https://doi.org/10.34740/kaggle/ds/1516536`; stripping only `kaggle.com/(datasets/)?` leaves the whole URL, `split("/")[:2]` gives `https:/`. `list_kaggle` then gets HTTP 404 and raises Manual. Handle `doi.org/10.34740/kaggle/ds/<id>` and `kaggle.com/ds/<id>` by following the redirect (kaggle.com/ds/1516536 -> /datasets/capsuleyolo/kyucapsule). Checked all 23 kaggle rows: see-ai is the only malformed one.
2. **fetch_hf, multi-config repos without a default (arcade).** `datasets.load_dataset(ref, streaming=True)` raises "Config name is missing" (expected from the datasets API; not run here, datasets is not installed in the audit venv). The fallback lists only `.parquet` (kind meta), so `fetch_entries` raises Manual "record lists only metadata files" and arcade yields 0 images. Iterate `get_dataset_config_names(ref)`.
3. **fetch_hf accepts preview or diagnostic image sets.** visem-tracking: streaming finds 45 YOLOv5 plots in `best_checkpoints/` (ClassLabel names `yolov5l..x`) and returns them as sperm images with no error. hvsmr-2-0: the 60-row preview parquet satisfies the loader, so `got=60` and the NIfTI fallback never runs (state partial). Suggested guard: if `got < 0.5*want` and the repo also holds media or archives, continue with the file route; flag ClassLabel names that look like model names.
4. **`exclude_paths` is applied only to top-level record files in src/bio_waila/data/download.py** (line 510) and to downloaded media in src/bio_waila/data/prepare.py (line 300). It is not applied to members of a sampled zip (`sample_zip_remote`, `sample_tar_stream`) nor in the HF file-listing fallback. `spread_sample` round-robins over parent folders, so for vce-anomalynet about 95% of the 2,800 drawn frames come from the 19 excluded `Anomalies(Combined)` folders and are discarded afterwards. Pass the regex into the zip/tar samplers and the HF fallback.
5. **Archive handling in `fetch_entries`.** (a) `not re.search("mask|label|seg", name)` drops real image archives: figshare 27308133 (GastroHUN) `Labeled Images.zip` (8,834 images) and `Labeled_Sequences_*.zip` are excluded, leaving the 10 GB `Videoendoscopies_*` zips. (b) A "whole" download credits the full quota (`got += want`) without knowing the contents: airc-labden's 1.2 GB code/weights zip, and `Pretrained weights.zip` (2.1 GB) in weighted-ensemble if it were reached. List the zip central directory first and require media members.
6. **Kaggle listing limits.** `datasets/list` returns at most 200 files per page (`pageSize=1000` in `list_kaggle` is ignored) and the API answers HTTP 429 after 4-6 consecutive pages; with the session's `Retry(total=5, backoff_factor=2)` this surfaces as `RetryError` (uncaught in `list_kaggle`) after about a minute. Datasets with thousands of files (see-ai about 18.5k jpg + txt, knee-oa about 10k png) will fail at listing. Fix: sleep 3-5 s between pages, catch and keep the partial list, or sample from the whole-dataset zip.
7. **src/bio_waila/data/prepare.py multipage TIFF.** `read_image` opens a TIFF and calls `seek(0)`: one frame per stack file. deepphd (3-5 GB movies) and three-photon (z-stacks) give 1 image per file. Use `tifffile` page sampling like `pick_slices`.
8. **metalabel join keys.** `key_candidates` uses `Path(name).name.split(".")[0]` as the stem: for virtual-staining files `1.0.jpg`, `1.1.jpg`, `1.2.jpg` all collapse to stem `1`. Harmless there (no table), but any table keyed by such names would collide.
9. **Zenodo small zips and remotezip.** `RemoteZip(url, session=S)` with default suffix-range fails on tiny files (`Invalid Content-Range ... bytes 18446744073709496827-10746/10747` for a 10 KB zip); `support_suffix_range=False` works. Irrelevant for the big zips actually sampled; note for tools that probe small archives. Also `remotezip` with `headers=None` raises `TypeError`; pass `{}`.

## 6. Notes per dataset (one line each; details in the jsonl)

| id | status | key finding |
|---|---|---|
| brats-2021 | ok | repo is BraTS 2023-GLI (1,251 cases x t1c/t1n/t2f/t2w + seg); labels right; filename suffix works through text_hits |
| tn5000 | ok | HF copy = 224x224 nodule crops; original frames on figshare 28455641 (5,000 JPEG + VOC xml); body_part -> neck |
| fives | ok | clean; image_id suffix A/D/G/N is diagnosis |
| hvsmr-2-0 | ok | HF streaming yields only 60 slice previews; route figshare 25226360 (orig.zip); repo triples the same 60 scans |
| hrf | ok | 45 images only (quota 81 includes the IQA set not in the repo) |
| octdl | ok | HF mirror has 1,618 images; Zenodo 10839556 has all 2,064 with patient_id/eye/sex |
| verse-2019 | ok | 96 volumes = 55 VerSe19 + 41 VerSe20; paths only, file-listing fallback |
| weighted-ensemble-bio-em-ulm | ok | submodality hard SEM wrong for 69% (TEM sets) |
| behsof-nafld-us | ok | re-audited: pixels present (113 patients, 1,598 jpg) |
| visem-tracking | wrong_source | HF streaming returns YOLO plots; use Zenodo 7293726 |
| chestx-det | ok | body_part -> multi [chest, lung] |
| arcade | ok_multiconfig_hf | fetch_hf fails on two configs without default |
| msd-task07-pancreas | ok | clean; body_part dataset-level |
| cardiosyntax | ok_array_format | 10 zips of npy videos; `artery` LCA/RCA in all.json |
| endoscopic-bladder-tissue-classification | ok | NBI 321 / WLI 1,433 in annotations.csv; submodality -> masked |
| toxofundus | ok | clean; masks/ excluded by is_mask |
| multi-organ-abdominal-ct-reference | masks_only | DROP |
| knee-oa-severity-chen-2018 | ok | Kaggle copy 212 MB (not Mendeley 7 GB); body_part -> knee; listing partial (429) |
| airc-labden | wrong_source | DROP; Mendeley only |
| humanembryo2-simula | ok | 5,500 png in stage folders, no csv (registry says jpg+csv) |
| see-ai | ok_ref_broken | ref `https:/` -> capsuleyolo/kyucapsule |
| paraguay-dr | ok | clean (DR-grade folders map to nothing) |
| vce-anomalynet | ok | 20.7 GB zip; aggregate of 4 other datasets; exclude Combined and Source_Dataset |
| three-photon-cerebral-organoids-rett | ok_stack_tif | 3 stacks only; soft drop |
| deepphd-two-photon | ok_stack_tif | 112 GB, high/low SNR duplicates; soft drop |
| bc-multiset | ok | clean; `types.csv` Breast resolves body_part |
| gastrohun | ok | HF = 5,202-image 6-class subset; Fundus_Cardia matcher false positive |
| puma-2024 | ok | `cohort` primary/metastatic -> skin / candidates |
| qpi-labelfree-segmentation-review-vicar | ok | DIC/HMC/PC/QPI folders; exclude reconstructions/ |
| carotid-us-color-doppler-kaggle | ok | 129 images (59 CD) of 18 patients vs plan 3,000; path UL/CD resolves submodality |
| virtual-staining-caco2-viability | ok | file suffix .0/.1/.2 carries modality/submodality; body_part colon from MMBU prior |
