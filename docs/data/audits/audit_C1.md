# Pass-2 labelling audit C1 (32 datasets from audit_todo_C1.json)

Metadata only: file listings, zip central directories (remotezip), table heads / small tables, HF datasets-server `/info` + `/first-rows`, Zenodo/figshare/BioStudies/DataCite/UCI APIs, Kaggle file listings with the token. No images or archives were downloaded. Per-dataset records: `audit_C1.jsonl` (schema of audit_A.jsonl; `per_field.*.suggested` is an extra key holding the proposed registry value).

Reading note: `metalabel` never overrides registry `hard`/`multi`; per-sample metadata only matters for `candidates` and `masked` fields. So matcher noise on hard fields is just logged conflicts, while a wrong registry label is silent.

## 1. Access status (32 datasets)

| status | n | datasets |
|---|---|---|
| ok | 21 | bone-marrow-cytomorphology, isles-2022, chaksu, tcga-lgg-mask, bcn20000, derm12345, bm-broncholc, brain-organoid-1400-heidelberg, msd-task03-liver, btxrd, multi-disease-fundus-structseg, cellfmcount, ptx-498, rsna-pneumonia-severity-3level, wcebleedgen, capsule-vision-2024, hvdropdb, covid-qu-ex, cystoscopy-atlas-stanford, freehand-us-forearm-v1, deepbacs-saureus-widefield-seg |
| ok_hf_mask_rows | 2 | raabin-wbc, dmid |
| manual_mendeley | 2 | bowen-nigeria-colposcopy, infant-fundus-od-macula-det |
| wrong_source | 1 | fracatlas |
| no_images_in_record | 1 | figshare-brain-tumor-cheng |
| ok_hf_parquet_only_slice | 1 | lctsc |
| ok_nested_zip | 1 | enid |
| ok_masks_mixed_with_images | 1 | uci-digital-colposcopy-quality |
| no_pixels_in_record | 1 | surgen |
| unsupported_host_4tu | 1 | echopiv-aortoiliac-stents-4tu |

Summary: usable as selected: bone-marrow-cytomorphology, isles-2022 (after exclude), chaksu, tcga-lgg-mask, lctsc (needs fetch_hf fix to reach the quota), bcn20000, derm12345, bm-broncholc, brain-organoid-1400-heidelberg, msd-task03-liver, btxrd, multi-disease-fundus-structseg, cellfmcount, ptx-498 (after exclude), rsna-pneumonia-severity-3level, wcebleedgen, capsule-vision-2024 (after exclude), hvdropdb, covid-qu-ex (listing partial), cystoscopy-atlas-stanford, freehand-us-forearm-v1, deepbacs-saureus-widefield-seg. Usable only with an exclude (83% of its files are masks): uci-digital-colposcopy-quality. Usable only after a route fix: fracatlas (HF repo is a loader script), figshare-brain-tumor-cheng (HF repo empty), dmid (HF imagefolder mixes masks; figshare route + excludes). Not usable: surgen, echopiv-aortoiliac-stents-4tu, enid. Manual: the two Mendeley sets. Kaggle sets were listed with the authorised token (rate limited by concurrent audits).

## 2. DROP list (no usable pixels / masks mixed in / unreachable / duplicate)

- **surgen**: Zenodo record holds only UNI patch embeddings (zarr); WSIs are on BioImages S-BIAD1285 (CZI / OME-Zarr), no handler or reader
- **echopiv-aortoiliac-stents-4tu**: 4TU.ResearchData host unsupported (API in maintenance; figshare mirror shows embargoed/confidential and no files); first listed record is manuscript+excel only
- **enid**: duplicate of glenda (re-organisation of the same frames, same group), 160 images, nested zips (3.3 GB incl. 2.4 GB model weights)

Manual (browser download needed, not dropped): bowen-nigeria-colposcopy, infant-fundus-od-macula-det (Mendeley 403).

## 3. LABEL FIXES (format of plan/registry_fixes.json)

```json
{
 "drop": {
  "surgen": "Zenodo record holds only UNI patch embeddings (zarr); WSIs are on BioImages S-BIAD1285 (CZI / OME-Zarr), no handler or reader",
  "echopiv-aortoiliac-stents-4tu": "4TU.ResearchData host unsupported (API in maintenance; figshare mirror shows embargoed/confidential and no files); first listed record is manuscript+excel only",
  "enid": "duplicate of glenda (re-organisation of the same frames, same group), 160 images, nested zips (3.3 GB incl. 2.4 GB model weights)"
 },
 "route": {
  "fracatlas": {
   "method": "figshare",
   "ref": "22363012"
  },
  "figshare-brain-tumor-cheng": {
   "method": "hf",
   "ref": "MedOtter/Figshare_Brain_Tumor"
  },
  "dmid": {
   "method": "figshare",
   "ref": "24522883"
  }
 },
 "labels": {
  "lctsc": {
   "body_part": {
    "mode": "multi",
    "values": [
     "chest",
     "lung"
    ]
   }
  },
  "fracatlas": {
   "body_part": {
    "mode": "candidates",
    "values": [
     "hand",
     "hip",
     "lower limbs",
     "shoulder"
    ]
   }
  },
  "btxrd": {
   "body_part": {
    "mode": "candidates",
    "values": [
     "lower limbs",
     "pelvis",
     "upper limbs"
    ]
   }
  },
  "cellfmcount": {
   "specimen": {
    "mode": "candidates",
    "values": [
     "cell line",
     "neuronal cells",
     "rat"
    ]
   },
   "stain": {
    "mode": "candidates",
    "values": [
     "dapi",
     "immunofluorescence",
     "propidium iodide"
    ]
   }
  },
  "brain-organoid-1400-heidelberg": {
   "specimen": {
    "mode": "hard",
    "values": [
     "organoid"
    ]
   }
  },
  "freehand-us-forearm-v1": {
   "submodality": {
    "mode": "hard",
    "values": [
     "b-mode ultrasound"
    ]
   }
  },
  "deepbacs-saureus-widefield-seg": {
   "domain": {
    "mode": "multi",
    "values": [
     "biology"
    ]
   }
  },
  "ptx-498": {
   "body_part": {
    "mode": "multi",
    "values": [
     "chest",
     "lung"
    ]
   }
  },
  "rsna-pneumonia-severity-3level": {
   "body_part": {
    "mode": "multi",
    "values": [
     "chest",
     "lung"
    ]
   }
  },
  "covid-qu-ex": {
   "body_part": {
    "mode": "multi",
    "values": [
     "chest",
     "lung"
    ]
   }
  }
 },
 "exclude_paths": {
  "isles-2022": "/derivatives/",
  "lctsc": "\\.simplified",
  "dmid": "Pixel-level|ROI Masks|DICOM Images|Reports",
  "ptx-498": "\\.merge\\.png$|\\.nii\\.gz$",
  "capsule-vision-2024": "/KVASIR/|~\\$",
  "btxrd": "~\\$",
  "uci-digital-colposcopy-quality": "/images/(artifacts|cervix|os|speculum|vaginal walls)/"
 }
}
```

Notes on the fixes: `exclude_paths` is applied by src/bio_waila/data/download.py to listing names only for non-HF routes (archive/loose file names) and by src/bio_waila/data/prepare.py to every member path; inside `fetch_hf`'s file fallback it is not applied (pipeline bug P3). `fracatlas`, `dmid`, `figshare-brain-tumor-cheng` need their `route`, otherwise they yield zero (or mask) images. Suggested but optional: brain-organoid specimen hard `organoid`, freehand submodality hard `b-mode ultrasound`.

## 4. Per-field verdicts (registry label right?)

| dataset | modality | submodality | body_part | stain | specimen | domain | per-sample resolution available |
|---|---|---|---|---|---|---|---|
| bone-marrow-cytomorphology | ok | ok | ok | ok (MGG) | ok | ok | none (ClassLabel = cell types) |
| isles-2022 | ok | hard structural mri (dwi/adc; eval has only this class) | ok | n/a | ok | ok | modality via hf:modality / BIDS dwi, adc |
| chaksu | ok | ok | ok | n/a | ok | ok | none |
| raabin-wbc | ok | ok | ok (blood) | ok | ok | ok | none; mask rows must be skipped |
| tcga-lgg-mask | ok | ok (3-ch composite) | ok | n/a | ok | ok | none |
| fracatlas | ok | ok | masked -> candidates [hand, hip, lower limbs, shoulder] | n/a | ok | ok | csv one-hot flags hand/leg/hip/shoulder/mixed (4,083 rows) |
| figshare-brain-tumor-cheng | ok | ok | ok | n/a | ok | ok | none |
| lctsc | ok | spiral ct is a guess | masked -> multi [chest, lung] | n/a | ok | ok | DICOM BodyPartExamined |
| dmid | ok | ok | ok | n/a | ok | ok | none |
| bcn20000 | ok | ok | ok (csv anatom_site only yields conflicts) | n/a | ok | ok | csv anatom_site_general (not used for hard) |
| derm12345 | ok | ok | ok | n/a | ok | ok | hf:image_type |
| bm-broncholc | ok | ok | candidates kept | n/a | ok | ok | none |
| brain-organoid-1400-heidelberg | ok | ok | ok | n/a | candidates -> hard organoid (optional) | ok | none |
| msd-task03-liver | ok | ok | candidates kept | n/a | ok | ok | hf:modality |
| btxrd | ok | ok | candidates lack pelvis -> [lower limbs, pelvis, upper limbs] | n/a | ok | ok | xlsx one-hot region/bone/joint flags |
| multi-disease-fundus-structseg | ok | ok | ok | n/a | ok | ok | Source_Dataset (not a Pass-2 field) |
| cellfmcount | ok | ok | ok | masked -> candidates [dapi, immunofluorescence, propidium iodide] | **human is wrong** -> [cell line, neuronal cells, rat] | ok | csv markers/staining |
| enid | ok | ok | candidates kept (maybe add abdomen) | n/a | ok | ok | none |
| ptx-498 | ok | ok | chest -> multi [chest, lung] | n/a | ok | ok | none |
| rsna-pneumonia-severity-3level | ok | ok | chest -> multi [chest, lung] | n/a | ok | ok | none |
| wcebleedgen | ok | ok | candidates kept | n/a | ok | ok | none |
| capsule-vision-2024 | ok | ok | candidates kept | n/a | ok | ok | csv/path Dataset (source) only |
| hvdropdb | ok | ok | ok | n/a | ok | ok | none |
| cystoscopy-atlas-stanford | ok | ok | ok | n/a | ok | ok | folder names agree |
| freehand-us-forearm-v1 | ok | candidates -> hard b-mode ultrasound | ok | n/a | ok | ok | none |
| deepbacs-saureus-widefield-seg | candidates | candidates | ok | n/a (Nile red) | ok | masked -> multi [biology] | path folder brightfield_dataset / fluorescence_dataset |
| uci-digital-colposcopy-quality | ok | ok | ok | n/a | ok | ok | none; 83% of files are masks (exclude_paths) |
| covid-qu-ex | ok | ok | chest -> multi [chest, lung] | n/a | ok | ok | none |
| bowen-nigeria-colposcopy, infant-fundus-od-macula-det | unverified (Mendeley 403), registry text consistent | | | | | | |
| surgen, echopiv-aortoiliac-stents-4tu, enid | dropped | | | | | | |

## 5. Matcher additions (src/bio_waila/data/rules.py, src/bio_waila/data/metalabel.py) with evidence

All tests were run with `match_value(value, column, classes)` / `bids_hits` / `dicom_hits`.

a. **Indicator (0/1) columns named after a body part.** FracAtlas `dataset.csv` (hand, leg, hip, shoulder, mixed) and BTXRD `dataset.xlsx` (upper limb, lower limb, pelvis, hand, foot, femur, tibia, knee-joint, ...). `match_value('1','csv:dataset.csv:hand') -> {}`, `match_value('1','csv:dataset.xlsx:upper limb') -> {}`. Proposal: in `load_tables`, for every column whose distinct values are a subset of {0,1,'0','1',True,False} and whose name matches a body_part class/synonym, add a synthetic column `csv:<file>:body_flags` holding the comma/space-joined names of the set flags per row, then match tokens (single flag -> hard, several -> multi). Also add synonyms `leg`, `legs`, `lower limb`, `upper limb` (singular; currently only `lower extremity`, `legs`, `arms` exist: `match_value('upper limb','csv:x:region') -> {}` while `pelvis -> pelvis`), `hip bone -> pelvis`, `*-joint -> <joint>`.
b. **Weak hits are never used.** rules.py's docstring says weak hits may pick between values the registry lists, but `metalabel.cmd_map` keeps only `strong` matches (`if strong: per_field[f][val] = ...`). deepbacs: `match_value('fluorescence_dataset','path:d1') -> modality ('fluorescence microscopy', False)` is dropped, so fluorescence images stay `candidates (fluorescence, light)`. Proposal: store weak hits in colmaps with a flag and let `resolve()` use them only in `candidates` mode and only for values inside the candidate set.
c. **Fluorescence folder token**. `fluorescence`/`fluorescent` as a whole folder or value token should be strong for modality (`fluorescein` is not matched by `fluorescen\w*`) unless a fundus/angiography token is also present; then `fluorescence -> epifluorescence microscopy` via cross_field when the registry lists it. deepbacs evidence above.
d. **Immunostain markers in fluorescence data**. `match_value('Ki67 + TuJ1','csv:metadata.csv:markers') -> stain ihc (strong)`, `GFAP + Ki67 -> ihc`; but cellfmcount is Cy3/AF488 immunocytochemistry. Make `ki67|tuj1|map2|gfap|nestin|sox2|rip` -> `immunofluorescence` when the registry modality is fluorescence microscopy (ihc only for brightfield/digital pathology). Add `pi -> propidium iodide` (stain-field column only; `match_value('PI', markers) -> {}`), keep `DAPI -> dapi` (works: `{'stain': ('dapi', True)}`). Gives dapi 1,373 / propidium iodide 552 / immunofluorescence 1,098 of 3,023.
e. **`msk`, `merge` mask tokens (files.py _MASK)**: `is_mask('ISLES/dataset/derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz')` is False, `is_mask('SiteA/1.3.merge.png')` is False. Add `msk`, `merge|merged`, and `derivatives` as a mask folder only for BIDS-like trees.
f. **Office lock files** `~$name.xlsx` (BTXRD `BTXRD/~$dataset(total).xlsx`, capsule-vision `Dataset/training/~$training_data.xlsx`): `is_junk()` should return True for a basename starting with `~$`.
g. **DICOM non-image objects**: LCTSC has 60 RTSTRUCT `.dcm` (`1.000000-.simplified-NNNNN/1-1.dcm`, 3.8 MB each) and `dicom_hits({'Modality':'RTSTRUCT'}) -> {}`; src/bio_waila/data/prepare.py should skip Modality in {RTSTRUCT, SEG, REG, PR, SR} instead of counting them as volumes. (`dicom_hits({'Modality':'CT','BodyPartExamined':'CHEST'})` correctly returns computed tomography / conventional ct / chest, so lctsc would conflict with registry hard `spiral ct`: DICOM_MODALITY['CT'] always claims conventional ct.)
h. **HF ClassLabel noise for diagnosis columns** (DERM12345): `match_value('mammary_paget','hf:sub_class') -> breast (strong)`, `cutaneous_horn -> skin`, `basal_cell_carcinoma -> cell (weak)`. Harmless for hard fields; add `sub_?class|super_?class|malign|main_?class` to SKIP_COLUMN to keep the audit logs clean. Also bcn20000 `anatom_site_general` ('anterior torso' -> thorax, 'head/neck' -> none, 'palms/soles' -> none, 'oral/genital' -> none): add `head/neck -> head and neck` if a dermatology set with masked body_part is ever selected.

## 6. Pipeline bugs found (src/bio_waila/data/download.py / files.py / src/bio_waila/data/metalabel.py)

- **P1 fetch_hf cannot tell mask rows from image rows** when masks are rows of the same Image column with a ClassLabel/string label: raabin-wbc (`Mask` 1,235 of 1,992 rows), dmid (`ROI Masks`, `Pixel-level annotation` 538 of 1,049 files). The filter in fetch_hf looks only at column names. Fix: for every ClassLabel feature (and str columns named label/class/type) skip rows whose class name matches `is_mask` / `\bmasks?\b|annotation`.
- **P2 fetch_hf never falls back when streaming returns few images**: lctsc parquet has 60 rows (`middle_slice`), so got=60>0 and the 9,569 DICOM files are not used (plan 2,000). Fix: fall back to `fetch_entries` when got < want/4 and the repo lists >= want/4 media files.
- **P3 `exclude_paths` is ignored inside fetch_hf's file fallback** (isles-2022: 250 `_msk.nii.gz` + folder `derivatives/`), only src/bio_waila/data/prepare.py applies it, after the download budget was spent on masks. Apply the regex to the `Entry` list there too.
- **P4 `.mat` pixel arrays unsupported** (Cheng brain tumor on figshare 1512427: four zips of HDF5 .mat). Use the MedOtter mirror (route) or add h5py/scipy.io reading to src/bio_waila/data/prepare.py (kind() returns `other`).
- **P5 nested zips** (enid): ENID_v1.0.zip contains only zips; remotezip sees no media, the 3.3 GB outer zip is downloaded whole (2.4 GB of model weights included) and src/bio_waila/data/prepare.py extracts nothing usable.
- **P6 whole-archive downloads for tiny quotas**: any archive <= `--max-gb` budget (50 GB) is fetched whole (bcn20000: 9.2 GB for 7,424 images; msd-task03-liver: 29 GB for 524 slices). Consider range-sampling above ~2 GB when the quota is a small fraction of the members.
- **P7 `is_mask`/`is_junk` gaps**: `msk`, `merge`, `~$` (see matcher e, f). Numerically named masks in the same folder as images (uci-digital-colposcopy-quality: 1,722 `N.jpg` for 287 photos) cannot be caught by names at all.
- **P8 weak hits dropped by cmd_map** (matcher b).
- **P9 Kaggle listing is fragile under rate limits**: list_kaggle requests `pageSize=1000` pages back to back; Kaggle caps a page at ~400 files and returns 429 (`Retry-After` 5-10 s) after a few calls; urllib3 `Retry` turns that into `RetryError` and the whole dataset ends as 'error' (observed for covid-qu-ex, hvdropdb, uci on the first pass, with other audits sharing the token). Add `time.sleep` between pages and honour Retry-After; also `view` returns `totalBytesNullable`, so `size = int(v.json().get("totalBytes") or -1)` is always -1.
- **P10 `resolve_doi` does not dispatch 4TU** (data.4tu.nl) and its fallback figshare API shows embargoed/empty records.
- **P11 Mendeley 403** (bowen-nigeria-colposcopy, infant-fundus-od-macula-det): manual by design.
- **P12 selection metadata errors**: tcga-lgg-mask dims '3d' / effective 6,480 (real: 3,929 2D PNG slices); raabin-wbc plan 3,923 vs 757 real images; cystoscopy-atlas 1,000 vs 606; deepbacs 1,000 vs ~150; hvdropdb 1,000 vs 485; freehand file_formats h5 (real .mha); fracatlas/figshare-cheng routes point at an empty or script-only HF repo.

## 7. Duplicates / derivatives / leakage

- capsule-vision-2024 contains 38,592 Kvasir-Capsule frames (d4 = KVASIR) that kvasir-capsule also supplies (same group vce-anomalynet).
- rsna-pneumonia-severity-3level is a repackaging of RSNA 2018 (rsna-pneumonia-2018, same group): keep one.
- tcga-lgg-mask and brats-tcga-segmentations / BraTS family share patients (group brats-africa links them, but they are duplicates in the pool).
- multi-disease-fundus-structseg embeds IDRiD, FIVES and Fundus-AVSeg images (resized + padded); idrid is in group rav-rotterdam (val) -> train/val leakage; fives and fundus-avseg are separate groups too.
- enid is a re-organisation of glenda (dropped above).
- hvdropdb: segmentation images repeat classification images (~121+ byte-size matches).
- dmid on figshare: TIFF Images.zip and DICOM Images.zip are the same 510 mammograms.
- isles-2022: ADC is derived from DWI for the same 250 cases.
- bone-marrow-cytomorphology, raabin-wbc share a split group with multifocus-wbc-park-2024 (distinct sources).

## 8. Licence flags

- raabin-wbc: HF mirror card says bigscience-bloom-rail-1.0 (use-restricted), registry says custom/tier A.
- tcga-lgg-mask: registry CC-BY-3.0 but the HF mirror card says cc-by-nc-sa-4.0.
