# Pass-2 labelling audit, part D (14 datasets from `audit_todo_D.json`)

Machine-readable records: `audit_D.jsonl` (one object per dataset, same schema as `audit_A.jsonl` / `audit_C3.jsonl`).

Scope and method: metadata only. Used: file listings (Zenodo, figshare, Kaggle with the authorised token, HF repo trees), zip and 7z central directories via ranged reads, the TCIA/NBIA series listing, BioStudies JSON file listing, HF datasets-server `/info`, `/first-rows`, `/statistics`, record descriptions, `match_value` / `text_hits` / `is_mask` / `kind` on the names and values seen, and `plan/eval_public.parquet` as a label prior. No image, volume or archive body was saved. Disclosure: my first ranged-zip helper issued a plain GET to Zenodo without streaming, so for the HER2 record (about 215 MB) it received the response bodies in memory before I fixed it (stream + close); nothing was written to disk and no other host was affected. The temp dir was deleted.

## 1. Access status (14 datasets)

| status | n | datasets |
|---|---|---|
| ok | 5 | bus-uclm, oraltrid, neojaundice-xuzhou, her2-tqi-valparaiso (only 174 images), shoulder-implant-manufacturer-uci |
| ok but needs an exclude / has a caveat | 4 | lfa1-immune-synapse-tirf-protocol (85% derived tifs), bracs (995 GB, 50 GiB cap), hmu-crc-hist550k (adipose class only), sdt4d-simulated-two-photon-calcium (28 GB of multipage TIFF) |
| no reader for the format | 2 | cryolithe-training-data (.mrc), acoustic-angiography-tumor-volumes (.mat plus 2 npz) |
| unsupported host behind a DOI (resolve_doi raises Manual) | 2 | biad3802-two-photon-calcium-v1-resolution (BioStudies), tcga-esca-ct (TCIA) |
| restricted | 1 | freehand-3d-ultrafast-doppler-brain-surgery (IEEE DataPort subscription, no files listed) |

## 2. DROP list

Hard drops:

| id | reason |
|---|---|
| cryolithe-training-data | 936 GB of `.mrc` volumes and tilt-angle text. `kind('.mrc')` is "other", no archive, so fetch_hf ends with "no media files or archives listed"; src/bio_waila/data/prepare.py has no MRC reader either. Also marked split=val, but 3 other cryo-ET sets are selected (all train). |
| tcga-esca-ct | `doi` -> TCIA landing page: unsupported by `resolve_doi`. The NBIA API is open (126 CT series, 20,593 slices, 10.2 GB, CC BY 3.0) but there is no lister. Reinstate after adding a TCIA lister (see section 5). |
| freehand-3d-ultrafast-doppler-brain-surgery | IEEE DataPort subscription, page states files are not uploaded; one complex-IQ HDF5 anyway. |

Soft drops (reachable or reachable-with-work, but not worth it):

| id | reason |
|---|---|
| biad3802-two-photon-calcium-v1-resolution | BioStudies host unsupported (Manual). Even with a lister: 95 TIFF movies x 927 MB = 81.9 GB; the 20 GB budget gives about 22 files, about 90 images vs plan 1,000. |
| sdt4d-simulated-two-photon-calcium | 41 noisy simulated TIFFs of 687 MB (28 GB); budget gives about 29 files, 4 frames each = about 116 images vs plan 2,000. |
| acoustic-angiography-tumor-volumes | 13.9 GB zip, whole download; only 2 readable arrays (`read_array` takes the first npz key and 4 slices) = at most 8 images; the 295 `.mat` are skipped. Registry specimen is also wrong (rat, not mouse). |

## 3. LABEL FIXES (exact format of `plan/registry_fixes.json`)

```json
{
  "drop": {
    "cryolithe-training-data": "936 GB of .mrc volumes; files.py has no .mrc kind and src/bio_waila/data/prepare.py no MRC reader, so nothing is fetched or read (3 other cryo-ET sets are selected)",
    "tcga-esca-ct": "doi routes to the TCIA landing page which resolve_doi does not support; no NBIA lister exists (reinstate after adding one: 126 CT series, 10.2 GB, open)",
    "freehand-3d-ultrafast-doppler-brain-surgery": "IEEE DataPort subscription-gated; record lists no files; one complex-IQ HDF5",
    "biad3802-two-photon-calcium-v1-resolution": "BioStudies doi unsupported; 95 TIFF movies of 927 MB (82 GB) give about 90 images under the 20 GB budget",
    "sdt4d-simulated-two-photon-calcium": "28 GB of 687 MB simulated TIFF movies; budget yields about 116 noisy frames vs plan 2000",
    "acoustic-angiography-tumor-volumes": "13.9 GB zip whose pixels are .mat plus 2 npz; at most 8 images via read_array (also registry specimen should be rat)"
  },
  "route": {},
  "labels": {
    "neojaundice-xuzhou": {"body_part": {"mode": "candidates", "values": ["chest", "face", "head", "skin"]}},
    "shoulder-implant-manufacturer-uci": {"body_part": {"mode": "hard", "values": ["shoulder"]}},
    "lfa1-immune-synapse-tirf-protocol": {"stain": {"mode": "candidates", "values": ["immunofluorescence", "phalloidin"]}}
  },
  "exclude_paths": {
    "lfa1-immune-synapse-tirf-protocol": "(^|/)(analysis_output|imagej_macros|r_scripts)/|_dic\\.tif$",
    "bracs": "(^|/)test_run/"
  }
}
```

Evidence and confidence per fix:

| id | field | evidence | confidence |
|---|---|---|---|
| neojaundice-xuzhou | body_part candidates [chest, face, head, skin] | web description of the paper: three photos per infant of head, face and chest (suffix `-1/-2/-3` in file names, order not documented); registry [face, skin] omits chest and head. No MMBU-public prior (31 clinical-photo images, body_part masked). | medium |
| shoulder-implant-manufacturer-uci | body_part hard shoulder | all 597 images are shoulder-prosthesis radiographs (UCI 517 description); registry [bone, shoulder] is not wrong but weaker. | high |
| lfa1-immune-synapse-tirf-protocol | stain candidates | channel tokens in file names: `lfa1`, `cd3` = antibody labels (immunofluorescence), `factin` most likely phalloidin (not stated in the record). | medium-low |
| lfa1-immune-synapse-tirf-protocol | exclude | zip = 352 raw tifs (88 synapses x cd3/dic/lfa1/factin) + about 2,300 derived tifs (crops, background-subtracted, spin stacks, mask tifs); `_dic` is DIC (transmitted light), contradicts hard fluorescence/TIRF. Regex leaves 264 raw fluorescence images (checked on the central directory names). | high |
| bracs | exclude test_run | 6 of the 553 `.svs` in the HF repo are duplicates under `test_run/svs/` (547 slides in README). | high |

Registry facts that are fine: bus-uclm (94% b-mode; see notes), oraltrid, her2-tqi-valparaiso, hmu-crc-hist550k, bracs (labels), shoulder submodality, neojaundice domain.

Kept-but-noted label caveat: bus-uclm `has_doppler` = True on 37 of 683 rows (5.4%) and `has_combined` on 6; the hard `b-mode ultrasound` is wrong for those. Not changed: with a hard label nothing overrides it, and candidates would weaken the other 94%.

Correction not needed because the dataset is dropped: acoustic-angiography-tumor-volumes specimen is `rat` (file names `ratL/ratR/ratN`, subcutaneous rat fibrosarcoma), not mouse. If it is ever reinstated: `{"specimen": {"mode": "hard", "values": ["rat"]}}`.

## 4. Matcher additions and per-sample rules (checked with `match_value` / `text_hits`)

Wrong match:
- `rules.py` line 97 matches bare `mus` as mouse (strong). `text_hits("MUS/1302097_0_19264_MUS.png")` and `match_value("MUS", "path:d1")` both return specimen mouse strong. HMU-CRC-Hist550K has a smooth-muscle class `MUS` (only the ADI article is selected now, so it does not fire today; it would if the MUS article is added). Keep `mus musculus`, drop bare `mus` (or require `mus` not upper-case class token).

Missing, resolves samples when added (all inside the registry candidate set, so safe):
- stain, TIRF channel tokens: `f-?actin|phalloidin` -> phalloidin (weak); `lfa-?1|cd3` -> immunofluorescence (weak) for lfa1-immune-synapse. Today `text_hits` returns nothing for `10min_10_factin.tif`, `..._lfa1.tif`, `..._cd3.tif` (only the TIRF submodality from the `raw_tirf_images` folder).
- BUS-UCLM boolean column `has_doppler` (True -> color doppler) is not a mappable value; only useful together with candidates mode, so not recommended.
- Dataset-specific per-sample tokens that would need a `name_rules` mechanism (not expressible today): NeoJaundice suffix `-1/-2/-3` (skin site, mapping undocumented); HER2 `Nueva Carpeta/` (second image set, probably vimentin).

Works as is (no change needed): `10min_10_dic.tif` -> contrast microscopy strong (irrelevant after the exclude); BRACS folder `svs` -> whole slide imaging strong (matches registry) with weak modality digital pathology (conflict only logged against registry light microscopy); DICOM `BodyPartExamined=ESOPHAGUS` -> esophagus inside the tcga-esca candidate set.

No new `overrides.csv` rows are needed.

## 5. Pipeline bugs and problems found

1. **resolve_doi has no BioStudies, TCIA or IEEE DataPort handler** (`src/bio_waila/data/download.py` line 236-251). `biad3802...`, `tcga-esca-ct`, `freehand...` all end in `Manual: DOI resolves to an unsupported host`. BioStudies is trivially listable: `GET https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD3802/files?limit=100` returns per-file `path`, `Size` and attribute columns (Animal_ID, axial resolution, depth, species, sensor). TCIA: `getSeries?Collection=TCGA-ESCA` is open and returns per-series SeriesInstanceUID, Modality, BodyPartExamined, SeriesDescription, ImageCount, FileSize; `getImage?SeriesInstanceUID=` returns one zip per series. The same BioStudies gap hits `oi-dic-smlm-correlative-biad3889` (not in this list).
2. **`.mrc` volumes unsupported** (`src/bio_waila/data/files.py` VOLUME_EXT, `src/bio_waila/data/prepare.py` read_volume): cryolithe is a 936 GB repo of `.mrc`; `kind()` returns "other" so `fetch_entries` reports no media. MRC is raw slices after a 1,024-byte header, so ranged HTTP reads of single slices would work if support is added (`mrcfile` is not in requirements.txt).
3. **`.mat` and multi-key npz** (acoustic-angiography): `find_media` skips `.mat` (kind other, no DICM); `read_array` uses only `a.files[0]` and 4 slices of an npz, so a 195-image 2-D MIP stack gives 4 images and an 8.2 GB 3-D npz is loaded in memory.
4. **fetch_hf hard-codes `50 * 2**30` as the budget** (both fallback calls to `fetch_entries`), ignoring `--max-gb 20`. For bracs (553 slides, 995 GB) it keeps the 70 smallest slides (a size-biased sample, about 1,120 tiles vs 2,000 planned). The streaming attempt also fails first (datasets-server 500 / format inference error), costing a round trip before the fallback.
5. **Derived-product zips are ingested wholesale** (lfa1): a Zenodo analysis repo with `analysis_output/` was downloaded whole and `extract_archives` extracts everything; 2,650 of 3,296 members pass `kind`/`is_mask`, 85% derived (background-subtracted copies, crops, spin stacks). Only a per-dataset `exclude_paths` fixes it; there is no generic derived-folder rule.
6. **xlsx with a 197 MB sheet** (her2-tqi): `DATA HER2 TQI.xlsx` is 20.8 MB zipped, below META_MAX 50 MB, so it is downloaded and given to metalabel; openpyxl full-mode loading a 197 MB sheet XML will take minutes and several GB of RAM. Use `read_only=True` or skip tables above a decompressed-size limit. (The xlsx also appears a second time inside the image zip.)
7. **Solid 7z** (hmu-crc-hist550k): prepare extracts all 72,006 PNG (about 6.7 GB) from `ADI.7z` to keep 2,000; and `figshare` refs are single articles, so the other seven class articles of the same dataset cannot be added without multi-ref support (`ref` could accept a comma list). `fallback_urls` in the registry lists six of eight articles and omits STR 28939169 and TUM 28939460.
8. **figshare range reads**: `remotezip` cannot open figshare `ndownloader` URLs (the signed S3 redirect expires in about 10 s and rejects HEAD with 403; the non-redirect URL ignores ranges). `sample_zip_remote` would fail for a figshare zip above the budget (not the case for neojaundice, 440 MB, which is downloaded whole). My audit used one ranged GET per 1 MB block with a fresh redirect each time.
9. **Quota not reachable, plan not adjusted**: her2-tqi (174 images vs 1,000), lfa1 (264 vs 3,000 after exclude), oraltrid/bus-uclm exact. `effective_images` for her2 is -1.

## 6. Notes per dataset (one line each; details in the jsonl)

| id | status | key finding |
|---|---|---|
| bus-uclm | ok | 683 breast US screenshots (normal 419, benign 174, malignant 90); 37 Doppler + 6 combined rows labelled hard b-mode |
| oraltrid | ok | 2.8 GB zip, 808 jpg + 808 LabelMe json (excluded as annotations); disease codes map to nothing |
| neojaundice-xuzhou | ok | 2,235 jpg (3 per infant: head/face/chest per paper); body_part candidates -> [chest, face, head, skin] |
| cryolithe-training-data | no reader | 936 GB .mrc; drop |
| her2-tqi-valparaiso | ok | only 174 images (87 + 87 in "Nueva Carpeta"); 197 MB-sheet xlsx; stain candidates right |
| shoulder-implant-manufacturer-uci | ok | 597 jpg, manufacturer = class; body_part -> hard shoulder |
| lfa1-immune-synapse-tirf-protocol | ok, needs exclude | 264 raw fluorescence images after excluding derived tifs and the 88 DIC images; stain candidates |
| biad3802-two-photon-calcium-v1-resolution | unsupported host | 95 x 927 MB TIFF movies (82 GB); soft drop |
| tcga-esca-ct | unsupported host | TCIA open (126 CT series, 10.2 GB, BodyPartExamined ESOPHAGUS on all); needs lister, drop for now |
| bracs | ok | 547 H&E WSIs (995 GB), 70 smallest slides within the 50 GiB cap; exclude test_run duplicates |
| hmu-crc-hist550k | ok | adipose class only (72,006 tiles, 499 slides); `mus` mouse matcher bug if MUS article added |
| sdt4d-simulated-two-photon-calcium | ok_stack_tif | 41 simulated TIFFs (28 GB) -> about 116 frames; soft drop |
| acoustic-angiography-tumor-volumes | no reader | .mat + 2 npz -> at most 8 images; specimen is rat; drop |
| freehand-3d-ultrafast-doppler-brain-surgery | restricted | IEEE DataPort subscription; drop |
