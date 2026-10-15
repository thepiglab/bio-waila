# Pass-2 labelling audit C2 (31 datasets from `audit_todo_C2.json`)

Machine-readable records: `audit_C2.jsonl` (same schema as `audit_A.jsonl`). Scripts: `_audit_C2_src/` (`collect.py` = A's collector plus Kaggle support, `digest.py`, `prior.py` = mmbu-public prior lookup, `kaggle_list.py` / `kaggle_probe.py` = throttled Kaggle listers, `ov_c2*.py` = hand-written per-dataset findings, `assemble.py`). Raw heads were deleted after the audit; re-run `collect.py` to rebuild them.

Metadata only: file listings, zip central directories via remotezip (plus one 400-byte read of an inner `.npz` local header to get the array key), the first 64 KB of tables, HF datasets-server `/info` + `/first-rows`, record descriptions from the Zenodo / figshare / Dataverse / BioStudies APIs, and the Kaggle API (`view`, paged `list`, and one 3.6 KB README). No image, archive or dataset was downloaded.

Caveat on Kaggle: the token works, but the API caps a list page at **200 files** (src/bio_waila/data/download.py asks for 1000 and gets 200) and answers HTTP 429 after roughly 5-11 requests. Full listings were therefore impossible for the three Kaggle sets (aml: 2,200 of ~18,365 files read; tbx11k: 600 files read; bcss: 1,000 of ~98k read). Their conclusions rest on the Kaggle `view` description, the first pages and the README.

## 1. Access status (31 datasets)

| status | n | datasets |
|---|---|---|
| ok (listable, pixels present; caveats in `problems`) | 22 | grazpedwri-dx, hushem, palm, bbbc039, kromp-human-blastocyst-gardner, kvasir-capsule, oct2017-kermany, crossmoda-2021, cmmd, biomeditech-rpe-maturation, cbis-ddsm, msd-task08-hepatic-vessel, msd-task10-colon, sustech-sysu, alphadent, iscrc-immunoscore, kermany-pediatric-pneumonia-cxr, intraoral-caries-ahmed-2025, knee-xray-osteoporosis-wani, microvision-fluorescence-cell-detection, pcle-rat-mosaicking-optica, multieye |
| ok_kaggle (token works, listing throttled) | 3 | aml-cytomorphology-lmu, tbx11k, bcss |
| ok_array_format (pixels in per-image `.npz`) | 1 | medsegmnist-derm2d |
| ok_unlistable_archive | 2 | five-d-us-ablation-deshmukh (5-part RAR), oocytor-oocyte-segmentation (single tar.gz) |
| ok_blocked_by_archive_filter (src/bio_waila/data/download.py would fetch nothing) | 1 (+oocytor above) | lightweight-instrument-seg-ush |
| unsupported_host_biostudies | 1 | biad1137-hht1-single-molecule-tracking-spombe |
| unsupported_host_ieee_dataport_login | 1 | color-doppler-twinkling-biopsy-markers |
| restricted / embargoed / masks_only / no_images / wrong_source | 0 | |

Fewer than a fifth of the set (6) cannot be used as selected; a further 3 have a very low yield (pcle-rat 47 frames, hushem <=216 repeated 131 px crops, crossmoda 105 volumes).

## 2. DROP list

| id | reason |
|---|---|
| five-d-us-ablation-deshmukh | 5-part RAR volume set (6.9 GB, "Timing_raw_data"), unlistable, no handler, no evidence of 2-D images; specimen/body_part unverified |
| color-doppler-twinkling-biopsy-markers | IEEE DataPort login (doi 10.21227/52wv-xv88), no listing, labels unverified |
| biad1137-hht1-single-molecule-tracking-spombe | BioStudies host unsupported by `resolve_doi` (API is open: 254 movie TIFF stacks of 131 MB + MIPs, 12.6 GB); drop unless a BioStudies handler is added; mostly dark single-molecule frames |
| multieye | derivative compilation (CLAHE/filtered, 224 px fundus; ~75% of the 109.7k OCT images are Kermany OCT2017 copies already selected as oct2017-kermany; fundus half repeats EyePACS/ODIR/RFMiD/JSIEC/FIVES), contamination-prone |
| lightweight-instrument-seg-ush, oocytor-oocyte-segmentation | not droppable on content, but they download nothing until the `fetch_entries` archive filter is fixed (section 6, bug 1); drop if the code is not fixed |

## 3. Label review per field

Registry modes in this set: modality 31 hard; submodality 28 hard / 3 candidates; body_part 19 hard / 12 candidates; stain 5 hard / 1 candidates / 25 masked; specimen 29 hard / 2 candidates; domain 29 multi / 2 masked.

| field | per-sample resolvers found | dataset-level decisions (registry edit) | notes |
|---|---|---|---|
| submodality | multieye (zip folder `cfp-clahe-224x224` vs `oct-filter-448x448`, strong `match_value` hits) | none needed | the 3 candidate sets: multieye (dropped), oocytor (unlistable), color-doppler (unreachable) |
| body_part | grazpedwri-dx (file token `WRI`), knee-xray (folder `Knee Osteoporosis Classification`), iscrc (coded `Tumour site`, needs Coding.txt) | grazpedwri wrist, crossmoda brain, MSD x2 abdomen, biomeditech retina, knee knee, oocytor cells, (kvasir optional) | alphadent, intraoral and pcle-rat stay candidates: nothing in names or labels separates them |
| stain | microvision (folder tokens dapi / phal / nucleopore / peroxisome / glycophorinA), iscrc (file token CD3/CD8, hard ihc is right) | microvision candidate set needs `immunofluorescence` | |
| specimen | microvision (folder: hek human, c127 mouse, N1E115 neuroblastoma mouse, cos7 monkey) | oocytor to masked (mouse + human + sea urchin, no class for sea urchin) | kromp candidates embryo/human stay a vocabulary choice (mmbu-public `contrast microscopy` embryo rows say `mouse embryo cells`) |
| modality | none | alphadent / intraoral: candidates [clinical photography, optical imaging] (convention is mixed in mmbu-public) | |
| domain | none | hushem add urology; biomeditech drop ophthalmology (both from mmbu-public rows of the same kind) | dental sets correctly masked |

mmbu-public conventions used as evidence (from `plan/eval_public.parquet`, via `_audit_C2_src/prior.py`):
- phase-contrast microscopy: 13 rows, all `retina` / `human` / domain `biology` (the BioMediTech RPE data);
- abdominal CT: 24 rows, all `abdomen`; `liver` occurs only on 2 ultrasound rows;
- sperm: 93 brightfield rows, `sperm cells`, `human`, domain [andrology, urology], stain masked;
- clinical photography: 61 rows (skin, eye) have modality `optical imaging`, 31 surgical-frame rows have modality `clinical photography`;
- fundus: `eye` 407 / `retina` ~40.

### Registry label errors (contradicted by the files)

| dataset | field | registry | evidence | fix |
|---|---|---|---|---|
| microvision-fluorescence-cell-detection | specimen | human (hard) | folders c127_dapi (mouse C127), neuroblastoma (N1E115 mouse, Cell Image Library), cos7_nucleopore (monkey): 916 of 4,127 jpg | candidates [human, mouse, non-human primate] |
| microvision-fluorescence-cell-detection | stain | candidates dapi, phalloidin | nucleopore, peroxisome, glycophorinA(+FOXO3) folders are antibody immunofluorescence (record text: Alexa Fluor 488 secondary) | add immunofluorescence |
| grazpedwri-dx | body_part, route | hand/upper limbs/wrist/appendicular skeleton; HF mirror | every file name carries `WRI`; HF mirror has only 1,010 of 20,327 images | hard wrist; route to figshare 14825193 |
| biomeditech-rpe-maturation | body_part, domain | cells/eye/retina; [biology, ophthalmology] | mmbu-public phase-contrast hPSC-RPE rows: retina, domain biology | hard retina; domain [biology] |
| oocytor-oocyte-segmentation | specimen | candidates embryo/human/mouse | record: mouse and human oocytes plus sea urchin eggs; no class for sea urchin; folders not listable | masked |
| hushem | domain | [andrology] | mmbu-public sperm rows: [andrology, urology] | add urology |
| alphadent, intraoral-caries-ahmed-2025 | modality | clinical photography (hard) | mmbu-public uses optical imaging for 61 of 92 clinical-photography rows | candidates [clinical photography, optical imaging] |
| crossmoda-2021 | body_part | candidates brain/ear | head ceT1 MRI; cochlea is a segmentation target | hard brain |
| msd-task08-hepatic-vessel, msd-task10-colon | body_part | abdomen+liver / abdomen+colon | portal-venous abdominal CT; mmbu-public abdominal CT is `abdomen` | hard abdomen |
| knee-xray-osteoporosis-wani | body_part | bone/knee | folder and title name the knee only | hard knee |
| five-d-us-ablation-deshmukh | specimen, body_part | human, soft tissue | record does not state specimen | unverified (dropped anyway) |

Verified consistent (no change): palm, bbbc039, kromp (all but the vocabulary choice), oct2017-kermany, cmmd, cbis-ddsm, sustech-sysu, iscrc-immunoscore, kermany-pediatric-pneumonia-cxr, tbx11k, aml-cytomorphology-lmu, bcss, medsegmnist-derm2d, pcle-rat-mosaicking-optica, lightweight-instrument-seg-ush (laparoscopy/abdomen plausible, source not stated).

## 4. LABEL FIXES (snippets for `plan/registry_fixes.json`)

```json
{
 "route": {
  "grazpedwri-dx": {"method": "figshare", "ref": "14825193"},
  "bbbc039": {"method": "direct", "ref": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"}
 },
 "drop": {
  "five-d-us-ablation-deshmukh": "5-part RAR, raw timing data, unlistable",
  "color-doppler-twinkling-biopsy-markers": "IEEE DataPort login",
  "biad1137-hht1-single-molecule-tracking-spombe": "BioStudies host unsupported; movie stacks",
  "multieye": "processed compilation of other selected datasets (Kermany OCT2017, EyePACS, ODIR, FIVES)"
 },
 "labels": {
  "grazpedwri-dx": {"body_part": {"mode": "hard", "values": ["wrist"]}},
  "hushem": {"domain": {"mode": "multi", "values": ["andrology", "urology"]}},
  "biomeditech-rpe-maturation": {"body_part": {"mode": "hard", "values": ["retina"]}, "domain": {"mode": "multi", "values": ["biology"]}},
  "crossmoda-2021": {"body_part": {"mode": "hard", "values": ["brain"]}},
  "msd-task08-hepatic-vessel": {"body_part": {"mode": "hard", "values": ["abdomen"]}},
  "msd-task10-colon": {"body_part": {"mode": "hard", "values": ["abdomen"]}},
  "knee-xray-osteoporosis-wani": {"body_part": {"mode": "hard", "values": ["knee"]}},
  "alphadent": {"modality": {"mode": "candidates", "values": ["clinical photography", "optical imaging"]}},
  "intraoral-caries-ahmed-2025": {"modality": {"mode": "candidates", "values": ["clinical photography", "optical imaging"]}},
  "microvision-fluorescence-cell-detection": {
   "specimen": {"mode": "candidates", "values": ["human", "mouse", "non-human primate"]},
   "stain": {"mode": "candidates", "values": ["dapi", "phalloidin", "immunofluorescence"]}
  },
  "oocytor-oocyte-segmentation": {
   "specimen": {"mode": "masked", "values": []},
   "body_part": {"mode": "hard", "values": ["cells"]}
  },
  "kvasir-capsule": {"body_part": {"mode": "hard", "values": ["gastrointestinal tract"]}}
 },
 "exclude_paths": {
  "cbis-ddsm": "/(Calc|Mass)-(Training|Test)_P_\\d+_(LEFT|RIGHT)_(CC|MLO)_\\d+/",
  "intraoral-caries-ahmed-2025": "^Benchmarking Dataset\\.zip$",
  "biomeditech-rpe-maturation": "RPE_dataset/Images/"
 }
}
```

Notes on the snippets:
- `kvasir-capsule` hard `gastrointestinal tract` is optional (confidence medium-low): the HF mirror has no per-sample label and mmbu-public labels whole-tract white-light endoscopy `gastrointestinal tract` (35 of 163 rows). Leave the candidate set if you prefer.
- `cbis-ddsm` regex verified on the HF listing: 10,239 DICOM entries -> 3,419 kept (all full mammograms); 6,820 ROI-mask and cropped-patch files (54 GB) removed.
- `biomeditech` exclusion only takes effect in `src/bio_waila/data/prepare.py` (bug 2): `Images/` (195 full 2560x1920 frames) are the parents of the 1,862 `Subwindows/` crops.
- After the `grazpedwri-dx` re-route the figshare `dataset.csv` (projection, laterality, age) is available to metalabel but carries no Pass-2 field; `WRI` in the file name does.
- Also keep these dataset-level choices in mind (no snippet): `tompei-cmmd` and `cmmd` hold the same CMMD images (same group, fine); `pcle-rat-mosaicking-optica` has 47 frames.

## 5. Matcher additions (`src/bio_waila/data/rules.py`), with evidence

All evidence is a real `match_value` run on real values from this audit.

| rule | evidence |
|---|---|
| specimen: remove `cos ?7` from the first (cell line) rule and keep it only as non-human primate | `match_value('cos7', 'path:d2')` and `'cos-7'` return `{}` (two strong rules conflict); `vero` alone resolves to non-human primate |
| specimen: add `\bhek\b` -> cell line (or human), `c127` -> mouse, `n1e115` -> mouse | `hek_peroxisome_class`, `c127_dapi_class`, `hek`, `c127`, `n1e115` all `{}`; only `hek ?293\w*` exists |
| stain: `phal(loidin)?` as a token -> phalloidin | `match_value('phal')` -> `{}`, `neuroblastoma_phal_class` -> `{}` (folder in microvision) |
| stain: `nucleopore|nuclear pore|peroxisome|glycophorin\w*|foxo3` -> immunofluorescence (strong) | folder names `cos7_nucleopore_class`, `hek_peroxisome_class`, `erythroblast_dapi_glycophorinA_class` give only `dapi` or nothing |
| stain: `cd3|cd8|cd20|cd45` (file-name marker) -> ihc | `match_value('1_CD3_CT.jpg')` -> `{}`; iscrc-immunoscore names are `<id>_{CD3,CD8}_{CT,MI}.jpg` |
| body_part: `\bwri\b` -> wrist | `0001_..._WRI-L1_M014` -> `{}` (grazpedwri-dx) |
| `_IDLIKE` tail `([_-]\w+)?$` -> `([_-]\d+)?$` | `match_value('c127_dapi_class')` -> `{}` while `'c127 dapi class'` and `'cos7_dapi_class'` give stain dapi: the id regex swallows a folder name |
| SKIP_COLUMN: add `mass|calc|margin|shape|distribution|abnormal|assessment|subtlety|density` | cbis-ddsm `mass shape` value `LYMPH_NODE` -> body_part lymph nodes (strong) |
| path tables to ignore: `results/`, `annotations_cache`, `models/` | microvision has 74 `retina_*_dn_n30/50.csv` training lists that crowd `meta[:50]` and give weak hits (`cell - c127 dapi` -> body_part cell) |

Checked and working: `hf:modality=MRI/CT`, `hf:label=fundus_images`, `path:d4=cfp-clahe-224x224` (fundus), `path:d4=oct-filter-448x448` and `path:d1=assemble_oct` (OCT), `csv:image view=CC/MLO` and `path:d2=cbis_ddsm` (mammography), path `Knee Osteoporosis Classification` (knee). Diagnosis labels (NORMAL/PNEUMONIA, CNV/DME/DRUSEN, Osteopenia, Frontal/Mandibular) correctly return nothing.

## 6. Pipeline bugs

1. **Archive filter discards the only archive** (`src/bio_waila/data/download.py: fetch_entries`): `archives = [e for e in archives if not re.search(r"mask|label|seg", e.name, re.I)] or archives[:0]` returns an empty list when every archive matches, so the dataset ends in "no media files or archives listed". Hit here: `lightweight-instrument-seg-ush` (`A Lightweight Segmentation Network ... .zip`) and `oocytor-oocyte-segmentation` (`SegmentationCortex.tar.gz`); both single-archive records. Patch: `or archives` (keep everything when all match) and drop the bare `seg` alternative (use `\b(masks?|labels?|annotations?)\b`). Probably affects other datasets whose only archive name contains "seg": grep `_status.json` for "no media files or archives listed".
2. **`exclude_paths` is not applied to remote zip/tar members**: `src/bio_waila/data/download.py` filters only listed entries (`fetch_entries` input); `sample_zip_remote` / `sample_tar_stream` pick members blindly, so excluded members use up the quota and the `n >= want*0.5` "done" test can pass falsely. Only `src/bio_waila/data/prepare.py` filters members later. Affects the zip-based exclusions here (biomeditech `Images/`).
3. **`fetch_hf` streaming path loses file names**: for imagefolder repos where masks sit beside images (bbbc039: 400 rows = 200 images + 200 `_masks.tif`), rows are PIL images with no filename and the mask regex looks only at column names, so ~50% of saved images are label maps. Fix: stream only when the repo siblings are parquet/arrow; if siblings are image files use the `hf-files` path (names + `is_mask`).
4. **Zip-range sampling ignores `--max-gb`; budget pick keeps the smallest files**: `medsegmnist-derm2d` has 2,594 members and want = 1.4 x 2000 = 2,800, so the whole 38.5 GB zip is extracted (budget 20 GB only gates whole-file downloads). For loose files (`cbis-ddsm`, 29 MB DICOM) `fetch_entries` sorts the pick ascending by size and keeps the smallest that fit the budget, which biases toward small/low-resolution images.
5. **`sample_tar_stream` takes the first N media in archive order** (not spread over folders): oocytor would give one species/condition only.
6. **Kaggle listing**: `list_kaggle` requests `pageSize=1000` but the API caps at 200; large sets (aml ~92 pages, tbx11k ~56, bcss ~490) hit HTTP 429 after 5-11 requests and `requests` raises `RetryError` (state `error`, retried by `--retry`). Honour `Retry-After`, sleep between pages, cache the listing, and for >20k-file sets use the whole-dataset zip with `sample_zip_remote` (already written to resolve the redirect without auth).
7. **`plan_images` vs available data**: 3-D sets give `volumes x SLICES_PER_VOLUME(4)`: crossmoda 105 -> ~420 images, msd-task10 126 -> 504, msd-task08 303 -> 1,212 (plan 2,000 each, state `partial`). Raise slices per volume for small cohorts or lower the plan. 2-D shortfalls: pcle-rat 47 frames (plan 3000), grazpedwri HF mirror 1,010 (plan 2000), hushem <=216 (plan 216 with 328 repeated rows).
8. **DICOM series folders**: `is_mask` sees only the last two path components, so CBIS-DDSM ROI masks / cropped patches (`<case>_<n>/<series>/<uuid>.dcm`) are never flagged; handled by `exclude_paths` above, but any other TCIA-style tree will need the same.
9. **Metadata table pick** (`meta[:50]`) is crowded by training lists in microvision (74 `retina_*.csv`): same defect as audit A item 7.
10. **`resolve_doi` rules missing** for `ebi.ac.uk/biostudies` (biad1137; files are reachable at `https://www.ebi.ac.uk/biostudies/files/S-BIAD1137/<path>` without auth) and `ieee-dataport.org` (login).
11. HF repos carry macOS AppleDouble junk (`._*` next to every file, e.g. 5,202 `._*.dcm` in cmmd): correctly filtered by `is_junk`; no action.

## 7. Duplicates and derivatives

| datasets | relation |
|---|---|
| cmmd, tompei-cmmd | same CMMD images (tompei adds lesion masks); same split group |
| oct2017-kermany, octmnist, multieye | OCTMNIST is a down-sized Kermany OCT2017; ~83k of multieye's OCT images are the same Kermany images (multieye dropped) |
| intraoral-caries-ahmed-2025 | `Benchmarking Dataset.zip` repeats `Dataset.zip` (same 6,265 jpg) |
| bcss | 224 px and 512 px versions are two resizes of the same TCGA ROIs |
| microvision | the same images re-appear in several class sets (erythroblast x3, erythroid x2, hek x2, neuroblastoma x2, mixed x3); prepare's within-dataset dHash removes them |
| knee-xray-osteoporosis-wani | Kaggle (1,947) + Mendeley (350) images, flipped/rotated to 1,800 per class |
| biomeditech-rpe-maturation | 1,862 `Subwindows/` are crops of the 195 `Images/` |
| hushem | 328 VQA rows for <=216 images (4 option permutations) |
| tbx11k | `imgs/extra/` holds 1,106 images of DA, DB, Montgomery and Shenzhen (also hosted elsewhere); group ju-tbv1 is an aggregate of TB sets |
| grazpedwri-dx, fracatlas; kromp, humanembryo2-simula; sustech-sysu, idrid, reta, rav-rotterdam | different datasets sharing a split group only |
| aml-cytomorphology-lmu | Kaggle copy of the TCIA original (Kaggle license "Unknown", registry CC-BY-3.0) |

## Files

- `audit_C2.jsonl` - 31 records.
- `_audit_C2_src/` - collector, digest, prior lookup, Kaggle probes, overrides, assembler.
