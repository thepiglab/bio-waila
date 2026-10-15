# V7: audit of the 14 non-commercial (nc) datasets

Worker w7, pilot 100 images, 2 GB per dataset (1 GB for derma-octa). Per-dataset records are in `V7_nc.jsonl`. Sample images were looked at on contact sheets of 6 per dataset (3 x 2, captions with the labels), and for mhsma, echoxflow and the ISIC routes also against the raw arrays/zips.

How this was run. Plain `waila-validate` crashes on some of these datasets (bugs B1, B6; V5's B1/B2), so most runs went through a scratch wrapper that only monkeypatches pandas-3 quirks (same idea as the V5 wrapper); routes, `exclude_paths` and the pannuke overrides were tested without touching `plan/` via `P2_TRY` row overrides and a copied plan dir (`WAILA_PLAN_DIR`). Consequently `data/cache/reports/{isic-2018,isic-2019,pitvis-2023,derma-octa,pannuke,ead2020}.json` describe the *tested* route/fix (the registry-route outcomes are in `V7_nc.jsonl`). Cache: 7.9 GB of 20 GB at the last check; raw samples are deleted (`--keep-raw` never used; one 504 MB EAD zip I fetched by hand was removed by the harness).

## Verdict in one paragraph

Only 6 of the 14 are worth keeping, and they carry the value for a specific reason: **MMBU-source domain match** (isic-2019 91 public rows, cholecinstanceseg 89, mhsma 68, pitvis-2023 36, cholecseg8k 18, pannuke 12 = 314 of 3,147 public rows, 10%; 225 = 7% without the blocked cholecinstanceseg). Almost none adds *classes* the permissive layer lacks: the NC datasets' classes are already covered by permissive data (dermoscopy 12 hard datasets / 23k images, laparoscopy 20 / 65k, clinical photography 16 / 25k, white-light endoscopy 24 / 44k, OCTA 3 / 8k). The one real gap-filler is **pitvis-2023** (the only substantial endoscopic head-and-neck / pituitary source; permissive `uaal-upper-airway-landmarks` has 0 public rows). Eight are redundant or unusable (echoxflow, darkfield-ir-slo-choroidal-nevi, isic-2018, fitzpatrick17k, ead2020, derma-octa, octa-mosaicking, bbbc041).

## Summary table

"Public rows" = `mmbu_public_rows` in `plan/selection.jsonl`. "Needs" lists what must happen before the dataset works.

| Dataset | Licence (authoritative) | Public rows | What it adds | Quality | Recommendation |
|---|---|---:|---|---|---|
| isic-2019 | CC BY-NC 4.0 (ISIC); BCN half also CC BY 4.0 on figshare | 91 | MMBU's largest accessible NC source. Image-level mostly redundant: contains all of ISIC 2018 T3 (HAM10000, 10,015), 2,593/2,594 of 2018 T1, about 12.4k BCN images that are in permissive `bcn20000`; genuinely new = HAM10000 + MSK ~13k | Registry HF route is 224x224 centre-cropped (poor). Direct S3 zip: full-res, correct. Labels correct | **keep-in-nc**; needs route to S3 zip |
| cholecinstanceseg | CC BY-NC-SA 4.0 (arXiv; code Apache-2.0) | 89 | Same Cholec80 appearance as 107 of 405 public laparoscopy rows with cholecseg8k; 41.9k frames | Not inspected: Synapse needs a login | **keep-in-nc**, blocked: manual Synapse download |
| pitvis-2023 | CC BY-NC-SA 4.0 (UCL RDR/figshare); HF mirror says NC-ND and is gated | 36 | Only head-and-neck/pituitary white-light endoscopy source; permissive body_part pituitary gland has 0 datasets | Frames correct; 1 of 6 blocky. Labels need 'head and neck' | **keep-in-nc**; needs figshare route + labels fix |
| mhsma | CC BY-NC-SA 4.0 | 68 | Best rows per image: all 93 public sperm rows are brightfield; permissive sperm sets (hushem, evisan, ...) look different | Pipeline output is 100% garbage strips today (bug B4); data are real but 128x128, blurry | **keep-in-nc** after the prepare fix |
| cholecseg8k | CC BY-NC-SA 4.0 | 18 | Accessible Cholec80 stand-in; 52% of its frames are near-dups of public images (dropped) | Correct frames, 512x288 | **keep-in-nc** |
| pannuke | CC BY-NC-SA 4.0 (research use, cite) | 12 | Only nc source of per-image organ labels at 40x H&E (16 organs after overrides) | 256x256 nuclei-level patches, labels correct; body_part 83% hard with overrides + bug B3 fix | **keep-in-nc** (marginal) |
| isic-2018 | Task 1-2 CC0; Task 3 CC BY-NC | 13 | Nothing: T1 is 100% inside isic-2019, 90% of permissive isic-2017, 90% of isic-2016, and is medsegmnist-derm2d | Correct; registry HF route crashes (B1) | **redundant** |
| fitzpatrick17k | CC BY-NC-SA 3.0; pixels owned by two atlases | 10 | Nothing the 16 permissive clinical-photography sets lack | Correct; VQA duplicates; 'SFS' watermarks | **redundant** (+ image-rights risk) |
| ead2020 | CC BY NC 3.0 | 6 | Same family as permissive edd2020 (43 rows) / polypgen | GI frames correct; text-overlay frames; 27/150 in-dataset dups | **redundant** |
| bbbc041-p-vivax-malaria | CC BY-NC-SA 3.0 | 7 | Giemsa smears already covered by 7 permissive datasets | Correct; 41% of files lost to Kaggle zip wrapper (B2); every image listed twice | **redundant** |
| derma-octa | CC BY-NC 4.0 | 0 | Skin OCTA only; OCTA target already met by permissive (8,037 vs 8,000) | Correct; 5 filter variants of the same 331 scans; stock route fetches 3D volumes | **redundant** |
| octa-mosaicking | CC BY-NC 4.0 | 0 | 71 retinal OCTA images | Correct, tiny | **redundant** |
| echoxflow | CC BY-NC-SA 4.0 | 0 | Would add colour Doppler, but only as raw zarr arrays | Beamspace (unwarped) B-mode in zarr; no reader; 441 GB | **drop** |
| darkfield-ir-slo-choroidal-nevi | CC BY NC 3.0 | 0 | Nothing: no images in the record | Record = one 11.6 KB participant spreadsheet | **drop** |

Licence notes beyond the table:
- Registry says `multiple` for isic-2018. Per `challenge.isic-archive.com/data`, Tasks 1-2 are CC-0 and Task 3 is CC-BY-NC, and the registry route is Task 1, so isic-2018 would belong in the permissive layer if it were wanted at all.
- pitvis-2023: three sources disagree. UCL RDR (figshare API `license`) and the registry say NC-SA; the HF mirror card says NC-ND. NC-ND would forbid the frame extraction/resizing this pipeline does. Use the UCL record.
- cholecinstanceseg: the licence comes from the dataset paper, not from the Synapse page (which needs JavaScript and a login). The GitHub repo says Apache-2.0, but that is the code.
- Share-alike on NC-SA data: if the Pass-2 teacher's labels train the submitted model, whether that counts as a derivative is a rules question, not something this audit settles.
- Possible licence-laundering in the permissive layer, not verified: `skin-lesion-yolo-ham-7class-makh` (MIT, "HAM-style" classes) is very likely HAM10000 (CC BY-NC) pixels; `medsegmnist-derm2d` (CC BY-SA) is ISIC 2018 Task 1 (CC-0, fine). Worth a look by whoever owns tier A.

## Registry fixes (`plan/registry_fixes.json` format)

Entries whose reason starts with "redundant" are optional: apply them if the layer should shrink to the six worth keeping. `isic-2019` and `pitvis-2023` routes were run end to end in the harness. For derma-octa or bbbc041, if someone keeps them anyway: derma-octa needs `exclude_paths` `3D|Contrast|Frangi|Morph_angio|Proj_art|codes|Manual` (verified: leaves `Norm - 2D.zip`, 993 PNG) and bbbc041 needs `^malaria/malaria/`.

```json
{
 "drop": {
  "echoxflow": "pixels are zarr v2 beamspace arrays inside per-exam tars (441 GB); no reader, B-mode not scan-converted, colour Doppler only as raw arrays",
  "darkfield-ir-slo-choroidal-nevi": "Mendeley record fdnb2kyz4f holds only an 11.6 KB participant spreadsheet, no images",
  "isic-2018": "redundant: Task 1 images are 100% inside isic-2019, 90% of permissive isic-2017 and in medsegmnist-derm2d; registry HF route also crashes (ArrowInvalid)",
  "fitzpatrick17k": "redundant: 10 public rows, 16 permissive clinical-photography datasets; atlas-owned pixels, unlicensed HF VQA mirror with duplicate rows",
  "ead2020": "redundant: same lineage as permissive edd2020/polypgen, 6 public rows, 24 permissive white-light endoscopy datasets",
  "derma-octa": "redundant: 0 public rows, OCTA target met by permissive data, 5 filter variants of the same 331 skin scans",
  "octa-mosaicking": "redundant: 71 images, 0 public rows, OCTA covered by 3 permissive datasets",
  "bbbc041-p-vivax-malaria": "redundant: 7 public rows, Giemsa smears covered by 7 permissive datasets; Kaggle record lists every image twice"
 },
 "route": {
  "isic-2019": {"method": "direct", "ref": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip"},
  "pitvis-2023": {"method": "figshare", "ref": "26531686"}
 },
 "labels": {
  "pitvis-2023": {
   "body_part": {"mode": "candidates", "values": ["head and neck", "nose", "pituitary gland"]},
   "domain": {"mode": "multi", "values": ["endocrinology", "neurology", "otolaryngology", "surgery"]}
  }
 },
 "exclude_paths": {
  "mhsma": "/(x_64|y_[a-z]+)_(train|valid|test)\\.npy$"
 },
 "force_include": {}
}
```

Notes on the block:
- isic-2019 route: 140 files = 56 MB via zip-range, 100 images kept, full resolution, no `hf_rows`. Replaces `flwrlabs/fed-isic2019` (224x224 centre crop, 23,247 images).
- pitvis-2023 labels: the 36 public PitVis rows are body_part `head and neck`, domain `otolaryngology` + `surgery`. That is evidence from the eval set; if the team does not want eval-derived conventions in training labels, keep only the `body_part` change (head and neck is also the natural class for an endonasal approach) and leave `domain` alone.
- mhsma `exclude_paths` drops the duplicate 64x64 copy and the 1-D label vectors; it is only useful together with the B4 fix.
- I did not change cholecseg8k or cholecinstanceseg labels, but note MMBU's laparoscopy rows are `abdomen` (337 of 405) and domain `surgery` (355 of 405), while the registry uses candidates abdomen/gallbladder/liver and domain gastroenterology + surgery. Not worth a fix, just a known drift.

## Overrides (`plan/overrides.csv` rows)

pannuke `hf:tissue` is a ClassLabel index (0-18). Adrenal Gland (0), Bile Duct (1) and Testis (16) have no class in `label_space.json` and stay masked. **Only effective together with the fix for bug B3**; verified in scratch: without it the column is "trusted" but body_part stays masked; with it 83 of 100 images get a hard body_part (breast 36, colon 23, cervix 7, lung 4, ...).

```csv
dataset_id,column,value,field,class
pannuke,hf:tissue,2,body_part,bladder
pannuke,hf:tissue,3,body_part,breast
pannuke,hf:tissue,4,body_part,cervix
pannuke,hf:tissue,5,body_part,colon
pannuke,hf:tissue,6,body_part,esophagus
pannuke,hf:tissue,7,body_part,head and neck
pannuke,hf:tissue,8,body_part,kidney
pannuke,hf:tissue,9,body_part,liver
pannuke,hf:tissue,10,body_part,lung
pannuke,hf:tissue,11,body_part,ovary
pannuke,hf:tissue,12,body_part,pancreas
pannuke,hf:tissue,13,body_part,prostate
pannuke,hf:tissue,14,body_part,skin
pannuke,hf:tissue,15,body_part,stomach
pannuke,hf:tissue,17,body_part,thyroid
pannuke,hf:tissue,18,body_part,uterus
```

## Pipeline bugs and reproductions

Existing from V5: B1 (NaN in `load_tables`) and B2 (empty review.csv) still apply; numbering below is new for V7 and independent of V5.

**B1. HF streaming shuffle buffer exhausts memory on large images; the failure is not caught as a route fallback** (`download.fetch_hf`, `dsd[sp].shuffle(seed=seed, buffer_size=2000)`). isic-2018 (`MedOtter/ISIC2018`, full-resolution dermoscopy) filled the buffer for ~13 min at ~14 GB RSS, then died with `ArrowInvalid: offset overflow while concatenating arrays`, reported as `download error` (the `except` that falls back to file listing wraps only `load_dataset`, not the iteration).
Repro: `uv run waila-validate --only isic-2018 --worker x --pilot 100 --max-gb 2`.
Fix: a small buffer (e.g. 50-200), or shuffle only when rows are small, and move the iteration inside the fallback `try`.

**B2. Kaggle per-file downloads sometimes return a zip wrapper; `get_file` saves it under the image name.** Hits bbbc041 (`kmader/malaria-bounding-boxes`): 11 of 12 random PNGs of 1.1-2.4 MB were `Zip archive data` with the PNG inside (the one at 0.94 MB was a real PNG), so `prepare` logged `reader_error:TiffFileError` for 57 of 140 files and the sample is biased to small files.
Repro: `uv run python -c "from bio_waila.data import download as dl; es=[e for e in dl.list_kaggle('kmader/malaria-bounding-boxes') if e.name.startswith('malaria/images/')][:12]; [dl.get_file(e, __import__('pathlib').Path('/tmp/bb')/e.name.split('/')[-1]) for e in es]"` then `file /tmp/bb/*`.
Fix: in `get_file`, if the bytes start with `PK\x03\x04` and the entry extension is not an archive type, extract the single member.
Related: the same Kaggle record lists every image twice (`malaria/images/` and `malaria/malaria/images/`), so the sampler draws duplicates unless `exclude_paths` is set.

**B3. `metalabel.apply` never matches non-string metadata values, so overrides on integer columns do nothing.** In `cmd_apply` the test is `isinstance(s.get(col), str) and s[col] in fmap[f]`, but `cmd_map` builds the map from `astype(str)` values. HF `ClassLabel` columns (pannuke `hf:tissue`) are ints in `samples.parquet`.
Repro: add the 16 CSV rows above to a copy of the plan (`WAILA_PLAN_DIR=...`), `uv run waila-validate --only pannuke --redo --worker x`: `colmaps` marks `hf:tissue` trusted for body_part but the labels stay `masked 140`.
Fix: compare `str(s[col])` (verified in scratch: body_part becomes hard for 83/100).

**B4. `prepare.read_array` cuts image stacks along the wrong axis.** `a.ndim == 3` with a last dim not in (1, 3, 4) goes to `pick_slices`, which slices along the *smallest* axis. For an `(N, H, W)` stack with N > W (mhsma `x_128_train.npy` is `(1000, 128, 128)`) it returns `(N*H) x 1` strips scaled to 66x512 or 33x512; the y_*.npy label vectors are also picked up as media. All 22 pilot images of mhsma are such strips and carry hard labels.
Repro: `uv run waila-validate --only mhsma --worker x --pilot 100`, open the sample images in `data/cache/reports/mhsma.json`.
Fix: if `a.ndim == 3 and a.shape[1] == a.shape[2] and a.shape[0] > a.shape[1]` (or generally when the first axis is the largest), sample along axis 0; skip arrays with `ndim < 2`. Correct frames: `np.load(...)[i]` gives 128x128 sperm crops.

**B5. Mendeley datasets are all marked `manual` because `requests` gets HTTP 403, but the public API and the file downloads work with curl.** `https://data.mendeley.com/public-api/datasets/<id>` returns 403 to python-requests (any User-Agent, including the pipeline's) and 200 to curl; `.../public-files/datasets/<id>/files/<fid>/file_downloaded` also 403s in requests and 302s to S3 in curl (looks like TLS-fingerprint blocking, not the UA).
Repro: `curl -s -o /dev/null -w '%{http_code}\n' https://data.mendeley.com/public-api/datasets/c7fjbxcgj9` (200) versus `uv run python -c "import requests; print(requests.get('https://data.mendeley.com/public-api/datasets/c7fjbxcgj9').status_code)"` (403).
Fix: shell out to curl (or use a TLS-impersonating client such as curl_cffi) for Mendeley listing and download. This affects every Mendeley row, not just the two NC ones (the README says Mendeley "blocks scripts").

**B6. `metalabel.load_tables` crashes with `TypeError: Cannot perform reduction 'mean' with string dtype`** (pandas-3 variant of V5 B1) when a table column has no non-null values, because `vals.map(...)` on an empty string Series stays string dtype. Hit on ead2020 (the harness reports `error: TypeError`).
Repro: with the EAD zip in `data/raw/ead2020/archives/`, `uv run waila-validate --only ead2020 --worker x`.
Fix: `.astype(bool).mean()` after guarding `vals.empty`.

**B7. zip-range sampling budget and archive choice.** `sample_zip_remote` budgets on `compress_size`, but extracted files are ~1.5x larger on disk (derma-octa 3D volumes: 15 MB compressed, 22 MB extracted), so `--max-gb 2` leaves ~3 GB in the cache; and `want_files` is not divided for datasets that list both `2d` and `3d`, so with `plan_images` 6,000 the stock run took more than 45 minutes for 2 GB of 90-slice volumes (and I aborted it at 2.8 GB). `archive_rank` prefers the 5 GB `Contrast - 3D.zip` over the 0.58 GB `Norm - 2D.zip`, which holds the same scans as PNGs.
Repro: `uv run waila-validate --only derma-octa --worker x --pilot 100 --max-gb 2`.

**B8 (data, not code). Mirror metadata disagrees with the source.** `drdreets/PitVis-2023` is `gated: auto` and tagged `cc-by-nc-nd-4.0`; UCL RDR (figshare 26531686) is open and `CC BY-NC-SA 4.0`. The registry kept the HF ref and the UCL licence, so the row is both mislicensed on the mirror and `manual` on download. Verify: `curl -s https://api.figshare.com/v2/articles/26531686 | python3 -c "import json,sys; print(json.load(sys.stdin)['license'])"`.

## Summary (about 190 words)

The NC layer is worth keeping only as a small, targeted layer, not as the 14-dataset block now in the plan. Its value is MMBU-source domain match, not new classes: the six datasets worth keeping explain 314 of 3,147 public rows (10%; 7% excluding blocked cholecinstanceseg), while their classes are already covered by permissive data. isic-2019 (91 rows) carries the most, but about half its images already sit in permissive bcn20000/ISIC 2016-17. pitvis-2023 (36 rows) is the only true gap-filler: the sole endoscopic head-and-neck source. The Cholec80-based pair (cholecinstanceseg 89, cholecseg8k 18) and mhsma (68 rows, but unusable today because of bug B4) follow, and pannuke is marginal. Eight datasets are redundant or unusable: echoxflow (zarr beamspace, no reader), darkfield (no images), isic-2018 (100% inside isic-2019), fitzpatrick17k, ead2020, derma-octa, octa-mosaicking and bbbc041. Four fixes are prerequisites: the isic-2019 S3 route, the pitvis figshare route plus label fix, the mhsma array fix (B4) and the integer-value fix (B3) for pannuke. cholecinstanceseg needs a manual Synapse download. Cache at the last check: 7.9 GB of 20 GB.
