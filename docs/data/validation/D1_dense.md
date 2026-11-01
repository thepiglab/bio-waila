# D1: dense-sidecar validation on 12 datasets (2026-10-30)

**What was validated:** the first real-data check of `waila-dense` (`src/bio_waila/data/dense.py`).

**What "ok" means: registered, not correct.** An ok dataset's sidecars carry the source's annotations for each image, placed on the same structures in our prepared JPEG as in the source image. That checks our pairing and geometry. It does not check the annotations themselves: whether a class is right, an outline accurate, or every object labelled. Nobody has checked that for these datasets; see "Label quality" below.

**How it ran:**
- Command: `uv run waila-validate --dense --root data/cache_dense --worker d<k> --pilot 40 --max-gb 2 --only <list>`. Workers were d1 to d4; d1 was stopped mid-run and its queue was restarted as d3 on fixed code.
- The scratch cache root kept the earlier V/R/N reports in `data/cache/` untouched.

**What was checked:**
- Automatically, on every sampled image (the n/m counts below are images with a sidecar out of images sampled): a label file was paired to the image, the mask decoded, its size equals the source image's, and it is not a plotted figure.
- By eye: the first 4 images per dataset, as 256 px thumbnails with the mask layers tinted and region boxes, seed points and file boxes, points and polygons drawn on. The script was a scratch file, not committed. A shift of a few pixels would not show at that size; the synthetic tests in `tests/test_dense.py` check the geometry exactly.
- Against the source, for bccd only: all 558 source boxes for the 40 images are in the sidecars, with the same classes per image.

**Which datasets:** the 12 cover every capture path:
- HF mask and box columns;
- whole zips;
- zip range reads with an annotation top-up, including masks in a second zip;
- loose Dataverse files;
- NIfTI label volumes;
- YOLO, VOC, labelme and Roboflow COCO annotation files.

## Counts

| Verdict | n | Datasets |
|---|---|---|
| ok (registered) | 10 | ccm-seg-2026, parasitobank-2025, murinecyto-det, rav-rotterdam-artery-vein, prostate-ppat-healthy-longitudinal-2026, asoct-msdc, l929-phc-cell-counting, fundus-avseg, fedstenonet-xca, bccd |
| rejected, correctly | 2 | papila (masks are plots), fdr26 (masks do not match the images) |

## Per dataset

- **ccm-seg-2026** (40/40, zenodo whole zip): binary corneal-nerve masks paired by stem; they lie on the nerves. This run named them "foreground"; `plan/dense_fixes.json` now names the class.
- **parasitobank-2025** (40/40, HF stream): `objects.bbox` boxes named from the nested ClassLabel (blastocystis, entamoebaHart, endolimaxNana, …); the boxes sit on cyst-sized objects. The species were not checked.
- **murinecyto-det** (40/40, zenodo whole zip):
  - Colour masks plus one COCO json per image. Mask colours are named by overlap with the dataset's own boxes, an inference rather than a legend: macrophage/monocyte 536, debris 498, neutrophil 418, eosinophil 43, lymphocyte 13 regions.
  - The json files say "Research Use Only" (Kaggle source) while Zenodo says CC-BY-4.0. This is noted in `dense_fixes.json`.
- **rav-rotterdam-artery-vein** (40/40, Dataverse files): 56 colour A/V masks fetched as loose files; the regions lie along the vessels. Colours stay unnamed until the legend is checked.
- **prostate-ppat-healthy-longitudinal-2026** (40/40, zip range):
  - The top-up fetched 28 `label.nii.gz` members (0.2 MB) from the zip the volumes came from.
  - Slices are taken with the image's canonical orientation and `rot90`; the labelled regions lie in the prostate area of the thumbnails.
  - The volumes hold label values 1 to 4 and the record has no legend. This run named value 1 "periprostatic adipose tissue" from the dataset title alone; that guess has been removed. Telling the gland from the fat around it needs the legend or a reader.
  - 11 slices are empty layers.
- **asoct-msdc** (40/40, figshare zip range): labelme polygons (anterior chamber, iris, lens) fetched for the sampled images; they trace those outlines.
- **l929-phc-cell-counting** (40/40, whole zip): YOLO boxes around cells, with class names from `classes.txt`.
- **fundus-avseg** (28/28, figshare whole zip): colour A/V masks paired by stem; the regions lie along the vessels.
- **fedstenonet-xca** (40/40, whole zip): VOC xml "stenosis" boxes. They lie on coronary vessels, so the coordinates are not flipped or scaled; whether each marks a real stenosis was not checked.
- **bccd** (40/40, direct zip): Roboflow `_annotations.coco.json` boxes (RBC, WBC, platelets). Every source box is carried over, but the source leaves many red cells unboxed (plain in the thumbnails).
- **papila** (0 usable, HF stream): the HF copy's cup/disc mask columns are matplotlib figures (axes, ticks, white margins). Build now rejects them as `figure`; the figshare original has contour files.
- **fdr26** (0/40, zenodo zip range + `vessels_FDR26.zip`):
  - The cross-zip top-up worked: 112 masks, 4.4 MB.
  - But every mask is 560×576, while the images are 1960×1958 or 2576×1958, so build rejects them as `size_mismatch`. The registry already notes the vessel masks are not original to the dataset.

## Label quality (not checked)

Registration says nothing about whether the source labels are right. From the thumbnails alone:
- **Incomplete:** bccd boxes only some of the red cells. A missing box does not mean a missing cell, so this dataset cannot back a claim such as "no other cells here".
- **Unnamed or inferred:** prostate-ppat values 1 to 4, the rav and fundus-avseg colours, and the murinecyto colours (named by box overlap).
- **Needs a domain reader:** fedstenonet stenoses, parasitobank species and murinecyto cell types. Thin structures such as the ccm-seg nerves cannot be judged at thumbnail size.

## Bugs found and fixed in this round

1. Roboflow's small `_annotations.coco.json` was skipped: tables were filtered by size. Tables are now chosen by name and type (found by the round-trip test).
2. A folder named `Annotated Images` made every per-image json look "dataset-level", so `fetch` pulled about 1,100 files instead of 56. The test is now on the file name only, with a case key of at most 2 tokens.
3. Per-image COCO files (one image per json) were not read.
4. Plotted "masks" (PAPILA) were accepted; they are now rejected as `figure`.
5. `waila-validate --dense` skipped the fetch when the raw data already existed.
6. A mask holding a single class was turned into a 0/1 mask, so it took class 1's name: a prostate slice holding only value 4 would be named after value 1. Such masks now keep their value, and colour masks are named only by their colour, never by their per-image index. This was found after the run, so this run's 7 single-class prostate slices may carry the wrong name.

Each fix has a case in `tests/test_dense.py`.
