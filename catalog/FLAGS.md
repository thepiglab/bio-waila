# Dataset flags for follow-up

Issues the catalog reviewers raised on 2026-10-08 while checking descriptions and sample counts (`notes_*.yaml`). Some came
from validation reports older than the latest pipeline fixes; re-check those on the cluster pilot before acting.

## Licence or provenance (check before training)
- **m4raw-abdomen**: the HF card calls it a private release-candidate snapshot; consent wording is not confirmed.
- **skin-lesion-yolo-ham-7class-makh**: likely HAM10000 pixels (CC BY-NC) re-uploaded under an MIT tag; treat as nc or drop.
- **asraf-covid-pneumonia-normal-pa**: we download a Kaggle mirror (6,902 files); the Mendeley original lists 4,575.

## Labels that look wrong
- **gb-wce-bleeding**: labelled capsule endoscopy; validators saw conventional upper-GI endoscopy frames.
- **pcle-pto**: mostly lens-paper videos (255 of 287) but tagged heart.
- **fucci-rpe1**: decoded as brightfield crops, but it is a FUCCI fluorescence dataset.
- **malhari-cervical-dataset**: about 70% Pap-smear microscopy and 30% colposcopy in one dataset label.
- **carotid-us-color-doppler-kaggle**: mixes B-mode and Doppler.
- **piur-thyroid-segmentation**, **shoulder-3d-us-mosaicking**: tagged 3D ultrasound, but the images we use are 2D B-mode slices (piur already multi-labelled).
- **ct-lymph-nodes**: slices span neck to pelvis, so the study-level body-part label is noisy.
- **openmind**: includes derived FA / T2 maps.

## Duplicates, synthetic or derived content
- **cecleanliness-wce**: about 82% synthetic corrupted copies.
- **rvo-me-oct**: every scan is stored three times (task folders); **covid-qu-ex**: the infection subset repeats lung-set images.
- **skin-lesion-yolo-det-10class**, **mpox-hsam**: Roboflow / augmented copies; mpox-hsam's mirror has 4 class folders, not 6.
- **evisan**: about 4,100 of 6,000 frames are near-duplicates.
- **drimdb**: about 8% non-fundus images; **ir-pupil-spectral-exposure**: archive includes spectrum plots and eye-tracker video.
- **fovea-intraoperative-fundus**: the 560 files include vessel / disc masks; **tcga-lgg-mask**: may be masks only.

## Coverage (pilot sampled only part, or few usable images)
- **cradi**: the only zip is named CHARDI_test (14.6 GB), maybe a subset of the 84k images.
- **bowen-nigeria-colposcopy**: the 119 MB Mendeley archive looks like a subset of the 3,356 images in the article.
- **midog-2022** (32 images of a 30 GB record), **sted-microtubules** (4 images), **cystofold** (~5 usable), **sbfsem-h446** (one 100-slice volume).
- **crc-ihc-cish-he-epithelium**: we use only the H&E zip; **glenda**: the zip we use is the no-pathology package.
- **camelyon17**: the HF source is ~1.06M 96x96 patches (WILDS-style), not the 1,000 WSIs.
- **bm-broncholc**: red and blue channels swapped in the source PNGs (blue cast); only the Lung_cancer folder was sampled.

## Format blockers (handle on the cluster)
- RAR archives (need unrar / bsdtar): aattct-ids, lunghist700, modus-urine-sediment-2026, cas-colon, renji-esd-esophagus (~212 GB), gim-endo, rarpl-workflow, reta-benchmark, cca-intima-media-cyprus, colorectal-polyps-gaze.
- Whole-slide images (need `uv sync --extra wsi`): bracs, mitos-wsi-ccmct. HDF5: cetus-2014. Unsupported zip compression: ispy2.
- Unsupported hosts: hagen-2021-fluorescence-denoising (GigaDB), biad2493 / biad3061 (BioStudies), tcga-coad-ct (TCIA), isic-clonal-sk (ISIC), gzmh-gland (scidb.cn).
