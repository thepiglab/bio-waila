# Pass-2 labelling audit, part E (14 datasets from `audit_todo_E.json`)

Machine-readable records: `audit_E.jsonl` (one object per dataset, same schema as `audit_D.jsonl`).

Scope and method: metadata only. Used: Zenodo / figshare / Dataverse / Dryad listings and record descriptions, DataCite, the UCSD object RDF, zip central directories via ranged reads (remotezip), `match_value` / `text_hits` / `is_mask` / `kind` on the names seen, `plan/eval_public.parquet` as a label prior, and a 4 KB AVI header range. No archive body was saved. Disclosure of reads beyond metadata is in section 6; the temp dir was deleted.

## 1. Access status (14 datasets)

| status | n | datasets |
|---|---|---|
| ok | 2 | chlamydomonas-axoneme-darkfield-geyer-2024 (but only about 148 sampled frames), mouse-rvo-ffa-slo-oct-sichuan-2024 |
| ok but needs a label fix / exclude | 4 | arabidopsis-ruthenium-red-darkfield-rrquant, murine-tumor-bmode-us-uj, nuinsseg, gut-on-chip-caco2-ht29mtx-zenodo-2025 |
| ok but yields few images | 1 | rat-retina-fluorescein-angiography-video-plos-2014 (1 video = 8 frames) |
| reachable only through a new route (unverified contents) | 1 | spc-darkfield-plankton-training-images-kenitz-2022 |
| readable format mismatch | 1 | fus-rat-neuroinflammation-pereira-2025 |
| wrong ref / too few images | 1 | ctc-darkfield-unstained-blood-ciurte-2018 |
| no pixels in record | 1 | empiar-11830-chlamydomonas-cryoet |
| unsupported host, token-gated downloads (Dryad) | 3 | large-aperture-spine-volumetric-us-dryad-2025, cmut-rca-3d-power-doppler-fingertip-dryad, brvo-bevacizumab-fa-oct-mferg-dryad-2016 |

## 2. DROP list

| id | reason |
|---|---|
| ctc-darkfield-unstained-blood-ciurte-2018 | Registry ref `13822151` is a figshare file id, so `list_figshare` raises Manual. The article (7463405) has one 9 MB zip with 6 PNG: 3 darkfield (`*_DF.png`) and 3 fluorescence. The registry's 263 images are not in the public record; 3 usable images is below the 30-image floor. |
| fus-rat-neuroinflammation-pereira-2025 | Pixels are fine (145 BIDS nii.gz in a 30 GB zip, range-samplable) but the shape is (1,112,128,825): a 2D power-Doppler movie with time last. `read_volume` does `a[0]` then `pick_slices` on argmin of the first 3 axes, i.e. axis 0 of (112,128,825), producing 128x825 space-time maps instead of Doppler frames. Reinstate once 4D arrays take frames from the time axis; labels need no change. |
| empiar-11830-chlamydomonas-cryoet | Zenodo 15615650 holds only `Chlamy_Tomo_Annotations_v2.xlsx`. Pixels are 32 TB of `.mrc` on EMPIAR / CZ portal (no host, no reader). Also auto-dropped (`no_images_in_record`). |
| large-aperture-spine-volumetric-us-dryad-2025 | Dryad: `resolve_doi` has no handler; the listing API is public but `/api/v2/files/<id>/download` returns 401 (bearer token) and `/downloads/file_stream/<id>` 403. 34 GB, probably `.mat` beamformed data; 3 of 5 zips are phantoms or a simulation under a hard `human` label. |
| cmut-rca-3d-power-doppler-fingertip-dryad | Same Dryad block; one 3.1 GB `data.zip`, one volunteer, format unstated (likely `.mat`). |
| brvo-bevacizumab-fa-oct-mferg-dryad-2016 | Same Dryad block; three RARs (FA 49 MB, OCT 6 MB, mfERG 6 MB) that cannot be listed; mfERG is waveform data. |

All three Dryad datasets are reinstatable together if a Dryad handler (OAuth client credentials, `GET /api/v2/versions/<id>/files` then `/api/v2/files/<id>/download`) is added.

## 3. LABEL FIXES (exact format of `plan/registry_fixes.json`)

```json
{
  "drop": {
    "ctc-darkfield-unstained-blood-ciurte-2018": "registry ref 13822151 is a figshare file id (list_figshare fails); the public S1 zip holds only 6 PNG (3 darkfield + 3 fluorescence), not 263",
    "fus-rat-neuroinflammation-pereira-2025": "NIfTI is (1,112,128,825) = 2D power-Doppler movie; prepare.read_volume would slice the wrong axis and yield space-time maps, not frames (reinstate after reading the last axis as time)",
    "empiar-11830-chlamydomonas-cryoet": "Zenodo record holds only an annotation xlsx; pixels are 32 TB of .mrc on EMPIAR / CZ cryoET portal",
    "large-aperture-spine-volumetric-us-dryad-2025": "Dryad: resolve_doi unsupported and downloads need an OAuth bearer token (401/403); likely .mat beamformed volumes mixed with phantoms",
    "cmut-rca-3d-power-doppler-fingertip-dryad": "Dryad download blocked without a token (401/403); beamformed volumes of a single volunteer, format unverified (likely .mat)",
    "brvo-bevacizumab-fa-oct-mferg-dryad-2016": "Dryad download blocked without a token (401/403); three RAR archives of unknown contents (one is mfERG waveforms)"
  },
  "route": {
    "spc-darkfield-plankton-training-images-kenitz-2022": {
      "method": "direct",
      "ref": "https://library.ucsd.edu/dc/object/bb3387869t/_1_1.zip"
    }
  },
  "labels": {
    "arabidopsis-ruthenium-red-darkfield-rrquant": {
      "body_part": {
        "mode": "hard",
        "values": [
          "hypocotyl"
        ]
      },
      "stain": {
        "mode": "masked",
        "values": []
      }
    },
    "nuinsseg": {
      "submodality": {
        "mode": "candidates",
        "values": [
          "brightfield microscopy",
          "whole slide imaging"
        ]
      }
    },
    "mouse-rvo-ffa-slo-oct-sichuan-2024": {
      "submodality": {
        "mode": "candidates",
        "values": [
          "fluorescein angiography",
          "fundus photography",
          "near-infrared imaging",
          "optical coherence tomography"
        ]
      }
    },
    "gut-on-chip-caco2-ht29mtx-zenodo-2025": {
      "submodality": {
        "mode": "candidates",
        "values": [
          "brightfield microscopy",
          "confocal microscopy"
        ]
      },
      "body_part": {
        "mode": "candidates",
        "values": [
          "cells",
          "colon",
          "gastrointestinal tract"
        ]
      },
      "stain": {
        "mode": "candidates",
        "values": [
          "dapi",
          "immunofluorescence",
          "mucin-specific stain",
          "phalloidin",
          "unstained"
        ]
      }
    }
  },
  "exclude_paths": {
    "murine-tumor-bmode-us-uj": "(^|/)(micro-CT|Ultrasound)/|(^|/)(training_dataset|ultrasound_datasets_varia|segmentation codes)\\.zip$",
    "nuinsseg": "(^|/)(distance maps|nuclei border|weighted_maps(_erode)?|label masks( modify)?|mask binary[^/]*|overlay_save_path|stacked mask|vague areas|Imagj_zips)/"
  }
}
```

Evidence and confidence per fix:

| id | field | evidence | confidence |
|---|---|---|---|
| arabidopsis-ruthenium-red-darkfield-rrquant | stain masked | Registry hard `unstained` comes from bio_waila.data.select ST2 (all darkfield -> unstained). The record is ruthenium-red stained, and file names split `--Stained--` / `--NonStained--` (about half each). Ruthenium red is not a label-space class, so masked; a `non ?stained` -> unstained rule (section 4) fills the NonStained half. | high |
| arabidopsis-ruthenium-red-darkfield-rrquant | body_part hard hypocotyl | Images are whole dark-grown hypocotyls (title, description); `cells` is the wrong scale; `hypocotyl` is a class (1 train dataset). | medium-high |
| nuinsseg | submodality candidates [brightfield, whole slide imaging] | 665 patches of 512x512 px at 40x; MMBU public H&E splits 139 brightfield / 80 WSI. | medium-low |
| nuinsseg | exclude_paths | `tissue images/` = 665 real patches; `nuclei border/disk1-10` (6,650 png), `distance maps` (665), `weighted_maps*` (1,330) pass `kind()` as images (8,645 of 9,310). Regex checked against all 48,399 member names: exactly 665 media files remain. | high |
| mouse-rvo-ffa-slo-oct-sichuan-2024 | submodality + fundus photography | 117 `Mcolor` (multicolor SLO) images fit none of FA / NIR / OCT; 186 `IR` images map to NIR but have no rule. FFA (88) and OCT (74) already resolve from the file name. | medium |
| gut-on-chip-caco2-ht29mtx-zenodo-2025 | stain candidates | Registry hard `mucin-specific stain` is true only for the 21 `Mucin staining` images; 95 brightfield are unstained and 73 confocal are DAPI / F-actin / Muc5AC / VIL1 / ZO1. DAPI resolves today (12); other tokens need rules. | high that hard is wrong; medium on the candidate set |
| gut-on-chip-caco2-ht29mtx-zenodo-2025 | submodality candidates | Folder names `Brightfield microscopy/` (weak hit) and `Confocal microscopy/` (strong); masked can only use strong hits, candidates uses both. | high |
| gut-on-chip-caco2-ht29mtx-zenodo-2025 | body_part + cells | Cell monolayers on chip; MMBU labels cell cultures `cells` (53 train datasets). | medium-low |
| murine-tumor-bmode-us-uj | exclude_paths | 25-file Dataverse record; `dataset_segmentation.zip` (666 MB) = 3,442 B-mode PNG + 1,551 masks (matches registry 3,442). Excluded: `micro-CT/4T1tumor_*.zip` (4.7 GB micro-CT tif/bmp, would be labelled ultrasound), `Ultrasound/*.zip` (mp4 incl. `z_maska` mask overlays and `Power_Doppler`), `training_dataset.zip` and `ultrasound_datasets_varia.zip` (mp4 sources of the same frames). Verified on the listing: leaves the one zip + readme/xlsx. | high |
| spc-darkfield-plankton-training-images-kenitz-2022 | route direct `_1_1.zip` | DOI -> UCSD page is unsupported by `resolve_doi`. Two components exist (`_1_1.zip` 322 MB protists / micro-SPC, `_2_1.zip` 281 MB copepods / mini-SPC), both direct HTTP 200 downloads. The `/download` suffix would be named `download` (kind other), so use the bare `_1_1.zip`. Contents could not be listed (no Range support). | medium |

Registry facts that are fine: chlamydomonas-axoneme (all fields), rat-retina FA (all fields), murine (modality, submodality, specimen, domain), nuinsseg (modality, stain, specimen candidates resolve per sample from the `human`/`mouse` path token), mouse-rvo (modality, body_part, specimen, domain), spc labels.

## 4. Matcher additions and per-sample rules (checked with `text_hits` / `match_value`)

Wrong or missing, all inside the registry candidate sets:
- stain: `\bnon ?stained\b` -> unstained (strong) for arabidopsis (`--NonStained--`); `f-?actin` -> phalloidin; `muc5ac|vil1|villin|zo1` -> immunofluorescence; `alcian|\bpas\b|mucin staining` -> mucin-specific stain (the last three for gut-on-chip, today only `DAPI/` resolves).
- submodality: `\bir\b` -> near-infrared imaging and `m ?color|multi ?colou?r` -> fundus photography (weak) for mouse-rvo; currently only `FFA` and `OCT` fire.
- bio_waila.data.select ST2: `darkfield -> unstained` is applied even when the registry lists a stain (arabidopsis). It should apply only when the registry stain is empty.

Works as is: `human oesophagus/...` -> esophagus (strong), `mouse muscle_tibia/` -> muscle, `human peritoneum/` -> abdomen, human/mouse specimen from the path token (nuinsseg); `Brightfield microscopy/` -> light microscopy (strong) and `Confocal microscopy/` -> confocal microscopy (strong) (gut-on-chip). Many nuinsseg organs (tonsil, salivary gland, cardia, pylorus, jejunum, thymus, femur, ...) map only to the weak `tissue` class and stay masked.

## 5. Pipeline bugs and problems found

1. `resolve_doi` has no Dryad, UCSD Library or BioStudies/TCIA handler (UCSD and 3 Dryad here). Dryad API listing is open; only downloads need a token.
2. A figshare file id in a registry ref (ctc `13822151`) fails in `list_figshare` ("not an article or collection"). Other registry entries built from `ndownloader.figshare.com/files/<id>` URLs may have the same problem.
3. `read_volume` on 4D NIfTI (fUS, fMRI-like): see the fus drop. `a.shape[-1] < a.shape[0]` is false for (1,112,128,825).
4. `read_video` / multipage tif give 8 / 4 frames per file, so single-video (rat retina, 8 images) and few-movie datasets (chlamydomonas about 148) fall far below plan counts; the download of 8 GB for 148 frames is poor value. `sample_zip_remote` is used only when a zip exceeds the budget, so the 8.08 GB set is fully downloaded.
5. Derived-image folders (NuInsSeg weights/distance maps/borders, 85% of its images) pass `kind()`/`is_mask`; only per-dataset `exclude_paths` fixes this. Mask-overlay videos named `z_maska` (Polish) are not caught by `is_mask`.
6. Several registry `selected_for` entries list a field whose label is masked (nuinsseg body_part:esophagus, fus color doppler); harmless, but the dataset does not deliver that class except through per-sample rules.

## 6. Disclosure: anything fetched beyond metadata

- fUS shape check: my header peek used `RemoteZip.open()`, which transfers the whole member in one range request, so about 88.6 MB of one nii.gz member (`rawdata/sub-r8b1497/...task-stim...nii.gz`) was received in memory to read its 348-byte header; nothing was written to disk.
- UCSD library: I read up to 60 KB of the object HTML page (and two 404/HTML pages) while looking for file links, and sent `Range: bytes=0-1` GETs (stream, closed immediately) to the `_1_1.zip` / `_2_1.zip` URLs; the host ignored Range and started a 200 response, so at most a socket buffer of zip bytes arrived before closing (not saved).
- Ranged central-directory reads: 65 KB to 6.1 MB per zip (the NuInsSeg zip has 48k members). Dryad `Range: bytes=0-99` probes were answered with 401/403 and no data. AVI header: 4 KB range of the figshare file. No Kaggle calls.
- Temp files under `audit_tmp_E/` only; deleted at the end (verified).
