# Rare-submodality discovery B (2026-10-07, agent tag `waila-rare-B`)

Classes: color doppler, 3d ultrasound, differential fluorescent microscopy, TIRF, darkfield.
27 new TOMLs written (validate: 0 errors in `ultrasound`, `fluor`, `lm-cellbio`). Searches: 8 WebSearch calls; the rest used DataCite, Zenodo, Dryad, figshare, BioStudies (BioImage Archive), Kaggle, HF and Europe PMC APIs. No data downloaded. All licenses are tier A; `license.verified = true` means the license field was read from the record's API (BioStudies, Zenodo, Dryad, figshare, CaltechDATA or DataCite).

Honest summary: TIRF and darkfield have usable new sources. Color Doppler and differential fluorescence do not have a faithful open dataset; what exists are proxies (see caveats). Most 3D-US additions are raw/ULM-type volumes, not clinical B-mode volumes (the registry already holds most open clinical 3D-US sets).

## 1. Color Doppler (slice `ultrasound`)
New ids (both are power Doppler, a colour-flow-like map, not B-mode with colour overlay):
| id | images | license | route |
|---|---|---|---|
| `cmut-rca-3d-power-doppler-fingertip-dryad` | unknown (3.1 GB zip) | CC0 | Dryad 10.5061/dryad.47d7wm3qg (browser download; API file fetch needs token) |
| `fus-rat-neuroinflammation-pereira-2025` | unknown (30 GB raw fUS power Doppler, 48 rats) | CC-BY | Zenodo 15194839 |

Existing registry color-Doppler entries (not new): `carotid-us-color-doppler-kaggle` (Kaggle, Apache-2.0, 14.7 MB, 18 patients, paired US + colour Doppler; the best true colour-Doppler image set found), `echoxflow` (NC), `color-doppler-twinkling-biopsy-markers` (IEEE DataPort), CREATIS `duplex-simulated-color-doppler` (no licence in Girder metadata). Note added to `ultrasound/_crossrefs.md`.

Rejected / not usable:
- Zenodo/Dryad/figshare/SciELO "colour Doppler" records (about 30 hits): tables, xlsx or article figures only, no image set (e.g. Zenodo 18523979, 10452106, Dryad 612jm649r).
- Mendeley / IEEE DataPort sets (excluded routes), e.g. "Multi-modal ultrasound for ovarian tumors v2" (already registered).
- MIMIC-IV-ECHO (credentialed), clinical MR-colour-Doppler echo cohorts (Cedars, VABC-UNet 315 images): not public.
- HF `Taylor658/Doppler-Ultrasound-Level-1`: synthetic text. HF `NaimuL0/Color_Doppler_ECG_Text_Reports`: text.
- ULM / sULM raw IQ-RF archives (Zenodo 17393921 50 GB, 11395562 35 GB, 6811910): no rendered images.
- Dryad `nk98sf891` (MUSIC epidural spinal-cord US, CC0, 8.9 MB): B-mode + Doppler-derived monitoring, but contents could not be verified (Dryad requires a token for files); worth a manual look.
Count of faithful open colour-Doppler sets found: 1 existing + 0 new. This class cannot reach 5 with open permissive data; recommend synthesising colour-Doppler-like frames only if needed, or rendering power-Doppler frames from the fUS archive.

## 2. 3D ultrasound (slice `ultrasound`)
| id | volumes/images | license | route |
|---|---|---|---|
| `cmut-rca-3d-power-doppler-fingertip-dryad` | unknown | CC0 | Dryad (see above) |
| `large-aperture-spine-volumetric-us-dryad-2025` | unknown (34 GB; probably raw RF; 7 volunteers + phantoms) | CC0 | Dryad 10.5061/dryad.9zw3r22sx |
| `3d-ulm-nhp-brain-xing-2025` | unknown (FRDR page unverifiable) | CC-BY (DataCite) | FRDR 10.20383/103.01508 |
| `3d-fulm-mouse-pericyte-zenodo-2025` | 35 mice, TIFF projections + .mat (0.98 GB) | CC-BY | Zenodo 15280877 |
| `takoyaki-3d-us-caltech-2025` | unknown (5.76 GB zip) | CC-BY | CaltechDATA 10.22002/knnyt-x4762 |
Rejected: UCL fetal-face 3D US shape models (meshes, not pixels); Harvard Dataverse 3D numerical breast phantoms (simulated sound-speed volumes); prostate MRI+US volumes Zenodo 609049 and TUS-REC 2024/25 (CC-BY-NC-SA); KU Leuven 3DUS muscle sets (NC); HF `Autzoko/3D_Ultrasound_Slices` and `hbcfx/Liver-3DUS-CT-registration` (no license); Zenodo 10549080 4D embryonic brain atlas (NC, already registered). Already in the registry and therefore not re-added: neonatal 3D US neuroimages, TUS-REC forearm (7740733), ABUS sets, CETUS, MITEA, CLUST, IVUS-EM, ReMIND2Reg, etc.
Caveat: only `3d-fulm-mouse-pericyte-zenodo-2025` has ready TIFF images; others need volume handling and are mostly preclinical or raw.

## 3. Differential fluorescent microscopy (slice `fluor`)
What MMBU means (from `docs/data/label_design.md`): 10 public-split items, 100% human / gastrointestinal tract / biology, never a stain. That matches the Caco-2 AO-PI / DiOC6-PI dual-fluorescence virtual-staining set already registered as `virtual-staining-caco2-viability` (lm-cellbio). No second open dataset of that technique was found. Added subject-matched proxies (human colon cell lines / intestinal organoids, multi-channel fluorescence), tagged `differential fluorescent microscopy`:
| id | images | license | route |
|---|---|---|---|
| `biad3207-hct116-hidro-two-colour-fish` | 22,179 TIFF (DAPI + 2 FISH colours, HCT116) | CC-BY | BioImage Archive S-BIAD3207 |
| `biad1043-hitips-hct116-high-throughput-imaging` | 67,828 TIFF (channel-split, HCT116 + HBEC3-KT) | CC-BY | S-BIAD1043 |
| `biad2215-crc-3d-collagen-calcein-screen` | 7,392 TIFF (BF + Calcein AM, human CRC colonies) | CC-BY | S-BIAD2215 |
| `biad3648-intestinal-organoid-photopatterning` | 1,017 files (ND2 confocal + JPG brightfield, human intestinal organoids) | CC0 | S-BIAD3648 |
| `gut-on-chip-caco2-ht29mtx-zenodo-2025` | 189 (brightfield + confocal, Caco-2/HT29-MTX) | CC-BY | Zenodo 14745113 |
Caveat: these share subject (human GI cell lines) and multi-channel fluorescence, not the live/dead dual-stain technique. Use with a lower weight, or only for body-part/specimen/stain-free targets. Rejected: S-BIAD1422 (2 files), S-BIAD955 (mostly U2OS, 670 MB files), Zenodo U2OS HCS sets (Aitslab, not GI). Largest existing Caco-2 fluorescence archive: IDR idr0094 / S-BIAD29 (CC0, 126,816 TIFF), registered in lm-cellbio as a phase-contrast entry; its fluorescence wells could be mapped to this class (note in `fluor/_crossrefs.md`).

## 4. TIRF (slice `fluor`)
Eleven new BioImage Archive deposits (the registry already had 13, e.g. biad2493, biad914, biad3061, bbbc052). Counts below are files (TIFF/CZI stacks), not frames.
| id | files | license |
|---|---|---|
| `biad597-her2-upaint-tirf-spt` | 2,400 TIFF stacks (HeLa, Cy3B) | CC-BY |
| `biad1733-arabidopsis-kinase-tirfm-smlm` | 3,715 (CZI + TIFF, plant; mixed TIRFM/confocal) | CC0 |
| `biad1347-met-receptor-smfret-dstorm-tirf` | 1,571 (about 1 GB each) | CC0 |
| `biad864-ht-paint-calr-kdel-tirf` | 11 stacks (U-2 OS ER, JF635) | CC-BY |
| `biad3361-rho-tag-smlm-tirf` | 6 TIFF + 42 HDF5 | CC-BY |
| `biad3793-adar1-p150-smtirf` | 46 (in vitro Cy3/Cy5 smTIRF) | CC0 |
| `biad2337-aspp-pp1-smtirf-celegans` | 2,226 (mixed) | CC0 |
| `biad1023-ire1-clustering-tirf` | 22,007 (TIRF + confocal, mixed) | CC0 |
| `biad1201-talin-tirf-frap-focal-adhesions` | 251 TIFF | CC-BY |
| `biad1683-arcos-px-tirf-podosomes` | 243 TIFF | CC0 |
| `biad2942-pmhc-dna-paint-tirf` | 4 image archives | CC0 |
Route: all via BioImage Archive HTTP/FTP (`ftp.ebi.ac.uk/biostudies/fire/...` or `/pub/databases/biostudies/...`). Caveats: mostly raw single-molecule movies (sparse spots), several are mixed-modality (TIRF identified from study text; flagged `verification=partial`). Best cell-level TIRF exemplars: biad1201, biad1683, biad597, biad1023. Rejected: S-BIAD415 and S-BSST712 (no licence field), S-BIAD3210 (CC BY-NC), S-BIAD1472/1473 (TIRF only incidental), S-BIAD3562 (cryo-FLIM).

## 5. Darkfield (slice `lm-cellbio`)
| id | images | license | route |
|---|---|---|---|
| `ctc-darkfield-unstained-blood-ciurte-2018` | 263 darkfield blood images (9 MB) | CC-BY | figshare 7463405 (PLOS ONE S1) |
| `arabidopsis-ruthenium-red-darkfield-rrquant` | unknown count, 5.7 GB, masks + quantification | CC-BY | Zenodo 15303477 |
| `chlamydomonas-axoneme-darkfield-geyer-2024` | movies, 8 GB (defocused darkfield) | CC-BY | Zenodo 13909287 |
| `fibroblast-aunp-cytoviva-darkfield-wiemann-2016` | 1 video, 20 MB | CC-BY | Zenodo 167072 |
| `spc-darkfield-plankton-training-images-kenitz-2022` | unknown (UCSD page blocked; verification=unverified) | CC-BY (DataCite) | UCSD DOI 10.6075/j00865gt |
Only the CTC set is a clinical-style darkfield cell dataset; the others are plant/organelle/plankton darkfield. Existing registry darkfield entries (not re-added): quadrant-darkfield-qdf-zangle, schistoscope-bf-df-cote-divoire-2024, ntdscope-2025, cell-microplastics-darkfield-hsi-2026, darkfield-ir-slo-choroidal-nevi.
Rejected: Kaggle `longnguyen2306/bacteria-detection-with-darkfield-microscopy` (366 images + masks; Kaggle page not retrievable, GitHub mirror PerceptiLabs/bacteria has no licence, tier D; worth manual confirmation because it is the closest clinical darkfield set), mezcal darkfield (non-biomedical), sub-30-image nanoparticle sets (figshare 9200315, 22094498, 8385203), video-only gold-nanoparticle darkfield (non-biological), sidestream-darkfield sublingual microcirculation sets (a different technique).

## Not done / follow-ups
- File contents were not opened (no downloads), so image counts for Dryad/Zenodo archives are unknown; set `num_images` after the first pull.
- Dryad file downloads need a token or the browser; handle manually.
- Pass-2 plan: TIRF can reach its floor from biad597 + biad1201 + biad1683 + biad1023 (cell-level) plus single-molecule sets; darkfield will stay thin (about 270 clinical images); colour Doppler and differential fluorescence need proxies or synthetic rendering.
