# Pass-2 rare-class discovery, batch A (2026-10-06)

Classes: specular microscopy, fluorescein angiography (FA), colonoscopy, gastroscopy, OCTA.
Agent tag: `waila-rare-A`. TOMLs are in `data_discovery/datasets/{fundus,endo-gi}/`; per-slice notes in each `_crossrefs.md` / `_progress.md`.
Method: Zenodo/figshare/DataCite/Dryad/Harvard Dataverse/OSF/ScienceDB/HF/Kaggle/GitHub APIs, Europe PMC/OpenAlex for dataset papers, 12 WebSearch calls (budget 30). Metadata only, no data downloaded; zips/rars were never opened, so several image counts are unknown.

Bottom line: the registry was already near-saturated for colonoscopy (all 40 sets of the cloneiq GI-endoscopy awesome list are in), so almost everything found is a small supplement-style set. Specular microscopy and gastroscopy (beyond one 758-image set) have no good permissive supply on open platforms.

## New entries (9 TOMLs, all tier A)

### 1. Specular microscopy: none
No CC-licensed specular microscopy image corpus exists on the platforms searched (see rejections).

### 2. Fluorescein angiography (slice `fundus`)
| id | images | license | route | notes |
|---|---|---|---|---|
| `mouse-rvo-ffa-slo-oct-sichuan-2024` | 465 (88 FFA, 186 IR SLO, 117 multicolor SLO, 74 OCT), 1.46 GB | CC BY 4.0 (verified, figshare API) | figshare 25346203, per-file download | mouse RVO model, Spectralis; best new FA supply; verified |
| `brvo-bevacizumab-fa-oct-mferg-dryad-2016` | unknown (FA rar 49.3 MB, OCT rar 6.4 MB, mfERG rar 5.7 MB; 27 BRVO eyes) | CC0 (Dryad API) | Dryad 10.5061/dryad.3cp54 | human clinical FA; rar contents not enumerated |
| `rat-retina-fluorescein-angiography-video-plos-2014` | 1 video (15.7 MB AVI) | CC BY 4.0 | figshare 1227889 | marginal, a few frames only |
Existing and eligible (note added, not duplicated): `cti-tongren-fa-icga-us` = figshare 28255265, single 0.82 GB zip, CC BY 4.0, 750 subjects (FA+ICGA+US), direct figshare download. This is the main human FA source.

### 3. Colonoscopy (slice `endo-gi`)
| id | size | license | route | notes |
|---|---|---|---|---|
| `endogaze-colonoscopy-eyetracking` | 467 GB, 319 files (counts of videos/frames unknown) | CC BY 4.0 (from DataCite; landing page not readable) | Science Data Bank DOI 10.57760/sciencedb.09204 | eye-tracking + colonoscopy withdrawal videos (with/without CADe overlay), 80 real-time segments; access via ScienceDB, probably needs a free login (unverified); verification = partial |
No other new colonoscopy set qualifies; see rejections.

### 4. Gastroscopy (slice `endo-gi`)
| id | images | license | route | notes |
|---|---|---|---|---|
| `gastric-polyp-ssd-gpnet-zhejiang-2019` | 758 jpg + 758 VOC xml (404 plain + 354 "_tb" variants; paper says 758 after augmentation) | CC BY 4.0 per README (no LICENSE file) | github.com/jiquan/Dataset-acess-for-PLOS-ONE (~28 MB) | gastric polyp boxes; verified via tree listing |
Only one new gastroscopy set found. Gastroscopy remains thin; the existing registry sets (Kvasir v1/v2, HyperKvasir, GastroVision, GastroHUN, BraGED, GIM-ENDO, ...) carry the load.

### 5. OCTA (slice `fundus`)
| id | images | license | route | notes |
|---|---|---|---|---|
| `octava-skin-octa-vivosight` | 24 en face tif (8 volunteers x 3), 30.6 MB | CC BY 4.0 (Zenodo API) | Zenodo 5564643 | skin (dorsal hand) OCTA, VivoSight Dx; verified |
| `octa-choriocapillaris-imh-seoul-2024` | unknown (27 IMH eyes), 28.7 MB zip | CC BY 4.0 | figshare 26956698 (PLOS ONE 0308292 S-file) | retinal en face choriocapillaris, Topcon Triton; zip not opened |
| `octa-choriocapillaris-ierm-seoul-2024` | unknown (28 iERM eyes), 10.2 MB zip | CC BY 4.0 | figshare 26193154 (PLOS ONE 0306735 S-file) | same group/device; zip not opened |
| `octa-multidevice-glaucoma-reproducibility-angers-2018` | unknown, 75.8 MB zip | CC BY 4.0 | figshare 6281525 (= PLOS S1 on figshare 6768974) | multi-vendor retinal OCTA, healthy + glaucoma |
All four are tiny; the OCTA registry already holds 27 entries (OCTA-500, ROSE, OCTAGON, FAZID, SOUL, ...).

## Rejected / not added, with reason
Specular microscopy: CLEAR-EC (already in registry, restricted license); Freiburg SREP-18-33533B (already in registry, research-only, tier C); Alizarine (Padova site unreachable, research terms) and Rotterdam ROD-REP (404); Vigueras-Guillen / Sierra datasets ("on reasonable request"; Zenodo 7378507 holds one validation image, "other-open"); Auckland/NZ eye bank figshare 5701087 and 31360192 (tables only); OSF 75kmu (CSV + code, no license); GitHub adriankucharski GAN repo (Alizarine/Gavet/Rotterdam copies, non-commercial); svdeepak99 (MIT code, ~240 tif of a third-party "Yoon 2019" set of unclear origin and license); sgrzegorz (482 png, no license); SciELO figshare supplements (figure panels only); TVST 2024 donor-cornea paper and Bioengineering 2024 (no data released).
FA: FIREFLY (request form); AngioReport (BJO 2025, 55k FFA/ICGA; hosting/license not stated); FFA-IR and HYAMD (PhysioNet credentialed); JRC-Multi-Modal (Zenodo, request); FLoRI21 and RECOVERY-FA19 (already present, IEEE DataPort, not allowed route); multi-lesion FFA (428 + 53 UWF, no public host); Karger/T&F case-report figshare supplements (2-10 figure tifs each).
Colonoscopy: PIBAdb (31,946 polyp images; biobank request); OSF r64qv synthetic colonoscopy (~20k synthetic images, no license set); figshare 32251902 CVC-ClinicDB (third-party mirror, mislabeled CC BY over research-only data) and Kaggle CC0/MIT mirrors of CVC-ClinicDB/Kvasir-SEG (same problem); figshare 12168684 (4 CADe demo AVIs); ScienceDB polyp-segmentation multi-dataset (CC BY-NC); Dataverse Colonoscopy-Coordinates (sensor poses); Cambridge hyperspectral colonoscopy (processed figures only); Kaggle re-uploads without descriptions (`shubhranshu331/colonoscopy-images`, `kholanaseem/colonoscopy-data`, `klaudiusandhika/new-combined-polyp-dataset`, `tonhuynhchi/polyp-depth`).
Gastroscopy: Zenodo 18962684 (3,313 WLE/NBI images with masks, CC BY, but files restricted); RARE25/26 Barrett's (CC BY-NC-SA, already present); Cambridge snapshot multispectral Barrett's (spectra tables); GPolypOD (CC BY-NC-SA, already present as `gastroenteroscopic-polyp-od-xmut`); Kaggle `chopinforest/esophageal-endoscopy-images` and small HF sets (no license); Fan1fei Gastroscope-tube-defect (instrument defects, not patient images).
OCTA: ROAD (permission by e-mail); MEMO (Google Drive, no license); HF syn-cerebral-octa-seg (no license); DERMA-OCTA and OCTA-Mosaicking (CC BY-NC, present); 3D microvascular labels figshare 25715604 (CC BY-NC-SA); Kaggle OCTA re-uploads without provenance (`octa200`, `octa-seg-data`, `half-octa-cleaned`).

## Leads not processed
- figshare 16911370 (Frontiers DataSheet zip, 66.7 MB, choroidal osteoma SS-OCTA; contents unknown).
- figshare 1161163 and 703467 (PLOS mouse retinal neovascularisation / EAU movies and tifs, CC BY, tiny; modality of tifs unverified).
- EndoGaze file list could only be read after a free ScienceDB login (not done per brief).
- Galar capsule video (figshare+ 25304616) is outside these classes.

## Notes for training use
- Supplement-style sets (figshare PLOS S-files, Dryad rars) need a local enumeration step before use; counts above marked "unknown" are exactly that.
- Animal FA (mouse, rat) is only useful as submodality exemplars, not as human FA training data.
- License caveat for `gastric-polyp-ssd-gpnet-zhejiang-2019`: CC BY stated only in README; the PLOS ONE paper is CC BY.
