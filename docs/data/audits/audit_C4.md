# Pass-2 labelling audit, part C4 (31 datasets from `audit_todo_C4.json`)

Machine-readable records: `audit_C4.jsonl` (same schema as `audit_A.jsonl`; a per-field `recommended` key carries the proposed registry change). Metadata only: file listings, zip central directories via remotezip (range requests), first KB of csv/xlsx, HF datasets-server `/info` + `/first-rows`, Zenodo/figshare/Dataverse/Kaggle APIs, README text. No images or archives were downloaded; the temp dir was deleted. Label priors quoted below come from `plan/eval_public.parquet` (mmbu-public) and are counts of public images.

## 1. Access status (31 datasets)

| status | n | datasets |
|---|---|---|
| ok as selected (listable, pixels present; caveats in `problems`) | 13 | dsad, mll23-peripheral-blood, totalsegmentator-v2, mm-mcd-myeloma-2026, laparoscopic-gastrectomy-annotated-images (was a drop in audit A: the truncated figshare id; real record has 6,832 jpg), bass-2017-axonal-synapses-2p, flare22, bbbc007, hepg2-dic-unstained-instance, totalsegmentator-mri, idrid, isic-2016, lacuna-malaria-2024 |
| ok, rar only (not listable / not range-samplable) | 3 | cmha-intracranial-aneurysm-cta, reta-benchmark, aattct-ids |
| ok but tiny (few usable images) | 2 | ankyrin-g-axon-initial-segment-cryoet-2026 (48 tif), cusi-mouse-4d-brain-hemodynamics (4 volumes) |
| selected source is wrong or blocked; a working source exists (route fix) | 8 | medfmc-chestdr (figshare collection of 5 tasks), drac-2022 (HF multi-config), fetal-planes-db (HF mirror is brain planes only), glenda (HF script repo), crc-ihc-cish-he-epithelium (zenodo preview mirror), rsna-pneumonia-2018 (manual, public S3 zip exists), fluorescent-neuronal-cells-v2 and -v1 (EPrints host) |
| blocked by a pipeline bug (needs code, no route) | 1 | c-nmc-2019 (HF `Image(decode=False)`) |
| unusable format | 1 | empiar-10164-hiv-vlp-cryoet (one `.mrc`, unsupported) |
| manual (Mendeley 403, verified) | 2 | afio-hr-dr-papilledema, asraf-covid-pneumonia-normal-pa |
| Kaggle | 1 | heel-bone-xray-kirkuk: per-file listing works but is rate-limited (429); 1,000 of ~3,956 files listed (all jpg, `heelspur` folder seen); counts unverified |

## 2. DROP list

| id | reason |
|---|---|
| empiar-10164-hiv-vlp-cryoet | Zenodo record = one reconstructed tomogram (`01_10.00Apx.mrc`, 471 MB) + `.star`; not the 865 GB EMPIAR tilt series. `.mrc` is in no extension set (`kind()` = other, no reader in src/bio_waila/data/prepare.py) so download ends in `error` with 0 media; even with a reader it is 1 volume (<= 4 slices) vs 2,580 planned |
| fluorescent-neuronal-cells-v1 | The 283 images are the yellow (CTb) collection of v2 (v2 README: yellow annotations start from v1). Keep v2 only; v1 is also on an unsupported host (EPrints) |
| (keep, do NOT drop) laparoscopic-gastrectomy-annotated-images | audit A dropped it for a truncated figshare id; with id 30058882 it has one 1.67 GB zip with 6,832 jpg frames (18 videos, Roboflow export with 13 augmentation duplicates) |
| (low value, your call) cusi-mouse-4d-brain-hemodynamics | 4 static 3D volumes -> <= 16 images, not 240 (the `3d+t` tag is wrong) |
| (low value, your call) ankyrin-g-axon-initial-segment-cryoet-2026 | <= 48 usable tifs (29 tomograms x 2 pixel sizes = near duplicates) vs 3,000 planned; the 16 `.mrc` tomograms are ignored |
| (overlap, choose one) totalsegmentator-v2 vs totalsegmentator-v3 | v2 (1,228 CT) is a strict subset of v3 CT (1,939 cases); v2-lite has the cleaner layout (flat `Images/` + `meta.csv`) |

Overlap flags (not drops): rsna-pneumonia-2018 images are NIH ChestX-ray14 images (mapping json `img_id` = `00012773_000.png`) already covered by nih-chestxray14 / nih-chestxray14-bbox; isic-2016 images are inside isic-2017 (same split group); lacuna-malaria-2024 `test` is a byte-identical copy of `validation` (275 rows, 313,614,856 bytes each); DRAC zips A and C are subsets of zip B.

Plan sizes that exceed what exists (cap `plan_images`): totalsegmentator-mri 616 (HF v2.0.0 mirror, one middle slice per volume), idrid 81 (HF mirror = segmentation subset; 516 planned), flare22 ~200 (50 volumes x 4 slices), mm-mcd ~2,840 jpg (text claims 20,000), ankyrin 48, cusi 16, fetal-planes-db 2,949 (HF) or 12,400 (Zenodo route).

## 3. LABEL FIXES (same format as `plan/registry_fixes.json`)

```json
{
 "drop": {
  "empiar-10164-hiv-vlp-cryoet": "one .mrc tomogram on Zenodo (not the EMPIAR tilt series); .mrc unsupported by files.py/prepare.py",
  "fluorescent-neuronal-cells-v1": "images are the yellow collection of fluorescent-neuronal-cells-v2 (subset); EPrints host"
 },
 "route": {
  "medfmc-chestdr": {"method": "figshare", "ref": "22302775"},
  "drac-2022": {"method": "zenodo", "ref": "10280358"},
  "fetal-planes-db": {"method": "zenodo", "ref": "3904280"},
  "glenda": {"method": "direct", "ref": "http://ftp.itec.aau.at/datasets/GLENDA/downloads/GLENDA_v1.5_no_pathology.zip"},
  "crc-ihc-cish-he-epithelium": {"method": "dataverse", "ref": "https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/DIGQGQ"},
  "rsna-pneumonia-2018": {"method": "direct", "ref": "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"},
  "fluorescent-neuronal-cells-v2": {"method": "direct", "ref": "https://amsacta.unibo.it/7347/17/raw_data.zip"}
 },
 "exclude_paths": {
  "dsad": "(^|/)(multilabel|merged|anno_\\d+)/|(^|/)mask\\d+[^/]*$",
  "hepg2-dic-unstained-instance": "/visualization/",
  "crc-ihc-cish-he-epithelium": "SPSS_databases|\\.png$",
  "drac-2022": "A\\. Segmentation|C\\. Diabetic Retinopathy|Groundtruths"
 },
 "labels": {
  "hepg2-dic-unstained-instance": {
   "modality": {"mode": "hard", "values": ["light microscopy"]},
   "submodality": {"mode": "hard", "values": ["contrast microscopy"]},
   "stain": {"mode": "hard", "values": ["unstained"]},
   "specimen": {"mode": "candidates", "values": ["cell line", "human"]}
  },
  "fetal-planes-db": {"body_part": {"mode": "candidates", "values": ["abdomen", "brain", "cervix", "fetus", "thorax"]}},
  "flare22": {"body_part": {"mode": "hard", "values": ["abdomen"]}},
  "laparoscopic-gastrectomy-annotated-images": {"body_part": {"mode": "hard", "values": ["abdomen"]}},
  "bass-2017-axonal-synapses-2p": {"body_part": {"mode": "hard", "values": ["brain"]}},
  "glenda": {"body_part": {"mode": "candidates", "values": ["abdomen", "ovary", "pelvis", "uterus"]}},
  "medfmc-chestdr": {"body_part": {"mode": "multi", "values": ["chest", "lung"]}},
  "rsna-pneumonia-2018": {"body_part": {"mode": "multi", "values": ["chest", "lung"]}},
  "asraf-covid-pneumonia-normal-pa": {"body_part": {"mode": "multi", "values": ["chest", "lung"]}},
  "fluorescent-neuronal-cells-v2": {"stain": {"mode": "candidates", "values": ["cfos", "immunofluorescence", "orexin"]}},
  "totalsegmentator-v2": {"submodality": {"mode": "candidates", "values": ["conventional ct", "spiral ct"]}},
  "mll23-peripheral-blood": {"stain": {"mode": "candidates", "values": ["giemsa", "may-grünwald-giemsa", "wright-giemsa"]}},
  "mm-mcd-myeloma-2026": {"stain": {"mode": "candidates", "values": ["giemsa", "may-grünwald-giemsa", "wright-giemsa"]}}
 }
}
```

Evidence and confidence for each change:

| dataset | change | evidence | conf |
|---|---|---|---|
| hepg2-dic-unstained-instance | modality/submodality/stain/specimen | Zenodo zip holds only `coco_format/*/images` (520 DIC png) + `visualization` overlays; title "unstained living HepG2"; calcein-AM images are not in the record. Registry says fluorescence candidates + stain calcein am: contradicts every pixel | high |
| fetal-planes-db | body_part candidates widened (+ route) | Zenodo csv `Plane`: Fetal brain 3,092 / thorax 1,718 / abdomen 711 / femur 1,040 / Maternal cervix 1,626 / Other 4,213. match_value gives brain/thorax/abdomen/cervix but candidates mode only accepts values inside the registry set [cervix, fetus]. The HF mirror used today has 2,949 brain-plane images only | high |
| medfmc-chestdr route | figshare 6476047 is a collection of 5 MedFMC tasks (ColonPath, ChestDR, Endo, Retino, NeoJaundice); article 22302775 = ChestDR | high |
| drac-2022 route + exclude | HF repo has 3 configs (load fails); Zenodo zip B has all 1,103 images, A and C are subsets, A contains 227 mask pngs | high |
| glenda route | HF repo is a loading script; the no_pathology zip has 13,438 frames of 20+ surgeries (868 MB, range ok) | high |
| crc route + exclude | zenodo 14800740 = 2 preview jpg; dataverse.no has 19 zips; each core ships a `.jpg` + same-stem `.png` epithelium mask; `archive_rank` puts `SPSS_databases.zip` (31 KB, GOOD token `data`) first, and the sampler would stop after it with zero images | high |
| rsna route | S3 zip HEAD 200, 30,000 dcm, 3.98 GB; the selection row is `manual` although `fallback_urls` has it | high |
| fluorescent-neuronal-cells-v2 route + stain | EPrints page; raw_data.zip (2 GB) has all 1,874 originals under `raw_data/{green,red,yellow}/images`; green = c-FOS, red = orexin (`Orx_fourth_round`), yellow = CTb (no class, use immunofluorescence); TH/TPH not evidenced in the record, so dropped from the candidate set | medium |
| dsad exclude | 116,025 of 130,650 pngs are `maskNN.png` / `anno_k` / `merged` copies not caught by is_mask; `multilabel/` has 1,430 frames = stomach folder count and the 11 organ folders sum to exactly the published 13,195 | high (masks), medium (multilabel duplicate) |
| flare22 body_part | Zenodo title "50 Labeled Abdomen CT Scans"; eval_public conventional ct: abdomen 24 / lung 20 / thoracoabdominal 19 | high |
| laparoscopic-gastrectomy body_part | no per-image metadata; eval_public laparoscopy: abdomen 337, abdominal organs 26, stomach 1 | medium |
| bass-2017 body_part | live mouse cortex two-photon; eval_public multiphoton: brain/cerebrum | medium |
| glenda body_part | gynecologic laparoscopy; eval laparoscopy priors abdomen 337 / male GU 23 / retroperitoneum 15; rectum dropped | low |
| chest radiographs multi [chest, lung] | eval_public chest x-ray: chest+lung 271, chest 126, thorax 20, lung 10 (same recommendation as audit A) | medium |
| totalsegmentator-v2 submodality | every `study_type` is only a weak `conventional ct`; no spiral/helical evidence; eval_public: spiral ct only for 9 lung images | low |
| mll23 / mm-mcd stain | record text names no stain; MLL = May-Grünwald-Giemsa (paper knowledge), bone-marrow smear = Wright-Giemsa/Giemsa in practice. Optional, partial-label only | low |

Per-sample resolutions through `plan/overrides.csv` (format `dataset_id,column,value,field,class`; value matched lower-case). The `path:dN` index is "last four non-generic folder parts", and src/bio_waila/data/download.py extracts zips to `zip_members/<zip stem>/<member path>`, so verify the column name with `src/bio_waila/data/metalabel.py profile` before copying:

```csv
dataset_id,column,value,field,class
crc-ihc-cish-he-epithelium,path:d1,HE,stain,h&e
crc-ihc-cish-he-epithelium,path:d1,ECAD,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,VIMENTIN,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,SMA,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,Ki67,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,SMAD3,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,MACC1,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,LASP1,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,CD44,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,NAIP,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,KLF5,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,FSCN1,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,CTNND1,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,KRAS,stain,ihc
crc-ihc-cish-he-epithelium,path:d1,miR-143,stain,in situ hybridization
crc-ihc-cish-he-epithelium,path:d1,miR-145,stain,in situ hybridization
crc-ihc-cish-he-epithelium,path:d1,U6,stain,in situ hybridization
crc-ihc-cish-he-epithelium,path:d1,Scr,stain,in situ hybridization
fluorescent-neuronal-cells-v2,path:d3,green,stain,cfos
fluorescent-neuronal-cells-v2,path:d3,red,stain,orexin
fluorescent-neuronal-cells-v2,path:d3,yellow,stain,immunofluorescence
fluorescent-neuronal-cells-v2,path:d3,green,specimen,mouse
fluorescent-neuronal-cells-v2,path:d3,yellow,specimen,mouse
fetal-planes-db,csv:FETAL_PLANES_DB_data.csv:Plane,Fetal femur,body_part,fetus
```

(CRC IHC marker names are exact zip stems from the Dataverse listing: HE, ECAD, Ki67, VIMENTIN, SMA, SMAD3, MACC1, LASP1, CD44, NAIP, KLF5, FSCN1, CTNND1, KRAS, miR-143, miR-145, U6, Scr. The red collection is 484 rat / 62 mouse in `metadata_v2.xlsx`, so it stays specimen candidates [mouse, rat].)

## 4. Per-sample metadata sources that work (what actually resolves a field)

| dataset | source | field resolved | scope |
|---|---|---|---|
| fetal-planes-db (Zenodo route) | csv `Plane` joined by `Image_name` = file stem | body_part cervix/abdomen/brain/thorax per image | 12,400 images |
| totalsegmentator-v2 | `meta.csv` `study_type` joined by `image_id` = stem (`;` csv with BOM) | body_part: single-region studies (neck, spine, pelvis, abdomen, heart, thorax, chest, hip, whole body) ~380/1,228; multi-region (~800) unresolved | per scan |
| crc-ihc-cish-he-epithelium | zip-stem folder = marker | stain h&e / ihc / in situ hybridization (needs overrides) | all |
| fluorescent-neuronal-cells-v2 | colour collection folder; `metadata_v2.xlsx` `animal_type` (marvin = mouse, rat) | stain, specimen | all |
| totalsegmentator-mri | HF columns (`manufacturer`, `scanner_model`, `source`), `meta.csv` `scanning_sequence` | none of Pass-2 fields (sequence could mask EPI rows) | - |
| bbbc007 | file name `_D_` / `_F_` (and `d`/`f`, `d0`/`d1`) | stain dna stain / phalloidin | 32 images |
| cusi | file name `Color_Doppler` vs `Power_Doppler` | submodality | 4 files |
| everything else | registry only (class folders / coco categories / diagnosis columns carry no Pass-2 field) | - | - |

Test values run with `match_value` (current rules.py): `path:d1 'liver' -> body_part liver (strong)` (DSAD, a conflict with hard abdomen, harmless); `'inferior_mesenteric_artery' -> {}`; `Plane 'Fetal femur' -> {}`, `'Other' -> {}`; `Brain_plane 'Not A Brain' -> brain (strong)` (bug, below); `'ct thorax-abdomen-pelvis' -> {}`; `'ct angiography head' -> head (weak)`; `'ct angiography abdomen-pelvis' -> CT only` (audit-A fix works); `path:d1 'HE' -> h&e (weak)`, `'Ki67' -> ihc (weak)`, other markers `{}`; `'Power_Doppler' -> color doppler (strong)`; `'smudge_cell' -> cell (weak)`; `'marvin' -> {}`.

## 5. Matcher additions (rules.py / src/bio_waila/data/metalabel.py) with evidence

1. Negation: skip values that start with `not a`, `non`, `no ` (or contain `not a <organ>`). Evidence: fetal-planes-db `Brain_plane='Not A Brain'` -> body_part brain (strong); it covers 9,308 of 12,400 rows (75%), so the column would be trusted and label non-brain planes as brain.
2. Multi-region study types: return a multi value (or a class) when >= 2 strong body hits are in one value. Evidence: totalsegmentator-v2 `study_type` (thorax-abdomen-pelvis 143, neck-thorax-abdomen-pelvis 114, abdomen-pelvis 87, thorax-neck 79, thorax-abdomen 54, polytrauma 48, ...) returns nothing for ~800 of 1,228 scans. Proposed targets: thorax+abdomen(+pelvis) -> `thoracoabdominal region` (class exists, 19 eval images), abdomen+pelvis -> multi [abdomen, pelvis], polytrauma / neck+thorax+abdomen+pelvis -> `whole body`. `pathology_location` ('thorax,abdomen', 'abdomen,pelvis') has the same problem.
3. `is_mask` tokens (tested with `is_mask()`): `mask\d+` (DSAD `liver/24/mask20.png`, `anno_3/mask03.png` all return False), `anno_\d+`, `annots` (GLENDA `annots/*.png`), `groundtruths?` (DRAC `2. Groundtruths/.../1. Intraretinal.../001.png`, the folder two levels up is never inspected), `visuali[sz]ations?` (HepG2 `visualization/0022.jpg`).
4. `Power_Doppler` / `power doppler` currently maps to `color doppler` (strong) through the generic `doppler` rule; the vocabulary has no power class -> leave unmapped. (cUSi would give both files color doppler.)
5. File-name tokens are never columns (`path_columns` only uses folders; `bids:suffix` needs a hyphen). Evidence: cusi `Color_Doppler`/`Power_Doppler`, bbbc007 `_D_`/`_F_`, fnc `_g/_y/_r` + `Mar`/`RT` prefixes. Suggest a `path:name` column (stem tokens) so overrides can key on it.
6. `PATH_CODES`/stain: the marker names `HE`, `Ki67` only reach weak hits as folders (CRC); fine once overrides exist.
7. `.mrc` / `.mrcs` / `.rec` must be added to `VOLUME_EXT` and a reader (mrcfile) to src/bio_waila/data/prepare.py (EMPIAR, ankyrin).

## 6. Pipeline bugs found

1. **exclude_paths is not applied to zip members at download time**: `run_one` filters only listed archive names; `sample_zip_remote` / `sample_tar_stream` ignore it, so masks are transferred and only dropped in src/bio_waila/data/prepare.py (DSAD: 89% of a random member sample would be masks). Pass the regex into the media/meta member filters.
2. **is_mask gaps**: see 5.3. Also the rule inspects only the last two path levels.
3. **fetch_hf**: (a) `Image(decode=False)` columns yield dicts, not PIL images -> "no image column" -> parquet-only fallback -> Manual (c-nmc-2019: whole dataset lost); decode `{'bytes':...}` or `cast_column(col, Image(decode=True))`. (b) repos with several configs (MedOtter/DRAC22: segmentation, image_quality_assessment, dr_grading) make `load_dataset` raise; iterate configs or accept `repo:config`. (c) script-based repos (MFreidank/glenda) fail with datasets>=4 and the fallback finds no media. (d) the column filter `mask|label|seg|gt` misses `overlay_middle_slice` (TotalSegmentatorMR) and relies on column order.
4. **figshare collections**: `list_figshare` silently falls back to `/collections/<id>/articles` and merges all articles (medfmc-chestdr: colon histology, colonoscopy, fundus, neonatal photos under chest x-ray labels). Fail loudly or require an article id.
5. **`archive_rank` GOOD token `data` matches `SPSS_databases.zip`** (31 KB statistics file, crc-ihc-cish-he-epithelium): it sorts before the 16-90 GB image zips, is downloaded whole, and `got += want` ends the loop with no images. Tighten the token (`\bdata\b`) or rank by media count.
6. **Archive sampler stops after one archive** (`got += want` for any archive that fits, break at the first zip-range): class-per-zip datasets collapse to one class/marker (MLL23 -> myeloblast.zip only; CRC -> miR-145.zip 89 GB only, an ISH marker; fnc colour zips). Round-robin `want` over comparable archives.
7. **`manual` rows ignore `fallback_urls`** (rsna-pneumonia-2018 has a public S3 zip).
8. **Hosts**: `resolve_doi` has no EPrints rule (amsacta.unibo.it); `list_direct` uses HEAD, which hangs on amsacta (60 s read timeout) while a `Range: bytes=0-1` GET returns `Content-Range` with the total size; Mendeley still 403.
9. **list_kaggle**: the `datasets/view` call returned HTTP 429 repeatedly (retries exhausted) and aborts the whole listing as Manual although `datasets/list` is independent and gave 200 on a first call. Make `view` optional (only needed for totalBytes) and back off on 429 across the list pages as well (pageSize 1000 vs 200 both got throttled after a few calls).
10. **Plan sizes vs reality**: `plan_images` is not checked against listings (see section 2): totalsegmentator-mri, idrid, flare22, ankyrin, cusi, empiar, mm-mcd.
11. **Column naming**: `path:dN` = last four non-generic folder parts, so the index depends on depth and on the extra `zip_members/<stem>` level (CRC `HE/HE/Cancer` -> d1=d2=HE, d3=Cancer; fnc raw_data `raw_data/raw_data/green` -> d3 = colour). Overrides keyed on `path:dN` need a profile run to confirm.
12. **Archive name filter `mask|label|seg`** (fetch_entries) also drops `neutrophil_segmented.zip` (MLL23, 1.75 GB, a regular cell class), because `seg` matches `segmented`. Use word boundaries.
13. **rar** (3 datasets, 17 GB+): `src/bio_waila/data/download.py` fetches whole rars (cmha patients.rar 10.7 GB first; controls never sampled); needs `unrar`/`unar` on HPG.

## 7. Per-dataset notes (short)

- **dsad**: ok; 13,195 laparoscopic frames in 11 organ folders (+ `weak_labels.csv` organ presence); masks/anno/merged copies must be excluded; `multilabel/` duplicates frames.
- **totalsegmentator-mri**: HF v2.0.0 mirror, 616 middle slices; `meta.csv` (`;`) has `scanning_sequence` (GR, SE, EP\SE, EP\RM), `study_type` = "unkown" everywhere.
- **idrid**: HF has the 81 segmentation images (54+27), masks in `mask_*` columns.
- **fetal-planes-db**: HF mirror = 2,949 fetal brain planes; Zenodo has 12,400 with `Plane`, `Brain_plane`, `Operator`, `US_Machine`, `Train`.
- **drac-2022**: B zip = all 1,103 UW-OCTA en-face pngs; A (174 + 227 masks) and C (997) are subsets; csv labels = quality/DR grade (no Pass-2 field).
- **mll23-peripheral-blood**: 18 class zips, 40k+ single-cell TIF; folder = cell class.
- **totalsegmentator-v2**: 1,228 CT, `Images.zip` 22.6 GB (range ok), `meta.csv` with `study_type`, `pathology_location`, `kvp`, `manufacturer`.
- **mm-mcd-myeloma-2026**: ~2,840 jpg in 2 zips (not 20,000), per-case folders, labelme json (cell classes), no stain text.
- **isic-2016**: HF 900+379 with mask column; overlaps isic-2017.
- **c-nmc-2019**: 10,661 single-cell crops, `decode=False` blocks the pipeline.
- **laparoscopic-gastrectomy-annotated-images**: 6,832 Roboflow frames, COCO categories = organs/instruments; consecutive frames, 18 videos.
- **lacuna-malaria-2024**: 3,022 rows incl. duplicated test/validation; objects = bbox + int categories only.
- **glenda**: script repo; use the no_pathology zip (13,438 frames); the classes zip has frames + 373 `annots` pngs.
- **cmha / reta / aattct**: rar only, nothing to list; registry labels unverified.
- **bass-2017**: 100 two-photon tif stacks in one 585 MB zip, labels folders excluded by is_mask.
- **medfmc-chestdr**: route to figshare 22302775; ChestDR.zip 4,848 png + 19-disease csv.
- **flare22**: 50 CT volumes in one 1.5 GB zip; `labels/` is a mask folder.
- **empiar-10164 / ankyrin-g**: `.mrc` unsupported; see drop list.
- **bbbc007**: 32 tif in 4 plates, channels as separate files.
- **heel-bone-xray-kirkuk**: Kaggle `datasets/list` ok page by page (200 files/page) but throttled; class folders = diagnoses, no Pass-2 metadata; registry body_part candidates [ankle, bone, foot] unchanged.
- **fluorescent-neuronal-cells-v2/v1**: see fixes; metadata_v2.xlsx maps `original_name` -> numbered pngs and carries `dataset`, `animal_type`, `marked_structure`, `brain_region`, `zoom`.
- **rsna-pneumonia-2018**: public S3 zip (30,000 dcm); NIH-derived.
- **afio / asraf**: Mendeley 403; browser download needed.
- **crc-ihc-cish-he-epithelium**: Dataverse 19 zips (470 GB) = H&E + 13 IHC + 4 ISH; same-stem `.png` masks.
- **cusi**: 4 uncompressed NIfTI-1 volumes (176x156x180 and 192x240x200 at 40 um).
- **hepg2-dic-unstained-instance**: 520 DIC images (+ 520 overlays in `visualization/`).
