from lib import *

o("microvision-fluorescence-cell-detection", src=[S("zip", "microvision.zip (zenodo 3894389, 0.25 GB; 14,003 members)", ["d2 = one VOC/YOLO training set per cell type: c127_dapi_class, cos7_nucleopore_class(+scale_10x/40x), erythroblast_dapi_class, erythroblast_dapi_glycophorinA_class, erythroblast_dapi_glycophorinA_FOXO3_class, erythroid_dapi_class, eukaryote_dapi_class, fibroblast_nucleopore_class, hek_peroxisome_class, neuroblastoma_phal_class, neuroblastoma_phal_dapi_class, mixed_col/hetero/homo", "d3/d4: JPEGImages (4,127 jpg), Annotations (xml), labels (YOLO txt), ImageSets, zips (46 tiny nested zips), results/models csv"], ["microvision/neuroblastoma_phal_class/JPEGImages/*.jpg", "retina_test_dn_n30.csv: 'cell - c127 dapi'"])],
  spec=F("top folder token: hek -> human; c127 -> mouse (C127 mouse mammary line); N1E115 neuroblastoma (Cell Image Library) -> mouse; cos7 -> African green monkey; erythroblast/erythroid/fibroblast/eukaryote unspecified", {"hek_peroxisome_class": "human", "c127_dapi_class": "mouse", "cos7_nucleopore_class": "non-human primate", "neuroblastoma_phal_class": "mouse (N1E115)"}, "medium"),
  stain=F("top folder tokens: dapi -> dapi, phal -> phalloidin, nucleopore / peroxisome / glycophorinA(+FOXO3) -> antibody immunofluorescence (Alexa Fluor 488 secondary, from the record text); registry candidates dapi/phalloidin omit immunofluorescence", {"c127_dapi_class": "dapi", "neuroblastoma_phal_class": "phalloidin", "cos7_nucleopore_class": "immunofluorescence", "hek_peroxisome_class": "immunofluorescence", "erythroblast_dapi_glycophorinA_class": "dapi (+ immunofluorescence)"}, "medium"),
  prob=["registry hard specimen=human is wrong for c127 (mouse), N1E115 (mouse) and cos7 (monkey) folders (916 of 4,127 jpg, 22%: c127 65 + cos7 131 + neuroblastoma 720)",
        "match_value on the folder names: 'c127_dapi_class' -> {} because _IDLIKE (^[a-z]{0,8}[_-]?\\d{3,}([_-]\\w+)?$) treats c127_dapi_class as an id; 'cos7' / 'cos-7' -> {} because two strong specimen rules conflict (cell line vs non-human primate); nucleopore, peroxisome, phal, hek, glycophorin are unknown tokens; 'neuroblastoma_phal_dapi_class' and 'erythroblast_..._dapi_...' give stain dapi (strong)",
        "same images re-appear in several training sets (erythroblast_dapi_* 169 each, erythroid 101 x2, hek_peroxisome 110 x2, neuroblastoma 360 x2, mixed_* 816/816/520): within-dataset dHash dedup in prepare.py is needed (present)",
        "74 retina_*.csv training lists under results/ are picked up as metadata tables (meta[:50]) and give weak junk hits (body_part 'cell' from 'cell - c127 dapi')",
        "__MACOSX / .DS_Store / .textClipping members are filtered by is_junk"],
  note="Fix: specimen candidates [human, mouse, non-human primate]; stain candidates [dapi, phalloidin, immunofluorescence]; with matcher additions the per-sample folder resolves both. body_part hard 'cells' and epifluorescence are right.")

o("biad1137-hht1-single-molecule-tracking-spombe", acc="unsupported_host_biostudies: doi.org/10.6019/S-BIAD1137 resolves to ebi.ac.uk/biostudies (resolve_doi has no BioStudies rule -> Manual); BioStudies API itself is open",
  src=[S("biostudies", "https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD1137 + file lists https://www.ebi.ac.uk/biostudies/files/S-BIAD1137/<list>.json", ["5 file lists: Histone_Live Cells.json (193 files), Histone_Fixed Cells.json (120), GFP_Live Cells.json (166), Optimization of JF646-HTL concentration.json (4), Comparison of filtered images with raw images.json (6)", "Biosample: Organism S. pombe, live vs fixed cells; Imaging method: Single molecule imaging / TIRF / fluorescence microscopy / HILO"], ["Histone_Live Cells/Hht1-HaloTag_fast imaging_(TrackIt)/15ms_Image (1).tif 131 MB", "..._MIP.tif 131 KB", ".mat tracks, .csv"])],
  prob=["no download path: unsupported host (BioStudies / BioImage Archive). Content = ~254 movie TIFF stacks (256x256, 200-1000 frames, 131 MB each, 12.6 GB total) + ~232 .mat track files + 3 csv; the only viewable 2-D images are *_MIP.tif projections and the 3-6 'filtered vs raw' frames",
        "stain masked is defensible (JF646-HaloTag ligand has no class; only the NLS-HaloTag-GFP subset would map to gfp)",
        "TIRF/HILO single-molecule movies are mostly dark frames with sparse spots (informative() filter will reject many frames)"],
  note="Needs a BioStudies handler (files/<accession>/<path> URLs work without auth) or drop. Registry labels (TIRF, yeast, cells) match the record metadata.")

o("pcle-rat-mosaicking-optica", src=[S("zip", "data.zip (figshare 32006034, 21 MB)", ["data/data1 (16 png), data/data2 (20), data/data3 (14), scan_direction.txt per folder"], ["scan_direction.txt: UP_DOWN false/true (MultiFig parameter)"])],
  bp=F("none: three sequences data1/data2/data3 are not named by organ (record text: 'rat colon, stomach, and kidney'); order of organs unverified", {}, "low"),
  prob=["only 47 PNG frames in total (3 sequences) vs plan_images 3000: effective yield 47 images",
        "registry candidates colon/kidney/stomach correct as a set; no per-sample decision possible from names"],
  note="Tiny but valid pCLE (confocal laser endomicroscopy) rat data; group with pcle-pto. Consider dropping for yield.")

o("lightweight-instrument-seg-ush", acc="ok_blocked_by_archive_filter (download.py discards the only archive: its name contains 'Segmentation')",
  src=[S("zip", "A Lightweight Segmentation Network for Endoscopic Surgical Instruments Based on Edge Refinement and Efficient Self-Attention.zip (zenodo 8098618, 2.55 GB)", ["dataset/{train,val,test}/{images,labels}/N.png (train 1,406 + val 469 + test 469 = 2,344 images, same number of label pngs)"], ["dataset/train/images/0.png"])],
  prob=["PIPELINE BLOCK: fetch_entries drops archives matching /mask|label|seg/i and uses `or archives[:0]`, so when every archive matches the result is the empty list -> 'no media files or archives listed' -> state error; here the single zip is named '... Segmentation Network ...'",
        "Zenodo description is one sentence ('we collected and annotated the surgical instrument dataset'): laparoscopic vs other endoscopy and the surgery (abdomen) are not stated; frames are sequential video frames (near-duplicates likely)",
        "labels/ folder is excluded by is_mask (last two components `labels/N.png`)"],
  note="Labels laparoscopy/abdomen/human are plausible but unverified (medium). Fix the filter in download.py (see md).")

o("five-d-us-ablation-deshmukh", spec=F("registry hard human; the record does not state the specimen", {}, "low"), bp=F("registry hard soft tissue; unverified", {}, "low"),
  acc="ok_unlistable_archive (5-part RAR volume set, 6.9 GB; raw 'Timing' data)",
  src=[S("dataverse", "doi:10.7910/DVN/KZDHYE (Harvard Dataverse, CC0)", ["Timing_raw_data.part1.rar ... part5.rar (1.5 GB x4 + 0.96 GB)"], ["description: 'Dataset containing 3D ultrasound volumes for ... Five dimensional Ultrasound Visualization Framework to Guide and Monitor Ablative Therapy' (IPCAI/IJCARS 2015)"])],
  prob=["split RAR volumes cannot be listed or range-sampled and download.py has no rar handler; contents are 'Timing raw data' (probably raw RF/volume files), not verifiable",
        "registry specimen=human and body_part soft tissue are unverified: the record only says 'Dataset containing 3D ultrasound volumes' for a 2015 ablation-guidance paper (specimen not stated in the record, which is only a 'Replication data' deposit)"],
  note="DROP candidate: unlistable multi-part RAR, no image evidence, labels unverified.")

o("oocytor-oocyte-segmentation", sub=F("none listable; record says transmitted light, settings differ per subset", {}, "low"),
  acc="ok_unlistable_archive + ok_blocked_by_archive_filter (single 1.15 GB tar.gz named SegmentationCortex.tar.gz)",
  src=[S("tar.gz", "SegmentationCortex.tar.gz (zenodo 6502830)", [], ["record text: 'mouse and human oocytes and sea urchin eggs, transmitted light, with membrane segmentation ground truth'"])],
  spec=F("none listable; the record names three species (mouse oocytes, human oocytes, sea urchin eggs): registry candidates embryo/human/mouse omit sea urchin, which has no class", {}, "low"),
  bp=F("dataset-level: oocytes (cells); 'ovary' candidate is not what is imaged", {"oocyte/egg images": "cells"}, "medium"),
  prob=["PIPELINE BLOCK: archive name matches /seg/ in fetch_entries, the only archive is filtered out -> nothing is downloaded (same `or archives[:0]` bug as lightweight-instrument-seg-ush)",
        "even after the fix, sample_tar_stream takes the first N media members in archive order (no spreading over species/conditions) and tar.gz cannot be listed, so folder-level metadata is unavailable until extraction",
        "submodality brightfield vs contrast microscopy: record says only 'transmitted light with different settings' (Hoffman/DIC settings possible)"],
  note="Specimen candidates would be wrong for the sea urchin part: mask specimen.")

o("multieye", bp=F("dataset-level; mmbu-public fundus rows: eye 407 / retina ~40; OCT rows eye", {"eye": "default"}, "medium"),
  acc="ok (HF repo = one 176k-image zip: a compilation of public datasets)",
  src=[S("zip", "multieye_data.zip (HF Luxuriant16/MultiEYE; central directory read via HF resolve URL)", ["assemble/{train,dev,test}/ImageData/cfp-clahe-224x224 (66,491 png)", "assemble_oct/{train,...}/ImageData/oct-filter-448x448 (109,714 png)", "dataset_source/{assemble,assemble_oct}/<split>/<source>.txt (aria, eyepacs, fives, fundoct, jsiec, messidor, mmcamd, odir, oiaddr, rfmid, rfmid2, stare, vietai, goal, oct2017, octid, private)", "private_paired_data/private_pair_{fundus,oct}.txt"], ["assemble/train/ImageData/cfp-clahe-224x224/(0001)aria_d_26.png", "..._left_eyepacs.png (17,563)", "CNV-#-#.png 37,068 / NORMAL 26,315 / DME 10,922 / DRUSEN 8,344 (OCT2017)"])],
  sub=F("zip path: folder `cfp-clahe-224x224` -> fundus photography, `oct-filter-448x448` / `assemble_oct` -> optical coherence tomography (match_value strong)", {"assemble/.../cfp-clahe-224x224/x.png": "fundus photography", "assemble_oct/.../oct-filter-448x448/x.png": "optical coherence tomography"}, "high"),
  prob=["DERIVATIVE COMPILATION: images are re-processed (CLAHE, 'filter') and down-sized (224 px fundus), pooled from EyePACS (35k), ODIR, Messidor, RFMiD, ARIA, FIVES, STARE, JSIEC, Kermany OCT2017 (~83k of the 109.7k OCT: CNV/NORMAL/DME/DRUSEN-#-#.png), GOAL, OCTID + a private set: OCT2017 is 75% of the OCT half (spread_sample splits by parent folder, so ~half of a sample is OCT and ~3/4 of those are oct2017-kermany copies); the fundus half repeats EyePACS (26%), FIVES (also selected), and ODIR/RFMiD/JSIEC (also re-compiled by selected amdnet23/rao sets)",
        "if fetch_hf's streaming succeeds on the zip, member paths are lost and the per-sample fundus/OCT split (only in the folder name) cannot be recovered; only the remotezip path keeps it",
        "registry full_size_gb -1 (unknown), group topcon-mm-fddm"],
  note="Recommend DROP (processed, duplicated, contamination-prone). If kept: exclude OCT2017 copies and use submodality from the path.")

o("color-doppler-twinkling-biopsy-markers", spec=F("registry hard phantom; unverified (page behind login)", {}, "low"), sub=F("registry candidates b-mode / color doppler; unverified", {}, "low"),
  acc="unsupported_host_ieee_dataport_login (doi 10.21227/52wv-xv88 -> ieee-dataport.org/documents/color-doppler-twinkling-change-between-ultrasound-platform-generations-compariso...)",
  src=[],
  prob=["IEEE DataPort needs a login; resolve_doi has no rule (Manual). No listing available; registry access 'registration'",
        "registry specimen=phantom / submodality candidates b-mode vs color doppler are from the registry description only (unverified)"],
  note="DROP: unreachable without credentials (same class as the other IEEE DataPort entries already dropped).")
