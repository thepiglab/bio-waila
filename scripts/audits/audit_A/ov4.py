from lib import *

o("ispy2", src=[S("hf-archives","ISPY2_T0_T3_DCE_npz.zip (428 .npz volumes); ISPY2_T0_T3.tar.gz (no listing)",[],[])],
  prob=["images are .npz arrays (DCE volumes): prepare.py needs array->slice logic; tar.gz cannot be listed (size 1750 GB claimed, sizes unknown)"],
  note="All fields hard and consistent (breast DCE-MRI).")
o("rdbmc-romanian-dense-breast", acc="needs Kaggle credentials (manual)", src=[], note="Kaggle listing unavailable. Registry mammography/breast hard is plausible for the title (Romanian dense breast mammography collection); DICOM header (ViewPosition/Laterality) would supply CC/MLO.")
o("whoi-plankton", src=[S("hf","nf-whoi/whoi-plankton parquet (957k images)",["label (ClassLabel, 100 plankton taxa)"],["Asterionellopsis","Chaetoceros","Ceratium"])],
  spec=F("registry masked: label names are plankton taxa; vocab has no plankton/phytoplankton class", {}, "low"),
  note="Imaging FlowCytobot images; label names are taxa (not Pass-2 fields). A whoi-plankton.tar.gz duplicate is also in the repo.")
o("tryptag-tbrucei-localisation", acc="ok but selected record has no images (tables + code only)", src=[S("tsv","localisations.tsv, raw_localisation_data.tsv, id_doi_index.tsv, plate_doi_index.tsv",["geneId","terminus","plate DOI"],[])],
  prob=["zenodo 17117768 is the TrypTag index: images are in per-plate Zenodo records referenced by plate_doi_index.tsv (10.5281/zenodo.6862289, ...): download.py would list only tables/code (media=0)"],
  note="stain mneongreen (mNG-tagged) is correct; specimen (Trypanosoma brucei) has no class in vocab.")
o("mslesseg-2024", acc="ok (single .tar, 2.78 GB)", src=[S("tar","MSLesSeg.tar",[],[])], bp=F("dataset-level (multiple sclerosis brain MRI)", {"-":"brain"}, "medium"),
  note="tar cannot be listed remotely: contents (T1/T2/FLAIR + masks) unverified.")
o("code-oral-mucosa", src=[S("zip-path","Version_N.zip: 01. NORMAL IMAGES | 02. VARIATIONS FROM NORMAL | 03. OPMD | 04. ORAL CANCER / SMITAnnnnn / ANNOTATED|UNANNOTATED / JSON",["path d2 category","Patient's Metadata.xlsx"],[])],
  bp=F("dataset-level (oral cavity photographs)", {"-":"oral cavity"}, "medium"),
  prob=["Version_1..5.zip are CUMULATIVE (1.7k, 3.4k, 5.6k, 8.0k, 11.3k images): sample Version_5 only, or the same photos repeat","folders named ANNOTATED/UNANNOTATED + JSON (labelme) coexist in images: annotated copies may carry overlays"],
  note="Category folders are diagnoses. Whether `teeth` images exist cannot be seen from listing.")
o("phlf-gd-eob-liver-mri-2026", src=[S("zip-path","PHLF/Center{1,2,3}/Image/N.nii.gz; Annotation_* masks; clinicopathological xlsx",[],[])],
  bp=F("dataset-level (Gd-EOB-DTPA liver MRI)", {"-":"liver (abdomen also valid)"}, "medium"), note="Annotation_* folders are masks (caught by is_mask).")
o("vertebral-bodies-ct-labels", acc="MASKS ONLY (no images in repo)", src=[S("hf","fhofmann/VertebralBodiesCT-Labels train/test/*.nii.gz + data.csv",["tseg-sNNNN / verse-glNNN labels"],[])],
  prob=["README: 'This dataset does not include imaging studies itself': 1460 label niftis only; CTs are in TotalSegmentator/VerSe. No usable images -> drop from selection (or re-point to TotalSegmentator)"])
o("mitoem", src=[S("hf-zip","EM30-{H,R}-mito-train-val-v2.zip (seg*.tif labels), mitoem_R_train_4um_{im,seg}.h5",[],[])],
  spec=F("file name EM30-H / EM30-R (human / rat)", {"EM30-H":"human","EM30-R":"rat"}, "high"),
  prob=["the tif files in the zips are named seg*.tif (instance labels): images are only in the h5 volumes; only mitoem_R_train_4um_im.h5 (rat) is visible in the repo: human images absent","h5 volumes are not handled by download.py (no h5 kind); sizes unknown (-1)"])
o("isles-2015", src=[S("hf-path","ISLES2015_{SISS,SPES}/training/N/VSD.Brain.XX.O.MR_{DWI,Flair,T1,T2,CBF,CBV,TTP,Tmax,...}.NNNNN/*.nii",["folder token MR_<seq>","subchallenge SISS|SPES"],["MR_DWI","MR_Flair","MR_CBF","MR_Tmax"])],
  bp=F("dataset-level (stroke brain MRI)", {"-":"brain"}, "medium"),
  prob=["SPES contains perfusion maps (CBF/CBV/TTP/Tmax) which are not structural MRI: filter on folder token (MR_CBF|MR_CBV|MR_TTP|MR_Tmax) or relabel; 'OT' folders are ground-truth masks (check is_mask: 'VSD.Brain.XX.O.OT.*')"])
o("mm-odir-129", src=[S("json","annotations.json, split.json",["conditions","glaucoma","cup_to_disc_ratio","transcription"],[])],
  bp=F("dataset-level (fundus)", {"-":"eye (MMBU prior 407/447)"}, "medium"),
  prob=["138 .png and 138 .npy next to the 129 .jpg are probably maps/embeddings, not fundus photos: select *.jpg only"])
o("ccm-seg-2026", src=[S("zip","Dataset.zip: images/ (410), annotations/ (410 masks), re-annotations/ (20); data.xlsx",["Healthy","Set ID"],[])],
  bp=F("dataset-level (corneal confocal microscopy, nerve fibres)", {"-":"cornea (MMBU confocal: eye 19, cornea 5)"}, "medium"), note="annotations/ png are masks: is_mask catches 'annotations'.")
