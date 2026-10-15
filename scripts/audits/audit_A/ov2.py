from lib import *

o("mri-pairedfield-four-field-strengths", src=[S("hf-path","data/<source_dataset>/ZN_PAIR_NNNN/*.nii.gz; dataset_summary.tsv",["path d2: penn|openneuro_hfc|unc|osf|zenodo_*|figshare_*","validation csv modality: T1W/T2W/T2-FLAIR"],[])],
  bp=F("dataset-level (paired 3T/7T/ULF brain MRI)", {"-":"brain"}, "medium"), note="All structural contrasts (T1W/T2W/FLAIR); includes ultra-low-field source (ULF-TR).")
o("mogambo-gynaecological-brachytherapy-mri", src=[S("hf-path","train/images/P1.nii.gz, train/masks/P1.nii.gz",[],[])],
  bp=F("dataset-level (gynaecological brachytherapy pelvic MRI)", {"-":"pelvis"}, "medium"), note="Registry body_part masked; pelvis (or cervix/uterus) is determinable from the dataset topic. Masks in separate folder.")
o("deepspot-smfish-simulated", src=[S("zip","data_xenodo.zip: hybrid_simulated/train/{original,GT,csv}/N.png|csv; data.zip has NESTED zips (Test.zip, Train.zip, wound_healing_assay.zip)",[],[])],
  prob=["data.zip (4.4 GB) contains only nested zips: sample_zip_remote cannot list/sample them; use data_xenodo.zip only","data.zip/wound_healing_assay.zip is a real (non-simulated) assay: would contradict hard 'synthetic fluorescence microscopy' + specimen 'synthetic'","GT/ png are label images (matched by is_mask regex 'gt'); csv are spot coordinates"])
o("feltrin-brain-tumor-mri-30c", src=[S("zip-path","<TumorClass> <Sequence>/…jpg e.g. 'Meningioma T1C+', 'Glioma T1', 'Normal T1', 'Meningioma T2'",["path d1: class + sequence (T1/T1C+/T2)","DATA.json"],[])],
  bp=F("dataset-level (brain tumor MRI)", {"-":"brain"}, "medium"),
  note="Folder name encodes MR sequence (T1, T1C+, T2) -> all structural mri. matcher weak-matches 'T1' inside 'Meningioma T1' but 'T1C+' normalizes to 't1c' only inside a submodality-named column (not for path levels): add t1c to PATH_CODES / allow trailing token codes.")
o("mu-glioma-post", src=[S("hf-path","MU-Glioma-Post/PatientID_NNNN/Timepoint_N/*.nii.gz; 2 clinical xlsx",[],[])], bp=F("dataset-level (post-treatment glioma brain MRI)", {"-":"brain"}, "medium"),
  prob=["30 files per timepoint include tumor segmentations and several sequences (names truncated in listing): filter *seg* and non-structural (DWI/ASL/perfusion) by suffix"])
o("soop-stroke", src=[S("hf-bids","sub-N/{anat,dwi}/…, derivatives/lesion_masks; participants.tsv",["BIDS suffix (T1w/FLAIR/dwi)"],[])], bp=F("dataset-level (acute stroke brain MRI)", {"-":"brain"}, "medium"),
  prob=["d2 listing shows `dwi` 6860 files: much of the image content is DWI/b-value volumes (+ bval/bvec): arguably not 'structural mri'; filter by BIDS suffix (T1w/T2w/FLAIR) or accept DWI as MRI"],
  note="lesion_masks (3109 files) are masks: exclude.")
o("brisc-2025", src=[S("csv","manifest.csv (also json)",["task","split","tumor_label","plane_label","sequence","is_mask","linked_image"],["sequence: T1"])],
  bp=F("dataset-level (brain tumor MRI)", {"-":"brain"}, "medium"),
  note="manifest.csv `is_mask` column gives exact mask filtering (stronger than the path regex). classification_task and segmentation_task may contain overlapping images: dedupe via linked_image.")
o("figshare-clinical-brain-mri-motion-artifacts", src=[S("zip-bids","anat/sub-NNN_desc-{artifactfree,motioncorrupted}_{T1w,T2w,FLAIR}.nii.gz; 2d_acquisitions vs 3d_acquisitions",["BIDS suffix"],[])],
  bp=F("dataset-level (clinical brain MRI)", {"-":"brain"}, "medium"),
  note="motion-corrupted vs artifact-free pairs: both labelled structural mri; consider dropping motioncorrupted copies to avoid duplicates.")
o("gbm-multi-institutional-ds007045", acc="needs Kaggle credentials; but same data is on OpenNeuro ds007045 (no auth)", src=[S("openneuro","ds007045 (checked): sub-<ID>/anat/sub-<ID>_run-NN_T1w.nii.gz (1452 niftis, 364 subjects)",["BIDS suffix T1w"],[])],
  bp=F("dataset-level (GBM brain MRI)", {"-":"brain"}, "medium"),
  prob=["selection method=kaggle (needs credentials) although homepage/fallback is OpenNeuro ds007045: switch method to openneuro"],
  note="Verified by listing openneuro S3: anat T1w only.")
o("riva-pap-2024", src=[S("zip-path","images/<Class>_<n>_<k>.png (e.g. HSIL_30_1.png); annotations/annotations.json",[],["HSIL_30_1.png"])],
  bp=F("dataset-level (Pap smear cervical cytology)", {"-":"cervix"}, "medium"), note="Bethesda class in the filename is a diagnosis, not a Pass-2 field.")
