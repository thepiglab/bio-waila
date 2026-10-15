from lib import *

o("nirschl-heart-failure-2018", src=[S("hf","nirschl-lab/nirschl_et_al_2018 (arrow cache files)",["modality","submodality","stain","domain","institution","label_name","normal_or_abnormal"],["modality: light microscopy","stain: H&E","submodality: brightfield microscopy"])],
  note="The HF rows already carry Pass-2-style columns (modality/submodality/stain); they agree with the hard registry labels. Repo layout is an HF arrow cache dir (all-<hash>/0.1.0/*.arrow): datasets-server works, load_dataset on the raw repo may need data_files.")
o("feltrin-brain-tumor-mri-44c", src=[S("hf","bombshelll/brain_14_tumor_classes (parquet)",["label (ClassLabel, 14 tumor classes, Portuguese names)"],["Astrocitoma","Glioblastoma","Meningioma"])],
  bp=F("dataset-level (brain tumor MRI)", {"-":"brain"}, "medium"),
  prob=["selection id says '44c' (44 classes) but the HF repo has 14 classes / 3,827 images: it is a different (smaller) packaging than the Zenodo 30-class record; overlaps heavily with feltrin-brain-tumor-mri-30c (same group, near-duplicate images)"])
o("tompei-cmmd", src=[S("hf","MedOtter/TOMPEI-CMMD parquet",["view","side","birads","breast_density","dicom_tag_laterality","mask_*"],["view: MLO","side: L/R"])],
  sub=HARD, note="`view` MLO/CC confirms mammography (CODES mlo/cc -> mammography ok). Several mask_* Image columns: only `image` is the mammogram.")
o("multifocus-wbc-park-2024", src=[S("csv-in-zip","multi-focus-wbc-dataset/labels.csv, slide_number.csv, cbc_result.csv",["img_num","label","slide_num"],["label: seg_neutrophil, myelocyte, band_neutrophil"])],
  prob=["zip has 91,858 `__MACOSX/._*.jpg` junk entries among 441k members (images are z-stack focal planes _1.._k of the same cell): filter __MACOSX and de-duplicate focal planes","dims 3d but images are 2D jpg focal planes"],
  note="Stain masked (peripheral smear: Wright-Giemsa probable, not stated in head).")
o("tom500-thyroid-eye-orbital-mri", src=[S("hf-path","TOM500/{train,val}/{image,label}/N.nii.gz",["path d3: image|label"],[])],
  bp=F("dataset-level (orbital MRI for thyroid eye disease)", {"-":"eye"}, "medium"))
o("rescale4dl-u2os-nuclei", src=[S("zip-path","<10x|20x|40x|60x>_{training,test}.zip: <mag>/{images,labels}/*.tif; 'Original Metadata - *.csv' (Key/Value)",["path d2: images|labels"],["60x_dic50ms ... = DIC channel"])],
  spec=F("dataset-level: U2OS is a human cell line", {"-":"cell line"}, "high"),
  prob=["metadata files include a DIC acquisition (60x_dic50ms_1-4mmFOV_006): if DIC tifs are inside the image zips they are NOT epifluorescence/Hoechst: check names after listing images/","every zip has .DS_Store + __MACOSX entries"])
o("senora-mri", src=[S("bids-zip","SENORA-MRI_v1.0_part{1,2}.zip: sub-NNN/anat/anat/sub-NNN_{FLAIR,T1w,T1wCE,T2w,MRA}.nii.gz; dwi/; participants.tsv; json sidecars",["BIDS suffix","participants.tsv"],[])],
  bp=F("dataset-level (stroke brain MRI)", {"-":"brain"}, "medium"),
  prob=["contains MRA (angiography) and DWI/ADC volumes (848+ dwi files): not all 'structural mri'; filter by BIDS suffix","lesion_mask files sit inside subject folders (sub-052_lesion_mask.nii.gz)"])
o("csxa-cervical-spine-xray-atlas", src=[S("hf-zip","datasets-PNG.zip (4963 png), datasets-JSON.zip (labelme json), datasets.xlsx",[],[])],
  bp=F("dataset-level (cervical spine radiographs)", {"-":"spine (+neck)"}, "medium"), note="datasets.xlsx size unknown (HF listing has no sizes), not read.")
o("celegans-l1-nuclei-3d", src=[S("zip-path","c_elegans_nuclei/{train,val,test}/{images,masks}/*.tif",[],[])],
  bp=F("dataset-level (whole L1 larva nuclei)", {"-":"whole body (or cells)"}, "low"),
  prob=["only 28 image volumes in 80 MB: plan_images 1680 requires slicing every z-plane"])
o("busi", src=[S("hf","MedOtter/BUSI parquet",["image","mask","class_label","has_multifocal_lesion"],[])], note="All hard, consistent.")
o("hcc-tace-seg", src=[S("hf-dicom","images/HCC_NNN/<SeriesUID>/*.dcm (51,863); series_to_patient.json",["Modality","SeriesDescription","StudyDesc"],[])],
  bp=F("dataset-level (liver HCC CT)", {"-":"liver (abdomen also valid)"}, "medium"),
  prob=["series_to_patient.json lists Modality/SeriesDescription per series: use it to keep only CT series (segmentations/ are DICOM-SEG)","51k single-slice .dcm files: slice sampling must be per-series"])
o("octmnist", src=[S("hf","danjacobellis/octmnist_224 parquet",["path","label (sequence)","image"],[])],
  bp=F("dataset-level (retinal OCT B-scans)", {"-":"retina (MMBU OCT: eye 12 / retina 10)"}, "low"),
  note="MedMNIST-derived 224px; label = CNV/DME/drusen/normal (diagnosis).")
