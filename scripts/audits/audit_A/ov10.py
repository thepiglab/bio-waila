from lib import *

o("aroma-octa-artifacts", src=[S("zip-path+xlsx","AROMA_{train,test}/<subject>_<OD|OG>/<id>_Angio_20(6mmx6mm)_<date>_N.png; artefacts_labels*.xlsx",["Motion/blink","Tilt","Defocus"],[])],
  bp=F("dataset-level (OCTA en-face)", {"-":"multi [eye, retina] (MMBU OCTA: eye 6, eye+retina 2)"}, "low"), note="Artifact grades are not Pass-2 fields.")
o("l929-phc-cell-counting", src=[S("zip","L929-PhC_Dataset.zip: {train,valid,test}/{images,labels}; classes.txt; agreement_analysis.xlsx",[],[])],
  spec=F("dataset-level: L929 is a mouse fibroblast CELL LINE", {"-":"cell line (mouse also valid)"}, "high"), note="YOLO label txt are not images.")
o("apl-aml-sidhom-2021", src=[S("csv+zip-path","Images_metadata_table.csv (Patient_ID,Diagnosis,Cohort,Age,Gender); Patient_NN/{Signed,Unsigned} slides/<cell type>/BL_*.jpg",["path d3: cell type folder","Diagnosis"],[])],
  prob=["matcher quirk: cell-type folders (Segmented neutrophils, Band neutrophils, Smudge cells...) map to body_part classes `neutrophils`/`cells`: harmless only while body_part stays registry-hard 'blood'"],
  note="stain masked: peripheral-blood smear stain not stated in head (likely May-Grunwald-Giemsa per source paper).")
o("tcga-breast-radiogenomics", src=[S("hf-path","volumes/<TCGA-id>/{image,mask}.nii.gz; preview series_description",["series_description","breast_side"],[])], note="All hard consistent (DCE breast MRI; sequences incl. T1 sagittal post fat sat).")
o("caiman-ground-truth", src=[S("zenodo-zips","images_K53.zip (113 GB, 116k tifs), images_J115.zip, images_J123.zip, images_N.*.zip, WEBSITE*.zip",[],[])],
  prob=["huge zips (up to 113 GB) of single frames image00000..N.tif from calcium-imaging movies: consecutive frames are near-duplicates; spread_sample must pick far-apart frames; WEBSITE*.zip are not images"],
  note="Labels all hard and plausible (two-photon GCaMP mouse brain).")
o("harvard-fairfedmed-oph", acc="ok but default HF config is the CHEST metadata CSVs; OCT lives in FairFedMed-Oph/Dataset/dataset.zip (npz)", src=[S("hf","FairFedMed-Chest/meta_chexpert_*.csv; FairFedMed-Oph/Dataset/dataset.zip (15165 npz)",["race","gender","age","filename"],[])],
  prob=["fetch_hf: load_dataset(streaming) succeeds on the CSV rows (no Image column) -> 'no image column found', 0 images; it only falls back to file listing when load_dataset raises: need data_files=FairFedMed-Oph/Dataset/dataset.zip","npz volumes (likely SLO fundus + OCT B-scans per sample): modality (fundus vs OCT) unresolved from head","FairFedMed-Chest is a chest X-ray set in the same repo: must not be mixed into optical imaging"])
o("ispy1-tumor-seg-radiomics", src=[S("hf-path","NIfTI-Files/{images_bias-corrected_nifti,images_bias-corrected_resample}/ISPY1_NNNN/*_DCE_000K.nii.gz; masks_*",[],[])],
  prob=["images_bias-corrected_nifti and images_bias-corrected_resample are two versions of the same volumes: use one","masks_ftv/masks_stv_* are masks"], note="All hard consistent (breast DCE-MRI).")
o("mfxr-monteggia", src=[S("zip-path","MFXR/{train,test}/{good,anomaly}/*.jpg",["path d3/d4: good|anomaly"],[])],
  bp=F("dataset-level (paediatric elbow/forearm radiographs, Monteggia)", {"-":"elbow"}, "medium"),
  prob=["zip carries a __MACOSX mirror (4,594 junk entries) and .DS_Store"])
o("bsubtilis-spore-germination-tracking", src=[S("zip-path","dataset_b_subtilis/<NN>/… and <NN>_GT/TRA/man_track*.tif; states.txt, man_track.txt",[],[])],
  note="*_GT/TRA tifs are Cell-Tracking-Challenge label images (is_mask catches 'GT'). Specimen B. subtilis = bacteria (hard).")
o("care-restoration-datasets", src=[S("zip-path","Archive.zip: {small_crops,large_crops}/{gt,noisy}/N.tif",["path d2: gt|noisy"],[])],
  prob=["`gt` folder = high-SNR ground-truth IMAGES, but is_mask regex treats 'gt' as a mask token: only the `noisy` half would ever be used (noisy = deliberately degraded images)","3D crops (dims 3d) of planaria/Tribolium: specimen has no vocab class"],
  note="body candidates cells/tissue/whole body and specimen masked: not resolvable from head.")
o("beamster", src=[S("csv","Spreadsheets/Table_clinical_data.csv; HF preview",["Origin / primary_origin","MRI Scanner Type","Sex"],["Origin: Non-Small Cell Lung Cancer, Melanoma, Gastrointestinal tract"])],
  prob=["matcher false positive: column `Origin` / `primary_origin` (primary tumor site of the brain metastasis) maps to body_part lung / gastrointestinal tract because FIELD_COLUMNS['body_part'] contains `primary`; the imaged organ is the brain (registry hard). Remove 'primary' from the body_part column regex or add origin|primary to SKIP_COLUMN"])
o("zenodo-infant-brain-mri-833", src=[S("zip+csv","zenodo_upload_v2.zip: sNNNN/{t1,t2}.nii.gz; meta.csv",["myelinisation","age","diagnosis","group"],[])],
  bp=F("dataset-level (infant brain MRI)", {"-":"brain"}, "medium"))
