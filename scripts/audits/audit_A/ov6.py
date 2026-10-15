from lib import *

o("exodeepfinder-lysosomal-exocytosis-tirf", acc="ok but only .rar archives (37 GB): cannot be range-sampled", src=[S("rar","Training Dataset.rar (19 GB), Robustness dataset.rar (19 GB)",[],[])],
  prob=["download.py has no .rar handler ('rar cannot be sampled'); whole 19 GB files would be needed + unrar on HPG"], note="Labels are all hard and plausible (TIRF, GFP-tagged lysosomes in cell line).")
o("magic-ct", src=[S("zip-names+json","liver_cyst.zip, kidney_cyst_scans{1..4}.zip, hcc.zip, lung_cancer.zip, ... (scans/*.nrrd, segmentations/*.nrrd); hcc.json etc.",["zip name = patient group","json patient_group, captions_ru"],["liver_cyst","kidney_cyst_scans4","hcc","lung_cancer"])],
  bp=F("zip file name / json `patient_group`", {"liver_cyst":"liver","hcc":"liver","kidney_cyst*":"kidney","lung_cancer":"lung","(other groups: see remaining zips)":"organ named in group"}, "high"),
  prob=["scans are .nrrd (kind() has VOLUME_EXT .nrrd ok); segmentations/ folders are masks (is_mask ok)"])
o("nes-plankton-ifcb-2022", src=[S("hf","sosiklab/NES-plankton-classifier-2022-dataset parquet",["label (155 plankton taxa)","ifcb_roi_pid"],[])],
  spec=F("registry masked: taxa labels, no plankton class in vocab", {}, "low"), note="Same situation as whoi-plankton.")
o("lapgc-kvad-30", acc="NO DATA in record: zenodo 18557082 lists a single 3 MB PDF", src=[S("zenodo","18557082",["14-Dynamic key vascular.pdf"],[])],
  prob=["registry claims 14.455 GB / 5303 images but the zenodo file list has only '14-Dynamic key vascular.pdf': files missing (maybe a different version/record or restricted files)"])
o("crcd", src=[S("hf-parquet","SITL-Eng/CRCD data/<episode>/train-*.parquet (e.g. A_1, B_2, C_3)",["video_id","frame_left","frame_right","kinematics columns"],[])],
  bp=F("dataset-level: robotic cholecystectomy (CRCD)", {"-":"multi [gallbladder, liver] (abdomen implied)"}, "low"),
  prob=["two image columns frame_left/frame_right (stereo): pick one","755k frames from few videos: sample across episodes"])
o("spinedl-mouse-spinal-cord", src=[S("xlsx","Image Information.xlsx",["Species","Sex(Strain)","Study","condition"],["Mus musculus C57BL/6J"]), S("zip-path","Structure_identification.zip / Neuron_identification.zip: train|test|validation/{raw,label}/",["zip name = task"],[])],
  stain=F("zip name: Neuron_identification -> NeuN; Structure_identification -> DAPI (verify in README)", {"Neuron_identification":"neun","Structure_identification":"dapi"}, "low"),
  note="`label*`, `label_old`, `save` folders (tif/xml) are annotations. raw/ is the image.")
o("pituitary-tumor-carotid-mri", src=[S("csv","metadata.csv",["t1_sequence","t2_sequence","has_t2","sex","functional_status"],["t1_sequence: Stealth","t2_sequence: TSE"])],
  bp=F("dataset-level (sellar/pituitary MRI with carotids)", {"-":"brain (arteries is also in candidates)"}, "low"),
  prob=["matcher false positive: metadata.csv cell_type='Mammosomatotroph' hits the `mammo\\w*` x-ray rule -> modality x-ray imaging; use \\bmammogra\\w*|\\bmammography\\b instead"],
  note="dataset/N/N_T1.nii.gz + N_T2.nii.gz are the images; original_masks/ masks (405 flagged).")
o("zebrafish-embryo-phenotypes-jeanray2015", src=[S("zip-path","JeanrayEtAl-Data-PlosOne.zip: Learning Sets|Validation Set|Test Sets|Stressed Set / <Phenotype> / <Treatment> / *.tif",["path d3 phenotype (Normal, Edema, Short_Tail, Dead...)","path d4 treatment (CT, 1mM, 4mgl ...)"],[])],
  prob=["matcher false positive: path level d4 named 'CT' (= control) is mapped to modality computed tomography by PATH_CODES['ct']; hard registry labels protect it but a conflict will be logged: drop 'ct' from PATH_CODES or require an imaging context"],
  note="16 GB zip of TIFFs (2655): zip-range sampling reads from the central directory: fine.")
o("lymphus-cervical-ln-ptc", acc="needs Kaggle credentials; homepage qamebi.com/cervical-lymph-node/ is a direct alternative", src=[], note="No listing possible without credentials. Try the original site download (QaMeBi) instead.")
o("hardwood-populus-stomata", src=[S("csv","Labeled Stomatal Images.csv",["FileName","Species","ScientificName","Magnification"],["Species: Nuttall oak"])],
  spec=HARD, note="Per-image species (hardwood trees) -> plant, consistent with registry. txt YOLO labels are not images.")
o("delta-ecoli-mother-machine-fluor", src=[S("zip-path","roi_mixed_200x800.zip: from_manual_mixed_200x800/{img,seg}/*.tiff (14k each)",[],["LED450NM_P6_X48775.7_Y-46828.6_..."])],
  prob=["is_mask regex bug: folder `from_manual_mixed_200x800` contains the token 'manual' (in the mask list `manual\\d?`), so ALL 28,027 tiffs are classified as masks and media=0: the dataset would yield no images. Fix: drop bare 'manual' from _MASK or require a word-boundary directory pattern","model file evo_roi_mixed_200x800_CKS5_2025-02-11.keras (739 MB) is not data"],
  note="img/ vs seg/ split is the right mask filter. LED450NM in names suggests the fluorescence channel.")
o("asoct-msdc", src=[S("zip-path+xlsx","asoct_dataset/<Cataract|Normal|Glaucoma|Glaucoma_Cataract>/{Original Images,Annotated Images}; Metadata_*.xlsx",["OD/OS","Category","Eyelid Occlusion"],[])],
  bp=F("dataset-level (anterior-segment OCT)", {"-":"anterior segment"}, "medium"), note="Annotated Images folders hold labelme json (no images); use Original Images.")
