from lib import *

o("fundus-avseg", src=[S("zip+xlsx","Fundus-AVSeg.zip: images/, annotation/ (masks); metadata.xlsx",["eye id","disease type","image quality"],[])], bp=F("dataset-level (fundus)", {"-":"eye"}, "medium"))
o("bdcxr-3257", src=[S("zip+csv","BDCXR-3257.zip: <Pneumonia Bacterial|Pneumonia Viral|Normal>/*.jpg; metadata.csv",["Class","Age","Gender"],[])],
  bp=F("dataset-level (chest radiographs)", {"-":"multi [chest, lung]"}, "medium"), note="Class folders are diagnoses.")
o("ridirp-rop-ostrava", src=[S("zip-names","images_stack/<id>_<sex>_GA<weeks>_BW<g>_PA<..>_DG<..>_PF<..>_D<..>_S<..>_<k>.jpg",["file name tokens (not Pass-2 fields)"],[])],
  bp=F("dataset-level (neonatal retina, ROP)", {"-":"eye"}, "medium"), note="`images_stack` = multiple frames per exam: near-duplicates.")
o("cervical-tct-heilongjiang-2025", acc="IMAGES MISSING: figshare 26533924 lists only Annotations.zip (8037 VOC xml)", src=[S("zip","Annotations.zip",["VOC xml"],[])],
  prob=["registry says 41 GB / 2000 images; only the 4 MB annotation zip is listed (images probably in another article/part): not downloadable as is"])
o("head-cta-intracranial-aneurysm-dicom-2025", src=[S("zip+csv","Dataset.zip: Studies/<id>/*.dcm (149k files), Annotations/an_*.stl (3D meshes); Metadata.csv",["Artery","Position","Type","Size","Rupture"],[])],
  bp=F("dataset-level (head CT angiography)", {"-":"brain (arteries also valid)"}, "medium"),
  prob=["~1,300 dcm per study: slice sampling per study; .stl meshes are annotations (not images)"])
o("stomataquant-2026", src=[S("zips","S1-1 detection (images/labels train|val), S1-2 segmentation, S1-3 stomata+pavement, S2 test sets (UNSM, Bangladesh Forests, Maize, AT), S3",["path d3 test-set name"],[])],
  prob=["five zips re-use the same leaf micrographs for detection/segmentation tasks: sample one zip's images/ folder only"],
  note="body masked: stomata/leaf epidermis has no vocab class (could stay masked).")
o("fovea-intraoperative-fundus", src=[S("zip+json","FOVEA.zip: FOVEANNN_i_img.png, FOVEANNN_i_od_K.png, ...mkv (40), metadata.json (dims, fps, condition)",[],["FOVEA001_i_img.png","FOVEA001_i_od_1.png"])],
  prob=["560 png include *_od_K.png (optic disc annotations) that is_mask does not catch (`od`): only *_img.png are photographs","40 intraoperative .mkv videos (frame extraction)"])
o("intrapartum-us-video-dataset", acc="ok but zip contains only READMEs + 3 nested zips (size -1)", src=[S("zip","DatasetV3.zip: README*.md + 3 inner .zip",[],[])],
  bp=F("dataset-level (intrapartum transperineal US: fetal head + pubic symphysis)", {"-":"multi [fetus, pelvis]"}, "low"),
  prob=["nested zips cannot be range-listed/sampled by sample_zip_remote (it will pick the inner .zip files)"])
o("dap-atlas", src=[S("hf-path","images/AutoPET_<id>_<n>.nii.gz, masks/…; labels.json (paper labels)",[],[])],
  bp=F("dataset-level (whole-body CT from AutoPET)", {"-":"whole body"}, "medium"))
o("curvas-2024", src=[S("zip-path","{training,validation,testing}_set.zip: UKCHLL###/{image, annotation_1..3}.nii.gz",[],[])],
  bp=F("dataset-level (multi-organ abdominal CT: liver, kidney, pancreas)", {"-":"abdomen"}, "medium"),
  prob=["3 of every 4 nifti per case are annotation_* masks (is_mask 'annotation' catches them)"])
o("rvo-me-oct", src=[S("zip-path","RVO-Lesion.zip: Image_Seg/{images,masks}, RVO_Lesion_Labelme, RVO_HF_Detection",["Demographics xlsx: Eye category"],[])],
  bp=F("dataset-level (retinal OCT)", {"-":"retina"}, "medium"),
  prob=["the same 3,012 OCT B-scans are repeated in Image_Seg/images, Labelme and HF_Detection folders: dedupe"])
o("cholec80-cvs", acc="NO IMAGES: figshare 22183885 is a 1-sheet xlsx (CVS annotations); frames come from Cholec80 (separate request)", src=[S("xlsx","cholec80-CVS.xlsx",["video","critical_view","initial_minute..final_second"],[])],
  prob=["only critical-view-of-safety annotations on Cholec80 videos; the videos/frames are not in the record"])
o("sonocorpus", acc="NO IMAGES: manifest/benchmark index of ~50 upstream ultrasound datasets", src=[S("csv","sonocorpus_datasets.csv, sonocorpus_manifest.csv, splits/, evaluation_records/",["dataset","application","source_url","access","modality"],[])],
  prob=["record holds manifests only (199 MB csv): pixels belong to upstream datasets; body masked because it spans organs. Not a usable image source; could instead be used to find per-organ upstream sets (application column)"])
o("octave", src=[S("zip-path","OCTAVE.zip: nnUNet_raw/Dataset001_OCTAVE/{imagesTr,labelsTr}/*.tif; external_tests/*_nnUNet/{imagesTs,labelsTs}",[],[])],
  bp=F("dataset-level (retinal OCT)", {"-":"retina"}, "medium"), note="Only ~200 image tifs; most tifs are labelsTr/labelsTs.")
