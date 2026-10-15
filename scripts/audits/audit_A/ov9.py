from lib import *

o("lnq2023", src=[S("hf-dicom","images/case_NNNN/<SeriesUID>/*.dcm (66k); series_to_patient.json; parquet full/partial",["Modality","SeriesDescription","annotation_regime"],[])],
  bp=F("DICOM BodyPartExamined after download (dicom_hits) else dataset-level mediastinal CT", {"CHEST":"chest"}, "medium"),
  note="Registry candidates chest/lymph nodes: the images are chest CT (lymph nodes is the target structure).")
o("deep-fish-bone-zebrafish", src=[S("zip+csv","ventral/<fish_id>/M<k>.png; metadata.csv",["genotype: wt|het","quality","experiment_code"],[])],
  bp=F("dataset-level (zebrafish larval skeleton)", {"-":"skeletal system"}, "low"),
  note="Likely alizarin-red bone staining (stain masked; no matching class in vocab). 190 of the pngs are masks.")
o("liver-mets-longitudinal-ct-isfahan", acc="RESTRICTED: zenodo 19432434 access_right=restricted", src=[], prob=["cannot be downloaded without author approval: drop or request access"])
o("recist-ct-hcuch", src=[S("zip-path","final-formatted/images/{abdomen,thorax}/{train,test}/<patient>_<SeriesUID>.nii.gz; metadata/patients.csv, series.json",["path d3: abdomen|thorax","path d4: train|test"],[])],
  bp=F("zip path level d3", {"abdomen":"abdomen","thorax":"chest (canonical thorax/chest)"}, "high"),
  note="Per-sample body part is explicit in the folder name.")
o("pediashf-dx", src=[S("zip-path","PediaSHF-DX.zip: 01.train, 02.train_txt, 03.validation, 04.yolov11 (ultralytics source tree), 05.detection_results",["path d2"],[])],
  bp=F("dataset-level (paediatric supracondylar humeral fracture elbow radiographs)", {"-":"elbow"}, "medium"),
  prob=["zip includes a full ultralytics code checkout (python/pyc/md/yaml) and 05.detection_results (8.3k YOLO output overlay jpgs, same count as 03.validation): sampling over 18.6k jpgs would pick detection overlays with boxes drawn; restrict to 01.train + 03.validation","__MACOSX resource forks (._results.csv)"])
o("pediurf", src=[S("csv+zip-path","train.csv/test.csv (Category,Gender,Age); PediURF/{train,test}/<fracture type>/<patient>/{front,side}.jpg",["path d3: Distal|Midshaft|Proximal ulna and radius fractures","Age: Y8"],[])],
  bp=F("zip path d3 (fracture level)", {"Distal ulna and radius fractures":"wrist","Midshaft/Proximal ulna and radius fractures":"upper limbs (forearm; no class)"}, "medium"),
  note="Paediatric forearm radiographs; front/side views are projections, not Pass-2 fields.")
o("fcd-bonn-ds004199", src=[S("openneuro","sub-NNNNN/anat/*_{T1w,FLAIR...}.nii.gz; participants.tsv",["BIDS suffix"],[])], bp=F("dataset-level (pre-surgical epilepsy brain MRI)", {"-":"brain"}, "medium"))
o("2p-ivm-leukocyte-cell-death", acc="ok but pixels only in .h5 / .ims (66 GB)", src=[S("zenodo","NeuN.h5 x34, {Neutrophils,Eosinophils,Dendritics}_crops.zip (.ims), Annotations_CSV.zip",[],[])],
  prob=["no 2D image files: .h5 and Imaris .ims 3D+t arrays are unsupported by download.py/prepare.py (kind() has no h5/ims)","matcher quirk: folder 'Neutrophils_crops' maps to body_part class 'neutrophils' (a cell type) although registry candidates are lymph nodes/spleen"],
  note="Intravital two-photon: registry multiphoton/mouse plausible.")
o("leishmania-canine-node-smears", src=[S("zip","Canine Leishmaniasis Infected Data.zip: dogs/<id>.jpg + YOLO .txt",["path d1: dogs"],[])],
  spec=F("zip folder name `dogs`", {"dogs":"dog (needs plural normalisation); leishmania parasites also visible: multi [dog, leishmania] is defensible"}, "medium"),
  note="201 images, 200 label txt files (YOLO).")
o("hycervix", src=[S("zip+csv","Patient_N.zip: Patient_N_Diagnostic.txt, _GT.mat/_GT.png, hyperspectral .dat/.hdr (ENVI) and an RGB .png; anonymizedClinicalData.csv",["AnonID","Group","ReasonStudy"],["Diagnostic: Healthy / CIN 2"])],
  prob=["77 zips x ~330 MB each (25 GB) for 77 planned images: one RGB png per patient is inside; hyperspectral cubes are .dat/.hdr (not decodable by prepare.py)"],
  note="All labels hard; _GT.png is a mask.")
o("lask", src=[S("zip","DatasetA_7DOF_train.zip / DatasetB_BAPES_val.zip / DatasetC_6DOF_test.zip: videos/*.mp4 + annotations/*.npz + kinematics csv",[],[])],
  prob=["phantom laparoscopic skill videos: only mp4 (frame extraction needed); body masked (phantom abdomen)"], note="spec phantom hard is right.")
o("nec-xpert", acc="ok (single .rar 175 MB)", src=[S("rar","Dataset.rar",[],[])],
  bp=F("dataset-level (neonatal abdominal radiograph, NEC)", {"-":"abdomen"}, "medium"),
  prob=["only a .rar: no remote listing; needs unrar after download (download.py whole-file branch ok at 175 MB, prepare.py extraction of .rar unverified)"])
