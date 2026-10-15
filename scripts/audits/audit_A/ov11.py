from lib import *

o("s-haematobium-eggs-urine-2023", src=[S("zip-path","SHdataset_12k.zip: {train,test}/{images,masks}/N.png; diagnosis_test_dataset.zip: DTNN/*.jpeg + microscopist_result.csv",[],[])],
  note="All hard; 26 GB + 12 GB zips (sampling by central directory ok). Body 'bladder' for urine-egg microscopy is a stretch but fixed by registry.")
o("sth-schistosoma-eggs-2025", src=[S("zip","STH_and_SM_Dataset_v1_0.zip: images/*.png (10,562, ~3 MB each), annotations/*.json (boxes)",[],[])], note="All hard (stool microscopy).")
o("avian-blood-cell-abc-thessaloniki-2026", src=[S("zip-path","ABC_dataset.zip: annotated/{train_set,test_set}/{images,yolo_annotations,mask_annotations,test_images,test_annotations}; unannotated/*.png",[],[])],
  note="All hard; mask_annotations tifs and yolo txt are not images.")
o("prostatex-seg-zones", src=[S("hf-path","volumes/ProstateX-NNNN/{image,mask}.nii.gz; label_map.json",[],[])], bp=F("dataset-level (prostate MRI)", {"-":"prostate"}, "medium"))
o("dr1-dr2-unifesp", src=[S("zip-path","DR1-images-by-lesions.zip (tif), DR2-images-by-lesions.zip, DR2-images-by-referral.zip, DR1-additional-marked-images.zip, markings.zip",["path d2: lesion type / referral"],[])],
  bp=F("dataset-level (fundus)", {"-":"eye"}, "medium"),
  prob=["DR2-images-by-lesions and DR2-images-by-referral hold the same DR2 photographs sorted two ways: sample one; DR1-additional-marked-images contain drawn markings"])
o("nci-isbi-2013-prostate", src=[S("hf-dicom","images/{Training,Test,Leaderboard}/ProstateDx-NN-NNNN/*.dcm; segmentations/*.nrrd",["scanner (parquet): 3T-Siemens"],[])],
  bp=F("dataset-level (prostate MRI) or DICOM BodyPartExamined after download", {"-":"prostate"}, "medium"), prob=["80 extension-less files (segmentation DICOM-SEG?) beside .nrrd"])
o("mcoa-corneal-opacity", src=[S("zip-path+xlsx","Images/{Opaque Cornea,Normal Cornea}/{AS-OCT,ASP}/N.jpg; json annotations; demographics xlsx",["path d3: Opaque|Normal Cornea","path d4: AS-OCT | ASP"],["AS-OCT (9264 files)","ASP (579 files)"])],
  sub=F("zip path level d4", {"AS-OCT":"optical coherence tomography","ASP":"anterior segment photograph: NOT OCT (no vocab class: mask or clinical photography)"}, "high"),
  bp=F("zip path d3 / dataset-level anterior segment", {"Opaque Cornea|Normal Cornea":"cornea"}, "medium"),
  prob=["registry hard submodality=OCT is wrong for ~579 ASP (anterior segment photography) images (~6%)"])
o("curvas-pdacvi", src=[S("zip-path","{training,testing,valididation}_set.zip: CURVASPDAC_NNNNN/{annotation_1..3,<phase images>}.nii.gz",[],[])],
  bp=F("dataset-level (pancreatic cancer CT)", {"-":"multi [abdomen, pancreas]"}, "medium"),
  prob=["80% of nifti entries are annotation_* masks (is_mask regex 'annotation' catches them); images are the remaining files per case (multi-phase)"])
o("modus-urine-sediment-2026", acc="ok (.rar only, 0.8 GB)", src=[S("rar","10x_magnification.rar, 40x_magnification.rar, VOC2COCO.py, VOC2YOLO.py",[],[])],
  prob=["only .rar archives (no listing, needs unrar)"], note="body candidates bladder/kidney: urine sediment microscopy: no per-sample metadata.")
o("ctpelvic1k", src=[S("hf-path+jsonl","CTPelvic1K_dataset{6,7}_{data,mask}/*.nii.gz; train.jsonl (modality,dataset,official_split)",["modality: CT"],[])],
  bp=F("dataset-level (pelvic bone CT)", {"-":"pelvis"}, "medium"))
o("mammo-mx", acc="ok but files inside zips have NO extension", src=[S("zip+csv","B1..B5.zip, B2_Part_I..VII_of_VII.zip: members like '000001_L_MLO_B2_D2' (no extension); Metadata.csv",["View: MLO|CC","Laterality","BIRADS","Breast Density"],[])],
  prob=["zip members are extension-less (likely DICOM): files.kind() returns 'other' -> media=0 -> download.py sees 'no media files or archives' / samples nothing; 69 GB total"],
  note="View (MLO/CC) is also encoded in the member name (L_MLO, R_CC): mammography already hard.")
o("hvm-hepatic-vessel-map", src=[S("zip-path+xlsx","HVM Dataset.zip: Center N/{Image,Annotation_Hepatic veins,Annotation_Portal veins,Annotation_Liver tumors}/NNN.nii.gz",[],[])],
  bp=F("dataset-level (liver CT, hepatic vessels)", {"-":"liver (abdomen also valid)"}, "medium"))
o("fiqs", src=[S("zip+xlsx","FIQSDataset.zip: full_size_images/ and images/ (2246 each), meta_data.xlsx (quality scores)",[],[])],
  bp=F("dataset-level (fundus)", {"-":"eye"}, "medium"),
  prob=["full_size_images/ and images/ are two resolutions of the same 2246 photographs: sample one"])
