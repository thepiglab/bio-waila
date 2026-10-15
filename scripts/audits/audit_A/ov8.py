from lib import *

o("murinecyto-det", src=[S("zip","data.zip: images/, masks/, masks_grayscale/, annotations/*.json (COCO)",[],["P01_10_10_162.json"])],
  note="333 images only (plan 333). body candidates cells/lung not resolvable from head; specimen mouse hard.")
o("dermoped", src=[S("zip-path+json","<diagnosis>.zip: patientN/{challenging,crops,masks}/NNNN.png (+_mask.png); patientN/metadata.json",["sex","access_age"],[])],
  note="All hard consistent. Frames look like video-derived sequences (challenging/0001..NNNN): spread sampling across patients; *_mask.png and masks/ are caught by is_mask. `crops` duplicate regions of `challenging` frames.")
o("s3d-spine-detection-2p", src=[S("zip","S3D.zip: train|valid|test/s3d_fovNNN_tpK_layerNNN.png; annotations/*.csv (boxes)",[],[])],
  bp=F("dataset-level (dendritic spines, mouse cortex two-photon)", {"-":"brain"}, "medium"))
o("cross-choleact", acc="ok (single 11 GB tar.gz, no remote listing)", src=[S("tar.gz","cross_CholeAct_v20260820.tar.gz",[],[])],
  prob=["tar.gz cannot be listed or range-sampled: tar-stream takes first N members only"],
  note="body candidates abdomen/gallbladder/liver all apply to cholecystectomy video: could be multi.")
o("grape", acc="NO DATA in record: figshare 575926 lists only 'Figure 8.tif' (2 MB)", src=[S("figshare","575926",["Figure 8.tif"],[])],
  prob=["registry expects 631 fundus images; figshare listing has one figure: wrong record / files hosted elsewhere (GRAPE glaucoma dataset)"])
o("pancreas-pda-segmentations-mayo", src=[S("zip+xlsx","Volume files.zip (91 nii.gz + __MACOSX copies); *_segmentations.zip; Metadata.xlsx",["Source: TCIA|MSD","Case"],[])],
  bp=F("dataset-level (pancreatic cancer CT)", {"-":"multi [abdomen, pancreas]"}, "medium"),
  prob=["Volume files.zip has `__MACOSX/Volume files/._*.nii.gz` junk entries (50) counted as volumes"])
o("drosophila-wing-database-gigadb", acc="NO DATA in record: figshare 437616 lists only Figure_2.tif", src=[S("figshare","437616",["Figure_2.tif"],[])],
  prob=["registry expects 8.3 GB / 1000 wing images; the figshare listing is one figure (real data on GigaDB FTP)","registry body_part hard='upper limbs' for a Drosophila wing is a questionable mapping (insect wing has no vocab class): prefer masked"])
o("spine-endoscopic-atlas", src=[S("zip-path","classified/<Small channel|Big channel>/<lumbar|cervical>/P##/<difficulty>/*.jpg; unclassified/; *.nrrd masks",["path d4: lumbar|cervical"],[])],
  note="body_part spine hard consistent with path d4. 10.6k .nrrd are masks (kind() would accept .nrrd as volume unless is_mask hits: it does for these paths).")
o("bicervi-2026", src=[S("zip-names","<NILM|ASC-US|ASC-H|LSIL|HSIL|AGC|...>.zip: <CLASS>/NEW/<CLASS>_N.jpg; README.pdf",["zip/folder name = Bethesda class"],[])],
  bp=F("dataset-level (Pap smear)", {"-":"cervix"}, "medium"),
  prob=["matcher false positive: folder 'ASC-US' (Bethesda class) weak-maps to modality ultrasound via the `us` token (path d1=ASC-US): protect Bethesda abbreviations (ASC-US, ASC-H, LSIL, HSIL, AGC) or ignore weak hits in registry-hard fields"])
o("qin-prostate-repeatability", src=[S("hf-dicom","images/PCAMPMRI-NNNNN/<SeriesUID>/*.dcm; series_to_patient.json; preview parquet `view`",["SeriesDescription","view: T2W|DCE|ADC"],["T2W","DCE","ADC"])],
  bp=F("dataset-level (prostate MRI)", {"-":"prostate"}, "medium"),
  prob=["330 extension-less DICOM files; ADC maps and DCE series are mixed with T2W: view/SeriesDescription lets you keep T2W only (structural) or relabel"])
o("sechenov-cystoscopy-bladder-cancer-videos", src=[S("xlsx","clinical_info.xlsx",["Patient Id Number","Histological Type of Lesion","Video Duration","Light mode"],["Light mode: CLARA + CHROMA / white light"])],
  sub=F("clinical_info.xlsx `Light mode` (per patient)", {"white light":"white-light endoscopy","CLARA + CHROMA":"white-light endoscopy (Storz digital enhancement); no vocab class for enhanced modes"}, "medium"),
  prob=["only 173 mp4 videos (frames need extraction); other Light mode values (blue light/NBI?) may exist beyond the first 50 rows: check full column before trusting hard white-light"],
  note="Light-mode column could split off non-WLE patients if present.")
o("uaal-upper-airway-landmarks", src=[S("zip-names","coco_ins_clinical.zip (3814 png), coco_ins_phantom.zip (2746 png); coco annotations",["zip name: clinical|phantom"],[])],
  spec=F("zip name", {"coco_ins_clinical":"human","coco_ins_phantom":"phantom"}, "high"),
  bp=F("dataset-level (upper airway endoscopy)", {"-":"head and neck (nose/larynx also valid)"}, "medium"))
