from lib import *

o("cisd-urine-cytology-2022", src=[S("zip-path","CISD/{stack,edf,center_slice}/<case>_<id>.jpg; CISD.json (labelme-style)",["path d2: stack|edf|center_slice"],[])],
  bp=F("dataset-level (urine cytology)", {"-":"multi [bladder, epithelial cells]"}, "low"),
  prob=["same fields appear three times: stack (82,131 z-planes, 21 per field), edf (3,911 extended depth-of-field) and center_slice (3,911): sample edf only or the plan is dominated by z-plane duplicates"])
o("drosophila-embryo-wound-healing", src=[S("zip","Dataset.zip: Videos/*.tif (62 time series), Segmentations/*.tif",[],[])],
  spec=F("dataset-level (Drosophila embryos)", {"-":"drosophila"}, "medium"), note="Videos are multi-frame tifs; Segmentations are masks (caught by is_mask 'segmentations').")
o("intelkamat-colposcopy", src=[S("csv","images.csv, patients.csv, annotation_audit.csv, reference_view_map.csv",["tz_type","findings","acquisition_type"],[])],
  note="All hard (colposcopy, cervix). annotated_references.zip are drawn-on photographs: exclude.")
o("sperm-scd-instance-seg-2026", src=[S("zip-path","SCD_Microscopy_YOLOv8seg_Repository.zip: dataset/{images,images_split/{train,val,test},labels_*}; annotations/coco_json; model/*.pt",[],[])],
  bp=F("dataset-level (sperm microscopy; MMBU brightfield: sperm cells 93)", {"-":"sperm cells"}, "medium"),
  prob=["dataset/images (1000) and dataset/images_split (1102) overlap (same images split into train/val/test)"])
o("crlm-ct-seg-refined-flr", acc="MASKS ONLY (no images in record)", src=[S("zip","CRLM-CT-Seg-Refined.zip: label/*.nii.gz (197)",[],[])],
  prob=["only refined segmentation labels (14 MB): CTs live in TCIA CRLM; no usable images -> drop or re-point"])
o("bratious-ious-glioma", src=[S("zip+xlsx","ioUS-BraTioUS-dataset.zip (imagesBraTioUS-public-dataset/BraTioUS-NNN_K.nii.gz), tumor-segmentation-…zip (labelsBraTioUS-public-dataset/…), BraTioUS - Metadata.xlsx",["Manufacturer","Type probe","Frecuency range"],[])],
  bp=F("dataset-level (intra-operative brain ultrasound)", {"-":"brain"}, "medium"),
  prob=["is_mask regex misses `labelsBraTioUS-public-dataset` (token 'labels' glued to the next word): the label zip is counted as 1,669 'media' images, and its archive name has neither 'mask' nor 'label' so download.py fetches it as image data"])
o("laparoscopic-gastrectomy-annotated-images", acc="WRONG/EMPTY RECORD: figshare id 058882 lists one unrelated jpg ('1471-2202-8-40-11.jpg', a BMC Neuroscience figure)", src=[S("figshare","058882",["1471-2202-8-40-11.jpg"],[])],
  prob=["registry ref '058882' (leading zero) looks truncated/mistyped: the listing is a paper figure, not 2000 laparoscopic frames (1.67 GB)"])
o("cryoet-jove-dl-training-data", src=[S("hf-path","tomogram/<Actin|Cofilactin|Membrane|Microtubule|Ribosom|TriC>/*.tiff (239 each), annotation/<same>/*.tiff (1434)",["path d1: tomogram|annotation","path d2: target class"],[])],
  prob=["the 239 tomogram slices appear under each of 6 class folders: probably the same slices repeated per target class (6x duplication): dedupe by file name/hash","body 'cells' hard may be wrong if samples are purified in-vitro reconstituted filaments (actin/cofilactin/TriC) - verify"])
o("tryp-v2-v3-2026", src=[S("zips","positive_images.zip (v1), positive_images_v2.zip, positive_images_v3.zip, ood.zip, negative_images.zip, videos.zip",["zip name","path d2: train|validation|test"],[])],
  prob=["positive_images.zip and positive_images_v2.zip have identical member counts (6,233): v1/v2/v3 are overlapping releases: use v3 (superset incl. T. cruzi) only","frames come from videos (positive_video_NNN_<frame>.jpg): near-duplicate frames"],
  note="specimen candidates (human/mouse) and stain (giemsa/unstained) not resolvable from head.")
o("celegans-embryo-tracking-confocal", src=[S("zip-path","mskcc_confocal_s{1,2,3}/images/*.tif (1200 volumes), tracks/ (txt + nuclei labels)",[],[])],
  spec=F("dataset-level (C. elegans embryo)", {"-":"c. elegans"}, "medium"))
o("murine-cell-lines-phase-xiong", src=[S("zip-path","dataset/{single_cell_patches (16,567 png), raw_images (243 bmp), instance_masks, annotations_labelme}; file_manifest.csv",["path d3"],[])],
  spec=F("dataset-level (murine cell lines)", {"-":"cell line"}, "medium"),
  prob=["single_cell_patches are crops of raw_images (same cells): choose raw_images (243) or patches, not both","dataset also carries code, checkpoints, evaluation tables (not images)"])
o("woives-uwf-octa", src=[S("zip-path","WOIVES_v1.0.zip: Image/{OCTA,SLO,CFP}/<layer: OCTA_superficial retina, OCTA_deep retina, Enface_choroid, ...>; assets/*_cfp.jpg, *_slo.jpg",["path d3: OCTA|SLO|CFP","path d4: layer"],["OCTA 994 / SLO 128 / CFP 63"])],
  sub=F("zip path level d3 (+ d4 en-face layers)", {"OCTA":"optical coherence tomography angiography","CFP":"fundus photography","SLO":"no exact class (scanning laser ophthalmoscopy): mask or near-infrared imaging","Enface_*":"OCT en-face (optical coherence tomography)"}, "high"),
  bp=F("dataset-level (UWF OCTA retina)", {"-":"retina (eye also valid)"}, "medium"),
  prob=["registry hard submodality=OCTA but the zip mixes OCTA (994), SLO (128), CFP (63) and OCT en-face choroid layers: per-sample path d3/d4 fixes this (PATH_CODES octa/cfp already handle)"])
o("camg-meibomian", src=[S("zip+xlsx","CAMG dataset.zip: img/, roi/, seg/ (100 jpg each), csv/ (gland morphology); CAMG dataset.xlsx",[],[])],
  bp=F("dataset-level (infrared meibography of the eyelid)", {"-":"eyelid (MMBU near-infrared prior: eye 14)"}, "low"),
  prob=["roi/ are cropped/overlaid derivatives of img/ (plan 1114 images vs 100 base images)"])
