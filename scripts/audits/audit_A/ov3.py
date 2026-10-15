from lib import *

o("brats-tcga-segmentations", src=[S("hf-path","dataset/{gbm,lgg}/TCGA-NN-NNNN/*.nii.gz; TCGA_*_radiomicFeatures.csv",["path d2: gbm|lgg"],[])],
  bp=F("dataset-level (TCGA GBM/LGG brain MRI)", {"-":"brain"}, "medium"), note="gbm/lgg folders are tumor grade, not Pass-2 fields.")
o("tb-sputum-ufam-figshare-plus-2025", acc="ok (single 127 GB tar.gz; cannot be range-sampled)", src=[S("tar.gz","ODS1.tar.gz + 'ODS1 - folders directory.pdf'",[],[])],
  prob=["only a 127 GB tar.gz: tar-stream takes the first N members in archive order (not spread across folders); budget likely exceeded; plan_images 2000 mixes 2D and 3D (z-stack) data"],
  note="Registry labels all hard (sputum smear, Ziehl-Neelsen, brightfield). specimen candidates bacteria/human are not resolvable from head; the directory PDF describes folders.")
o("brats-africa", src=[S("hf-path","ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/BraTS-SSA-NNNNN-000/*.nii.gz",["file suffix t1n/t1c/t2w/t2f/seg"],[])],
  bp=F("dataset-level (adult glioma, sub-Saharan Africa brain MRI)", {"-":"brain"}, "medium"))
o("harris-2015-hippocampal-neuropil", acc="ok but pixels are zarr chunks inside .zarr.zip", src=[S("hf-zarr","data/{apical,oblique,spine}.zarr.zip (raw/s0..s2, labels, labels_mask)",[],[])],
  sub=F("registry masked: serial-section TEM (no TEM class in vocab)", {}, "low"),
  stain=F("dataset-level: ssTEM sections post-stained with uranyl acetate + lead citrate (Harris 2015)", {"-":"multi [uranyl acetate, lead citrate]"}, "medium"),
  prob=["zarr chunk members ('.0', '.1', ...) are not image files: download.py zip-range sampling would pick raw chunks; needs a zarr reader (zarr over ZipStore) to cut 2D slices","HF info 501 (script-based); size -1","vocab has no transmission EM submodality: submodality stays masked"])
o("bcdd-sust-dengue-2026", src=[S("zip","BCDD_SUST.zip: train|valid|test/{images,labels}; image_metadata.xlsx, metadata.xlsx",["Dengue(NS1)","Gender","Age"],[])],
  note="All hard fields consistent (blood smear). Slide-level dengue/NS1 status is a diagnosis. YOLO label txt files are not images. Stain masked: README/protocol pdf may name it (Leishman/Giemsa) - not readable from head.")
o("rhuh-gbm", src=[S("hf-path","RHUH-NNNN/<timepoint 0|1|2>/RHUH-NNNN_<t>_{adc,flair,t1ce,t1,t2,...}.nii.gz; subjects_manifest.json",["file suffix"],[])],
  bp=F("dataset-level (GBM brain MRI)", {"-":"brain"}, "medium"), note="Sequence suffix includes adc (diffusion-derived): still MRI; maps to structural mri only loosely.")
o("deepcell-brightfield-single-cells", src=[S("hf-zip","images_bf_ic.zip (12000 png), embeddings_bf_ic.parquet",[],[])],
  prob=["zip contains a parallel `__MACOSX/._*.png` entry for every image (12001 junk members): filter","embeddings_bf_ic.parquet holds embeddings, not images (size -1)"],
  note="Immune-cell single-cell crops; body/specimen candidates (blood/cells; cell line/human) not resolvable from head. Likely human PBMC-derived: recommend body 'blood cells'?? (not in candidates) - keep candidates.")
o("hep2-cell-cycle-2026", src=[S("zip-path","HEp-2-CNN-D.zip / HEp-2-ExpD.zip: <Phase>/<Phase>_NNNN.webp; DAPI_Full_FOV.zip; HEp-2 dataset-metadata.txt",["path d1: Interphase|Prophase|Metaphase|Anaphase|Telophase|Artefact|Triple"],[])],
  spec=F("dataset-level: HEp-2 is a human epithelial CELL LINE", {"-":"cell line"}, "high"),
  bp=F("dataset-level", {"-":"cells"}, "medium"),
  prob=["100k + 10k webp crops: Artefact (6.5k) and Triple (2.5k) folders are not clean single cells: drop","DAPI_Full_FOV.zip is full-field images (1000)"])
o("clinical-bacteria-gram-mrose-2024", src=[S("zip","RawImageDataSet.zip (1705 jpg); DeepDataSet.zip (640DataSet/images, json, labels)",[],[])],
  prob=["DeepDataSet.zip is crops/derivatives of the raw images (640DataSet, DetectionDataSet, SegmentationDataSet): sample from RawImageDataSet only to avoid triplicates"],
  note="specimen candidates (bacteria/human) not resolvable from head.")
o("biotisr", acc="ok but selected record has no images (supplementary only)", src=[S("xlsx","dataset_urls.xlsx + Supplementary zip/imaging_conditions.xlsx",["Structure","Imaging method","Sample","Label"],["GI-SIM, 3D-SIM, TIRF-SIM, NL-SIM","COS-7, SUM-159","Lifeact-mEmerald, 2×mEmerald-Tomm20, Lamp1-Halo"])],
  sub=F("imaging_conditions.xlsx `Imaging method`", {"TIRF-SIM":"total internal reflection fluorescence microscopy (tirf)","GI-SIM / 3D-SIM / NL-SIM":"no vocab class: mask (or epifluorescence?)"}, "medium"),
  stain=F("imaging_conditions.xlsx `Label` (fluorescent protein fusions)", {"mEmerald fusions":"gfp","Lifeact-*":"gfp (NOT phalloidin)","Lamp1-Halo":"(Halo dye, no class)"}, "medium"),
  spec=F("imaging_conditions.xlsx `Sample`", {"COS-7":"cell line","SUM-159":"cell line"}, "high"),
  prob=["zenodo 15354981 contains only code, OTF files and a spreadsheet of URLs: image data are in other Zenodo records (dataset_urls.xlsx)","registry stain hard=phalloidin is wrong (labels are mEmerald/Lifeact/Halo fusions, live-cell SIM)","registry submodality hard=TIRF is only true for TIRF-SIM subsets; others are GI-SIM/3D-SIM/NL-SIM"])
o("tvus-ovarian-carcinoma-meir-2026", src=[S("zip-path","images_{normal,benign,malignant}.zip: images/<class>/NNN.vK.J.png; yolo_labels.zip; morphological_features.zip",["path d2: normal|benign|malignant"],[])],
  bp=F("dataset-level (transvaginal pelvic US)", {"-":"multi [ovary, pelvis]"}, "low"),
  prob=["record also lists 4 model checkpoint files (.pth/.pt, ~0.9 GB): not data, avoid in archive sort"], note="class folder = diagnosis (not Pass-2).")
o("nsclc-radiogenomics", src=[S("hf-dicom","images/R01-NNN/<SeriesUID>/<instance> (593 extension-less DICOM + 156 .dcm); series_to_patient.json",["Modality","SeriesDescription","StudyDesc"],[])],
  bp=F("dataset-level (NSCLC chest CT)", {"-":"lung (or multi [chest, lung])"}, "medium"),
  prob=["most DICOM files have NO extension: pass2data.files.kind() returns 'other' so download.py would skip 593 of 749 files","NSCLC Radiogenomics TCIA series include PET (Modality PT) next to CT: filter with series_to_patient.json Modality=CT to keep registry modality valid"])
