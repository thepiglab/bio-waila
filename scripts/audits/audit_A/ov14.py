from lib import *

o("hspc2-brightfield-hspc", src=[S("zip+csv","hspc2.zip: images/NNNN.tif, masks/NNNN_label.tif; manifest.csv",["n_cells"],[])],
  bp=F("dataset-level (haematopoietic stem/progenitor cell culture)", {"-":"cells"}, "low"))
o("cradi", src=[S("zip+xlsx","CHARDI_test.zip (10,440 png); CHARDI_label.xlsx (7 MB, skipped by the 3 MB head cap)",[],[])],
  bp=F("dataset-level (chest radiographs)", {"-":"multi [chest, lung]"}, "medium"), note="Only the 'test' release of CheXpert-derived CHARDI images is on this record.")
o("syke-plankton-ifcb-2025", src=[S("zip-path+txt","<Taxon>/D<date>T<time>_IFCB114_NNNNN.png; SYKE-plankton_IFCB_2025_classlist.txt; sample-metadata.txt",["path d2 = taxon"],[])],
  spec=F("registry masked: taxa are phytoplankton/cyanobacteria, no vocab class", {}, "low"), note="Same situation as whoi-plankton and nes-plankton.")
o("popeye-nir-ocular", src=[S("zip-path+csv","PopEYE_2022_images/<correct|closed|up|right|left|down>/PIC__NNNN.png; labels.csv",["label = gaze/position class"],[])],
  note="All hard consistent (NIR eye images); folder = eye-position QC class (not Pass-2).")
o("bacteria-species-mother-machine-7", src=[S("zip+csv","replication_package_seven_species.zip (122 GB): data/img/<exp>/<Pos>/..png, data/segmentation, labeling/all_data_multi_and_empty.csv, scripts/labels.csv",["species: kleb|acin|psed|ecol|ente|...","experiment","position"],["species: acin, kleb, psed, ecol, ente"])],
  spec=F("labels.csv `species` joined by path", {"ecol":"e. coli","acin|kleb|psed|ente|...":"bacteria"}, "medium"),
  prob=["122 GB single zip incl. code/checkpoints/plots; only data/img/** are images (is_mask catches segmentation/)","images are consecutive time frames of mother-machine traps (near-duplicates)"])
o("spheromix", src=[S("csv","manifest.csv, known_overlap.csv (also inside SpheroMix.zip)",["path","split","type","cell_line","source_dataset","dup_group_id","cross_split_dup"],[])],
  spec=F("manifest.csv `cell_line` present for every sample", {"<any cell line name>":"cell line"}, "high"),
  prob=["SpheroMix is a superset of SpheroHQ (identical file names such as mcf_mcf_pokus2_po_1-2024_11_30_21_46_33_G3.png appear in both zips): selecting both duplicates data; manifest dup_group_id/cross_split_dup columns allow dedupe"])
o("angiocad", acc="ok but 4-part .rar (15 GB): no remote listing, unsupported archive", src=[S("rar+xlsx","AngioCAD_Dataset.part{1..4}.rar; AngioCAD_Labels.xlsx; AngioCAD_Features.xlsx",["Right/Left Coronary Series","stenosis %"],[])],
  bp=F("dataset-level (coronary angiography)", {"-":"coronary arteries (+heart)"}, "medium"),
  prob=["multi-volume .rar: needs all 4 parts + unrar; download.py cannot sample it"])
o("hcfmrp-covid19-lid", acc="ok but single 10.8 GB .rar", src=[S("rar","HCFMRP COVID-19 & LID (v1).rar",[],[])],
  bp=F("dataset-level (chest radiographs)", {"-":"multi [chest, lung]"}, "medium"), prob=["only a .rar archive: no remote listing, cannot be sampled"])
o("dermlens-5k", acc="RESTRICTED: zenodo 22945830 access_right=restricted", src=[], prob=["request access from authors or drop"])
o("ptc-thyroid-fnac-elazig-2025", acc="EMBARGOED: zenodo 17311696", src=[], prob=["embargoed until release date: drop for now"])
o("detecdiv-yeast-lifespan", src=[S("zip-path","lifespanAnnotation.zip: {trainingdataset,testdataset}/images/<large|dead|small|unbud|clog|empty>/Pos*_frame_NNNN.tif; timeseries/*.mat",["path d3 = cell state class"],[])],
  note="All hard; 222k frames from microfluidic time-lapse (near-duplicates): stratify across states/positions.")
o("zebrafish-pronephros-cystic", src=[S("csv+zip","images.zip (290 tif); Annotations-*.csv (Folder,Image,Slice,Category Normal|Cystic)",["Category"],[])],
  note="All hard consistent; trainedModel.zip is weights (archive branch would sort it first by size: 110 MB vs 193 MB images: ok).")
o("tracking-one-in-a-million", src=[S("zip-path","ctc_format.zip: 00..04 (800 frames each) + 00_GT..04_GT/{TRA,SEG}; videos.zip (5 avi)",[],[])],
  note="GT folders are label images (is_mask 'GT' catches them). Specimen bacteria (hard).")
