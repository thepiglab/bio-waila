from lib import *

o("cystods", src=[S("csv", "cystods_split.csv / cystods.csv (HF mirror + zip xvdhy-osfstorage-archive.zip)", ["filename","class","subclass","modality","pid"], ["modality: WLC / BLC"])],
  sub=F("csv column `modality` (WLC/BLC)", {"WLC": "white-light endoscopy", "BLC": "blue-light cystoscopy -> no class in vocab: drop or mask these images"}, "high"),
  prob=["registry hard submodality=white-light endoscopy is wrong for ~5.6% of images (BLC=blue-light, 450 of 8067); BLC appears in cystods_split.csv (9/50 head rows)",
        "HF repo has no loose images: pixels are inside xvdhy-osfstorage-archive.zip (images/, segmentations/ json) -> download needs zip-range sampling", "junk file test_uploads/test_checkpoint_ping.txt"],
  note="rules.py CODES['submodality'] lacks WLC/BLC; add wlc->white-light endoscopy, blc->(exclude).")
o("papila", src=[S("hf", "ai4ophth/PAPILA-dataset (parquet)", ["retina","cup_exp1","cup_exp2","disc_exp1","disc_exp2","opht_cont","sparse text","metadata"], ["metadata: 'Eye side: OS, Age: 47, G'"])],
  bp=F("no per-sample metadata; set dataset-level hard label", {"eye": "MMBU fundus prior: eye 407 vs retina 40 of 447"}, "medium"),
  prob=["8 image columns: only `retina` is the fundus photo, cup/disc exp1/exp2 are segmentation images (names do not match the mask regex in fetch_hf; works only because `retina` is first)"],
  note="body_part candidates eye/optic disc/retina cannot be resolved per sample; recommend hard 'eye' (or multi eye+retina).")
o("openorganelle", src=[S("hf", "eminorhan/openorganelle-2d (parquet 2.67M slices)", ["volume_name","axis","slice","part_id"], ["jrc_mus-kidney","jrc_mus-liver-3","jrc_mus-choroid-plexus-3","jrc_hela-2 (cell line)"])],
  bp=F("hf column volume_name", {"jrc_mus-kidney*":"kidney","jrc_mus-liver*":"liver","jrc_mus-heart-1":"heart","jrc_mus-pancreas-4":"pancreas","jrc_mus-skin-1":"skin","jrc_mus-hippocampus/choroid-plexus":"brain (matcher misses choroid-plexus/hippocampus tokens)","jrc_hela/jurkat/macrophage/cos7/sum159":"cells"}, "medium"),
  spec=F("hf column volume_name prefix", {"jrc_mus-*":"mouse","jrc_hela|jrc_jurkat|jrc_macrophage|jrc_cos7|jrc_sum159|jrc_ctl":"cell line","jrc_fly-*":"drosophila","jrc_zf-*":"zebrafish (NOT in registry candidates)"}, "medium"),
  stain=F("none per sample; all FIB-SEM prep uses OsO4(+uranyl acetate)", {}, "low"),
  prob=["registry specimen candidates omit zebrafish (jrc_zf-cardiac) and plant/other volumes if present"],
  note="rules.py has no 'mus'->mouse token rule: volume names like jrc_mus-kidney only give the organ. Add prefix rules (mus, hela, jurkat, fly, zf, ce).")
o("cellmap-challenge-2025", src=[S("hf", "eminorhan/cellmap-2d (parquet 366k slices)", ["crop_name","axis","slice","label(Image mask)"], ["jrc_hela-2/recon-1/crop2","jrc_mus-liver-zon-2/recon-1/crop133","jrc_zf-cardiac-1/recon-1/crop3"])],
  bp=F("hf column crop_name (organ token after jrc_mus-)", {"jrc_mus-liver*":"liver","jrc_mus-kidney*":"kidney","jrc_zf-cardiac-1":"heart","jrc_hela/jurkat":"cells"}, "medium"),
  spec=F("hf column crop_name prefix", {"jrc_mus-":"mouse","jrc_hela|jrc_jurkat|jrc_sum159":"cell line","jrc_zf-":"zebrafish (not in candidates)"}, "medium"),
  prob=["`label` column is an Image (segmentation); excluded by fetch_hf regex but make sure `image` is used"], note="Same prefix-token rule need as openorganelle.")
o("scin", src=[S("csv/hf", "metadata.csv + HF columns", ["fitzpatrick_skin_type","age_group","sex_at_birth","primary_diagnosis","condition_duration"], [])],
  note="All Pass-2 fields hard in registry; metadata has no extra Pass-2 info. Mixed user phone photos (clinical photography) - consistent.")
o("pad-ufes-20", src=[S("csv", "metadata.csv", ["region","diagnostic","gender","fitspatrick"], ["region: NECK/FACE/HAND/CHEST/NOSE/EAR"])],
  note="Registry body_part hard=skin; `region` column maps to finer parts (neck, face, hand, chest, nose, ear) via matcher, but hard registry labels are kept. Optional: make body_part multi [skin, region].")
o("wbcatt-plus-2026", acc="ok (single .tar, no remote listing)", src=[S("tar", "pbcseg_final_v1.tar (HF)", [], [])],
  note="Only a tar archive (0.19 GB) + teaser.jpg; download.py tar-stream handles it. All labels hard and consistent (PBC peripheral blood smears, MGG). Cell-type annotations are not Pass-2 fields.")
o("brats-peds", src=[S("hf", "chehablab/BraTS_PED_2024 (parquet, slices)", ["t1c","t1n","t2f","t2w","seg"], [])],
  bp=F("dataset-level: skull-stripped BraTS brain MRI", {"-":"brain"}, "medium"),
  note="4 image columns = 4 MRI sequences (all structural mri); `seg` excluded. Recommend registry body_part hard 'brain'.")
o("taix-ray", src=[S("hf", "TLAIM/TAIX-Ray (parquet, configs data + original)", ["UID","Split","Age","Sex","HeartSize","PulmonaryCongestion","PleuralEffusion_*"], [])],
  bp=F("no per-sample metadata; dataset-level", {"-":"multi [chest, lung] (MMBU chest x-ray: chest,lung 271 / chest 126)"}, "medium"),
  prob=["two configs (`data`, `original`) hold the same 215k rows: use one"], note="Findings columns are diagnoses (skipped by SKIP_COLUMN).")
o("pbc-acevedo-2020", src=[S("hf", "Docty/Blood-Cells", ["label (ClassLabel)"], ["monocyte","ig","neutrophil","basophil","lymphocyte","erythroblast","eosinophil","platelet"])],
  note="All hard. Consistent with PBC dataset (MGG-stained smears).")
o("mitoem-2", acc="ACCESS PROBLEM: HF repo is README-only; data moved to Zenodo 20417683", src=[S("zenodo", "20417683 metadata.csv + zip names", ["dataset_id","organism","tissue","cell_type","modality"], ["Dataset001_ME2-Beta,Mus musculus,Pancreas (islet of Langerhans),Beta cell,FIB-SEM","Dataset004_ME2-Mossy,Mus musculus,Brain (cerebellum),ssSEM","Dataset006_ME2-Pyra,Homo sapiens,Brain (cortex)"])],
  bp=F("zenodo metadata.csv `tissue` / zip name", {"Beta":"pancreas","Podo":"kidney","Mossy":"brain","Pyra":"brain","Sperm":"sperm cells","Stem":"bone marrow","Jurkat/Macro":"cells"}, "high"),
  spec=F("zenodo metadata.csv `organism`", {"Mus musculus":"mouse","Homo sapiens":"human","Homo sapiens (Jurkat cell line)":"cell line","Homo sapiens (macrophage cell line)":"cell line"}, "high"),
  prob=["selection method hf:pytc/MitoEM2.0 is empty (README says moved to Zenodo 20417683, superseded by pytc/MitoVerse); switch ref to zenodo:20417683 (16 zips incl. OMEZarr duplicates, 17 GB+ Pyra)", "registry says submodality scanning EM; metadata modality is FIB-SEM / ssSEM / SBF-SEM (all SEM-family: ok)"])
o("ucsf-pdgm", src=[S("hf", "MedOtter/UCSF-PDGM file names + csv", ["<id>_FLAIR|T1|T1c|T2|tumor_segmentation .nii.gz","UCSF-PDGM-metadata_v5.csv"], [])],
  bp=F("dataset-level (glioma, skull-stripped brain MRI)", {"-":"brain"}, "medium"),
  note="Per-patient files FLAIR/T1/T1c/T2 (+ segmentation, exclude). All structural mri.")
o("openswisshcc-2026", src=[S("bids-json-in-zip", "sub-*.zip -> sub-*/anat|dyn|dwi/*.json", ["Modality","BodyPartExamined","SeriesDescription","ProtocolName"], ["Modality: MR","BodyPartExamined: ABDOMEN"]), S("tsv","annotation_status.tsv / participants.tsv",["sequence","Machine"],["sequence: T1 native, Arterial, Venous, Delayed, T2, DWI (TRACE), ADC"])],
  bp=F("BIDS sidecar BodyPartExamined (needs json reads inside zips)", {"ABDOMEN":"abdomen"}, "high"),
  note="BIDS: anat (T2w blade/haste), dyn (dynamic contrast T1), dwi (ADC/TRACE). derivatives.zip = masks/transforms: skip. Liver is also visible: candidate liver is true as well (multi [abdomen, liver] is defensible).")
o("galar", acc="WRONG SOURCE: figshare 31065262 holds only split CSVs (no images)", src=[S("csv","train.csv/val.csv/test.csv/galar_results.csv",["path","anomaly"],["61/frame_002385.PNG"])],
  prob=["figshare 31065262 = 'Noise corrected Galar dataset splits' (CSV only; frames referenced as <video>/frame_XXXXXX.PNG are hosted in the original Galar release by Le Floch et al. 2025, not here)","registry size 603 GB belongs to the real Galar release"],
  note="Needs the original Galar figshare/Zenodo record. Video capsule endoscopy: body_part would be GI (esophagus/stomach/small intestine/colon) - not available per frame.")
o("pan-multiplex-nimbus", src=[S("hf-zip", "JLrumberger/Pan-Multiplex: mibi_decidua.zip (listable), codex_colon.zip.001.. (multi-volume, not listable)", ["gt_pred_ie_consolidated.csv: dataset,fov,labels","image_data/<fov>/<marker>.tif"], ["dataset: codex_colon"])],
  bp=F("csv column `dataset` / zip name", {"codex_colon":"colon","mibi_decidua":"uterus (decidua) - no exact class"}, "medium"),
  prob=["MIBI subset is multiplexed ion beam imaging (mass spec, metal tags), not fluorescence microscopy -> registry modality/submodality wrong for those","images are one grayscale TIFF per marker channel (Au.tif, CD3.tif ...): no RGB image; need channel selection/stack","multi-volume zip (.001-.009) cannot be remote-listed: full download"])
o("lidc-idri", src=[S("csv","aux/LIDC-IDRI_MetaData.csv, scans.csv",["Modality","Study Description","Series Description"],["Modality: CT 40 / DX 9 / CR 1"]) ],
  bp=F("dataset-level: lung nodule CT", {"-":"lung (or multi [chest, lung])"}, "medium"),
  note="images/ = 1018 CT volumes (nii.gz). MetaData.csv lists DX/CR series (chest radiographs) that are NOT among the 1018 nifti volumes; do not use its Modality column to relabel.")
o("isic-2017", src=[S("hf","MedOtter/ISIC2017",["image","mask","sex","age_approximate"],[])], note="All hard, consistent (dermoscopy).")
o("vestibular-schwannoma-seg", src=[S("hf-path","dataset/vs_gk_NNN/{t1,t2}/…nii.gz; preview.parquet",["path d3: t1|t2","modality: ceT1|hrT2"],[])],
  bp=F("dataset-level (cerebellopontine angle / vestibular schwannoma, head MRI)", {"-":"brain (low confidence; ear/head also in candidates)"}, "low"),
  note="Per-sample sequence from d3 (t1=contrast T1, t2=hrT2): both structural mri. masks are in separate files.")
o("spherohq", src=[S("csv","manifest.csv (also in SpheroHQ.zip)",["path","split","type","cell_line","source_dataset"],["cell_line: BxPC-3","type: image/mask"])],
  spec=F("manifest.csv `cell_line` present for every sample", {"<any cell line name>":"cell line"}, "high"),
  note="zip d3 = images/masks: mask half excluded by is_mask (verify `type` column). Brightfield spheroid images.")
o("glioma-warsaw-multiparametric-2026", src=[S("zip-names","<id>_<t>_brain_{t1c,t1n,t2f,t2w,swi,adc,dwi}.nii.gz",["file suffix"],[]), S("csv","1_0_clinical_molecular_data.csv, *_scans_meta.csv",["ScannerName","FieldStrengthLabel"],[])],
  bp=F("dataset-level: skull-stripped brain MRI (dir `skullstripped_*`)", {"-":"brain"}, "medium"),
  note="Raw + coreg x pre/postop zips; masks only in 3_2_postop_masks.zip (excluded).")
o("vetxray-canine-feline-thorax", src=[S("xlsx","File list with tags.xlsx",["FileName","specie","Projection","Quality","TAG","NOTE"],["specie: Dog/Cat","Projection: LL/DV","Quality: correct/positioning/underexposed/exclude"])],
  spec=F("xlsx column `specie` joined by FileName", {"Dog":"dog","Cat":"cat"}, "high"),
  bp=F("dataset-level thorax radiographs (DICOM BodyPartExamined unreadable remotely)", {"-":"thorax (or multi chest/heart/lung)"}, "low"),
  prob=["every zip has a mirrored __MACOSX/._*.dcm resource-fork entry per file (half the members): filter '__MACOSX' and '._' names or half the sampled 'dicoms' are junk","Quality=exclude/positioning rows should be dropped"],
  note="Projection LL/DV are not in rules.py (view column). They are lateral/dorsoventral, not submodality.")
o("spider-lumbar-mri", src=[S("hf+csv","parquet + overview.csv",["sequence: t1/t2/t2_SPACE","BodyPartExamined: LSPINE/TSPINE","SeriesDescription"],[])], note="All hard, consistent (spine MRI).")
o("spinal-multiple-myeloma-seg", src=[S("hf-path","dataset/Myel_NNN/Myel_NNN_convct.nii.gz (+lesions/spine segmentation)",["Table_of_acquisition_parameters.tsv: ScanOptions=HELIX"],[])],
  bp=F("dataset-level: whole-spine CT", {"-":"spine (+axial skeleton)"}, "medium"),
  note="ScanOptions=HELIX confirms registry spiral ct. convct recon vs monoenergetic recon exist in the project but only convct appears in the repo.")
o("rodent-estrous-cytology-compendium-hf", acc="ok but images NOT in HF repo (image_url -> EBI BioStudies S-BIAD545 etc.)", src=[S("hf-parquet","CytologySample/CytologyImage/CellAnnotation/dataset",["cytology_sample.stain: shorr...","notes: raw_strain=Sprague Dawley WT","cytology_image.image_url"],["stain: shorr","image_url: https://ftp.ebi.ac.uk/biostudies/.../metestrus/*.JPG"])],
  stain=F("hf cytology_sample.stain joined via cytology_image.sample", {"shorr":"(no class; closest = other/unmapped)","giemsa|H&E":"map if present"}, "medium"),
  spec=F("cytology_sample.notes raw_strain / source dataset (Sprague Dawley -> rat; C57BL/6J -> mouse)", {"Sprague Dawley":"rat","C57BL/6J":"mouse"}, "medium"),
  prob=["no image column: fetch_hf finds no PIL images ('no image column found'); needs a URL downloader for image_url (EBI FTP, GitHub, MDPI)","7 sub-corpora, many licenses 'TODO_VERIFY_LICENSE'","Shorr/Papanicolaou stains: only papanicolaou is in the vocab"])
o("protlocnet-2026", acc="ok (single 19.6 GB tar.gz: structure unknown remotely)", src=[S("tar.gz","ProtLocNetData.tar.gz",[],[])],
  prob=["tar.gz cannot be range-listed; tar-stream samples the first N members in archive order (not spread across folders)"],
  note="Cannot audit metadata without downloading; likely multi-channel confocal (protein/nucleus/ER/MT) with per-image csv.")
o("totalsegmentator-v3", src=[S("csv-in-zip","Totalsegmentator_dataset_v300/meta.csv",["study_type","pathology_location","institute","manufacturer","scanner_model","convolution_kernel"],["study_type: ct polytrauma / ct abdomen-pelvis / ct angiography abdomen-pelvis-…"])],
  bp=F("meta.csv `study_type` (+ pathology_location)", {"ct abdomen-pelvis":"abdomen + pelvis","ct thorax-abdomen":"thoracoabdominal region","ct head/neck":"head and neck","ct polytrauma":"whole body"}, "medium"),
  prob=["matcher bug: 'ct angiography abdomen-pelvis-…' maps to x-ray imaging via the `angiogra\\w*` rule although it says CT","one zip of 35 GB with 226,863 segmentation niftis + ~1.9k ct.nii.gz: spread_sample over 228k members is slow; filter members ending /ct.nii.gz"],
  note="Mask-heavy archive: only s*/ct.nii.gz are images.")
o("miqr-cc-ercp-fluoroscopy", src=[S("csv-in-zip","MIQR-CC-Dataset/metadata.csv",["raw_image_path","processed_image_path","Label","Keep","equipment_model"],["Label: Unlabelled/Malignant Stricture/Normal/Lithiasis","Keep: Keep/Discard"])],
  bp=F("no per-sample; ERCP biliary/pancreatic duct images", {"-":"abdomen (safest) / multi [abdomen,pancreas,gallbladder]"}, "low"),
  prob=["zip holds both raw/ and processed/ copies of every frame (19k each): sample one only","metadata Keep=Discard rows (about half in head) should be filtered","5 loose PNG examples at figshare top level"])
o("afids-anatomical-fiducials", src=[S("zip","data-part001.zip Images/*.nii.gz (72)",[],[])],
  bp=F("dataset-level T1w brain MRI", {"-":"brain"}, "medium"), note="HF datasets-server fails (500); file listing works. Only 72 volumes.")
o("embryonet-zebrafish", acc="ok but record is code + demo only", src=[S("zip","mueller-lab/EmbryoNet-Release_1_0_1.zip",["Demo/experiment_1/images (720 jpg)","benchmark json"],[])],
  spec=F("folder name / dataset title", {"ZebraFish":"zebrafish"}, "high"),
  prob=["zenodo 7430639 = 0.62 GB GitHub code release + 725 demo jpgs; the registry's 1347 GB / 2110 planned images do not exist here","plan_images (2110) > available (~725)"])
o("openmind", src=[S("hf-bids","OpenMind/dsNNNNNN/sub-*/anat/sub-*_<suffix>.nii.gz",["BIDS suffix T1w/T2w/FLAIR …"],[])],
  bp=F("dataset-level (BIDS anat brain MRI)", {"-":"brain"}, "medium"),
  prob=["HF file listing is capped at exactly 100,000 entries (truncated); 55,908 are masks (fb_mask/deface_mask)","may include non-structural suffixes (e.g. angio, swi, dwi b0) - filter by BIDS suffix"])
o("ivygap-radiomics", src=[S("hf-path","W<n>/W<n>_{flair,t1,t1gd,t2}.nii.gz (+seg-CWRU/UPenn)",["ivygap_metadata.csv: location, gender"],[])],
  bp=F("dataset-level (GBM brain MRI)", {"-":"brain"}, "medium"))
o("tissue-fold-teaching-slides", src=[S("zip-path","images.zip: images/<Organ>/{clean,tissue_fold}/*.jpg",["path d2: Brain|Kidney|Liver|Testis|Small_Intestine","metadata.csv: organ,class"],[])],
  bp=F("path level d2 / metadata.csv `organ`", {"Brain":"brain","Kidney":"kidney","Liver":"liver","Testis":"testis (no class -> reproductive system?)","Small_Intestine":"small intestine"}, "high"),
  note="masks.zip excluded. Specimen not stated in head (masked); likely mouse or human teaching slides - check README.")
o("ideas-epilepsy", src=[S("bids","sub-N/anat/sub-N_{T1w,FLAIR}.nii.gz (+json)",["BIDS suffix"],[])], bp=F("dataset-level epilepsy brain MRI", {"-":"brain"}, "medium"))
o("nih-chestxray14", src=[S("csv","data/Data_Entry_2017_v2020.csv",["View Position: PA/AP","Finding Labels"],["PA","AP"])],
  bp=F("dataset-level", {"-":"multi [chest, lung]"}, "medium"),
  prob=["12 image zips contain `__MACOSX/._*.png` entries (counted as images by kind()): filter","HF repo is script-based (NIH-Chest-X-ray-dataset.py): datasets-server returns 501; load_dataset(streaming) fails -> falls back to file listing (ok)"],
  note="View Position PA/AP -> chest x-ray via CODES (matches registry).")
o("parasitobank-2025", src=[S("hf","andsfonseca/parasito-bank",["image","objects(bbox)"],[])], note="All hard, consistent.")
o("popsicle-cryoet-benchmark", acc="ok but tomograms are NOT in the HF repo (Croissant metadata -> CZ cryoET data portal)", src=[S("hf-config","bacterial/motorbench/phantom/yeast Croissant features.csv",["run","voxel_size","tomo_type","url"],[])],
  spec=F("HF config / folder name", {"bacterial":"bacteria","motorbench":"bacteria","phantom":"phantom","yeast":"yeast"}, "high"),
  prob=["HF repo only has Croissant/copick metadata; pixel data lives on the CZ cryoET portal (external URLs in features.csv): download.py cannot fetch","size -1"])
o("synthstrip-dataset", src=[S("csv","scan_metadata.csv",["modality","modality_full","modality_type","subset"],["modality: t1,t2,pd,epi,qt1,mra,flair; modality_type: MRI"])],
  bp=F("dataset-level head MRI", {"-":"brain"}, "medium"),
  prob=["upstream SynthStrip release also includes CT and PET scans (README: 'MRI, PET, and CT scans'); check modality_type: any CT/PET rows would contradict hard MRI/structural labels (HF subset's 30 head rows are all MRI)"],
  note="modality values like epi (ASL EPI), mra, qt1 are non-structural MR; consider mapping epi/mra to modality MRI only and masking submodality.")
o("cardiag", src=[S("zip","CARDIAG.zip: <patient>/{img.png,NN.png,metadata.json}, other_keyframes/",["metadata.json: sod,sid,spacing,primary_angle,secondary_angle,patient_id,center_id"],[])],
  bp=F("dataset-level coronary angiography", {"-":"coronary arteries (+heart, blood vessels)"}, "medium"),
  prob=["numbered pngs (03.png, 16a.png) next to img.png are probably vessel masks/overlays: not caught by is_mask -> risk of training on masks","models.zip (17 GB of .ckpt) is not data: download.py archives branch sorts biggest first and would fetch it"])
