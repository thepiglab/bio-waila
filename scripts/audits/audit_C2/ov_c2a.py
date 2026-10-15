from lib import *

o("grazpedwri-dx", acc="ok (HF mirror MuhammadJazib/grazpedwri_dx is a 1,010-image subset, no labels; the full 20,327-image release is figshare 14825193 -> re-route)",
  src=[S("hf", "MuhammadJazib/grazpedwri_dx (parquet, 2 shards, 0.82 GB)", ["image"], ["splits: train 1010 rows"]),
       S("csv", "figshare 14825193 dataset.csv (NOT in the selected HF repo; listed + 337-row head read)", ["filestem","patient_id","study_number","timehash","gender","age","laterality","projection","initial_exam","ao_classification","cast","diagnosis_uncertain","osteopenia","fracture_visible","metal","pixel_spacing","device_manufacturer"],
         ["filestem: 0001_1297860395_01_WRI-L1_M014", "projection: 1/2", "laterality: L/R", "pixel_spacing: 0.144 (all)", "device_manufacturer: Siemens (all)"])],
  bp=F("file-name token `WRI` (wrist) in every figshare image name; dataset is pediatric wrist trauma only", {"0001_..._WRI-L1_M014.png": "wrist", "0002_..._WRI-R1_F012.png": "wrist"}, "high"),
  prob=["HF mirror holds only 1,010 images (0.82 GB, no csv/yolo labels) vs 20,327 in the release: plan_images=2000 cannot be met and `registry full_size_gb=16.25` refers to the figshare release",
        "figshare 14825193 = dataset.csv + folder_structure.zip + images_part1..4.zip (4 x ~4 GB, 5,031 PNG in part1): download.py zip-range sampling works on it",
        "registry candidates hand / upper limbs / appendicular skeleton are all coarser than 'wrist'; every file name carries WRI; rules.py has no `wri` token (match_value('WRI-L1') -> {})"],
  note="Fix: route to figshare 14825193 and set body_part hard wrist (mmbu-public has 76 appendicular-skeleton rows but only 6 are x-ray, so no wrist prior exists). Also add a `wri` -> wrist token if candidates are kept.")

o("hushem", acc="ok (HF repo is a VQA/MCQ re-format of HuSHeM: 328 rows of 131x131 images, repeats of the 216 originals)",
  src=[S("hf", "ZYXue/HuSHeM (parquet; train 256 + validation 72 rows)", ["image","question","pure_question","reasoning_path","answer","choices","is_correct_prompt"], ["answer: Sperm (every row)", "choices: ['Bacteria','Sperm','Cancer cells','Kidney cells'] (4 permutations of the same image)"])],
  prob=["MCQ/VQA re-format of HuSHeM: 328 rows (256 train + 72 validation) but the original HuSHeM has only 216 images (Mendeley), so >=112 rows are repeats; first-rows show groups of 4 consecutive rows with identical question + choices (option permutations, is_correct_prompt marks one). Unique image count not verifiable without reading pixels",
        "`answer` is the generic word 'Sperm', not the head-morphology class; no Pass-2 metadata beyond the registry; fetch_hf saves the question/choices strings into hf_rows.jsonl (harmless: SKIP_COLUMN covers question/answer/prompt)",
        "selection rules say no VQA corpora; pixels are real HuSHeM but 4x duplicated (prepare.py dHash dedup within dataset removes the repeats)"],
  note="Expect <=216 unique 131x131 images; keep as a tiny set or drop. mmbu-public sperm rows (93) carry domain [andrology, urology]: add urology. Stain diff-quik is true for HuSHeM per the original paper but mmbu-public leaves stain masked for sperm.")

o("palm", src=[S("hf", "MedOtter/iChallenge-PALM19 (parquet; train/validation/test 400 each)", ["image_id","image","pm_label","fovea_x","fovea_y","fovea_visible","device","shooting_center","width","height","id_prefix","disc_mask","has_disc_mask","atrophy_mask","detachment_mask"], ["device: Zeiss Visucam 500 / Canon CR-2", "shooting_center: fovea / optic_disc", "pm_label 1/0"])],
  prob=["image columns disc_mask/atrophy_mask/detachment_mask are excluded by fetch_hf regex (mask|label|seg|gt); only `image` is used: ok"],
  note="All six fields hard/multi and consistent (README: color fundus, 720 myopia-clinic subjects, CC0). device column is correctly skipped by SKIP_COLUMN (Zeiss Visucam would not map).")

o("bbbc039", acc="ok (HF mirror, but ~half of the streamed rows are label masks, see problems)",
  src=[S("hf", "niccogreek/bbbc039-cellpose (imagefolder of 400 TIFF)", ["image","label (ClassLabel: test/train/val)"], ["files: test/IXMtest_A09_s1_w1<GUID>.tif and ..._masks.tif side by side"]),
       S("folder", "repo paths", ["d1: train 200, test 100, val 100 files"], ["each image has a `_masks.tif` partner"])],
  prob=["HF imagefolder exposes every .tif as one row (train 200 + validation 100 + test 100 = 400 rows = 200 images + 200 `_masks.tif`): fetch_hf streams PIL images, cannot see file names and its mask regex only looks at column names, so ~50% of the saved 'images' are 16-bit instance label maps (prepare.informative() may not reject them)",
        "`label` ClassLabel is just the split name (train/val/test): not a Pass-2 field",
        "original BBBC039 is available as a direct zip: https://data.broadinstitute.org/bbbc/BBBC039/images.zip (77.9 MB, HEAD 200) + masks.zip (2.7 MB, not needed); the other BBBC entries in the registry already use method direct"],
  note="Fix: route to direct images.zip. Fields hard and correct (U2OS, Hoechst, epifluorescence); prior: mmbu-public Hoechst rows use specimen kc167 cells / vero / human for other BBBC sets, so U2OS -> human matches the cell-row convention.")

o("kromp-human-blastocyst-gardner", src=[S("zip", "Blastocyst_Dataset.zip (figshare 20123153, 0.58 GB)", ["Clincial_annotations.csv: Image, Fond, d, EXP_silver, ICM_silver, TE_silver, AMH, Age, Endo, COC, MII, SS, HA, LB", "Gardner_test_gold.xlsx", "Gardner_train_silver.csv"], ["Images/0001_01.png ... 2344 PNG", "EXP/ICM/TE = Gardner grades 0-4/ND (diagnostic, not Pass-2)"])],
  spec=F("registry candidates embryo/human; every image is a day-5 human blastocyst (record description) but mmbu-public labels contrast-microscopy embryo rows specimen 'mouse embryo cells' (17 rows), so 'human' vs 'embryo' is a vocabulary choice", {"day-5 human blastocysts (IVF)": "human (primary) / embryo"}, "medium"),
  note="Nothing per-sample. Images only (no masks); 2,344 PNG vs plan 2000. clinical csv has Age (maternal) / Endo / COC columns that are not image metadata.")

o("kvasir-capsule", acc="ok (HF zip mirror 'processed': 158,932 jpg frames in 1000 anonymous folders, not the original 47k labelled frames)",
  src=[S("zip", "Kvasir_Capsule_processed.zip (HF, size not reported by API)", ["d2 folder 00001..01000 (ClassLabel with 1000 numeric names)", "Kvasir_Capsul_list.txt (member list)"], ["Kvasir_Capsule_processed/00710/000091.jpg"])],
  bp=F("none per sample: numeric clip/cluster folders; README is empty. Capsule traverses stomach, small bowel and colon", {"gastrointestinal tract": "MMBU WLE GI rows use 'gastrointestinal tract' (35 of 163) / colon (57)"}, "low"),
  prob=["HF README is empty (license header only apache-2.0, registry says CC-BY-4.0); the 1000 numeric folders are not the 14 Kvasir-Capsule finding classes: the 'processed' meaning is undocumented (resize? subsampled videos?)",
        "no repo file sizes in the plain listing (size -1); load_dataset streaming on a single-zip repo may or may not work: fetch_hf falls back to remotezip sampling of the zip (listing via HF resolve URL works, central directory read fine)",
        "member names carry no body-part information; dataset-level candidates stay [colon, GI tract, small intestine, stomach]"],
  note="Group vce-anomalynet (8 capsule datasets) keeps it in one split. Possible dataset-level choice: hard 'gastrointestinal tract'.")

o("oct2017-kermany", src=[S("hf", "zacharielegault/Kermany2017-OCT (parquet, 17 shards)", ["image","label (ClassLabel CNV/DME/DRUSEN/NORMAL)"], ["train 108309 rows, test 1000"])],
  note="Labels hard and correct (Kermany 2018 OCT, Heidelberg Spectralis). label = diagnosis (SKIP). Same images as octmnist (same group). No fix.")

o("crossmoda-2021", src=[S("hf", "MedOtter/crossmoda2021 (train.jsonl + source_training/*.nii.gz)", ["image","mask","label (list)","modality","dataset","official_split","patient_id"], ["modality: MRI", "label: ['vestibular schwannoma','cochlea']", "files crossmoda_100_ceT1.nii.gz / _Label.nii.gz"])],
  bp=F("dataset-level: head ceT1 MRI of vestibular-schwannoma patients (internal auditory canal); cochlea is the 2nd segmentation target, not the imaged organ", {"head MRI slices": "brain (mmbu-public structural-mri brain rows 80)"}, "medium"),
  prob=["only 105 ceT1 volumes (+105 _Label masks, excluded by is_mask: `_Label.` matches) => at SLICES_PER_VOLUME=4 only ~420 images vs plan_images 2000 (state will be 'partial')",
        "jsonl image paths (data/nii/CrossMoDA2021/source_training/...) do not exist in the repo (files are at source_training/...): irrelevant for download.py (file listing), but metalabel hf_rows would not join",
        "only source (ceT1) training set is present; no hrT2 target volumes"],
  note="HF load_dataset on train.jsonl succeeds but has no PIL column -> fetch_hf falls back to the file listing, as intended.")

o("cmmd", src=[S("hf", "MinhHuy1304/CMMD (TCIA DICOM tree + CMMD_clinicaldata_revision.xlsx + metadata.csv)", ["xlsx: ID1, LeftRight, Age, number, abnormality, classification, subtype", "metadata.csv: PatientID, StudyInstanceUID, SeriesInstanceUID, Collection, FileSize, DownloadURL, ..."], ["ID1 D1-0001, LeftRight R, abnormality calcification, classification Benign", "dcm path: .../cmmd/D1-0021/<study>/<series>/1-1.dcm"])],
  prob=["HF repo carries macOS AppleDouble junk (`._*` for every file and folder, incl. 5,202 `._*.dcm` of 4 KB): is_junk() filters them (kind=junk), so media=5,202 real DICOM",
        "same images as tompei-cmmd (HF MedOtter/TOMPEI-CMMD, lesion segmentations) - both selected, same group so same split; consider keeping only one",
        "datasets-server profile is the metadata.csv (no image column) -> fetch_hf falls back to file listing"],
  note="DICOM header (ViewPosition CC/MLO, Laterality, BodyPartExamined) readable after download; registry labels all hard and correct.")

o("biomeditech-rpe-maturation", src=[S("zip", "RPE_dataset.zip (3.14 GB) + ToolRPE.rar (code, 8 MB)", ["Subwindows/ (1862 tif: <initial class>_<image no>_<subwindow>_<class>_<inclusion bit>.tif)", "Images/ (195 full 2560x1920 tif)", "license.txt (CC BY 4.0)"], ["classes Fusiform/Epithelioid/Cobblestone/Mixed are in the file names (maturation, not Pass-2)"])],
  bp=F("dataset-level; mmbu-public phase-contrast rows are exactly this kind of data: 13 rows, body_part retina, specimen human, domain biology", {"hPSC-derived RPE cultures": "retina"}, "high"),
  prob=["Images/ (195 full frames) are the parents of the Subwindows/ crops (16 per frame): both would be sampled; plan_images 1862 = Subwindows count (use exclude_paths for Images/; note sample_zip_remote does not apply exclude_paths, only prepare does)",
        "registry domain [biology, ophthalmology]: mmbu-public phase-contrast/retina rows have domain biology only",
        "ToolRPE.rar is code (skipped by archive_rank _ARCH_BAD? name has 'Tool', not matched: could be downloaded whole, 8 MB, harmless)"],
  note="Fix: body_part hard retina, domain [biology], exclude Images/.")

o("cbis-ddsm", acc="ok (HF DICOM tree, 163 GB; two thirds of the DICOM are ROI masks / cropped patches)",
  src=[S("csv", "calc_case_description_{train,test}_set.csv, mass_case_description_{train,test}_set.csv (loose in the repo)", ["patient_id","breast density","left or right breast","image view","abnormality id","abnormality type","calc type | mass shape","calc distribution | mass margins","assessment","pathology","subtlety","image file path","cropped image file path","ROI mask file path"], ["image view: CC / MLO", "left or right breast: LEFT / RIGHT"]),
       S("folder", "cbis_ddsm/<Calc|Mass>-<Training|Test>_P_<id>_<LEFT|RIGHT>_<CC|MLO>[_<n>]/<study>/<series>/<uuid>.dcm", ["d3 case folder"], ["full mammogram: ..._LEFT_CC (1 dcm)", "ROI mask + cropped patch: ..._LEFT_CC_1 (2 dcm)"])],
  prob=["MASKS/PATCHES AS IMAGES: 10,239 .dcm files = 3,419 full mammograms (3,261 case folders `..._CC|MLO`) + 6,820 files in 3,410 folders `..._CC|MLO_<n>`, each with 2 dcm (ROI mask + cropped lesion patch, 54 GB). is_mask() looks only at the last two components (`<seriesUID>/<uuid>.dcm`) so none is flagged; the lesion crop passes informative(), ROI masks mostly fail it",
        "csv `mass shape` value LYMPH_NODE -> match_value gives body_part lymph nodes (strong); `calc type`/`mass margins` etc. are not skipped",
        "average full mammogram is 29 MB: want = 1.4 x 2000 = 2800 files would be 82 GB; fetch_entries budget (20 GB) keeps the SMALLEST files first (sorted ascending by size), which biases toward small/low-res images",
        "listing shows `.tcia` manifest + AppleDouble junk (filtered)"],
  note="Fix exclude_paths: /(Calc|Mass)-(Training|Test)_P_\\d+_(LEFT|RIGHT)_(CC|MLO)_\\d+/ (verified: keeps exactly 3,419 of 10,239). Labels hard and correct.")

o("msd-task08-hepatic-vessel", src=[S("hf", "MedOtter/msd-hepatic-vessel (train.jsonl + imagesTr/labelsTr .nii.gz)", ["image","mask","label (list)","modality","dataset","official_split","patient_id"], ["modality: CT", "label: ['liver vessel','liver tumor']"])],
  bp=F("dataset-level; mmbu-public abdominal CT rows (24) are all body_part abdomen, liver appears only for 2 ultrasound rows", {"portal-venous abdominal CT": "abdomen"}, "medium"),
  prob=["303 volumes x 4 slices = 1,212 images vs plan 2000 (state 'partial'); labelsTr excluded by is_mask",
        "hf modality=CT maps (strong) -> computed tomography; body_part from the jsonl image path 'Task08_HepaticVessel' gives liver (weak)"],
  note="Same MSKCC portal-venous CT family as MSD task 03/07/10 (separate groups). Recommend body_part hard abdomen.")

o("msd-task10-colon", src=[S("hf", "MedOtter/msd-colon (train.jsonl + imagesTr/labelsTr .nii.gz)", ["image","mask","label (list)","modality","dataset","official_split","patient_id"], ["modality: CT", "files imagesTr/colon_001.nii.gz"])],
  bp=F("dataset-level; abdominal CT rows in mmbu-public are all 'abdomen'", {"portal-venous abdominal CT, colon primary": "abdomen"}, "medium"),
  prob=["126 volumes x 4 slices = 504 images vs plan 2000 ('partial')", "path token 'Task10_Colon' gives body_part colon (weak) - the scan is the whole abdomen"],
  note="Recommend body_part hard abdomen.")

o("sustech-sysu", src=[S("zip", "1030The SUSTech-SYSU dataset ... grading.zip (figshare 12570770, 0.41 GB)", ["originalImages/drLabels.csv: Fundus_images, left_versus_right_eye, DR_grade (3 protocols)", "originalImages/c5_DR_reclassified.csv", "odFoveaLabels/*.xml (1219)", "exudatesLabels/*.xml (564)"], ["1,219 jpg in originalImages/"])],
  note="Labels hard and correct; annotation .xml files are not media (kind=mask). Grouped with rav-rotterdam / idrid / reta in one split.")

o("alphadent", mod=F("registry hard clinical photography; mmbu-public convention is mixed (clinical photos of skin/eye: modality optical imaging, 61 rows; surgical frames: clinical photography, 31 rows)", {}, "medium"),
  bp=F("none per sample: intraoral photographs of the dentition (teeth); no 'oral cavity' vs 'teeth' signal in names or labels", {}, "low"),
  acc="ok (HF repo = AlphaDent.zip only; images/{train,valid,test} + labels/ YOLO polygons)",
  src=[S("zip", "AlphaDent.zip (HF)", ["images/{train,valid,test}/*.jpg (1,455)", "labels/{train,valid,test}/pNNN_<sex>_<age>_NNN.txt (1,320 YOLO polygons)", "data yaml"], ["p002_F_52_001.txt: '7 0.135 0.290 ...' (class 7 + polygon)"]), S("hf", "datasets-server default config", ["image","label (ClassLabel = split name train/test/valid)"], ["first-rows HTTP 501"])],
  prob=["datasets-server first-rows returns 501 for this repo (imagefolder-from-zip); fetch_hf streaming may work or fall back to remotezip: either way no member path is kept, which is fine (nothing path-dependent)",
        "ClassLabel `label` is the split folder (train/valid/test), not a Pass-2 field",
        "modality=clinical photography: in mmbu-public modality 'clinical photography' appears only on 31 surgical-video rows (body_part masked); skin/eye clinical photos (61 rows) are modality 'optical imaging' with submodality 'clinical photography'"],
  note="Intraoral dental photographs; domain masked is correct (no dentistry class). Modality choice is convention-dependent: candidates [clinical photography, optical imaging] is safer.")

o("medsegmnist-derm2d", acc="ok_array_format (2,594 per-image .npz inside one 38.5 GB zip; first array key 'image')",
  src=[S("zip", "derm2d_native.zip (zenodo 20786399)", ["derm2d_native/ISIC_0000000.npz ... 2594"], ["first member's first zip entry is image.npy (read from the inner local header, 400 bytes)"])],
  prob=["TRANSFER SIZE: want = 1.4 x 2000 = 2800 >= 2594 members, so sample_zip_remote extracts ALL 2,594 npz = 38.5 GB (member avg 14.9 MB, up to 2848x4288x3); the --max-gb 20 budget only gates whole-file downloads, not zip-range extraction",
        "ISIC 2018 task-1 images (HAM10000 subset); the MedSegMNIST npz also holds a mask array (2nd key) - read_array takes files[0]='image' which is correct only if that order holds for every member",
        "same source family as other selected ISIC sets (isic-2016/2017, derm12345, bcn20000 are different image sets; no overlap check possible remotely)"],
  note="Reduce plan_images (e.g. 600-800 -> ~12 GB) or cap by compress_size in sample_zip_remote.")

o("iscrc-immunoscore", src=[S("zip", "immunoscore.zip (figshare 24007515, 1.94 GB)", ["Images/<patient>/<patient>_{CD3|CD8}_{CT|MI}.jpg (385)", "data_csv.csv / data_excel.xlsx (Tumour site code, specimen type, ...)", "labels.csv (ID;Label %)", "Coding.txt (code book)"], ["Images/1/1_CD3_CT.jpg", "Tumour site 1a/1b/1c colon, 2a rectum, 3 multiple, 4 appendix (coded)"])],
  bp=F("data_csv `Tumour site` is coded (1 colon / 2 rectum ...: needs Coding.txt); ~96% of the head rows are colon", {"1c": "colon", "2a": "rectum (2 of 50 head rows)"}, "medium"),
  stain=F("file name marker CD3 / CD8 (DAB immunohistochemistry, TMA cores); registry hard ihc is right", {"1_CD3_CT.jpg": "ihc", "1_CD8_MI.jpg": "ihc"}, "high"),
  prob=["file tokens `_CT` (= tumour centre) and `_MI` (= invasive margin): match_value('CT') would read computed tomography on a short-code basis (hard registry wins; only spams conflicts); `CD3`/`CD8` are not matcher tokens",
        "5.2 MB JPEG per TMA core, 385 images vs plan 312 (fine)"],
  note="All hard labels correct.")

o("kermany-pediatric-pneumonia-cxr", src=[S("hf", "hf-vision/chest-xray-pneumonia (parquet 9 shards)", ["image","label (ClassLabel NORMAL/PNEUMONIA)"], ["train 5216, validation 16, test 624"])],
  note="Labels hard and correct (Guangzhou pediatric CXR, Kermany 2018). label = diagnosis (SKIP). Group bdcxr-3257 (different figshare set) shares the split. No fix.")

o("intraoral-caries-ahmed-2025", mod=F("registry hard clinical photography; see alphadent (mmbu-public convention is mixed)", {}, "medium"),
  src=[S("zip", "Dataset.zip (1.47 GB) and Benchmarking Dataset.zip (1.46 GB), zenodo 14827784", ["Dataset/Images/{retractors,no_retractors,pilot}/{Frontal,Maxillary_Occlusal,Mandibular,Left_Lateral,Right_Lateral}/*.jpg (6,265)", "Dataset/Annotations/{Darknet_YOLO,Labelme,pascal-voc,MS_coco}", "Benchmarking Dataset/{train,valid,test}/{images,labelme,pascal,yolo,coco}"], ["anonymous_003-007-...jpg", "view folder = photo view (intraoral), retractors/no_retractors"])],
  bp=F("path levels name the intraoral VIEW (Frontal, Maxillary_Occlusal, Mandibular, Left/Right_Lateral): all show teeth and gingiva; none maps to a body_part class", {"Frontal": "teeth/oral cavity (no class change)"}, "low"),
  prob=["DUPLICATE ZIP: Benchmarking Dataset.zip holds the same 6,266 jpg as Dataset.zip re-split into train/valid/test (both are listed; fetch_entries would download/sample both)",
        "annotation folders (json/xml/txt) are not images; is_mask('Annotations/...') fine"],
  note="Fix exclude_paths '^Benchmarking Dataset\\.zip$' (matches the listed entry name). Domain masked is correct.")

o("knee-xray-osteoporosis-wani", src=[S("zip", "Knee_Osteoporosis_Dataset.zip (zenodo 20053147, 33 MB)", ["Knee Osteoporosis Classification/{train 3780,val 1080,test 540}/{Normal,Osteopenia,Osteoporosis}/*.jpeg|jpg|png (5,400)", "Readme.txt"], ["1,800 images per class (augmented to balance)"])],
  bp=F("folder level d2 `Knee Osteoporosis Classification` (match_value -> knee, strong) and the record title: all images are knee radiographs", {"Knee Osteoporosis Classification": "knee"}, "high"),
  prob=["composite + augmented: record description = Kaggle set (1,947 images) + Wani/Arora Mendeley set (350 images), expanded with random flips/rotations to 1,800 per class: <=2,297 distinct source images, many near-duplicates (dHash dedup catches flips only if hash distance <=4)",
        "33 MB for 5,400 images: very small files (low resolution)"],
  note="Fix: body_part hard knee. Diagnosis folders (Normal/Osteopenia/Osteoporosis) are not Pass-2.")
