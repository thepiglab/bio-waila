from lib import *

o("aml-cytomorphology-lmu", acc="ok (Kaggle livepriyanka09/aml-cytology via token, 11.76 GB; per-file listing throttled by HTTP 429 after 11 pages: first 2,200 of ~18,365 files read)",
  src=[S("folder", "AML-Cytomorphology_LMU/<cell type>/<ABBR>_NNNN.tiff (Kaggle file list, alphabetical first 2,200)", ["d2 class folders seen: BAS 79, EBO 78, EOS 424, KSC 15, LYA 11, LYT 1,593 (15 classes in the original)"], ["AML-Cytomorphology_LMU/BAS/BAS_0001.tiff (640,154 bytes each = 400x400x4 + header)"]),
       S("kaggle-view", "datasets/view/livepriyanka09/aml-cytology", ["title AML-Cytomorphology_LMU", "subtitle: Single cell Leukocytes from AML Patients and Non-malignant Controls", "licenseName Unknown", "description empty"], [])],
  prob=["Kaggle copy has license 'Unknown' and an empty description (original TCIA Matek et al. 2019 is CC BY 3.0); registry license CC-BY-3.0 refers to the original",
        "list_kaggle asks pageSize=1000 but the API caps a page at 200 (measured: 200 returned): 18k files = ~92 sequential requests; the token is rate-limited (HTTP 429 after ~11 requests at 8 s spacing) so download.py will end in RetryError/state 'error' unless it sleeps per Retry-After and caches the listing",
        "folder = cell-type class (not a Pass-2 field); all 15 class folders are single-cell crops: spread_sample spreads over class folders, fine",
        "group multifocus-wbc-park-2024 shares the split with 6 other WBC sets"],
  note="Registry labels (MGG, blood, human, brightfield) match the dataset (peripheral blood smears of AML patients and controls, Matek 2019).")

o("bcss", acc="ok (Kaggle whats2000/breast-cancer-semantic-segmentation-bcss, 8.26 GB, CC0; listing throttled: first 1,000 files = BCSS/test/*.png)",
  src=[S("kaggle-view", "datasets/view/whats2000/breast-cancer-semantic-segmentation-bcss (description read)", ["224x224 version: train 30,760 + val 5,429 + test 4,021 (masks merged to 3 classes)", "512x512 version: train 6,000 + val 2,768 (22 classes)"], ["file names TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0_0_0_size224.png (TCGA slide id + ROI origin + tile offset)"])],
  prob=["DERIVATIVE: 224 and 512 versions are two resizes of the same annotated TCGA ROIs; ~49k images + ~49k masks = ~98k files -> ~490 list requests at 200 per page: not listable under the 429 rate limit (only BCSS/test first 1,000 names read; masks are expected in separate folders and are excluded by is_mask if the folder/suffix says mask: unverified)",
        "tiles are 224 px (or 512 px) crops of TCGA diagnostic slides at 0.25-0.5 MPP: stain h&e / submodality whole slide imaging are right; no per-sample metadata beyond TCGA case id in the file name (case id -> breast is dataset-level)",
        "registry plan eff_images 151,000 does not match the 49k images in the record"],
  note="Per-sample stain/body fields cannot vary; registry hard labels are correct. Use the whole-dataset zip + sample_zip_remote instead of per-file listing for this one.")

o("tbx11k", acc="ok (Kaggle usmanshams/tbx-11, 4.25 GB; listing throttled: first 600 files = README, paper pdf, annotations/json (10) and annotations/xml (588 of ~800); imgs/ not reached)",
  src=[S("readme", "TBX11K/README.md (3.6 KB, read via the Kaggle per-file endpoint)", ["imgs/ (11,200 TBX11K 512x512 X-rays: train 6,600 / val 1,800 / test 2,800) + imgs/extra/ (DA 156, DB 150, Montgomery 138, Shenzhen 662)", "annotations/xml (TB boxes, trainval only), annotations/json (COCO), TBX11K_*.txt / all_*.txt image lists", "5 categories: Healthy, Sick but Non-TB, Active TB, Latent TB, Uncertain TB"], ["all images are 512x512"])],
  prob=["Kaggle view licenseName 'Unknown' (registry says CC-BY-4.0); the official TBX11K release is for research use and the test-set ground truth is withheld: license/redistribution needs a check",
        "imgs/extra/ contains 1,106 images re-used from the DA/DB/Montgomery/Shenzhen TB sets (Montgomery and Shenzhen are also hosted elsewhere); group ju-tbv1 (an aggregate of other TB sets) shares the split",
        "annotation .xml files are kind 'other' (not meta, not mask): not downloaded as media (fetch_entries only takes image/volume/... kinds), fine; images are 512x512 PNG so Pass-2 pixel quality is limited",
        "full per-file listing needs >=56 pages at 200/page; throttled as for the other Kaggle sets (HTTP 429)"],
  note="Registry hard labels (chest x-ray, chest, human) are right; Kaggle copy has no per-sample metadata beyond the diagnosis folders.")
