from lib import *

o("fd3611", src=[S("zip-path+csv","FD3611/<Normal|Media_Hazy|Diabetic_Retinopathy|Myopic_Retinopathy|Optic_Disc_Disorder>/*_left|right.png; metadata/fd3611_master_metadata.csv",["Left","Right","Gender"],[])],
  bp=F("dataset-level (fundus)", {"-":"eye"}, "medium"),
  note="Folders/columns are diagnoses (the matcher maps 'Optic_Disc_Disorder'/'Retinal Hemorrhage' to optic disc/retina by keyword: these are findings, not the imaged part): ensure diagnosis columns stay skipped (add 'left|right' disease columns to SKIP or accept 'eye').")
o("ultracortex-9t4", src=[S("openneuro","sub-N/ses-K/anat/*.nii (uncompressed); derivatives/{brainmasks,skullstrips,freesurfer_segmentation,manual_segmentation}; derivatives/scanning_parameters.tsv",["Sequence: MP2RAGE"],[])],
  bp=F("dataset-level (9.4T brain MRI)", {"-":"brain"}, "medium"),
  prob=["list_openneuro keeps only /anat/ non-derivatives (ok) but derivatives/skullstrips are near-duplicates: keep excluded"])
o("behsof-nafld-us", acc="NO DATA in record: figshare 389069 holds only 'Figure_8.tif'", src=[S("figshare","389069",["Figure_8.tif"],[])],
  prob=["registry says 0.8 GB / 1000 ultrasound images; figshare article lists a single figure TIFF: wrong record or files removed"])
o("plasmodium-4species-rtdetr-france-2023", acc="ok but 42 GB multi-volume zip (.zip.001-.005): no range listing", src=[S("zenodo","Plasmodium-v1.zip.001 ... .005 + models.zip",[],[])],
  prob=["split zip volumes must be concatenated: cannot be range-sampled; models.zip is weights only"],
  note="specimen candidates human/plasmodium: smears of Plasmodium-infected human blood: could be multi [human, plasmodium] but not resolvable from head.")
o("biliary-atresia-cholestasis", src=[S("zip-path+xlsx","Raw Images.zip: BA|non-BA/<id>/{Fasting,Post-prandial,Others}/N.jpg; List Data.xlsx; 5-fold.zip (nnU-Net validation outputs)",["path d2: BA|non-BA (diagnosis)","path d4: Fasting|Post-prandial"],[])],
  bp=F("dataset-level (gallbladder ultrasound for biliary atresia)", {"-":"gallbladder (liver also valid)"}, "medium"),
  prob=["5-fold.zip (264 MB > Raw Images.zip 201 MB) holds 2.5k derived PNGs (progress/validation plots, savePic niftis): download.py sorts archives by size and would sample it first: restrict to Raw Images.zip","nnUNet-master.zip is source code"])
o("skdrepadata-v2-2025", src=[S("zip-path","SKDrepaData-v2/<sickle_cell|other_cell|normale_cell>/blood_cell_NN.jpg",[],[])], note="All hard. Folder = class (cell type): not a Pass-2 field; matcher weakly maps 'sickle_cell' to body_part 'cell' (ignore).")
o("single-cell-cancer-vs-normal-fluorescence-hf", src=[S("hf-path","cell_40x/{normal,cancer}/type<k>_40x_*_cell_N.tif; label ClassLabel cancer|normal",[],[])], note="All hard (DAPI, cell lines). 1133 tifs.")
o("wormswin-celegans", src=[S("zip-path","synthetic_images_dataset.zip (images 10000 + masks), csb-1_dataset.zip (4631 images + masks + coco), md_dataset.zip (429 images)",[],[])],
  prob=["synthetic_images_dataset.zip is GENERATED worm images (10,000 of ~15,060 images): contradicts hard brightfield/c. elegans real-specimen labels and would dominate a proportional sample: exclude it (or label synthetic)"],
  note="masks/ folders caught by is_mask.")
o("rcc-aid", src=[S("csv","metadata.csv (TCIA export)",["Modality","BodyPartExamined","Collection","StudyDesc","SeriesDescription"],["Modality: CT","BodyPartExamined: KIDNEY","Collection: TCGA-KIRC"])],
  bp=F("csv column BodyPartExamined (DICOM tag copy)", {"KIDNEY":"kidney"}, "high"),
  note="images.zip = 129 nifti CTs named <Collection>_<SeriesUID with underscores>; join to metadata.csv by series UID. labels.zip are masks.")
o("bosque-test-set", src=[S("dataverse-tab","BOSQUE_metadata.tab (tab-separated; Content-Type .tab)",["file_name","sex","skin_phototype","age_group","anatomical_site","lesion_nature","clinical_label","histology_label"],["anatomical_site: head_neck / trunk"])],
  prob=["`.tab` is not in META_EXT (files.py) so metalabel/download would skip the metadata table: add .tab","zip has a __MACOSX/._*.jpg mirror entry per image (151 junk)","registry says dermoscopy but the metadata (skin_phototype, evolution, size buckets, clinical_label) reads like clinical photography: verify on a few images"],
  note="anatomical_site (head_neck, trunk...) could refine body_part but registry hard=skin.")
o("smart-om", acc="RESTRICTED: zenodo 16899860 'SMART intraoral images' is access_right=restricted (request access)", src=[], prob=["cannot be downloaded without author approval: drop or request access"])
o("m4raw-abdomen", src=[S("hf","mylyu/M4Raw_abdomen: nifti/<T1|T2|T2FB>/..., multicoil_* .h5 (k-space), metadata/*.tsv",["contrast: T1|T2","sequence_id: T2FB","acquisition_condition: free_breathing"],[])],
  prob=["591 .h5 files are raw multicoil k-space (not images): use nifti/ only (393 files; masks folder 131)","size -1"],
  note="T1/T2 (+free-breathing T2) all structural mri; abdomen hard.")
