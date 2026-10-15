"""File-type helpers shared by download.py (what to fetch) and prepare.py (what to read)."""
import re

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".pgm", ".ppm", ".pnm", ".cri"}  # .cri: CRi/Nuance scanner TIFFs
VOLUME_EXT = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WSI_EXT = {".svs", ".ndpi", ".mrxs", ".scn", ".vsi", ".bif"}
ARRAY_EXT = {".npy", ".npz"}
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z", ".rar"}
META_EXT = {".csv", ".tsv", ".tab", ".json", ".jsonl", ".txt", ".xlsx", ".xls", ".yaml", ".yml", ".parquet"}
MEDIA_EXT = IMAGE_EXT | VOLUME_EXT | VIDEO_EXT | WSI_EXT | ARRAY_EXT

# segmentation masks, label maps and overlays live next to the images in most datasets; they are not "the image"
_MASK = re.compile(r"(^|[/_\-. ])(masks?\d*|msk|merged?_?masks?|labels?|labelsTr|labelsTs|seg|segs|segmentations?|segmented|gt|groundtruth|ground_truth|"
                   r"annotations?|annotated|annot|overlays?|contours?|marks?|merged?|merge_images|detected|teaser|trimap|lesion_mask|binary_?mask|weight_?maps?|"
                   r"visuali[sz]ed_masks|curvature|training_target|target_masks?)([/_\-. ]|$)|(^|/)labels[A-Za-z][^/]*/", re.I)
_MASK_SUFFIX = re.compile(r"(_m|_mask|_seg|_label|_gt|_lbl|-mask|-label)\.(png|tif|tiff|jpg|nii|nii\.gz)$", re.I)


def ext(path):
    p = str(path).lower()
    for e in (".nii.gz", ".tar.gz", ".tar.bz2", ".tar.xz"):
        if p.endswith(e):
            return e
    i = p.rfind(".")
    return p[i:] if i > p.rfind("/") else ""


# unambiguous mask-tree names may sit one level higher (EDD2020/masksPerClass/BE/x.tif, .../validation/targets/x.png)
_MASK_TREE = re.compile(r"(^|/)(masks?\w*|targets?|labels|ground_?truth|gt|annotations)(/|$)", re.I)


def is_mask(path):
    """Judge the file's own folder and name (a dataset root like 'lesions-seg/' must not hide every image), plus
    unambiguous mask-tree folder names up to two levels above the file."""
    p = str(path).replace("\\", "/")
    parts = p.split("/")
    tail = re.sub(r"([a-z])([A-Z])", r"\1_\2", "/".join(parts[-2:]))  # BinaryMask -> Binary_Mask
    upper = "/".join(parts[-3:-1])
    return bool(_MASK.search(tail) or _MASK_SUFFIX.search(tail) or _MASK_TREE.search(upper))


def is_junk(path):
    """macOS resource forks and metadata (__MACOSX/, ._file, .DS_Store) carry image extensions but no image."""
    p = str(path)
    base = p.rsplit("/", 1)[-1]
    return "__MACOSX" in p or base.startswith(("._", "~$")) or base in (".DS_Store", "Thumbs.db", "desktop.ini")


def kind(path):
    if is_junk(path):
        return "junk"
    e = ext(path)
    if e in ARCHIVE_EXT:
        return "archive"
    if e in META_EXT:  # labels.csv / annotations.json are metadata, not masks
        return "meta"
    if is_mask(path):
        return "mask"
    for k, s in (("image", IMAGE_EXT), ("volume", VOLUME_EXT), ("video", VIDEO_EXT), ("wsi", WSI_EXT), ("array", ARRAY_EXT), ("meta", META_EXT)):
        if e in s:
            return k
    return "other"
