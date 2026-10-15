"""bio_waila.data.files: file kinds and the mask/junk filters that decide what counts as 'the image'."""
import pytest

from bio_waila.data.files import ext, is_junk, is_mask, kind


@pytest.mark.parametrize("path", [
    "masks/img1.png", "a/Masks/img.png", "data/BinaryMask/1.png", "data/mask12.png", "x/img_mask.png",
    "labelsTr/case1.nii.gz", "a/segmentation/1.png", "gt/001.png", "Ground_Truth/001.png", "x/img_seg.png",
    "dataset/images/img_gt.png", "annotations/001.png"])
def test_masks(path):
    assert is_mask(path)
    assert kind(path) == "mask"


@pytest.mark.parametrize("path", [
    "images/img_001.jpg", "img.png", "train/CT/ct1.png", "images/trainmask.png",
    "lesions-seg/images/img001.png",       # dataset root folder 'lesions-seg/' must not hide every image
    "lesions-seg/nested/deeper/img001.png"])
def test_not_masks(path):
    assert not is_mask(path)
    assert kind(path) == "image"


def test_only_the_files_own_folder_and_name_are_judged():
    assert is_mask("lesions-seg/img001.png")             # direct parent folder says segmentation
    assert not is_mask("lesions-seg/images/img001.png")  # grandparent does not


@pytest.mark.parametrize("path", ["__MACOSX/x/img.png", "x/__MACOSX/._img.png", "~$notes.xlsx", "x/._img.png",
                                  "a/.DS_Store", "a/Thumbs.db", "desktop.ini"])
def test_junk(path):
    assert is_junk(path)
    assert kind(path) == "junk"


def test_not_junk():
    assert not is_junk("x/img.png") and not is_junk("x/my_file.csv")


@pytest.mark.parametrize("path,expected", [
    ("labels.csv", "meta"), ("annotations.json", "meta"), ("meta.tab", "meta"), ("x.TAB", "meta"),
    ("sheet.xlsx", "meta"), ("data.parquet", "meta"),               # metadata files are never masks, whatever their name
    ("masks/labels.csv", "meta"),
    ("a.zip", "archive"), ("a.tar.gz", "archive"), ("a.7z", "archive"),
    ("x.nii.gz", "volume"), ("x.dcm", "volume"), ("x.mha", "volume"),
    ("x.svs", "wsi"), ("x.mp4", "video"), ("x.npy", "array"), ("readme.pdf", "other"), ("noext", "other")])
def test_kind(path, expected):
    assert kind(path) == expected


def test_ext_handles_compound_suffixes():
    assert ext("a/b.nii.gz") == ".nii.gz"
    assert ext("a/b.tar.gz") == ".tar.gz"
    assert ext("a.b/c") == ""
    assert ext("a/B.PNG") == ".png"
