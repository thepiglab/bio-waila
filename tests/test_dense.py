"""Dense-annotation sidecars (dense.py), offline: pairing on real-world layouts, mask decoding, annotation formats,
geometry (EXIF turns, resize, volume slices), a prepare → build round trip and HF row capture."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bio_waila.data import dense, download, prepare
from bio_waila.data.dense import Frame, PairIndex, decode_mask, layer_name


# ── pairing ─────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("media,anns,level,want", [
    ("images/a1.png", ["masks/a1.png", "masks/b2.png"], "stem", ["masks/a1.png"]),
    ("files/ISIC_0000001.jpg", ["files/ISIC_0000001_segmentation.png"], "core", ["files/ISIC_0000001_segmentation.png"]),
    ("CAMG dataset/img/Image 98.jpg", ["CAMG dataset/seg/Seg 98.jpg", "CAMG dataset/seg/Seg 9.jpg"], "core",
     ["CAMG dataset/seg/Seg 98.jpg"]),
    ("Patient_71/Patient_71_RGB.png", ["Patient_71/Patient_71_GT.png"], "core", ["Patient_71/Patient_71_GT.png"]),
    ("phase/train/243_img.tif", ["phase/train/243_binary_mask.tif", "phase/train/339_binary_mask.tif"], "core",
     ["phase/train/243_binary_mask.tif"]),
    ("Dataset006/imagesTr/me2_train12_0000.nii.gz", ["Dataset006/labelsTr/me2_train12.nii.gz"], "core",
     ["Dataset006/labelsTr/me2_train12.nii.gz"]),
    ("BraTS2021_00001/BraTS2021_00001_flair.nii.gz", ["BraTS2021_00001/BraTS2021_00001_seg.nii.gz"], "core",
     ["BraTS2021_00001/BraTS2021_00001_seg.nii.gz"]),
    ("Volume files/C3L_03126.nii.gz", ["Pancreas_segmentations/C3L-03126_pancreas.nii.gz"], "core+class",
     ["Pancreas_segmentations/C3L-03126_pancreas.nii.gz"]),
    ("Totalseg/s0001/ct.nii.gz", ["Totalseg/s0001/segmentations/liver.nii.gz", "Totalseg/s0001/segmentations/spleen.nii.gz",
                                  "Totalseg/s0002/segmentations/liver.nii.gz"], "folder",
     ["Totalseg/s0001/segmentations/liver.nii.gz", "Totalseg/s0001/segmentations/spleen.nii.gz"]),
    ("Dataset 3/Image/case54/Time1/AX T2/volume.nii.gz", ["Dataset 3/Annotations/case54/Time1/AX T2/label.nii.gz",
                                                           "Dataset 3/Annotations/case58/Time1/AX T2/label.nii.gz"], "folder",
     ["Dataset 3/Annotations/case54/Time1/AX T2/label.nii.gz"]),
    ("ctc/01/t0576.tif", ["ctc/01_GT/SEG/man_seg0576.tif", "ctc/02_GT/SEG/man_seg0576.tif"], "core",
     ["ctc/01_GT/SEG/man_seg0576.tif"]),
    ("worms/images/601.png", ["worms/masks/601_1.png", "worms/masks/601_2.png", "worms/masks/6011_1.png"], "core+class",
     ["worms/masks/601_1.png", "worms/masks/601_2.png"]),
    ("sub-071/anat/sub-071_FLAIR.nii.gz", ["sub-071/sub-071_lesion_mask.nii.gz", "sub-072/sub-072_lesion_mask.nii.gz"],
     "core+class", ["sub-071/sub-071_lesion_mask.nii.gz"]),
    ("MITO/M2_U2OS/M2_004/ratio.tif", ["MITO/M2_U2OS/M2_004/label.tif", "MITO/M2_U2OS/M2_012/label.tif"], "folder",
     ["MITO/M2_U2OS/M2_004/label.tif"]),
])
def test_pairing_layouts(media, anns, level, want):
    assert PairIndex(anns).match(media) == (level, want)


def test_numeric_keys_need_parallel_folders():
    idx = PairIndex(["Datasets/Dataset 7/Ground truth mask/61.png"])
    assert idx.match("Datasets/Dataset 5/Images/61.png") == ("", [])
    idx = PairIndex(["Datasets/Dataset 7/Ground truth mask/61.png", "Datasets/Dataset 5/Ground truth mask/61.png"])
    assert idx.match("Datasets/Dataset 5/Images/61.png")[1] == ["Datasets/Dataset 5/Ground truth mask/61.png"]


def test_specific_image_does_not_own_other_cases():
    # a folder rule must not hand case b's mask to image a when a has none of its own
    assert PairIndex(["d/masks/b_mask.png"]).match("d/images/a.png") == ("", [])


@pytest.mark.parametrize("name,want", [
    ("x/train/_annotations.coco.json", True), ("coco_ins/annotations/instances_train2017.json", True),
    ("S3D/annotations/valid.csv", False), ("d/bbox_labels.csv", True), ("CVAT/annotations.xml", True),
    ("asoct/Normal/Annotated Images/106.json", False), ("dataset2/G1/G2208842_10_jpg_Label/G2208842_10_jpg_Label.json", False),
])
def test_dataset_level_tables(name, want):
    assert bool(dense._dataset_level(name)) is want


def test_layer_names():
    assert layer_name("Totalseg/s0001/ct.nii.gz", "Totalseg/s0001/segmentations/liver.nii.gz") == "liver"
    assert layer_name("C/C/image/C_0118.png", "C/C/vessel_mask/C_0118.png") == "vessel"
    assert layer_name("case/image.nii.gz", "case/annotation_3.nii.gz") == "annotation_3"
    assert layer_name("a/images/x.png", "a/masks/x_mask.png") == ""


# ── masks ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_decode_mask_kinds():
    b = np.zeros((40, 40), np.uint8)
    b[5:15, 5:15] = 255
    lab, k, _ = decode_mask(b)
    assert k == "binary" and lab.max() == 1 and lab.sum() == 100
    m = np.zeros((40, 40), np.uint8)
    m[:10], m[10:20], m[20:30] = 1, 2, 3
    assert decode_mask(m)[1] == "labels" and set(decode_mask(m)[2]) == {1, 2, 3}
    one = np.zeros((40, 40), np.uint8)
    one[:10] = 4                                     # a slice of a 1-4 label map that holds only class 4
    lab, k, cls = decode_mask(one)
    assert k == "labels" and set(cls) == {4} and lab.max() == 4
    assert decode_mask(np.zeros((8, 8), np.uint8))[1] == "empty"
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:10] = (255, 0, 0)
    rgb[10:20] = (0, 255, 0)
    lab, k, cls = decode_mask(rgb)
    assert k == "labels" and sorted(cls.values()) == ["rgb(0,255,0)", "rgb(255,0,0)"] and (lab[30:] == 0).all()
    photo = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    assert decode_mask(photo)[1] == "overlay"
    noisy = np.clip(b.astype(int) + np.random.default_rng(1).integers(-12, 12, b.shape), 0, 255).astype(np.uint8)
    lab, k, _ = decode_mask(noisy, lossy=True)
    assert k == "binary" and lab.sum() == 100
    inst = np.zeros((64, 64), np.uint16)
    for i in range(100):
        inst[(i // 10) * 6:(i // 10) * 6 + 3, (i % 10) * 6:(i % 10) * 6 + 3] = i + 1
    assert decode_mask(inst)[1] == "instances"
    assert decode_mask(np.random.default_rng(2).random((20, 20)))[1] == "not_mask"


def test_plotted_masks_are_rejected():
    fig = np.full((300, 400), 255, np.uint8)       # white figure margins with tick labels
    fig[30:270, 50:370] = 0                          # black axes box
    fig[120:180, 200:260] = 255                      # the shape, drawn in plot coordinates
    fig[280:290, 60:70] = 0                          # a tick label
    assert dense.looks_like_figure(decode_mask(fig)[0])
    real = np.zeros((300, 400), np.uint8)
    real[120:180, 200:260] = 255
    tissue = np.full((300, 400), 255, np.uint8)
    tissue[100:140, 100:300] = 0                     # tissue everywhere but a gap
    assert not dense.looks_like_figure(decode_mask(real)[0]) and not dense.looks_like_figure(decode_mask(tissue)[0])
    assert dense.fit_mask_2d(fig, Frame(400, 300, 400, 300))[1] == "figure"


def test_regions_and_seeds():
    lab = np.zeros((50, 50), np.uint16)
    lab[10:20, 10:30] = 1
    lab[35:45, 35:45] = 1
    regs = dense.layer_regions(lab, "binary", {1: "polyp"}, 0)
    assert sorted(r["box"] for r in regs) == [[10, 10, 20, 10], [35, 35, 10, 10]]
    for r in regs:
        x, y, w, h = r["box"]
        assert x <= r["seed"][0] < x + w and y <= r["seed"][1] < y + h and r["label"] == "polyp"


# ── annotation formats ──────────────────────────────────────────────────────────────────────────────────────────
def test_yolo_det_seg_dota():
    s = dense.parse_yolo("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.2 0.3\n", ["cell", "egg"])
    assert s[0] == {"type": "box", "label": "cell", "box": pytest.approx([0.4, 0.3, 0.2, 0.4]), "norm": True}
    assert s[1]["type"] == "polygon" and s[1]["label"] == "egg" and len(s[1]["xy"]) == 3
    d = dense.parse_yolo("10 10 50 10 50 30 10 30 needle 0\n")
    assert d[0]["type"] == "polygon" and d[0]["label"] == "needle" and not d[0]["norm"]
    assert dense.parse_yolo("patient reports pain\n") == []


def test_voc_and_labelme(tmp_path):
    (tmp_path / "a.xml").write_text("<annotation><object><name>sperm</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
                                     "<xmax>30</xmax><ymax>50</ymax></bndbox></object></annotation>")
    assert dense.read_annotation_file(tmp_path / "a.xml") == [{"type": "box", "label": "sperm", "box": [10, 20, 20, 30], "norm": False}]
    lm = {"shapes": [{"label": "lesion", "points": [[1, 2], [11, 22]], "shape_type": "rectangle"},
                     {"label": "edge", "points": [[0, 0], [5, 0], [5, 5]], "shape_type": "polygon"},
                     {"label": "fovea", "points": [[7, 8]], "shape_type": "point"}]}
    (tmp_path / "a.json").write_text(json.dumps(lm))
    got = dense.read_annotation_file(tmp_path / "a.json")
    assert [g["type"] for g in got] == ["box", "polygon", "point"] and got[0]["box"] == [1, 2, 10, 20]


def _rle_to_string(cnts):  # pycocotools' rleToString, to round-trip the decoder
    out = []
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def test_coco_boxes_polygons_keypoints_rle():
    m = np.zeros((6, 5), np.uint8)
    m[1:4, 2:4] = 1
    flat = m.T.reshape(-1)  # column-major runs, starting with zeros
    runs, cur, n = [], 0, 0
    for v in flat:
        if v != cur:
            runs.append(n)
            cur, n = v, 0
        n += 1
    runs.append(n)
    j = {"images": [{"id": 1, "file_name": "dir/x.jpg", "width": 5, "height": 6}],
         "categories": [{"id": 3, "name": "tool"}],
         "annotations": [{"image_id": 1, "category_id": 3, "bbox": [1, 1, 2, 3], "segmentation": [[0, 0, 4, 0, 4, 4]],
                          "keypoints": [2, 2, 2, 0, 0, 0]},
                         {"image_id": 1, "category_id": 3, "segmentation": {"size": [6, 5], "counts": _rle_to_string(runs)}}]}
    idx = dense.CocoIndex(j)
    shapes, im = idx.get("zip_members/z/x.jpg")
    assert {s["type"] for s in shapes} == {"box", "polygon", "point", "rle"}
    rle = next(s for s in shapes if s["type"] == "rle")
    assert (dense.coco_rle_mask(rle["seg"], 6, 5) == m).all()
    assert dense._coco_rle_counts(_rle_to_string(runs)) == runs


def test_mask_colours_named_by_boxes(tmp_path):
    regions = [{"layer": 0, "label": "rgb(0,111,166)", "box": [10, 10, 20, 20], "area": 400, "seed": [20, 20]},
               {"layer": 0, "label": "rgb(0,111,166)", "box": [50, 50, 18, 22], "area": 380, "seed": [59, 61]},
               {"layer": 0, "label": "rgb(255,74,70)", "box": [90, 10, 10, 10], "area": 100, "seed": [95, 15]}]
    layers = [{"kind": "labels", "classes": {"1": "rgb(0,111,166)", "2": "rgb(255,74,70)"}}]
    boxes = [{"label": "Neutrophil", "box": [11, 10, 19, 21]}, {"label": "Neutrophil", "box": [50, 51, 18, 21]},
             {"label": "Eosinophil", "box": [89, 9, 12, 12]}]
    dense.name_regions_by_boxes(regions, boxes, layers)
    assert [g["label"] for g in regions] == ["Neutrophil", "Neutrophil", "Eosinophil"]
    assert layers[0]["classes"] == {"1": "Neutrophil", "2": "Eosinophil"}
    j = {"images": [{"id": 1, "file_name": "P01.png", "width": 9, "height": 9}], "categories": [{"id": 2, "name": "Neutrophil"}],
         "annotations": [{"image_id": 1, "category_id": 2, "bbox": [1, 1, 3, 3]}]}
    (tmp_path / "P01.json").write_text(json.dumps(j))  # per-image COCO file
    assert dense.read_annotation_file(tmp_path / "P01.json") == [{"type": "box", "label": "Neutrophil", "box": [1.0, 1.0, 3.0, 3.0], "norm": False}]


def test_tables_boxes_and_points():
    df = pd.DataFrame({"filename": ["a.png", "a.png", "b.png"], "xmin": [1, 5, 2], "ymin": [1, 5, 2], "xmax": [3, 9, 4],
                       "ymax": [4, 9, 4], "class": ["rbc", "wbc", "rbc"]})
    t = dense.table_shapes(df)
    assert [s["box"] for s in t["a.png"]] == [[1, 1, 2, 3], [5, 5, 4, 4]] and t["a.png"][1]["label"] == "wbc"
    pts = dense.table_shapes(pd.DataFrame({"image_id": ["c"], "x": [0.5], "y": [0.25]}))
    assert pts["c"][0] == {"type": "point", "label": "", "xy": [0.5, 0.25], "norm": True}


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_exif_turn_moves_points_like_pixels(tmp_path):
    a = np.zeros((20, 40), np.uint8)  # W0=40, H0=20
    a[5, 10] = 255
    im = Image.fromarray(a)
    ex = im.getexif()
    ex[0x0112] = 6  # stored sideways; viewers (and prepare.read_image) turn it upright
    im.save(tmp_path / "x.jpg", exif=ex, quality=100)
    upright = np.asarray(Image.open(tmp_path / "x.jpg").transpose(Image.ROTATE_270))
    fr = dense.image_frame(tmp_path / "x.jpg", upright.shape[1], upright.shape[0])
    assert fr.op == Image.ROTATE_270
    x, y = fr.pt(10.5, 5.5)  # centre of the bright pixel
    yy, xx = np.unravel_index(np.argmax(upright), upright.shape)
    assert (int(x), int(y)) == (xx, yy)
    lab = np.zeros((20, 40), np.uint16)
    lab[5, 10] = 1
    assert np.unravel_index(np.argmax(fr.label_map(lab)), upright.shape) == (yy, xx)


def test_frame_scales_and_clips_boxes():
    fr = Frame(1000, 500, 512, 256)
    assert fr.box([100, 50, 200, 100]) == [51.2, 25.6, 102.4, 51.2]
    assert fr.box([0.5, 0.5, 0.6, 0.2], norm=True) == [256.0, 128.0, 256.0, 51.2]
    assert fr.box([2000, 10, 5, 5]) is None


def _nifti_world(tmp_path, affine):
    import nibabel as nib
    lab = np.zeros((64, 48, 24), np.uint8)
    lab[10:24, 5:16, 8:16] = 1
    lab[35:50, 25:40, 8:16] = 2
    img = (lab > 0).astype(np.float32) * 400 + np.random.default_rng(0).normal(100, 5, lab.shape)
    nib.save(nib.Nifti1Image(img.astype(np.float32), affine), str(tmp_path / "case_0000.nii.gz"))  # zooms from the affine
    nib.save(nib.Nifti1Image(lab, affine), str(tmp_path / "case.nii.gz"))


@pytest.mark.parametrize("affine", [np.diag([1.0, 1.0, 2.5, 1.0]), np.diag([-1.0, -1.0, 2.5, 1.0])])
def test_nifti_label_slices_match_prepared_slices(tmp_path, affine):
    _nifti_world(tmp_path, affine)
    sl = list(prepare.read_volume(tmp_path / "case_0000.nii.gz"))
    assert sl
    got = dense.volume_label_slices(tmp_path / "case.nii.gz", tmp_path / "case_0000.nii.gz", [l for _, l, _ in sl])
    checked = 0
    for im, loc, _ in sl:
        if not got[loc].any():
            continue  # slices without the structure: per-slice contrast stretching makes their noise look bright
        im = prepare.fit(im)
        bright = np.asarray(im.convert("L")) > 128
        lab = np.asarray(Image.fromarray(got[loc].astype(np.uint8)).resize(im.size, Image.NEAREST)) > 0
        assert (bright & lab).sum() / (bright | lab).sum() > 0.8, loc
        checked += 1
    assert checked >= 3


def test_sitk_label_slices(tmp_path):
    import SimpleITK as sitk
    lab = np.zeros((12, 64, 48), np.uint8)
    lab[3:9, 10:30, 5:20] = 1
    sitk.WriteImage(sitk.GetImageFromArray(lab.astype(np.int16) * 400 + 50), str(tmp_path / "x1.mha"))
    sitk.WriteImage(sitk.GetImageFromArray(lab), str(tmp_path / "masksX1.mha"))
    sl = list(prepare.read_volume(tmp_path / "x1.mha"))
    got = dense.volume_label_slices(tmp_path / "masksX1.mha", tmp_path / "x1.mha", [l for _, l, _ in sl])
    for im, loc, _ in sl:
        assert ((np.asarray(im.convert("L")) > 128) == (got[loc] > 0)).mean() > 0.99


# ── prepare → build round trip ──────────────────────────────────────────────────────────────────────────────────
def _scene(seed, box):
    a = np.random.default_rng(seed).integers(30, 90, (300, 600), dtype=np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = 230
    return a


@pytest.fixture
def world(tmp_path):
    raw, prep = tmp_path / "raw", tmp_path / "prepared"
    root = raw / "toy"
    for d in ("files/images", "files/masks", "files/labels"):
        (root / d).mkdir(parents=True)
    boxes = {"a": (100, 50, 200, 150), "b": (300, 100, 420, 260), "c": (50, 200, 150, 280)}
    for n, b in boxes.items():
        Image.fromarray(_scene(ord(n), b)).save(root / "files/images" / f"{n}.png")
    m = np.zeros((300, 600), np.uint8)
    m[50:150, 100:200] = 255
    Image.fromarray(m).save(root / "files/masks/a.png")
    m = np.zeros((300, 600), np.uint8)
    m[100:260, 300:420] = 1
    Image.fromarray(m).save(root / "files/masks/b_mask.png")
    (root / "files/labels/b.txt").write_text("0 0.6 0.6 0.2 0.5333333\n")
    (root / "files/classes.txt").write_text("lesion\n")
    coco = {"images": [{"id": 7, "file_name": "c.png", "width": 600, "height": 300}], "categories": [{"id": 1, "name": "spot"}],
            "annotations": [{"image_id": 7, "category_id": 1, "bbox": [50, 200, 100, 80]}]}
    (root / "files/_annotations.coco.json").write_text(json.dumps(coco))
    (root / "_status.json").write_text(json.dumps({"state": "done", "strategy": "files"}))
    none = {"mode": "masked", "values": []}
    row = {"id": "toy", "plan_images": 10, "split": "train", "group": "toy", "exclude_paths": "",
           "labels": {f: none for f in ("modality", "submodality", "body_part", "stain", "specimen", "domain")}}
    return dict(raw=raw, prep=prep, row=row, boxes=boxes)


def test_prepare_then_build(world):
    st = prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(), None, seed=0)
    assert st["kept"] == 3  # the masks are not images
    out = dense.build_dataset(world["row"], world["raw"], world["prep"], {})
    assert out["with_dense"] == 3, out
    man = pd.read_parquet(world["prep"] / "manifest" / "toy.parquet")
    dman = pd.read_parquet(world["prep"] / "dense" / "manifest" / "toy.parquet")
    assert set(dman.image) == set(man.image)
    for r in man.itertuples():
        j = json.loads((world["prep"] / dman.set_index("image").at[r.image, "dense_json"]).read_text())
        name = r.source.rsplit("/", 1)[-1][0]
        s = r.width / 600
        x0, y0, x1, y1 = (v * s for v in world["boxes"][name])
        if name == "a":  # mask only
            assert len(j["layers"]) == 1 and j["layers"][0]["kind"] == "binary" and len(j["regions"]) == 1
            assert j["regions"][0]["box"] == pytest.approx([x0, y0, x1 - x0, y1 - y0], abs=1.5)
            png = np.asarray(Image.open(world["prep"] / j["layers"][0]["path"]))
            assert png.shape == (r.height, r.width) and png.max() == 1
        elif name == "b":  # mask + YOLO box, with the class name from classes.txt
            assert j["regions"] and j["boxes"][0]["label"] == "lesion"
            assert j["boxes"][0]["box"] == pytest.approx([x0, y0, x1 - x0, y1 - y0], abs=1.5)
        else:  # COCO only
            assert not j["layers"] and j["boxes"] == [{"label": "spot", "box": pytest.approx([x0, y0, x1 - x0, y1 - y0], abs=0.6)}]
    dense.merge(world["prep"])
    assert len(pd.read_parquet(world["prep"] / "dense.parquet")) == 3


def test_overlays_are_not_masks(world):
    root = world["raw"] / "toy"
    (root / "files/masks/a.png").unlink()
    (root / "files/annotated").mkdir()
    overlay = np.stack([_scene(1, world["boxes"]["a"])] * 3, -1)
    overlay[50:150, 100:105] = (255, 0, 0)
    overlay = overlay + np.random.default_rng(3).integers(0, 9, overlay.shape, dtype=np.uint8)
    Image.fromarray(overlay).save(root / "files/annotated/a.png")
    prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(), None, seed=0)
    out = dense.build_dataset(world["row"], world["raw"], world["prep"], {})
    assert out["skip:overlay"] == 1 and out["with_dense"] == 2


def test_one_class_masks_keep_their_class_name(world):
    """A mask holding only class 4 is named class 4, not class 1; a colour's per-image index never takes a value name."""
    root = world["raw"] / "toy"
    m = np.zeros((300, 600), np.uint8)
    m[50:150, 100:200] = 4
    Image.fromarray(m).save(root / "files/masks/a.png")
    rgb = np.zeros((300, 600, 3), np.uint8)
    rgb[100:260, 300:420] = (0, 255, 0)
    Image.fromarray(rgb).save(root / "files/masks/b_mask.png")
    (root / "files/labels/b.txt").unlink()
    (root / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "organ": 1, "vessel": 4}}))
    prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(), None, seed=0)
    dense.build_dataset(world["row"], world["raw"], world["prep"], {})
    man = pd.read_parquet(world["prep"] / "manifest" / "toy.parquet")
    dman = pd.read_parquet(world["prep"] / "dense" / "manifest" / "toy.parquet").set_index("image")
    got = {}
    for r in man.itertuples():
        j = json.loads((world["prep"] / dman.at[r.image, "dense_json"]).read_text())
        got[r.source.rsplit("/", 1)[-1][0]] = sorted({g["label"] for g in j["regions"]})
    assert got["a"] == ["vessel"] and got["b"] == ["rgb(0,255,0)"]


# ── HF capture ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_hf_capture_and_dense_only_restream(tmp_path, monkeypatch):
    import datasets
    feats = datasets.Features({"image": datasets.Image(), "mask": datasets.Image(), "view": datasets.Image(),
                               "objects": {"bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                                           "label": datasets.Sequence(datasets.ClassLabel(names=["giardia", "coli"]))}})
    rows = {"image": [], "mask": [], "view": [], "objects": []}
    for i in range(4):
        rows["image"].append(Image.fromarray(_scene(i, (10, 10, 40, 30))[:64, :64]))
        m = np.zeros((64, 64), np.uint8)
        m[10:30, 10:40] = 255
        rows["mask"].append(Image.fromarray(m))
        rows["view"].append(Image.fromarray(_scene(i + 9, (0, 0, 5, 5))[:64, :64]))  # a second photo, not a mask
        rows["objects"].append({"bbox": [[10.0, 10.0, 30.0, 20.0]], "label": [1]})
    ds = datasets.Dataset.from_dict(rows, features=feats).to_iterable_dataset()
    dest = tmp_path / "hf"
    dest.mkdir()
    monkeypatch.setattr(download, "DENSE", True)
    got, _, strat, _ = download._stream_hf({"train": ds}, "x/y", dest, 3, 0, [], None, [])
    assert (got, strat) == (3, "hf-stream")
    recs = [json.loads(l) for l in open(dest / "_hf_dense.jsonl")]
    assert len(recs) == 3 and all(len(r["masks"]) == 1 and r["masks"][0]["column"] == "mask" for r in recs)
    assert recs[0]["class_names"]["objects.label"] == ["giardia", "coli"]
    shapes = dense.hf_shapes(recs[0], 64, 64)
    assert shapes == [{"type": "box", "label": "coli", "box": [10.0, 10.0, 30.0, 20.0], "norm": False}]
    # top-up: same rows again, annotations only, nothing new saved as an image
    (dest / "_hf_dense.jsonl").unlink()
    monkeypatch.setattr(download, "DENSE", False)
    n_img = len(list((dest / "hf_images").iterdir()))
    assert download._stream_hf({"train": ds}, "x/y", dest, 3, 0, [], None, [], dense_only=True) == 3
    assert len(list((dest / "hf_images").iterdir())) == n_img
