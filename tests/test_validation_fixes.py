"""Regressions for bugs found by the local validation workers (docs/data/validation/V5.md B1–B8)."""
import io

import pandas as pd

from bio_waila.data import metalabel as ml
from bio_waila.data.files import is_mask


def test_kaggle_slug_folder_is_not_a_metadata_column():  # B3
    cols = ml.path_columns("extracted/pahunichoudhary__carotid-artery-ultrasound-and-color-doppler/Common Carotid Artery/US/img1.png")
    assert not any("__" in v for v in cols.values())
    assert "Common Carotid Artery" in cols.values() and "US" in cols.values()


def test_archive_stem_folder_is_dropped():  # B3
    cols = ml.path_columns("zip_members/TRUSTED_dataset_for_nsd/TRUSTED_dataset_for_nsd/US_DATA/US_images/a.nii.gz")
    assert "US_DATA" in cols.values()


def test_overrides_allow_several_fields_per_value(tmp_path, monkeypatch):  # B5
    (tmp_path / "overrides.csv").write_text(
        "dataset_id,column,value,field,class\nds,*,us_data,modality,ultrasound imaging\nds,*,us_data,submodality,3d ultrasound\n")
    monkeypatch.setattr(ml, "PLAN_DIR", tmp_path)
    ov = ml.load_overrides()
    assert sorted(ov[("ds", "*", "us_data")]) == [("modality", "ultrasound imaging"), ("submodality", "3d ultrasound")]


def test_mask_and_overlay_folders():  # B7/B8
    assert is_mask("shseg/mark/0001.png")
    assert is_mask("vip20k/Merge/frame.png")
    assert is_mask("rsua/merge_images/x.png")
    assert not is_mask("rsua/images/x.png")


def test_nan_cells_do_not_break_key_detection():  # B1
    s = pd.read_csv(io.StringIO("a,b\nx,\n"), dtype=str)["b"]
    assert s.dropna().astype(str).tolist() == []


def test_npy_image_stack_is_sampled_along_items(tmp_path):  # V7 B4
    import numpy as np
    from bio_waila.data.prepare import read_array
    a = (np.random.default_rng(0).random((1000, 128, 128)) * 255).astype("uint8")
    np.save(tmp_path / "x.npy", a)
    np.save(tmp_path / "y.npy", np.zeros(1000))
    ims = [im for im, _, _ in read_array(tmp_path / "x.npy")]
    assert ims and all(im.size == (128, 128) for im in ims)
    assert list(read_array(tmp_path / "y.npy")) == []


def test_zip_wrapped_single_file_is_unwrapped(tmp_path, monkeypatch):  # V7 B2
    import io as _io
    import zipfile
    from bio_waila.data import download as dl
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 100
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("img.png", png)

    class R:
        def __init__(self): self.status_code = 200
        def raise_for_status(self): pass
        def iter_content(self, n): yield buf.getvalue()
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(dl.S, "get", lambda *a, **k: R())
    out = tmp_path / "img.png"
    dl.get_file(dl.Entry("img.png", -1, "http://x"), out)
    assert out.read_bytes() == png


def test_mask_tree_one_level_up_and_pgm():  # V6 B2/B5
    from bio_waila.data.files import kind
    assert is_mask("EDD2020/masksPerClass/BE/EDD2020_B0003_BE.tif")
    assert is_mask("polypgen/validation/targets/x.png")
    assert not is_mask("skin-lesion-yolo-seg/lesions-seg/images/a.jpg")
    assert kind("mias/mdb001.pgm") == "image"


def test_mask_content_check_and_multichannel():  # V1 P21 / V3 mask leaks
    import numpy as np
    from PIL import Image
    from bio_waila.data.prepare import looks_like_mask, to_pil
    m = np.zeros((128, 128), np.uint8); m[30:60, 30:90] = 255
    assert looks_like_mask(Image.fromarray(m))
    photo = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    assert not looks_like_mask(Image.fromarray(photo))
    tif = np.zeros((64, 64, 4), np.uint16); tif[..., 0] = np.arange(64)[None, :] * 50; tif[..., 3] = 65535
    im = to_pil(tif)
    assert im.mode == "RGB" and np.asarray(im)[..., 0].std() > 10


def test_nifti_slices_are_axial(tmp_path):
    import nibabel as nib
    import numpy as np
    from bio_waila.data.prepare import read_volume
    v = np.random.default_rng(0).random((40, 50, 30)).astype(np.float32)
    nib.save(nib.Nifti1Image(v, np.eye(4)), tmp_path / "v.nii.gz")
    out = list(read_volume(tmp_path / "v.nii.gz"))
    assert out and all(im.size == (40, 50) for im, _, _ in out)  # rot90 of a 40x50 axial plane -> 40 wide, 50 high


def test_measurement_columns_are_not_indicators():
    from bio_waila.data.metalabel import MEASURE_COLUMN
    for c in ("cervix_area", "lesion_size", "liver_volume", "tumor_diameter_mm", "nucleus count"):
        assert MEASURE_COLUMN.search(c), c
    for c in ("cervix", "has_cervix", "lung", "stain_he", "is_fundus", "Atelectasis", "Pleural_Effusion"):
        assert not MEASURE_COLUMN.search(c), c


def _blobs(label_ids):
    import numpy as np
    rng = np.random.default_rng(0)
    a = np.zeros((256, 256), np.float32)
    yy, xx = np.mgrid[:256, :256]
    for k in range(30):
        cy, cx, r = rng.integers(20, 236), rng.integers(20, 236), rng.integers(5, 14)
        d = (yy - cy) ** 2 + (xx - cx) ** 2
        a[d < r * r] = (k + 1) * 7 if label_ids else 0
        if not label_ids:
            a += 200 * np.exp(-d / (2 * r * r)) + rng.normal(0, 4, a.shape) * (d < r * r)
    return a


def test_instance_label_maps_are_masks_but_dark_fluorescence_is_not():
    import numpy as np
    from PIL import Image
    from bio_waila.data.prepare import looks_like_mask
    lab = Image.fromarray(_blobs(True).astype(np.uint8))
    fl = Image.fromarray(np.clip(_blobs(False), 0, 255).astype(np.uint8))
    assert looks_like_mask(lab)
    assert not looks_like_mask(fl)


def test_nifti_slice_axis_and_square_pixels():
    from PIL import Image
    from bio_waila.data.prepare import nifti_slice_axis, square_pixels
    assert nifti_slice_axis((512, 512, 300), (0.7, 0.7, 1.0)) == 2          # ordinary axial CT
    assert nifti_slice_axis((140, 256, 209), (0.65, 1.055, 1.055)) == 0     # sagittal slab (hvsmr-2-0)
    im = square_pixels(Image.new("L", (256, 140)), 0.65, 1.055)             # rows 0.65 mm apart, columns 1.055 mm
    assert im.size == (416, 140)


def test_dim_single_channel_fluorescence_is_informative():
    import numpy as np
    from PIL import Image
    from bio_waila.data.prepare import informative
    rng = np.random.default_rng(0)
    a = np.zeros((384, 512, 3), np.uint8)
    a[..., 0] = rng.integers(8, 15, (384, 512))
    m = rng.random((384, 512)) < 0.15
    a[..., 0][m] = rng.integers(28, 65, m.sum())
    assert informative(Image.fromarray(a))
    assert not informative(Image.fromarray(np.full((64, 64, 3), 3, np.uint8)))


def test_exif_rotation_is_applied(tmp_path):
    from PIL import Image
    from bio_waila.data.prepare import read_image
    im = Image.new("RGB", (40, 20), (200, 10, 10))
    ex = im.getexif()
    ex[0x0112] = 6  # rotate 90° when displayed
    p = tmp_path / "phone.jpg"
    im.save(p, exif=ex)
    out = next(read_image(p))[0]
    assert out.size == (20, 40)


def test_public_index_catches_reencoded_copy_not_dark_lookalike():
    import io
    import numpy as np
    from PIL import Image
    from bio_waila.data.prepare import PublicIndex, dhash256, thumb_vec
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    pub_img = Image.fromarray((np.clip(120 + 60 * np.sin(xx / 9) * np.cos(yy / 13) + rng.normal(0, 5, (256, 256)), 0, 255)).astype(np.uint8))
    idx = PublicIndex(np.stack([dhash256(pub_img)]), np.stack([thumb_vec(pub_img)]))
    b = io.BytesIO()
    pub_img.resize((200, 200)).save(b, "JPEG", quality=80)
    assert idx.matches(Image.open(b))
    dark = np.zeros((256, 256), np.uint8)
    dark[(yy - 128) ** 2 + (xx - 120) ** 2 < 90 ** 2] = 90  # a round body section on black, like a CT slice
    assert not idx.matches(Image.fromarray(dark))


def test_tar_stream_reservoir_spreads(tmp_path, monkeypatch):
    import io
    import tarfile
    from bio_waila.data import download as dl
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for i in range(200):
            data = b"x" * 10
            ti = tarfile.TarInfo(f"patient{i // 20:02d}/frame{i:03d}.png")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()

    class R:
        def __init__(self):
            self.raw = io.BytesIO(raw)
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(dl.S, "get", lambda *a, **k: R())
    log = []
    n, _ = dl.sample_tar_stream(dl.Entry("x.tar", len(raw), "http://x"), tmp_path, 10, log)
    files = list((tmp_path / "tar_members").rglob("*.png"))
    assert n == 10 and len(files) == 10
    assert len({f.parent.name for f in files}) >= 4  # several patients, not just patient00


def test_full_table_matches_a_sampled_download(tmp_path):
    import pandas as pd
    from bio_waila.data.metalabel import load_tables
    pd.DataFrame({"image_id": [f"IMG{i:06d}.jpeg" for i in range(3746)], "hand": ["0", "1"] * 1873}).to_csv(tmp_path / "dataset.csv", index=False)
    sampled = {f"IMG{i:06d}" for i in range(0, 3746, 13)} | {f"IMG{i:06d}.jpeg" for i in range(0, 3746, 13)}
    tabs = load_tables(tmp_path, sampled, n_media=len(range(0, 3746, 13)))
    assert len(tabs) == 1  # only ~7.7% of rows match the sample, but the table covers every sampled file


def test_override_columns_ignore_download_container():
    from bio_waila.data.metalabel import norm_col
    a = norm_col("csv:extracted/BTXRD/BTXRD/dataset.xlsx:hand")
    assert a == norm_col("csv:zip_members/BTXRD/BTXRD/dataset.xlsx:hand") == "csv:BTXRD/dataset.xlsx:hand"
    assert norm_col("csv:files/cystods_simple.csv:label") == "csv:cystods_simple.csv:label"
    assert norm_col("path:d1") == "path:d1"
