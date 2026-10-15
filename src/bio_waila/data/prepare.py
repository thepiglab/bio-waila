#!/usr/bin/env python3
"""Turn downloaded raw datasets into MMBU-style training images with Pass-2 labels (auto-labelling layers 1–2).

Per dataset (status done/partial in RAW/<id>/_status.json):
  1. extract whole archives (safely; no path traversal)
  2. find media: images, DICOM, NIfTI/MHA/NRRD volumes, videos, WSIs, .npy/.npz arrays; masks are skipped
  3. sample up to plan_images: volumes → SLICES_PER_VOLUME slices, videos → 8 frames, WSIs → 16 tissue tiles
  4. convert to 8-bit, long side ≤ 512 px (MMBU's images are ≤ 512 px), JPEG
  5. drop exact/near duplicates within the dataset and any image near-identical to an mmbu-public image
  6. label: per-sample labels from metalabel.py (registry targets refined by each sample's own metadata: DICOM tags,
     CSV/JSON/HF columns, BIDS entities, folder names). Files metalabel did not cover fall back to registry +
     DICOM header + path rules. Every field records value(s), mode, source, confidence and the evidence used.
Then `--merge` concatenates the per-dataset manifests and writes stats on how much auto-labelling resolved.

Layer 3 (model-based pseudo-labels) is autolabel.py, which reads this manifest.

  uv run waila-prepare --raw $RAW --out $PREP --shard 0/8
  uv run waila-prepare --out $PREP --merge
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tarfile
import warnings
import zipfile

try:  # registers deflate64 support in zipfile when installed
    import zipfile_deflate64  # noqa: F401
except ImportError:
    pass
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from bio_waila.data.files import ext, is_mask, kind
from bio_waila.data.rules import bids_hits, dicom_hits, resolve, text_hits
from bio_waila.data.vocab import FIELDS, STAINED_MODALITIES

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 400_000_000
from bio_waila.paths import PLAN, PREP, PUBLIC, RAW
LONG_SIDE = 512
MIN_SIDE = 32
MAX_ASPECT = 8  # longer side / shorter side; OCT B-scans and panoramics stay well below
SLICES_PER_VOLUME = 4
STACK_ITEMS = 2000  # images taken from one .npy stack of independent images (the plan's per-dataset cap still applies)
FRAMES_PER_VIDEO = 8
TILES_PER_WSI = 16
NEAR_DUP_BITS = 4  # 64-bit dHash distance treated as the same image *within* a dataset (cheap; frames, repeats)
# mmbu-public near-duplicates need a stricter test: a 64-bit dHash alone matches unrelated dark-background images (CT
# slices, endoscopy frames; cystods lost 245 of 280). Measured on 600 resized / re-encoded / cropped public images and
# 15k training images: 256-bit dHash ≤ 36 AND 32x32 thumbnail correlation ≥ 0.9 → 98.7% recall, 0.3% false positives.
PUB_DH256_BITS = 36
PUB_CORR = 0.9


# ── pixels ──────────────────────────────────────────────────────────────────────────────────────────────────────
def to_uint8(a, lo=None, hi=None):
    a = np.asarray(a)
    if a.dtype == np.uint8 and lo is None:
        return a
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    if lo is None:
        lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        return None
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_pil(a):
    """2D gray, HxWx3/4 colour, or HxWxC multichannel (first 3 channels as RGB) → PIL image."""
    a = np.squeeze(np.asarray(a))
    if a.ndim == 3 and a.shape[0] in (2, 3, 4, 5) and a.shape[-1] > 5:  # channels-first
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] == 2:
        a = np.concatenate([a, np.zeros_like(a[..., :1])], -1)
    if a.ndim == 3 and a.shape[-1] > 4:
        a = a[..., :3]
    if a.ndim not in (2, 3):
        return None
    if a.ndim == 3 and a.dtype != np.uint8:  # multichannel 16-bit/float: normalise each channel on its own
        if a.shape[-1] == 4:
            a = a[..., :3]
        chans = [to_uint8(a[..., c]) for c in range(a.shape[-1])]
        chans = [c if c is not None else np.zeros(a.shape[:2], np.uint8) for c in chans]
        return Image.fromarray(np.stack(chans, -1))
    a = to_uint8(a)
    if a is None:
        return None
    return Image.fromarray(a[..., :3] if a.ndim == 3 else a)


def fit(im):
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:  # thumbnails; 19x512 mother-machine strips
        return None
    s = LONG_SIDE / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im


def informative(im):
    """Not blank and not almost all background. Uses the brightest channel, not luma: a dim single-colour fluorescence
    image (red signal on black) has a luma std of ~4 but is a real image."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32).max(2) if im.mode not in ("L", "I", "F", "I;16") else np.asarray(im.convert("L"), dtype=np.float32)
    return a.std() > 6 and (a > 10).mean() > 0.05


def looks_like_mask(im):
    """Label maps and binary masks: ≤ 4 distinct grey levels, or two levels covering ≥ 98% of pixels (and few colours)."""
    small = im.convert("RGB").resize((128, 128), Image.NEAREST)
    a = np.asarray(small).reshape(-1, 3)
    cols, counts = np.unique(a, axis=0, return_counts=True)
    if len(cols) <= 4:
        return True
    top2 = np.sort(counts)[-2:].sum() / counts.sum()
    if len(cols) <= 16 and top2 >= 0.98:
        return True
    # instance-label maps (one grey id per object on a zero background): mostly exactly 0, and the nonzero pixels
    # repeat few values (real images have smooth intensity ramps with many values per object)
    g = np.asarray(im.convert("L").resize((256, 256), Image.NEAREST)).astype(np.int16)
    if (g == 0).mean() < 0.5 or (g > 0).sum() < 200:
        return False
    l, r = g[:, :-1], g[:, 1:]
    both = (l > 0) & (r > 0)  # horizontally adjacent pixel pairs inside objects
    return both.sum() >= 100 and (l[both] == r[both]).mean() >= 0.97 and len(np.unique(g[g > 0])) >= 5


def dhash(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(h, arr):
    x = np.bitwise_xor(arr, np.uint64(h))
    return np.array([bin(int(v)).count("1") for v in x]) if len(arr) < 64 else np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


# ── readers: yield (PIL image, sub-location string, extra evidence dict) ───────────────────────────────────────
def _tiff_highbit(p):
    """(pages, n) for a TIFF whose samples are > 8 bit with several channels: PIL opens those as truncated 8-bit RGBA
    without an error, so they must go through tifffile. None otherwise."""
    if ext(p) not in (".tif", ".tiff"):
        return None
    try:
        import tifffile
        with tifffile.TiffFile(p) as tf:
            pg = tf.pages[0]
            if pg.bitspersample > 8 and pg.samplesperpixel > 1:
                n = len(tf.pages)
                idx = [0] if n == 1 else np.linspace(int(n * 0.3), max(int(n * 0.7), int(n * 0.3)), min(SLICES_PER_VOLUME, n)).round().astype(int)
                return [(int(i), tf.pages[int(i)].asarray()) for i in idx], n
    except Exception:
        return None
    return None


def read_image(p):
    hb = _tiff_highbit(p)
    if hb:
        pages, n = hb
        for i, a in pages:
            yield to_pil(a), (f"frame{i}/{n}" if n > 1 else ""), {}
        return
    try:
        im = Image.open(p)
        n = getattr(im, "n_frames", 1)
        if n > 1:  # multipage TIFF stack (z-planes or time): sample frames from the middle like a volume
            for i in np.linspace(int(n * 0.3), max(int(n * 0.7), int(n * 0.3)), min(SLICES_PER_VOLUME, n)).round().astype(int):
                im.seek(int(i))
                yield to_pil(np.array(im)), f"frame{i}/{n}", {}
            return
        if im.mode in ("I;16", "I;16B", "I", "F"):
            yield to_pil(np.array(im)), "", {}
            return
        from PIL import ImageOps
        try:
            im = ImageOps.exif_transpose(im)  # phone / camera photos store their rotation in EXIF
        except Exception:
            pass
        yield im.convert("RGB") if im.mode not in ("RGB", "L") else im, "", {}
    except Exception:
        import tifffile
        yield to_pil(tifffile.imread(p)), "", {}


def nifti_slice_axis(shape, zooms):
    """Slice axis for a RAS-canonical volume: axial (2) unless axial slices would be a thin strip (> 2:1 in mm), as with
    sagittal slabs or cropped volumes; then the acquisition axis (fewest voxels)."""
    ext_mm = [s * z for s, z in zip(shape, zooms)]
    if max(ext_mm[0], ext_mm[1]) / max(1e-6, min(ext_mm[0], ext_mm[1])) <= 2:
        return 2
    return int(np.argmin(shape))


def square_pixels(im, row_mm, col_mm):
    """Resample so one pixel is the same size in mm both ways (row_mm: spacing between rows, col_mm: between columns)."""
    if im is None or not row_mm or not col_mm or abs(row_mm / col_mm - 1) < 0.05:
        return im
    w, h = im.size
    if row_mm > col_mm:
        h = max(1, round(h * row_mm / col_mm))
    else:
        w = max(1, round(w * col_mm / row_mm))
    return im.resize((w, h), Image.BILINEAR)


def pick_slices(vol, n):
    """n informative slices from the middle 30-70% of the slice axis (smallest dimension = slice axis)."""
    ax = int(np.argmin(vol.shape[:3]))
    vol = np.moveaxis(vol, ax, 0)
    lo, hi = int(vol.shape[0] * 0.3), max(int(vol.shape[0] * 0.7), int(vol.shape[0] * 0.3) + 1)
    idx = np.linspace(lo, hi - 1, n).round().astype(int) if hi - lo >= n else range(lo, hi)
    for i in idx:
        yield vol[i], f"slice{i}/{vol.shape[0]}"


def read_volume(p):
    e = ext(p)
    tags = {}
    if e == ".dcm" or kind(p) == "other":
        import pydicom
        ds = pydicom.dcmread(p, force=True)
        tags = {k: str(getattr(ds, k, "")) for k in ("Modality", "BodyPartExamined", "StudyDescription", "SeriesDescription")}
        a = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        lo = hi = None
        wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(wc[0] if hasattr(wc, "__len__") and not isinstance(wc, str) else wc)
            ww = float(ww[0] if hasattr(ww, "__len__") and not isinstance(ww, str) else ww)
            lo, hi = wc - ww / 2, wc + ww / 2
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        if a.ndim == 3 and a.shape[-1] not in (3, 4):  # multi-frame
            for s, loc in pick_slices(a, SLICES_PER_VOLUME):
                yield Image.fromarray(to_uint8(s, lo, hi)) if to_uint8(s, lo, hi) is not None else None, loc, {"dicom": tags}
            return
        u = to_uint8(a, lo, hi)
        yield (Image.fromarray(u) if u is not None else None), "", {"dicom": tags}
        return
    if e in (".nii", ".nii.gz"):
        import nibabel as nib
        img = nib.as_closest_canonical(nib.load(str(p)))  # RAS+: axis 2 is axial (inferior→superior)
        a = np.asanyarray(img.dataobj)
        if a.ndim == 4:
            a = a[..., 0]
        if a.ndim == 3 and min(a.shape) > 1:
            zooms = [float(x) or 1.0 for x in img.header.get_zooms()[:3]]
            ax = nifti_slice_axis(a.shape[:3], zooms)
            z = a.shape[ax]
            lo, hi = int(z * 0.3), max(int(z * 0.7), int(z * 0.3) + 1)
            name = ("sagittal", "coronal", "axial")[ax]
            inplane = [zooms[d] for d in range(3) if d != ax]
            for i in np.linspace(lo, hi - 1, min(SLICES_PER_VOLUME, hi - lo)).round().astype(int):
                s = np.rot90(np.take(a, i, axis=ax))  # rot90: anterior / superior up, as viewers show it
                im = square_pixels(to_pil(s), inplane[1], inplane[0])
                if im is not None and max(im.size) / max(1, min(im.size)) <= 3:
                    yield im, f"{name}{i}/{z}", {}
            return
    else:
        import SimpleITK as sitk
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    if a.ndim == 4:
        a = a[..., 0] if a.shape[-1] < a.shape[0] else a[0]
    if a.ndim == 2:
        yield to_pil(a), "", {}
        return
    for s, loc in pick_slices(a, SLICES_PER_VOLUME):
        yield to_pil(s), loc, {}


def read_array(p):
    a = np.load(p, allow_pickle=False)
    if isinstance(a, np.lib.npyio.NpzFile):
        a = a[a.files[0]]
    if a.ndim < 2:
        return  # label vectors (y_*.npy) are not images
    if a.ndim == 2 or (a.ndim == 3 and a.shape[-1] in (1, 3, 4)):
        yield to_pil(a), "", {}
    elif a.ndim == 3 and a.shape[0] > max(a.shape[1], a.shape[2]) and min(a.shape[1:]) >= 16:
        # an (N, H, W) stack of independent images (e.g. 1000 x 128 x 128 crops): each item is its own image
        for i in np.linspace(0, a.shape[0] - 1, min(a.shape[0], STACK_ITEMS)).round().astype(int):
            yield to_pil(a[i]), f"item{i}/{a.shape[0]}", {}
    elif a.ndim == 4 and a.shape[-1] in (1, 3, 4) and a.shape[0] > 4:  # (N, H, W, C) stack of images
        for i in np.linspace(0, a.shape[0] - 1, min(a.shape[0], STACK_ITEMS)).round().astype(int):
            yield to_pil(a[i]), f"item{i}/{a.shape[0]}", {}
    elif a.ndim >= 3:
        for s, loc in pick_slices(a if a.ndim == 3 else a[0], SLICES_PER_VOLUME):
            yield to_pil(s), loc, {}


def read_video(p):
    import cv2
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    for i in np.linspace(0, max(n - 1, 0), FRAMES_PER_VIDEO).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            yield Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), f"frame{i}/{n}", {}
    cap.release()


def read_wsi(p):
    import openslide
    s = openslide.OpenSlide(str(p))
    lvl = min(range(s.level_count), key=lambda l: abs(s.level_downsamples[l] - 2))  # ≈ 2× downsample (~0.5 µm/px)
    thumb = np.asarray(s.get_thumbnail((512, 512)).convert("RGB")).astype(np.int16)
    sat = thumb.max(-1) - thumb.min(-1)
    ys, xs = np.where(sat > 25)  # tissue is coloured, background is white/grey
    if len(ys) == 0:
        return
    W, H = s.dimensions
    rng = random.Random(str(p))
    for k in rng.sample(range(len(ys)), min(TILES_PER_WSI, len(ys))):
        x = int(xs[k] / thumb.shape[1] * W)
        y = int(ys[k] / thumb.shape[0] * H)
        size = int(LONG_SIDE * s.level_downsamples[lvl])
        tile = s.read_region((max(0, x - size // 2), max(0, y - size // 2)), lvl, (LONG_SIDE, LONG_SIDE)).convert("RGB")
        yield tile, f"tile{x}_{y}@L{lvl}", {}


READERS = {"image": read_image, "volume": read_volume, "array": read_array, "video": read_video, "wsi": read_wsi}
PER_FILE = {"image": 1, "volume": SLICES_PER_VOLUME, "array": SLICES_PER_VOLUME, "video": FRAMES_PER_VIDEO, "wsi": TILES_PER_WSI}


# ── dataset-level steps ─────────────────────────────────────────────────────────────────────────────────────────
def extract_archives(root: Path):
    for a in [p for p in (root / "archives").rglob("*") if p.is_file()] if (root / "archives").exists() else []:
        done = root / "extracted" / (a.name + ".ok")
        if done.exists():
            continue
        dest = root / "extracted" / a.name.split(".")[0]
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if zipfile.is_zipfile(a):
                with zipfile.ZipFile(a) as z:
                    for m in z.infolist():
                        if m.is_dir() or ".." in m.filename or m.filename.startswith("/"):
                            continue
                        if kind(m.filename) in ("image", "volume", "video", "wsi", "array", "meta", "other"):
                            z.extract(m, dest)
            elif tarfile.is_tarfile(a):
                with tarfile.open(a) as t:
                    t.extractall(dest, filter="data")
            elif a.suffix.lower() == ".rar":  # needs the `unrar` or `unar` binary on PATH (rarfile wraps it)
                import rarfile
                with rarfile.RarFile(a) as r:
                    for m in r.infolist():
                        if not m.isdir() and ".." not in m.filename and not m.filename.startswith("/"):
                            r.extract(m, dest)
            elif a.suffix.lower() == ".7z":
                import py7zr
                with py7zr.SevenZipFile(a) as z:
                    z.extractall(dest)
            else:
                continue
            done.write_text("ok")
        except Exception as ex:
            (root / "extracted" / (a.name + ".err")).write_text(str(ex))


def find_media(root: Path):
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("_") or p.suffix == ".part" or "archives" in p.parts or kind(p.name) == "junk" or "__MACOSX" in p.parts:
            continue
        rel = str(p.relative_to(root))
        if is_mask(rel):
            continue
        k = kind(p.name)
        if k == "other" and p.stat().st_size > 1024:
            with open(p, "rb") as f:  # extension-less DICOM
                f.seek(128)
                if f.read(4) == b"DICM":
                    k = "volume"
        if k in READERS:
            out.append((p, rel, k))
    return out


def dhash256(im):
    a = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def thumb_vec(im):
    a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).flatten()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


class PublicIndex:
    """mmbu-public fingerprints for the near-duplicate test (see PUB_DH256_BITS / PUB_CORR)."""

    def __init__(self, dh256, thumbs):
        self.dh256, self.thumbs = dh256, thumbs.astype(np.float32)

    def __len__(self):
        return len(self.dh256)

    def matches(self, im) -> bool:
        if not len(self):
            return False
        c = self.thumbs @ thumb_vec(im)
        cand = np.nonzero(c >= PUB_CORR)[0]
        if not len(cand):
            return False
        d = np.unpackbits(np.bitwise_xor(self.dh256[cand], dhash256(im)), axis=1).sum(1)
        return bool((d <= PUB_DH256_BITS).any())


def public_hashes(public_dir: Path, cache: Path):
    cache = cache.with_name(cache.stem + "_v2.npz")
    if cache.exists():
        z = np.load(cache)
        return PublicIndex(z["dh256"], z["thumbs"])
    dh, th = [], []
    for p in sorted((public_dir / "images").glob("*")):
        try:
            im = Image.open(p)
            dh.append(dhash256(im))
            th.append(thumb_vec(im))
        except Exception:
            pass
    idx = PublicIndex(np.stack(dh) if dh else np.zeros((0, 32), np.uint8), np.stack(th) if th else np.zeros((0, 1024), np.float32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, dh256=idx.dh256, thumbs=idx.thumbs.astype(np.float16))
    return idx


def prepare_dataset(row, raw: Path, out: Path, body_classes, pub, seed):
    import pandas as pd
    root = raw / row["id"]
    st_p = root / "_status.json"
    if not st_p.exists() or json.loads(st_p.read_text())["state"] not in ("done", "partial"):
        return {"id": row["id"], "state": "not downloaded"}
    extract_archives(root)
    media = find_media(root)
    if row.get("exclude_paths"):
        media = [m for m in media if not re.search(row["exclude_paths"], m[1])]
    if not media:
        return {"id": row["id"], "state": "no media found"}
    rng = random.Random(f"{seed}:{row['id']}")
    rng.shuffle(media)
    n_files = sum(1 for _ in media)
    target = row["plan_images"]
    # how many files to open: enough to reach the target given per-file yield, spread over folders
    by_dir = defaultdict(list)
    for m in media:
        by_dir[str(Path(m[1]).parent)].append(m)
    order, dirs = [], list(by_dir.values())
    while dirs:
        for d in dirs:
            order.append(d.pop())
        dirs = [d for d in dirs if d]
    lab_p = out / "labels" / f"{row['id']}.parquet"  # from metalabel.py apply (per-sample metadata labels)
    meta_labels = pd.read_parquet(lab_p).set_index("rel").to_dict("index") if lab_p.exists() else {}
    img_dir = out / "images" / row["id"]
    img_dir.mkdir(parents=True, exist_ok=True)
    seen = np.zeros(0, dtype=np.uint64)
    recs, stats = [], Counter()
    for p, rel, k in order:
        if len(recs) >= target:
            break
        try:
            for im, loc, extra in READERS[k](p):
                if im is None:
                    stats["unreadable"] += 1
                    continue
                im = fit(im)
                if im is None or not informative(im):
                    stats["blank_or_tiny"] += 1
                    continue
                if looks_like_mask(im):
                    stats["looks_like_mask"] += 1
                    continue
                h = dhash(im)
                if len(seen) and hamming(h, seen).min() <= 1:
                    stats["dup_in_dataset"] += 1
                    continue
                if pub is not None and len(pub) and pub.matches(im):
                    stats["near_mmbu_public"] += 1
                    continue
                seen = np.append(seen, np.uint64(h))
                name = hashlib.sha1(f"{row['id']}/{rel}/{loc}".encode()).hexdigest()[:20] + ".jpg"
                im.convert("RGB").save(img_dir / name, quality=92)
                ml = meta_labels.get(rel)
                ev = []
                if ml is None:  # no metalabel row (e.g. metalabel not run): fall back to header + path rules
                    if extra.get("dicom"):
                        ev.append(("dicom", dicom_hits(extra["dicom"])))
                    ph = text_hits(rel, body_classes)
                    if not set(row["labels"]["modality"]["values"]) & STAINED_MODALITIES:
                        ph.pop("stain", None)
                    for f, v in bids_hits(rel).items():
                        ph.setdefault(f, {}).update(v)
                    ev.append(("path", ph))
                rec = {"image": f"images/{row['id']}/{name}", "dataset_id": row["id"], "split": row["split"],
                       "license_group": row.get("license_group", "permissive"), "license": row.get("license", ""),
                       "group": row["group"], "source": rel, "loc": loc, "kind": k, "width": im.size[0],
                       "height": im.size[1], "gray": im.mode == "L", "dhash": f"{h:016x}"}
                for f in FIELDS:
                    if ml is not None:
                        r = {"mode": ml[f"{f}_mode"], "values": list(ml[f"{f}_values"]), "source": ml[f"{f}_source"],
                             "confidence": ml[f"{f}_conf"], "conflict": ml[f"{f}_conflict"]}
                        rec[f"{f}_evidence"] = ml.get(f"{f}_evidence", "")
                    else:
                        r = resolve(row["labels"][f], [(s_, e[f]) for s_, e in ev if f in e], f)
                        rec[f"{f}_evidence"] = ""
                    rec[f"{f}_mode"], rec[f"{f}_values"] = r["mode"], r["values"]
                    rec[f"{f}_source"], rec[f"{f}_conf"] = r["source"], r["confidence"]
                    rec[f"{f}_conflict"] = r.get("conflict", "")
                recs.append(rec)
                stats["kept"] += 1
                if len(recs) >= target:
                    break
        except Exception as ex:
            stats["reader_error:" + type(ex).__name__] += 1
    if recs:
        (out / "manifest").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(out / "manifest" / f"{row['id']}.parquet", index=False)
    return {"id": row["id"], "state": "ok" if recs else "empty", "media_files": n_files, "target": target, **stats}


def merge(out: Path):
    import pandas as pd
    parts = sorted((out / "manifest").glob("*.parquet"))
    if not parts:
        print(f"nothing to merge: no manifests in {out / 'manifest'}")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / "manifest.parquet", index=False)
    L = [f"# Prepared Pass-2 training set\n", f"{len(df):,} images from {df.dataset_id.nunique()} datasets "
         f"({(df.split == 'train').sum():,} train / {(df.split == 'val').sum():,} val).\n",
         "## Label status per head after layers 1–2 (registry + rules)\n",
         "| Head | hard (registry) | hard (auto) | candidates | multi | masked | conflicts |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FIELDS:
        m, s = df[f"{f}_mode"], df[f"{f}_source"]
        pct = lambda x: f"{100 * x.mean():.1f}%"
        L.append(f"| {f} | {pct((m == 'hard') & (s == 'registry'))} | {pct((m == 'hard') & (s != 'registry'))} | "
                 f"{pct(m == 'candidates')} | {pct(m == 'multi')} | {pct(m == 'masked')} | {(df[f + '_conflict'] != '').sum():,} |")
    L.append("\n## Auto-label sources\n")
    for f in FIELDS:
        c = df.loc[(df[f"{f}_mode"] == "hard") & (df[f"{f}_source"] != "registry"), f"{f}_source"].value_counts()
        if len(c):
            L.append(f"- **{f}**: " + ", ".join(f"{k} {v:,}" for k, v in c.items()))
    L.append("\n## Datasets with the most label conflicts (check their registry entries)\n")
    conf = df.assign(n=sum((df[f + "_conflict"] != "").astype(int) for f in FIELDS)).groupby("dataset_id").n.sum().sort_values(ascending=False)
    for k, v in conf[conf > 0].head(25).items():
        L.append(f"- {k}: {v}")
    (out / "stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(PLAN / "selection.jsonl"))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(PREP))
    ap.add_argument("--public", default=str(PUBLIC), help="mmbu-public dir, for the contamination filter")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", type=int, default=0, help="cap images per dataset (use a separate --out, e.g. prepared_pilot)")
    args = ap.parse_args()
    out = Path(args.out)
    if args.merge:
        return merge(out)
    space = json.loads((PLAN / "label_space.json").read_text())
    body_classes = set(space["body_part"]["classes"])
    pub = public_hashes(Path(args.public), out / "_cache" / "mmbu_public_dhash.npy")
    rows = [json.loads(l) for l in open(args.plan)]
    if args.pilot:
        for r in rows:
            r["plan_images"] = min(r["plan_images"], args.pilot)
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only.split(","))]
    i, n = map(int, args.shard.split("/"))
    log = out / "_logs" / f"prepare_shard{i}of{n}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        for row in rows[i::n]:
            r = prepare_dataset(row, Path(args.raw), out, body_classes, pub, args.seed)
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
