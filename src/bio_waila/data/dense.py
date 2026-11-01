#!/usr/bin/env python3
"""Dense annotations (masks, boxes, points, polygons) for prepared images, as a sidecar to the Pass-2 manifest.

The point-seeded reveal traces (../data_discovery/research/grounded_reveal_flywheel.md) need the regions the Pass-2
pipeline throws away: download samples media only and prepare skips masks. This module adds them back without changing
any image:

  waila-dense fetch --raw RAW [--only ids]    top up raw downloads with the annotation files that pair with the media
                                             already there. Loose files and zip members are re-listed and only the pairs
                                             are fetched (masks often sit in another zip); tars are re-streamed; HF rows
                                             are re-streamed with the download's seed and checked against the saved
                                             image. Archives downloaded whole need nothing: build extracts their masks.
  waila-dense build --raw RAW --out PREP      per prepared image: find its annotation, map it into the prepared JPEG's
                                             pixel frame (EXIF turn, resize, volume slice), write
                                             dense/<id>/<image stem>.json (+ one PNG per mask layer) and
                                             dense/manifest/<id>.parquet
  waila-dense build --out PREP --merge        dense.parquet + dense_stats.md

Targets: plan datasets whose registry entry lists spatial labels and that may train the student (registry train_ok);
--all-spatial adds the teacher-only ones, --only picks ids. New downloads capture annotations directly with
`waila-download --dense`. Per-dataset corrections go in plan/dense_fixes.json (a "_note" key is free text):
  {"<id>": {"classes": {"1": "polyp"}, "ignore": "<regex on annotation paths>", "box_format": "xywh|xyxy|cxcywh"}}

Sidecar JSON; coordinates are pixels of the prepared image and boxes are [x, y, w, h]:
  image, size [w, h], match, sources,
  layers   [{path, name, kind: binary|labels|instances|empty, classes {value: name}}]   one PNG per mask layer
  regions  [{layer, label, box, area, seed}]   one per connected component or instance; seed = most interior pixel
  boxes [{label, box}], points [{label, xy}], polygons [{label, xy: [[x, y], ...]}]   from annotation files
"""
import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from bio_waila.data.files import IMAGE_EXT, VOLUME_EXT, ext, is_junk, is_mask, kind
from bio_waila.paths import PLAN, PREP, RAW, REGISTRY

Image.MAX_IMAGE_PIXELS = 400_000_000
MASK_EXT = IMAGE_EXT | VOLUME_EXT | {".npy", ".npz"}
ANN_EXT = {".xml", ".json", ".txt", ".csv", ".tsv"}
ANN_MAX = 500 * 2**20       # annotation files fetched by `fetch` (COCO json of a large set can be ~100s of MB)
MAX_LAYERS = 32             # mask files combined per image (TotalSegmentator ships 117 per case: the largest are kept)
MAX_REGIONS = 1000          # regions written per image (nuclei maps can have more; n_regions_total says so)
MAX_COLOURS = 64            # an RGB "mask" with more colours is an overlay or a photo
MIN_REGION_PX = 2
SPATIAL = {"bounding_box", "rotated_box", "segmentation_mask", "instance_mask", "polygon", "keypoints", "points"}
OWN_FILES = {"_status.json", "_files.jsonl", "_hf_dense.jsonl", "hf_rows.jsonl"}

# ── names → pairing keys ────────────────────────────────────────────────────────────────────────────────────────
# Tokens that say what role a file or folder plays rather than which case it is; they are removed before matching.
IMAGE_ROLE = {"image", "images", "img", "imgs", "im", "input", "inputs", "raw", "rgb", "orig", "original", "originals",
              "source", "src", "frame", "frames", "volume", "volumes", "vol", "scan", "scans", "photo", "photos", "data",
              "picture", "pictures", "x", "t", "ct", "mr", "mri", "t1", "t1w", "t1ce", "t1c", "t1n", "t2", "t2w", "t2f",
              "flair", "adc", "dwi", "pd", "pdw", "tiff"}
MASK_ROLE = {"mask", "masks", "msk", "msks", "seg", "segs", "segmentation", "segmentations", "segmented", "label",
             "labels", "labeled", "labelled", "lbl", "lbls", "gt", "groundtruth", "ground", "truth", "annotation",
             "annotations", "annotated", "annot", "annots", "anno", "annos", "target", "targets", "binary", "bin", "man",
             "instance", "instances", "ids", "mark", "marks", "manual", "contour", "contours", "semantic", "map", "maps",
             "full"}
SPLIT_ROLE = {"train", "training", "test", "testing", "val", "valid", "validation", "dev", "tr", "ts", "set"}
PIPE_ROLE = {"zip", "members", "extracted", "archives", "tar", "files", "file", "hf", "jpg", "jpeg", "png", "tif",
             "bmp", "nii", "gz", "dcm", "mha", "nrrd", "npy", "xml", "json", "txt", "csv", "yolo", "coco", "voc",
             "labelme", "dataset", "datasets"}
ROLE = IMAGE_ROLE | MASK_ROLE | SPLIT_ROLE | PIPE_ROLE
STRONG_MASK = re.compile(r"mask|seg|label|gt|ground|annotation|target|manual", re.I)
CHANNEL = re.compile(r"^000\d$")          # nnU-Net channel suffix: case_0000.nii.gz ↔ case.nii.gz
ANNOTATOR = re.compile(r"^(exp|rater|reader|annotator|expert|rad|grader|observer|obs)\d+$|^rad[a-z]$|^consensus$")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def strip_ext(name):
    e = ext(name)
    return name[: -len(e)] if e else name


def sep_tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", _CAMEL.sub("_", s).lower()) if t]


def core_tokens(s):
    """Tokens that identify the case: role words, nnU-Net channels, letter/digit runs re-split, numbers without padding."""
    t = sep_tokens(s)
    t = [x for i, x in enumerate(t) if x not in ROLE and not (CHANNEL.match(x) and len(t) > 1 and i == len(t) - 1)]
    out = []
    for x in t:
        out += re.findall(r"[a-z]+|\d+", x)
    return [str(int(x)) if x.isdigit() else x for x in out if x not in ROLE]


def norm_dirs(rel):
    """Directory path with role-only folders dropped (images/, masks/, imagesTr/, zip_members/ ...)."""
    out = []
    for d in Path(rel).parts[:-1]:
        c = core_tokens(d)
        if c:
            out.append("_".join(c))
    return out


def file_keys(rel):
    stem = strip_ext(Path(rel).name)
    strict = "_".join(sep_tokens(stem))
    core = core_tokens(stem)
    return strict, "_".join(core), core


def bids_key(rel):
    m = re.search(r"(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?", Path(rel).name)
    return "_".join(x for x in m.groups() if x) if m else ""


class PairIndex:
    """Annotation files (masks and per-image xml/json/txt/csv) indexed for matching to media files.

    Levels, best first: identical stem (parallel folders) · same core key (role words removed: x.png ↔ x_mask.png,
    'Image 98' ↔ 'Seg 98', case_0000 ↔ case) · core key plus a class name (C3L_03126 ↔ C3L-03126_pancreas) · same
    owner folder (case/image.nii.gz ↔ case/label.nii.gz, s0001/ct ↔ s0001/segmentations/liver) · BIDS subject/session.
    Numeric-only keys must come from parallel folders, otherwise '61.png' would pair across unrelated subsets."""

    def __init__(self, names, ignore=None):
        self.names = [n for n in names if not (ignore and re.search(ignore, n))]
        self.strict, self.core, self.group, self.bids = (defaultdict(list) for _ in range(4))
        self.prefix = defaultdict(list)
        for n in self.names:
            s, c, ct = file_keys(n)
            self.strict[s].append(n)
            if c:
                self.core[c].append(n)
                for k in range(1, min(len(ct), 6)):
                    self.prefix["_".join(ct[:k])].append((n, ct[k:]))
            self.group["/".join(norm_dirs(n))].append(n)
            if (b := bids_key(n)):
                self.bids[b].append(n)

    @staticmethod
    def _score(media_dirs, cand):
        d = norm_dirs(cand)
        k = 0
        while k < min(len(media_dirs), len(d)) and media_dirs[k] == d[k]:
            k += 1
        return k - 0.1 * (len(media_dirs) + len(d) - 2 * k)

    def _best(self, media_rel, cands, numeric):
        md = norm_dirs(media_rel)
        cands = [c for c in dict.fromkeys(cands) if c != media_rel]
        if numeric:  # a bare number only identifies the case inside parallel folders
            cands = [c for c in cands if norm_dirs(c)[: len(md)] == md or md[: len(norm_dirs(c))] == norm_dirs(c)]
        if not cands:
            return []
        sc = {c: self._score(md, c) for c in cands}
        top = max(sc.values())
        return sorted(c for c in cands if sc[c] >= top - 1e-9)

    def match(self, media_rel):
        """(level, [annotation names]) for one media file; level is '' when nothing pairs."""
        s, c, ct = file_keys(media_rel)
        numeric = bool(c) and all(x.isdigit() for x in ct)
        if (got := self._best(media_rel, self.strict.get(s, []), numeric)):
            return "stem", got
        if c and (got := self._best(media_rel, self.core.get(c, []), numeric)):
            return "core", got
        if c:
            extra = [n for n, rest in self.prefix.get(c, []) if len(rest) <= 3 and file_keys(n)[1] != c]
            if (got := self._best(media_rel, extra, numeric)):
                return "core+class", got
        grp = "/".join(norm_dirs(media_rel))
        if not c or not self.core.get(c):
            members = [n for n in self.group.get(grp, []) if file_keys(n)[1] != c or not c]
            # a generic image name owns the annotations in its folder; a specific one only those not naming another case
            if members and (not c or all(not self.core.get(file_keys(n)[1]) or file_keys(n)[1] == c for n in members)):
                if not c or len(members) <= MAX_LAYERS * 4:
                    return "folder", sorted(members)
        if (b := bids_key(media_rel)) and (got := self._best(media_rel, self.bids.get(b, []), False)):
            return "bids", got
        return "", []


def layer_name(media_rel, ann_rel):
    """What distinguishes this annotation from its image: a class folder (vessel_mask/), a class stem (liver.nii.gz,
    C3L_03126_pancreas) or an annotator suffix; '' when nothing does."""
    m_core = set(core_tokens(strip_ext(Path(media_rel).name)))
    raw = sep_tokens(strip_ext(Path(ann_rel).name))
    stem_extra = [t for t in core_tokens(strip_ext(Path(ann_rel).name)) if t not in m_core]
    if any(re.search(r"[a-z]", t) for t in stem_extra):
        return "_".join(stem_extra)
    md = set(norm_dirs(media_rel))
    dir_extra = [d for d in norm_dirs(ann_rel) if d not in md]
    if dir_extra:
        return dir_extra[-1]
    if stem_extra:  # numbers only: annotator or part index; keep the raw words so 'annotation_3' stays readable
        return "_".join(t for t in raw if t not in m_core and t not in IMAGE_ROLE)
    return ""


# ── masks → label maps ──────────────────────────────────────────────────────────────────────────────────────────
def decode_mask(a, lossy=False):
    """Array from a mask file → (label map uint16 or None, kind, classes). kind is binary (0/1), labels (class values),
    instances (one id per object), empty (all background) or why it is not a mask (overlay, not_mask). lossy: the file
    was a JPEG, so a black/white mask arrives with compression noise around both levels."""
    a = np.asarray(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3]
        if (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all():
            a = rgb[..., 0]
        else:
            flat = rgb.reshape(-1, 3)
            cols, inv, cnt = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
            if len(cols) > MAX_COLOURS:
                grey = rgb.astype(np.int16).max(-1)
                if ((grey < 32) | (grey > 223)).mean() >= 0.97:  # lossy (JPEG) black/white mask saved as RGB
                    a = rgb.max(-1)
                else:
                    return None, "overlay", {}
            else:
                dark = np.where((cols.astype(int).sum(1) == 0))[0]
                bg = int(dark[0]) if len(dark) else int(np.argmax(cnt))
                order = [bg] + [i for i in range(len(cols)) if i != bg]
                remap = np.zeros(len(cols), np.uint16)
                for new, old in enumerate(order):
                    remap[old] = new
                lab = remap[inv.reshape(-1)].reshape(rgb.shape[:2])
                classes = {new: "rgb({},{},{})".format(*map(int, cols[old])) for new, old in enumerate(order) if new}
                return lab, ("empty" if len(cols) == 1 else "binary" if len(cols) == 2 else "labels"), classes
    if a.ndim != 2:
        return None, "not_mask", {}
    if a.dtype.kind == "b":
        a = a.astype(np.uint8)
    if a.dtype.kind == "f":
        if not np.isfinite(a).all() or (np.abs(a - np.round(a)) > 1e-3).any():
            return None, "not_mask", {}  # probability or intensity maps
        a = np.round(a).astype(np.int64)
    vals = np.unique(a)
    if vals.min() < 0:
        return None, "not_mask", {}
    if len(vals) == 1:
        return np.zeros(a.shape, np.uint16), "empty", {}
    if len(vals) == 2 and vals[0] == 0:
        if vals[1] in (1, 255, 65535) or vals[1] > 65535:  # a 0/1, 0/255 or 0/65535 mask file, or one instance
            return (a > 0).astype(np.uint16), "binary", {1: ""}
        return a.astype(np.uint16), "labels", {int(vals[1]): ""}  # one class of a label map keeps its value (and name)
    if a.dtype == np.uint8 and len(vals) > 2 and (lossy or len(vals) > 64):
        lo, hi = int(vals.min()), int(vals.max())
        near = ((a <= lo + 32) | (a >= hi - 32)).mean()
        if hi - lo >= 128 and near >= 0.97:  # lossy (JPEG) black/white mask
            return (a > (lo + hi) / 2).astype(np.uint16), "binary", {1: ""}
    if len(vals) <= 64:
        if vals.max() > 65535:
            return None, "not_mask", {}
        return a.astype(np.uint16), "labels", {int(v): "" for v in vals if v}
    if (a == 0).mean() >= 0.3:  # many ids on a zero background: an instance map
        if vals.max() > 65535:
            _, inv = np.unique(a, return_inverse=True)
            a = inv.reshape(a.shape)
        return a.astype(np.uint16), "instances", {}
    return None, "not_mask", {}


def load_mask_array(p: Path):
    """Raw array of a 2D mask file (PNG/JPG/BMP/TIFF/NPY). Multi-page TIFFs come back as (pages, H, W)."""
    e = ext(p)
    if e in (".npy", ".npz"):
        a = np.load(p, allow_pickle=False)
        return a[a.files[0]] if hasattr(a, "files") else a
    if e in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(p)
    im = Image.open(p)
    if im.mode == "P":
        return np.asarray(im)  # palette indices are the class values
    if getattr(im, "n_frames", 1) > 1:
        frames = []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(np.asarray(im))
        return np.stack(frames)
    return np.asarray(im)


# ── annotation files → shapes in the original image's pixels ───────────────────────────────────────────────────
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def class_names_near(root: Path):
    """YOLO class names from classes.txt / obj.names / data.yaml anywhere in the dataset."""
    for p in list(root.rglob("classes.txt")) + list(root.rglob("*.names")):
        names = [l.strip() for l in p.read_text(errors="ignore").splitlines() if l.strip()]
        if names and len(names) <= 1000:
            return names
    for p in list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml")):
        t = p.read_text(errors="ignore")
        m = re.search(r"^names\s*:\s*\[(.*?)\]", t, re.M | re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        m = re.search(r"^names\s*:\s*\n((?:\s+.+\n?)+)", t, re.M)
        if m:
            return [re.sub(r"^\s*(-|\d+\s*:)\s*", "", l).strip().strip("'\"") for l in m.group(1).splitlines() if l.strip()]
    return []


def parse_yolo(text, names=()):
    """YOLO detection (cls cx cy w h), YOLO segmentation (cls x1 y1 x2 y2 …) or DOTA (x1 y1 … x4 y4 class [difficult]),
    all normalized except DOTA. Returns shapes with norm=True for normalized coordinates."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        nums = [_num(x) for x in p]
        if len(p) >= 9 and all(v is not None for v in nums[:8]) and nums[8] is None:  # DOTA oriented box
            xy = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
            out.append({"type": "polygon", "label": p[8], "xy": xy, "norm": False})
            continue
        if any(v is None for v in nums):
            return []  # not a YOLO file (free text, clinical notes)
        c = int(nums[0]) if float(nums[0]).is_integer() else None
        if c is None:
            return []
        lab = names[c] if 0 <= c < len(names) else str(c)
        rest = nums[1:]
        if len(rest) == 4:
            cx, cy, w, h = rest
            out.append({"type": "box", "label": lab, "box": [cx - w / 2, cy - h / 2, w, h], "norm": max(rest) <= 1.0001})
        elif len(rest) >= 6 and len(rest) % 2 == 0:
            out.append({"type": "polygon", "label": lab, "xy": [[rest[i], rest[i + 1]] for i in range(0, len(rest), 2)],
                        "norm": max(rest) <= 1.0001})
        elif len(rest) == 2:
            out.append({"type": "point", "label": lab, "xy": rest, "norm": max(rest) <= 1.0001})
        else:
            return []
    return out


def parse_voc(root_el):
    out = []
    for ob in root_el.iter("object"):
        lab = (ob.findtext("name") or "").strip()
        bb = ob.find("bndbox")
        if bb is not None:
            x0, y0, x1, y1 = (_num(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            if None not in (x0, y0, x1, y1):
                out.append({"type": "box", "label": lab, "box": [x0, y0, x1 - x0, y1 - y0], "norm": False})
        pg = ob.find("polygon")
        if pg is not None:
            vals = [_num(c.text) for c in pg]
            if len(vals) >= 6 and None not in vals:
                out.append({"type": "polygon", "label": lab, "xy": [[vals[i], vals[i + 1]] for i in range(0, len(vals) - 1, 2)], "norm": False})
        pt = ob.find("point")
        if pt is not None and _num(pt.findtext("x")) is not None:
            out.append({"type": "point", "label": lab, "xy": [_num(pt.findtext("x")), _num(pt.findtext("y"))], "norm": False})
    return out


def _cvat_shapes(img_el):
    out = []
    for el in img_el:
        lab = el.get("label", "")
        if el.tag == "box":
            x0, y0, x1, y1 = (_num(el.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            if None not in (x0, y0, x1, y1):
                out.append({"type": "box", "label": lab, "box": [x0, y0, x1 - x0, y1 - y0], "norm": False})
        elif el.tag in ("polygon", "polyline", "points"):
            xy = [[_num(v) for v in pt.split(",")] for pt in (el.get("points") or "").split(";") if "," in pt]
            if xy:
                out.append({"type": "point" if el.tag == "points" and len(xy) == 1 else "polygon", "label": lab,
                            "xy": xy[0] if el.tag == "points" and len(xy) == 1 else xy, "norm": False})
    return out


def parse_labelme(j):
    out = []
    for s in j.get("shapes", []) or []:
        pts = s.get("points") or []
        lab, t = str(s.get("label", "")), s.get("shape_type") or "polygon"
        if not pts:
            continue
        if t == "rectangle" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            out.append({"type": "box", "label": lab, "box": [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)], "norm": False})
        elif t == "point" or len(pts) == 1:
            out.append({"type": "point", "label": lab, "xy": list(pts[0]), "norm": False})
        elif t == "circle" and len(pts) >= 2:
            (cx, cy), (px, py) = pts[0], pts[1]
            r = float(np.hypot(px - cx, py - cy))
            out.append({"type": "box", "label": lab, "box": [cx - r, cy - r, 2 * r, 2 * r], "norm": False})
        else:
            out.append({"type": "polygon", "label": lab, "xy": [list(p) for p in pts], "norm": False})
    return out


def _coco_rle_counts(s):
    """Decode a compressed COCO RLE string into run lengths (pycocotools' LEB128-like format)."""
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def coco_rle_mask(seg, h, w):
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = _coco_rle_counts(counts)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T  # column-major


class CocoIndex:
    """A COCO-style json: shapes per image file, looked up by basename or stem."""

    def __init__(self, j):
        cats = {c.get("id"): str(c.get("name", c.get("id"))) for c in j.get("categories", []) or []}
        self.by = {}
        imgs = {im.get("id"): im for im in j.get("images", []) or []}
        anns = defaultdict(list)
        for a in j.get("annotations", []) or []:
            anns[a.get("image_id")].append(a)
        for iid, im in imgs.items():
            fn = str(im.get("file_name", ""))
            shapes = []
            for a in anns.get(iid, []):
                lab = cats.get(a.get("category_id"), str(a.get("category_id", "")))
                bb = a.get("bbox")
                if bb and len(bb) == 4 and None not in [_num(v) for v in bb]:
                    shapes.append({"type": "box", "label": lab, "box": [float(v) for v in bb], "norm": False})
                seg = a.get("segmentation")
                if isinstance(seg, list):
                    for poly in seg:
                        if isinstance(poly, list) and len(poly) >= 6:
                            shapes.append({"type": "polygon", "label": lab, "xy": [[poly[i], poly[i + 1]] for i in range(0, len(poly) - 1, 2)], "norm": False})
                elif isinstance(seg, dict) and "counts" in seg and im.get("height") and im.get("width"):
                    shapes.append({"type": "rle", "label": lab, "seg": seg, "hw": [im["height"], im["width"]]})
                kp = a.get("keypoints")
                if isinstance(kp, list) and len(kp) >= 3:
                    for i in range(0, len(kp) - 2, 3):
                        if kp[i + 2]:
                            shapes.append({"type": "point", "label": lab, "xy": [kp[i], kp[i + 1]], "norm": False})
            self.by[Path(fn).name.lower()] = (shapes, im)
            self.by.setdefault(strip_ext(Path(fn).name).lower(), (shapes, im))

    def get(self, rel):
        n = Path(rel).name.lower()
        return self.by.get(n) or self.by.get(strip_ext(n)) or ((None, None))


_FILE_COL = re.compile(r"^(file(_?name)?|image(_?(id|name|file|path))?|img(_?name)?|name|path|filepath|image_?path|id|sample_?id)$", re.I)
_LABEL_COL = re.compile(r"^(label|class(_?name)?|category(_?name)?|finding|type|name|object|cls)$", re.I)
_BOX_SETS = [("xmin", "ymin", "xmax", "ymax", "xyxy"), ("x_min", "y_min", "x_max", "y_max", "xyxy"), ("x1", "y1", "x2", "y2", "xyxy"),
             ("left", "top", "right", "bottom", "xyxy"), ("x", "y", "w", "h", "xywh"), ("x", "y", "width", "height", "xywh"),
             ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "xywh"), ("cx", "cy", "w", "h", "cxcywh")]
_POINT_SETS = [("x", "y"), ("cx", "cy"), ("center_x", "center_y"), ("centroid_x", "centroid_y"), ("coord_x", "coord_y"),
               ("point_x", "point_y"), ("col", "row")]


def table_shapes(df):
    """{file key: [shapes]} from a table with a file column and box or point columns; {} if it has neither."""
    cols = {c.lower().strip(): c for c in df.columns}
    fcol = next((cols[c] for c in cols if _FILE_COL.match(c)), None)
    lcol = next((cols[c] for c in cols if _LABEL_COL.match(c) and cols[c] != fcol), None)
    box = next((s for s in _BOX_SETS if all(k in cols for k in s[:4])), None)
    pt = None if box else next((s for s in _POINT_SETS if all(k in cols for k in s)), None)
    if not (box or pt):
        return {}
    out = defaultdict(list)
    for _, r in df.iterrows():
        key = str(r[fcol]) if fcol else ""
        lab = str(r[lcol]) if lcol else ""
        if box:
            v = [_num(r[cols[k]]) for k in box[:4]]
            if None in v:
                continue
            a, b, c, d = v
            bx = [a, b, c - a, d - b] if box[4] == "xyxy" else [a, b, c, d] if box[4] == "xywh" else [a - c / 2, b - d / 2, c, d]
            out[key].append({"type": "box", "label": lab, "box": bx, "norm": max(abs(x) for x in v) <= 1.0001})
        else:
            v = [_num(r[cols[k]]) for k in pt]
            if None in v:
                continue
            xy = v[::-1] if pt == ("col", "row") else v
            out[key].append({"type": "point", "label": lab, "xy": xy, "norm": max(abs(x) for x in v) <= 1.0001})
    return dict(out)


def read_annotation_file(p: Path, names=()):
    """Shapes in a per-image annotation file (YOLO txt, VOC xml, labelme json, per-image csv). None if not one."""
    e = ext(p)
    try:
        if p.stat().st_size > ANN_MAX:
            return None
        if e == ".txt":
            s = parse_yolo(p.read_text(errors="ignore"), names)
            return s or None
        if e == ".xml":
            r = ET.parse(p).getroot()
            return parse_voc(r) if r.tag == "annotation" else None
        if e == ".json":
            j = json.loads(p.read_text(errors="ignore"))
            if isinstance(j, dict) and "shapes" in j:
                return parse_labelme(j)
            if isinstance(j, dict) and "annotations" in j and len(j.get("images") or []) == 1:  # COCO, one image per file
                shapes, _ = next(iter(CocoIndex(j).by.values()))
                return shapes or None
            return None
        if e in (".csv", ".tsv"):
            import pandas as pd
            df = pd.read_csv(p, sep=None, engine="python", nrows=200_000)
            t = table_shapes(df)  # a per-image table lists one object per row; a file column, if any, names this image
            return t.get("") or (next(iter(t.values())) if len(t) == 1 else None)
    except Exception:
        return None
    return None


class TableIndex:
    """Dataset-level annotation files (COCO json, CVAT xml, csv/tsv/json tables) keyed by image basename or stem."""

    def __init__(self, root: Path, files):
        self.cocos, self.rows = [], defaultdict(list)
        for p in files:
            e = ext(p)
            try:
                if p.stat().st_size > ANN_MAX:
                    continue
                if e == ".json":
                    j = json.loads(p.read_text(errors="ignore"))
                    if isinstance(j, dict) and "images" in j and "annotations" in j:
                        self.cocos.append(CocoIndex(j))
                    elif isinstance(j, (list, dict)):
                        import pandas as pd
                        recs = j if isinstance(j, list) else next((v for v in j.values() if isinstance(v, list)), None)
                        if recs and isinstance(recs[0], dict):
                            self._add(table_shapes(pd.json_normalize(recs)))
                elif e == ".xml":
                    r = ET.parse(p).getroot()
                    if r.tag == "annotations":  # CVAT for images
                        for im in r.iter("image"):
                            self._add({im.get("name", ""): _cvat_shapes(im)})
                elif e in (".csv", ".tsv"):
                    import pandas as pd
                    self._add(table_shapes(pd.read_csv(p, sep=None, engine="python", nrows=2_000_000)))
            except Exception:
                continue

    def _add(self, t):
        for k, v in t.items():
            if k:
                kk = Path(str(k).replace("\\", "/")).name.lower()
                self.rows[kk] += v
                if strip_ext(kk) != kk:
                    self.rows[strip_ext(kk)] += v

    def get(self, rel):
        n = Path(rel).name.lower()
        for c in self.cocos:
            shapes, im = c.get(rel)
            if shapes is not None:
                return shapes, "coco"
        got = self.rows.get(n) or self.rows.get(strip_ext(n))
        return (got, "table") if got else (None, "")


# ── geometry: original pixels → prepared pixels ────────────────────────────────────────────────────────────────
EXIF_OPS = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
            6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}


def exif_point(x, y, op, W, H):
    """Where (x, y) of a W×H image lands after PIL's transpose(op)."""
    return {Image.FLIP_LEFT_RIGHT: (W - x, y), Image.ROTATE_180: (W - x, H - y), Image.FLIP_TOP_BOTTOM: (x, H - y),
            Image.TRANSPOSE: (y, x), Image.ROTATE_270: (H - y, x), Image.TRANSVERSE: (H - y, W - x),
            Image.ROTATE_90: (y, W - x)}.get(op, (x, y))


class Frame:
    """How prepare.py turned the original pixels into the prepared JPEG: an optional EXIF transpose of a W0×H0 image,
    then a resize to (w, h)."""

    def __init__(self, W0, H0, w, h, op=None):
        self.W0, self.H0, self.w, self.h, self.op = W0, H0, w, h, op
        W1, H1 = (H0, W0) if op in (Image.TRANSPOSE, Image.TRANSVERSE, Image.ROTATE_90, Image.ROTATE_270) else (W0, H0)
        self.sx, self.sy = w / W1, h / H1

    def pt(self, x, y, norm=False):
        if norm:
            x, y = x * self.W0, y * self.H0
        if self.op is not None:
            x, y = exif_point(x, y, self.op, self.W0, self.H0)
        return [round(x * self.sx, 1), round(y * self.sy, 1)]

    def box(self, b, norm=False):
        x, y, bw, bh = b
        pts = [self.pt(x, y, norm), self.pt(x + bw, y + bh, norm)]
        x0, x1 = sorted(p[0] for p in pts)
        y0, y1 = sorted(p[1] for p in pts)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(self.w), x1), min(float(self.h), y1)
        return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)] if x1 > x0 and y1 > y0 else None

    def label_map(self, lab):
        im = Image.fromarray(lab.astype(np.uint16) if lab.max() > 255 else lab.astype(np.uint8))
        if self.op is not None:
            im = im.transpose(self.op)
        return np.asarray(im.resize((self.w, self.h), Image.NEAREST))


def image_frame(p: Path, w, h):
    """Frame of a prepared 2D image, mirroring prepare.read_image: EXIF turns apply to single-frame 8-bit images only."""
    e = ext(p)
    if e in (".npy", ".npz"):
        a = np.load(p, mmap_mode="r", allow_pickle=False) if e == ".npy" else load_mask_array(p)
        a = np.squeeze(a)
        h0, w0 = (a.shape[1], a.shape[2]) if a.ndim == 3 and a.shape[0] in (2, 3, 4, 5) and a.shape[-1] > 5 else a.shape[:2]
        return Frame(int(w0), int(h0), w, h)
    if e in (".dcm",) or kind(p.name) == "other":
        import pydicom
        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        return Frame(int(ds.Columns), int(ds.Rows), w, h)
    if e in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(p) as tf:
                pg = tf.pages[0]
                if pg.bitspersample > 8 and pg.samplesperpixel > 1:
                    return Frame(int(pg.imagewidth), int(pg.imagelength), w, h)
        except Exception:
            pass
    im = Image.open(p)
    W0, H0 = im.size
    op = None
    if getattr(im, "n_frames", 1) == 1 and im.mode not in ("I;16", "I;16B", "I", "F"):
        try:
            op = EXIF_OPS.get(im.getexif().get(0x0112, 1))
        except Exception:
            op = None
    return Frame(W0, H0, w, h, op)


def looks_like_figure(lab):
    """A plotted mask (matplotlib: white margins, tick labels, a black axes box with the shape inside) rather than a
    mask: foreground frames the image and the largest background component is a filled rectangle. PAPILA's HF copy
    ships its disc/cup masks like this; they do not line up with the photo."""
    import cv2
    fg = lab > 0
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    if border.mean() < 0.9 or fg.mean() > 0.9:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), connectivity=4)
    if n < 2:
        return False
    x, y, w, h, area = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return w * h >= 0.3 * fg.size and area >= 0.85 * w * h


def fit_mask_2d(arr, frame: Frame, lossy=False):
    """A mask array for a 2D image → prepared-frame label map, or (None, reason) when sizes cannot correspond."""
    lab, k, cls = decode_mask(arr, lossy)
    if lab is None:
        return None, k, cls
    if k in ("binary", "labels") and looks_like_figure(lab):
        return None, "figure", cls
    Hm, Wm = lab.shape
    if (Wm, Hm) != (frame.W0, frame.H0):
        if frame.op is not None and (Wm, Hm) == (frame.H0, frame.W0):  # mask stored already upright
            im = Image.fromarray(lab.astype(np.uint16) if lab.max() > 255 else lab.astype(np.uint8))
            return np.asarray(im.resize((frame.w, frame.h), Image.NEAREST)), k, cls
        if abs(Wm / Hm - frame.W0 / frame.H0) > 0.02 * frame.W0 / frame.H0:
            return None, "size_mismatch", cls
    return frame.label_map(lab), k, cls


_LOC = re.compile(r"^(sagittal|coronal|axial|slice|frame|item)(\d+)/(\d+)$")


def volume_label_slices(label_path: Path, image_path: Path, locs):
    """{loc: 2D label slice} for slices prepare.read_volume took from image_path, read the same way from the label
    volume (canonical orientation and rot90 for NIfTI, the shortest axis for SimpleITK volumes)."""
    e = ext(image_path)
    out = {}
    if e in (".nii", ".nii.gz"):
        import nibabel as nib
        img, lab = nib.load(str(image_path)), nib.load(str(label_path))
        if tuple(img.shape[:3]) != tuple(lab.shape[:3]):
            return {}
        a = np.asanyarray(nib.as_closest_canonical(lab).dataobj)
        if a.ndim == 4:
            a = (a.argmax(-1) + 1) * (a.max(-1) > 0) if a.shape[-1] <= 16 else a[..., 0]
        for loc in locs:
            m = _LOC.match(loc or "")
            if not m or m.group(1) not in ("sagittal", "coronal", "axial"):
                continue
            ax, i = ("sagittal", "coronal", "axial").index(m.group(1)), int(m.group(2))
            if i < a.shape[ax]:
                out[loc] = np.rot90(np.take(a, i, axis=ax))
        return out
    if e in (".mha", ".mhd", ".nrrd"):
        import SimpleITK as sitk
        ia = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        la = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        if ia.ndim == 4:
            ia = ia[..., 0] if ia.shape[-1] < ia.shape[0] else ia[0]
        if la.ndim == 4:
            la = la[..., 0] if la.shape[-1] < la.shape[0] else la[0]
        if ia.shape[:3] != la.shape[:3] or ia.ndim != 3:
            return {}
        ax = int(np.argmin(ia.shape[:3]))
        lv = np.moveaxis(la, ax, 0)
        for loc in locs:
            m = _LOC.match(loc or "")
            if m and m.group(1) == "slice" and int(m.group(2)) < lv.shape[0]:
                out[loc] = lv[int(m.group(2))]
        return out
    if e in (".npy", ".npz"):
        ia = np.load(image_path, mmap_mode="r") if e == ".npy" else None
        la = load_mask_array(label_path)
        if ia is None or ia.shape[0] != la.shape[0]:
            return {}
        for loc in locs:
            m = _LOC.match(loc or "")
            if m and m.group(1) == "item" and int(m.group(2)) < la.shape[0]:
                out[loc] = la[int(m.group(2))]
        return out
    if e in (".tif", ".tiff"):
        la = load_mask_array(label_path)
        for loc in locs:
            m = _LOC.match(loc or "")
            if m and m.group(1) == "frame" and la.ndim == 3 and la.shape[0] == int(m.group(3)):
                out[loc] = la[int(m.group(2))]
        return out
    return out


# ── regions ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _groups(labels, values):
    """Boxes, areas and most-interior points per id of an integer map (ids > 0), without a per-id pass over the image."""
    import cv2
    ys, xs = np.nonzero(labels)
    if not len(ys):
        return []
    ids = labels[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    dist = values[ys, xs] if values is not None else None
    out = []
    for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
        y0, y1, x0, x1 = ys[s:e].min(), ys[s:e].max(), xs[s:e].min(), xs[s:e].max()
        k = s + int(np.argmax(dist[s:e])) if dist is not None else s + (e - s) // 2
        out.append((int(ids[s]), [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)], int(e - s), [int(xs[k]), int(ys[k])]))
    return out


def layer_regions(lab, kind_, classes, layer_i):
    import cv2
    regs = []
    if kind_ == "instances":
        binary = (lab > 0).astype(np.uint8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3) if binary.any() else None
        for iid, box, area, seed in _groups(lab, dist):
            if area >= MIN_REGION_PX:
                regs.append({"layer": layer_i, "label": classes.get(iid) or "instance", "box": box, "area": area, "seed": seed})
        return regs
    for v in [int(x) for x in np.unique(lab) if x]:
        m = (lab == v).astype(np.uint8)
        n, cc = cv2.connectedComponents(m, connectivity=8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        for cid, box, area, seed in _groups(cc, dist):
            if area >= MIN_REGION_PX:
                regs.append({"layer": layer_i, "label": classes.get(v) or str(v), "box": box, "area": area, "seed": seed})
    return regs


def box_iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax1, bx1) - max(a[0], b[0])), max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


UNNAMED = re.compile(r"^(rgb\(.*\)|\d+|foreground|instance)$")


def name_regions_by_boxes(regions, boxes, layers):
    """Mask values without a name (a colour, a number, 'foreground') take the label of the labelled boxes their regions
    coincide with (IoU ≥ 0.5): a class when ≥ 60% of its regions agree, else each matched instance region alone.
    MurineCyto ships colour masks and per-image COCO boxes with the cell types."""
    votes, hits = defaultdict(Counter), {}
    for gi, g in enumerate(regions):
        if not UNNAMED.match(str(g["label"])):
            continue
        best = max(((box_iou(g["box"], b["box"]), b["label"]) for b in boxes if b.get("label")), default=(0, None))
        if best[0] >= 0.5:
            votes[(g["layer"], g["label"])][best[1]] += 1
            hits[gi] = best[1]
    for (li, lab), c in votes.items():
        name, n = c.most_common(1)[0]
        if layers[li]["kind"] == "instances" or n < 0.6 * sum(c.values()):
            continue
        for v, nm in layers[li]["classes"].items():
            if nm == lab:
                layers[li]["classes"][v] = name
        for g in regions:
            if g["layer"] == li and g["label"] == lab:
                g["label"] = name
    for gi, name in hits.items():
        if layers[regions[gi]["layer"]]["kind"] == "instances":
            regions[gi]["label"] = name


# ── per-dataset build ───────────────────────────────────────────────────────────────────────────────────────────
def load_fixes():
    p = PLAN / "dense_fixes.json"
    return json.loads(p.read_text()) if p.exists() else {}


def extract_dense_members(root: Path):
    """Masks and annotation files inside archives downloaded whole: prepare.extract_archives leaves them out of zips."""
    import zipfile
    arch = root / "archives"
    for a in [p for p in arch.rglob("*") if p.is_file()] if arch.exists() else []:
        done = root / "extracted" / (a.name + ".dense.ok")
        if done.exists() or not zipfile.is_zipfile(a):
            continue
        dest = root / "extracted" / a.name.split(".")[0]
        with zipfile.ZipFile(a) as z:
            for m in z.infolist():
                n = m.filename
                if m.is_dir() or ".." in n or n.startswith("/") or is_junk(n) or m.file_size > ANN_MAX:
                    continue
                if (kind(n) == "mask" and ext(n) in MASK_EXT) or ext(n) in ANN_EXT:
                    if not (dest / n).exists():
                        try:
                            z.extract(m, dest)
                        except Exception:
                            pass
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok")


def dataset_class_names(root: Path):
    """Label-value names shipped with a dataset: nnU-Net dataset.json ("labels") or a CamVid-style class_dict.csv."""
    names, colours = {}, {}
    for p in root.rglob("dataset.json"):
        try:
            j = json.loads(p.read_text())
            for k, v in (j.get("labels") or {}).items():
                if str(v).isdigit():
                    names[int(v)] = k
                elif str(k).isdigit():
                    names[int(k)] = str(v)
        except Exception:
            pass
    for p in root.rglob("class_dict*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(p)
            cols = {c.lower().strip(): c for c in df.columns}
            if {"r", "g", "b"} <= set(cols):
                nc = next((cols[c] for c in cols if c in ("name", "class", "label", "class_name")), df.columns[0])
                for _, r in df.iterrows():
                    colours["rgb({},{},{})".format(int(r[cols["r"]]), int(r[cols["g"]]), int(r[cols["b"]]))] = str(r[nc])
        except Exception:
            pass
    return names, colours


def scan_annotations(root: Path, ignore=None):
    """(mask files, annotation files) under a dataset root, as relative paths. Annotation files are both per-image
    sidecars (YOLO txt, VOC xml, labelme json) and dataset-level tables (COCO json, CSV); readers sort them out."""
    masks, per_image = [], []
    for p in root.rglob("*"):
        if not p.is_file() or is_junk(str(p)) or p.name in OWN_FILES or p.suffix == ".part" or "archives" in p.relative_to(root).parts:
            continue
        rel = str(p.relative_to(root))
        if rel.startswith("hf_masks/") or (ignore and re.search(ignore, rel)):
            continue
        e = ext(rel)
        if e in MASK_EXT and (kind(rel) == "mask" or is_mask(rel)):
            masks.append(rel)
        elif e in ANN_EXT and rel != "hf_rows.jsonl":
            per_image.append(rel)
    return masks, per_image


def _hf_dense(root: Path):
    out = {}
    p = root / "_hf_dense.jsonl"
    if p.exists():
        for line in open(p):
            j = json.loads(line)
            out[j["file"]] = j
    return out


def hf_shapes(rec, W, H, box_format=None):
    """Boxes/points/polygons from the structured columns captured with an HF row (see capture_hf_row)."""
    out = []
    names = rec.get("class_names") or {}

    def lab_of(col, sub, v):
        nm = names.get(f"{col}.{sub}") or names.get(col) or []
        return nm[v] if isinstance(v, int) and 0 <= v < len(nm) else ("" if v is None else str(v))

    def add_boxes(col, boxes, labels, sub):
        for i, b in enumerate(boxes or []):
            if not isinstance(b, (list, tuple)) or len(b) != 4 or None in [_num(v) for v in b]:
                continue
            b = [float(v) for v in b]
            norm = max(abs(v) for v in b) <= 1.0001
            fmt = box_format or ("xyxy" if (b[2] > b[0] and b[3] > b[1] and (b[0] + b[2] > (1 if norm else W) * 1.02
                                                                            or b[1] + b[3] > (1 if norm else H) * 1.02)) else "xywh")
            if fmt == "xyxy":
                b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            elif fmt == "cxcywh":
                b = [b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3]]
            lab = labels[i] if isinstance(labels, list) and i < len(labels) else None
            out.append({"type": "box", "label": lab_of(col, sub, lab), "box": b, "norm": norm})

    for col, v in (rec.get("columns") or {}).items():
        if isinstance(v, dict):
            bkey = next((k for k in v if re.fullmatch(r"bbox(es)?|boxes", k, re.I)), None)
            lkey = next((k for k in v if re.fullmatch(r"labels?|category(_id)?|categories|class(es)?", k, re.I)), None)
            if bkey:
                add_boxes(col, v[bkey], v.get(lkey) if lkey else None, lkey or "label")
            pkey = next((k for k in v if re.fullmatch(r"points?|keypoints?|landmarks?|coords?|coordinates", k, re.I)), None)
            if pkey:
                for p_ in v[pkey] or []:
                    if isinstance(p_, (list, tuple)) and len(p_) >= 2 and None not in [_num(x) for x in p_[:2]]:
                        out.append({"type": "point", "label": col, "xy": [float(p_[0]), float(p_[1])], "norm": max(abs(float(p_[0])), abs(float(p_[1]))) <= 1.0001})
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for d in v:
                bkey = next((k for k in d if re.fullmatch(r"bbox|box", k, re.I)), None)
                lkey = next((k for k in d if re.fullmatch(r"labels?|category(_id)?|class(_name)?|name", k, re.I)), None)
                if bkey:
                    add_boxes(col, [d[bkey]], [d.get(lkey)] if lkey else None, lkey or "label")
                elif {"x", "y"} <= set(d):
                    out.append({"type": "point", "label": str(d.get(lkey, col)) if lkey else col, "xy": [d["x"], d["y"]],
                                "norm": max(abs(float(d["x"])), abs(float(d["y"]))) <= 1.0001})
        elif isinstance(v, list) and re.search(r"bbox|box", col, re.I):
            add_boxes(col, v if v and isinstance(v[0], (list, tuple)) else [v], None, "label")
        elif isinstance(v, list) and re.search(r"point|keypoint|landmark|coord|centroid", col, re.I):
            pts = v if v and isinstance(v[0], (list, tuple)) else [v[i:i + 2] for i in range(0, len(v) - 1, 2)]
            for p_ in pts:
                if len(p_) >= 2 and None not in [_num(x) for x in p_[:2]]:
                    out.append({"type": "point", "label": col, "xy": [float(p_[0]), float(p_[1])],
                                "norm": max(abs(float(p_[0])), abs(float(p_[1]))) <= 1.0001})
    return out


def _shapes_to_frame(shapes, frame: Frame, layers_out, fix_classes):
    """Annotation-file shapes → (boxes, points, polygons) in prepared pixels; RLE masks become extra layers."""
    boxes, points, polys = [], [], []
    for s in shapes or []:
        lab = fix_classes.get(str(s.get("label")), s.get("label", ""))
        if s["type"] == "box":
            b = frame.box(s["box"], s.get("norm", False))
            if b:
                boxes.append({"label": lab, "box": b})
        elif s["type"] == "point":
            x, y = s["xy"][:2]
            p = frame.pt(float(x), float(y), s.get("norm", False))
            if 0 <= p[0] <= frame.w and 0 <= p[1] <= frame.h:
                points.append({"label": lab, "xy": p})
        elif s["type"] == "polygon":
            xy = [frame.pt(float(x), float(y), s.get("norm", False)) for x, y in s["xy"]]
            polys.append({"label": lab, "xy": xy})
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            b = [max(0.0, min(xs)), max(0.0, min(ys))]
            b += [round(min(frame.w, max(xs)) - b[0], 1), round(min(frame.h, max(ys)) - b[1], 1)]
            if b[2] > 0 and b[3] > 0:
                boxes.append({"label": lab, "box": [round(b[0], 1), round(b[1], 1), b[2], b[3]], "from": "polygon"})
        elif s["type"] == "rle":
            h, w = s["hw"]
            m = coco_rle_mask(s["seg"], h, w)
            layers_out.append((frame.label_map(m), "binary", {1: lab}, lab or "rle"))
    return boxes, points, polys


def write_png(lab, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lab.astype(np.uint16) if lab.max() > 255 else lab.astype(np.uint8)).save(path)


def build_dataset(row, raw: Path, out: Path, fixes=None):
    """Sidecars for every prepared image of one dataset. Returns a stats dict."""
    import pandas as pd
    did = row["id"]
    root = raw / did
    man_p = out / "manifest" / f"{did}.parquet"
    st = Counter()
    if not man_p.exists():
        return {"id": did, "state": "not prepared"}
    if not root.exists():
        return {"id": did, "state": "no raw data"}
    fx = (fixes or {}).get(did, {})
    fix_classes = {str(k): v for k, v in (fx.get("classes") or {}).items()}
    extract_dense_members(root)
    masks, per_image = scan_annotations(root, fx.get("ignore"))
    m_index = PairIndex(masks)
    a_index = PairIndex(per_image)
    tables = TableIndex(root, [root / r for r in per_image if ext(r) in (".csv", ".tsv") or (ext(r) in (".json", ".xml")
                               and (_dataset_level(r) or (root / r).stat().st_size > 2**20))])
    yolo_names = class_names_near(root) if any(ext(r) == ".txt" for r in per_image) else []
    value_names, colour_names = dataset_class_names(root)
    hf = _hf_dense(root)
    df = pd.read_parquet(man_p)
    dense_dir = out / "dense" / did
    recs = []
    vol_cache = {}
    for r in df.itertuples(index=False):
        st["images"] += 1
        src, loc, k = r.source, r.loc or "", r.kind
        name = Path(r.image).stem
        p = root / src
        layers, sources, matches = [], [], []
        shapes = []
        try:
            if k in ("video", "wsi"):
                st[f"unsupported:{k}"] += 1
                continue
            if src.startswith("hf_images/") and (h := hf.get(Path(src).name)):
                frame = Frame(*Image.open(p).size, r.width, r.height)
                for m in h.get("masks", []):
                    arr = load_mask_array(root / m["path"])
                    lab, kd, cls = fit_mask_2d(arr, frame)
                    if lab is not None:
                        layers.append((lab, kd, cls, m.get("name", "")))
                        sources.append(m["path"])
                    else:
                        st[f"skip:{kd}"] += 1
                shapes += hf_shapes(h, frame.W0, frame.H0, fx.get("box_format"))
                matches.append("hf")
            elif k == "image" or (k == "volume" and not loc and ext(src) in (".dcm", "") ) or (k == "array" and not loc):
                frame = image_frame(p, r.width, r.height)
                level, cands = m_index.match(src)
                cands = cands[:MAX_LAYERS * 8]
                names_ = [layer_name(src, c) for c in cands]
                # x_1.png, x_2.png … or <case>/masks/<n>.png: one binary file per object → a single instance layer
                inst_set = len(cands) > 1 and all(re.fullmatch(r"[\d_]*", nm or "") for nm in names_)
                inst = None
                for c, nm in zip(cands, names_):
                    try:
                        arr = load_mask_array(root / c)
                    except Exception:
                        st["skip:unreadable"] += 1
                        continue
                    if arr.ndim == 3 and arr.shape[0] <= 64 and arr.shape[1:] == (frame.H0, frame.W0) and arr.shape[-1] not in (3, 4):
                        arr = np.where(arr > 0, np.arange(1, arr.shape[0] + 1)[:, None, None], 0).max(0)  # page per instance
                    lab, kd, cls = fit_mask_2d(arr, frame, lossy=ext(c) in (".jpg", ".jpeg"))
                    if lab is None:
                        st[f"skip:{kd}"] += 1
                        continue
                    if inst_set:
                        inst = np.zeros_like(lab, dtype=np.uint16) if inst is None else inst
                        inst[(lab > 0) & (inst == 0)] = inst.max() + 1 if (lab > 0).any() else 0
                        sources.append(c)
                        continue
                    if kd == "labels" and len(cls) > 3 and re.search(r"instance|_ids|nuclei|nucleus|cells?\b|objects", c, re.I):
                        kd = "instances"
                    layers.append((lab, kd, cls, nm))
                    sources.append(c)
                if inst is not None:
                    layers.append((inst, "instances" if inst.any() else "empty", {}, "instance"))
                if layers:
                    matches.append(f"mask:{level}")
                lvl_a, anns = a_index.match(src)
                for a_ in anns:
                    got = read_annotation_file(root / a_, yolo_names)
                    if got:
                        shapes += got
                        sources.append(a_)
                        matches.append(f"file:{lvl_a}")
                got, how = tables.get(src)
                if got:
                    shapes += got
                    matches.append(how)
            elif k in ("volume", "array", "image") and loc:
                level, cands = m_index.match(src)
                cands = sorted(cands, key=lambda c: -(root / c).stat().st_size)[:MAX_LAYERS]  # largest structures first
                for c in cands:
                    key = (src, c)
                    if key not in vol_cache:
                        locs = list(df.loc[df.source == src, "loc"])
                        try:
                            vol_cache[key] = volume_label_slices(root / c, p, locs)
                        except Exception:
                            vol_cache[key] = {}
                    sl = vol_cache[key].get(loc)
                    if sl is None:
                        st["skip:volume_mismatch"] += 1
                        continue
                    lab, kd, cls = decode_mask(sl)
                    if lab is None:
                        st[f"skip:{kd}"] += 1
                        continue
                    lab = np.asarray(Image.fromarray(lab.astype(np.uint16) if lab.max() > 255 else lab.astype(np.uint8))
                                     .resize((r.width, r.height), Image.NEAREST))
                    layers.append((lab, kd, cls, layer_name(src, c)))
                    sources.append(c)
                if layers:
                    matches.append(f"volume:{level}")
                frame = None
            else:
                st[f"unsupported:{k}"] += 1
                continue
            if frame is not None:
                boxes, points, polys = _shapes_to_frame(shapes, frame, layers, fix_classes)
            else:
                boxes, points, polys = [], [], []
        except Exception as ex:
            st[f"error:{type(ex).__name__}"] += 1
            continue
        if not layers and not (boxes or points or polys):
            st["no_annotation"] += 1
            continue
        if len(layers) > MAX_LAYERS:  # keep the largest structures
            layers = sorted(layers, key=lambda t: -int((t[0] > 0).sum()))[:MAX_LAYERS]
            st["layers_truncated"] += 1
        out_layers, regions = [], []
        for i, (lab, kd, cls, lname) in enumerate(layers):
            names = {}
            for v in (cls or {}):
                key = str(cls[v])
                if key.startswith("rgb("):  # colour indices differ per image, so only the colour can name the class
                    nm = fix_classes.get(key) or colour_names.get(key) or key
                else:
                    nm = fix_classes.get(str(v)) or fix_classes.get(key) or value_names.get(v) or key
                names[v] = nm or ((lname or "foreground") if kd == "binary" else f"{lname}:{v}" if lname else str(v))
            if kd == "binary" and 1 not in names:
                names[1] = fix_classes.get("1") or lname or "foreground"
            pth = dense_dir / f"{name}_{i}.png"
            write_png(lab, pth)
            out_layers.append({"path": str(pth.relative_to(out)), "name": lname or names.get(1, ""), "kind": kd,
                               "classes": {str(v): n for v, n in names.items()}})
            if kd != "empty":
                regions += layer_regions(lab, kd, names, i)
        if boxes and regions:
            name_regions_by_boxes(regions, boxes, out_layers)
        n_total = len(regions)
        regions = sorted(regions, key=lambda g: -g["area"])[:MAX_REGIONS]
        rec = {"image": r.image, "size": [int(r.width), int(r.height)], "match": sorted(set(matches)), "sources": sources,
               "layers": out_layers, "regions": regions, "n_regions_total": n_total, "boxes": boxes, "points": points,
               "polygons": polys}
        dense_dir.mkdir(parents=True, exist_ok=True)
        (dense_dir / f"{name}.json").write_text(json.dumps(rec))
        empty = all(l["kind"] == "empty" for l in out_layers) and not (boxes or points or polys)
        labels = sorted({g["label"] for g in regions} | {b["label"] for b in boxes} | {p_["label"] for p_ in points})
        recs.append({"image": r.image, "dataset_id": did, "dense_json": str((dense_dir / f"{name}.json").relative_to(out)),
                     "n_layers": len(out_layers), "layer_kinds": [l["kind"] for l in out_layers], "n_regions": n_total,
                     "n_boxes": len(boxes), "n_points": len(points), "n_polygons": len(polys), "labels": labels,
                     "match": ",".join(sorted(set(matches))), "empty": bool(empty)})
        st["with_dense"] += 1
        st["empty"] += int(empty)
        for m_ in set(matches):
            st[f"match:{m_}"] += 1
    if recs:
        (out / "dense" / "manifest").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(out / "dense" / "manifest" / f"{did}.parquet", index=False)
    return {"id": did, "state": "ok" if recs else "none", "n_masks_found": len(masks), "n_annotation_files": len(per_image), **st}


def merge(out: Path):
    import pandas as pd
    parts = sorted((out / "dense" / "manifest").glob("*.parquet"))
    if not parts:
        print(f"nothing to merge in {out / 'dense' / 'manifest'}")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / "dense.parquet", index=False)
    man = pd.read_parquet(out / "manifest.parquet") if (out / "manifest.parquet").exists() else None
    L = ["# Dense annotations for the prepared set\n",
         f"{len(df):,} images with a sidecar across {df.dataset_id.nunique()} datasets"
         + (f" (of {len(man):,} prepared images)" if man is not None else "") + ".\n",
         "| dataset | images | prepared | empty | regions / image | boxes | points | top labels | match |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    tot = man.groupby("dataset_id").size() if man is not None else {}
    for did, g in df.groupby("dataset_id"):
        labs = Counter(x for v in g.labels for x in v).most_common(4)
        L.append(f"| {did} | {len(g):,} | {int(tot.get(did, 0)) if len(tot) else ''} | {int(g["empty"].sum())} | "
                 f"{g.n_regions.mean():.1f} | {int(g.n_boxes.sum())} | {int(g.n_points.sum())} | "
                 f"{', '.join(f'{k} ({v})' for k, v in labs)} | {Counter(g['match']).most_common(1)[0][0]} |")
    (out / "dense_stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:4]) + f"\n… {df.dataset_id.nunique()} datasets → {out / 'dense_stats.md'}")


# ── HF capture (download time and top-up) ───────────────────────────────────────────────────────────────────────
_STRUCT_COL = re.compile(r"bbox|box|object|point|keypoint|landmark|coord|polygon|segment|annot|anno|region|roi|spot|"
                         r"centroid|location|rle", re.I)
_MASK_COL = re.compile(r"mask|seg|label|gt|ground|annot|anno|target|disc|cup|lesion|roi", re.I)


def class_label_paths(features, prefix=""):
    """{'objects.label': [names]} for every ClassLabel inside an HF Features tree."""
    out = {}
    if features is None:
        return out
    if hasattr(features, "names") and getattr(features, "names", None):
        out[prefix] = list(features.names)
        return out
    if isinstance(features, dict):
        for k, v in features.items():
            out.update(class_label_paths(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(features, (list, tuple)) and features:
        out.update(class_label_paths(features[0], prefix))
    elif hasattr(features, "feature"):
        out.update(class_label_paths(features.feature, prefix))
    return out


def _jsonable(v, budget=1_000_000):
    try:
        s = json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception:
        return None
    return json.loads(s) if len(s) <= budget else None


def capture_hf_row(ex, image_col, fn, dest: Path, class_paths):
    """Save an HF row's mask images and structured annotation columns next to its saved image `fn`. Returns the
    _hf_dense.jsonl record, or None when the row carries nothing spatial. A column counts as a mask when its name says
    so or its content looks like one (papila's disc_exp1/cup_exp2 say neither)."""
    from bio_waila.data.prepare import looks_like_mask
    stem = Path(fn).stem
    masks, cols = [], {}
    mdir = dest / "hf_masks"
    for col, v in ex.items():
        if col == image_col or re.search(r"overlay|preview|thumb", col, re.I):
            continue
        items = v if isinstance(v, list) and v and all(isinstance(x, (Image.Image, dict)) for x in v) else [v]
        ims = []
        for x in items:
            if isinstance(x, dict) and x.get("bytes"):
                try:
                    x = Image.open(io.BytesIO(x["bytes"]))
                except Exception:
                    continue
            if isinstance(x, Image.Image):
                ims.append(x)
        if ims:
            if not _MASK_COL.search(col) and not looks_like_mask(ims[0]):
                continue  # a second image (other view, other contrast), not an annotation
            mdir.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(ims[:256]):
                safe = re.sub(r"[^A-Za-z0-9]+", "-", col).strip("-") or "mask"
                p = mdir / f"{stem}__{safe}{'_' + str(i) if len(ims) > 1 else ''}_mask.png"
                if im.mode in ("I", "F"):
                    a = np.asarray(im)
                    im = Image.fromarray(np.clip(np.round(a), 0, 65535).astype(np.uint16))
                (im if im.mode in ("1", "L", "P", "I;16", "RGB", "RGBA") else im.convert("L")).save(p)
                masks.append({"path": str(p.relative_to(dest)), "column": col, "name": re.sub(r"[_\-]?(exp|rater|annotator)\d+$", "", col),
                              **({"instance": i} if len(ims) > 1 else {})})
            continue
        if isinstance(v, (list, dict)) and v and (_STRUCT_COL.search(col) or (isinstance(v, dict) and any(_STRUCT_COL.search(k) for k in v))):
            j = _jsonable(v)
            if j is not None:
                cols[col] = j
    if not masks and not cols:
        return None
    return {"file": fn, "masks": masks, "columns": cols,
            "class_names": {k: v for k, v in (class_paths or {}).items() if k.split(".")[0] in cols}}


# ── fetch: top up existing downloads ────────────────────────────────────────────────────────────────────────────
def _fetched_media(root: Path):
    """(loose file names, {zip/tar source: [member names]}) that the download fetched, from _files.jsonl."""
    loose, members = set(), defaultdict(set)
    p = root / "_files.jsonl"
    if not p.exists():
        return loose, members
    for line in open(p):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("dense"):
            continue  # annotation members a previous fetch added
        if j.get("member"):
            members[j["src"]].add(j["member"])
        elif j.get("src") and not str(j["src"]).startswith("hf:") and kind(j["src"]) in ("image", "volume", "video", "wsi", "array"):
            loose.add(j["src"])
    return loose, members


def _media_in_use(root: Path, prep: Path, did):
    """Media names (relative to their file listing or archive) behind the prepared images, when a manifest exists; else
    every media member of the archives downloaded whole. Fetch pairs annotations to these."""
    man = prep / "manifest" / f"{did}.parquet"
    out = set()
    if man.exists():
        import pandas as pd
        for src in pd.read_parquet(man, columns=["source"]).source:
            parts = Path(src).parts
            if parts and parts[0] in ("zip_members", "tar_members", "extracted") and len(parts) > 2:
                out.add(str(Path(*parts[2:])))
            elif parts and parts[0] == "files" and len(parts) > 1:
                out.add(str(Path(*parts[1:])))
        return out
    import zipfile
    for a in (root / "archives").glob("*") if (root / "archives").exists() else []:
        if zipfile.is_zipfile(a):
            with zipfile.ZipFile(a) as z:
                out.update(m for m in z.namelist() if kind(m) in ("image", "volume", "video", "wsi", "array"))
    return out


def _is_ann_name(n):
    e = ext(n)
    return not is_junk(n) and ((e in MASK_EXT and (kind(n) == "mask" or is_mask(n))) or e in ANN_EXT)


def _dataset_level(n):
    """A table for many images (instances_train2017.json, _annotations.coco.json, bbox.csv), judged by its own file
    name: a folder called 'Annotated Images' full of per-image json files must not make each of them one."""
    b = Path(n).name
    return (ext(b) in (".json", ".xml", ".csv", ".tsv")
            and bool(re.search(r"annot|coco|instances|labels?\b|label_|bbox|boxes|points|coords?|roi|ground|cvat", b, re.I))
            and len(core_tokens(strip_ext(b))) <= 2)  # G2208842_10_jpg_Label.json names a case: per-image



def pick_pairs(media, names, ignore=None):
    """Annotation names (from a listing) that pair with any of `media`, plus dataset-level annotation tables."""
    cands = [n for n in names if _is_ann_name(n)]
    idx = PairIndex(cands, ignore)
    got = set()
    for m in media:
        got.update(idx.match(m)[1])
    got.update(n for n in cands if _dataset_level(n) and kind(n) != "mask")
    return sorted(got)


def fetch_dataset(row, raw: Path, budget_gb=20.0, seed=0, prep: Path = PREP):
    """Top up one downloaded dataset with paired annotation files. Returns a stats dict."""
    from bio_waila.data import download as dl
    did = row["id"]
    root = raw / did
    st_p = root / "_status.json"
    if not st_p.exists():
        return {"id": did, "state": "not downloaded"}
    st = json.loads(st_p.read_text())
    if st.get("state") not in ("done", "partial"):
        return {"id": did, "state": st.get("state")}
    strat = st.get("strategy", "")
    ignore = (load_fixes().get(did) or {}).get("ignore")
    out = {"id": did, "strategy": strat}
    if strat == "hf-stream":
        n = dl.fetch_hf_dense(row["ref"], root, st.get("want", 0), seed)
        out.update(state="ok", hf_rows_with_dense=n)
        return out
    if strat == "manual-drop":
        out.update(state="ok", note="archives on disk; build extracts their annotations")
        return out
    method, ref = st.get("method", row["method"]), st.get("ref", row["ref"])
    if strat.startswith("hf-files"):
        from huggingface_hub import HfApi
        import os
        hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
        entries = [dl.Entry(s.rfilename, s.size or -1, f"https://huggingface.co/datasets/{ref}/resolve/main/{s.rfilename}", headers=hdr)
                   for s in HfApi().dataset_info(ref, files_metadata=True).siblings]
    else:
        entries = dl.LISTERS[method](ref)
    loose, members = _fetched_media(root)
    media_names = set(loose) | {m for v in members.values() for m in v} | _media_in_use(root, prep, did)
    budget, used, n_files = int(budget_gb * 2**30), 0, 0
    # loose annotation files next to loose media (or anywhere in the listing, for media inside zips)
    by_name = {e.name: e for e in entries}
    for n in pick_pairs(media_names, list(by_name), ignore):
        e = by_name[n]
        if (root / "files" / n).exists() or 0 <= e.size > ANN_MAX or used + max(e.size, 0) > budget:
            continue
        used += dl.get_file(e, root / "files" / n)
        n_files += 1
    # members of zips that were not downloaded whole: masks often live in a different zip than the images
    for e in entries:
        if ext(e.name) != ".zip" or (root / "archives" / e.name).exists():
            continue
        try:
            from remotezip import RemoteZip
            url = e.url
            if e.auth or e.headers:
                r = dl.S.get(url, auth=e.auth, headers=e.headers, allow_redirects=False, timeout=60)
                url = r.headers.get("Location", url)
            with RemoteZip(url, session=dl.S) as z:
                infos = {i.filename: i for i in z.infolist() if not i.is_dir()}
                for n in pick_pairs(media_names, list(infos), ignore):
                    i = infos[n]
                    dest = root / "zip_members" / Path(e.name).stem
                    if (dest / n).exists() or i.file_size > ANN_MAX or used + i.file_size > budget:
                        continue
                    z.extract(i, dest)
                    used += i.file_size
                    n_files += 1
                    with open(root / "_files.jsonl", "a") as f:
                        f.write(json.dumps({"src": e.name, "member": n, "bytes": i.compress_size, "dense": True}) + "\n")
        except Exception as ex:
            out.setdefault("errors", []).append(f"{e.name}: {type(ex).__name__}")
    # tars can only be streamed: re-stream those the download sampled and keep members that pair with its media
    for src, mem in members.items():
        if ext(src).startswith(".tar") or ext(src) == ".tgz":
            e = by_name.get(src)
            if e is not None:
                n_files += _restream_tar(dl, e, root, mem, ignore, budget - used)
    out.update(state="ok", fetched=n_files, bytes=used)
    return out


class StreamMatcher:
    """Does a single annotation name pair with any of a fixed set of media names? (For streamed tars, where names
    arrive one by one; same keys as PairIndex, without the folder scoring.)"""

    def __init__(self, media):
        self.strict, self.core, self.groups = set(), set(), set()
        for m in media:
            s, c, _ = file_keys(m)
            self.strict.add(s)
            if c:
                self.core.add(c)
            else:
                self.groups.add("/".join(norm_dirs(m)))

    def pairs(self, name):
        s, c, ct = file_keys(name)
        if s in self.strict or (c and c in self.core) or any("_".join(ct[:k]) in self.core for k in range(1, len(ct))):
            return True
        return not c and "/".join(norm_dirs(name)) in self.groups


def _restream_tar(dl, e, root: Path, media, ignore, budget):
    import shutil
    import tarfile
    sm = StreamMatcher(media)
    base = root / "tar_members" / Path(e.name).name.split(".")[0]
    n, streamed = 0, 0
    with dl.S.get(e.url, auth=e.auth, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=r.raw, mode="r|*") as t:
            for m in t:
                streamed += m.size
                if streamed > budget:
                    break
                if not m.isfile() or not _is_ann_name(m.name) or (ignore and re.search(ignore, m.name)):
                    continue
                if not (sm.pairs(m.name) or _dataset_level(m.name)):
                    continue
                outp = base / m.name.lstrip("/")
                outp.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(outp, "wb") as f:
                    shutil.copyfileobj(src, f)
                n += 1
    return n


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────────────────
def target_rows(plan: Path, only="", all_spatial=False):
    rows = [json.loads(l) for l in open(plan)]
    if only:
        keep = set(only.split(","))
        return [r for r in rows if r["id"] in keep]
    reg = {d["id"]: d for d in json.loads(REGISTRY.read_text())} if REGISTRY.exists() else {}
    out = []
    for r in rows:
        d = reg.get(r["id"])
        if d and set(d["annotations"]["annotation_types"]) & SPATIAL and (all_spatial or d["derived"].get("train_ok")):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "build"])
    ap.add_argument("--plan", default=str(PLAN / "selection.jsonl"))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(PREP))
    ap.add_argument("--only", default="")
    ap.add_argument("--all-spatial", action="store_true", help="also datasets that may train teachers only")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-gb", type=float, default=20.0, help="per-dataset transfer budget for fetch")
    ap.add_argument("--seed", type=int, default=0, help="the seed waila-download used (HF rows are re-streamed with it)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.cmd == "build" and args.merge:
        return merge(out)
    rows = target_rows(Path(args.plan), args.only, args.all_spatial)
    i, n = map(int, args.shard.split("/"))
    fixes = load_fixes()
    log = out / "dense" / "_logs" / f"{args.cmd}_shard{i}of{n}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        for row in rows[i::n]:
            try:
                r = (fetch_dataset(row, Path(args.raw), args.max_gb, args.seed, out) if args.cmd == "fetch"
                     else build_dataset(row, Path(args.raw), out, fixes))
            except Exception as ex:
                r = {"id": row["id"], "state": f"error: {type(ex).__name__}: {ex}"}
            f.write(json.dumps(r, default=str) + "\n")
            print(json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    main()
