#!/usr/bin/env python3
"""Label every sample from its own metadata: the main auto-labeller for the Pass-2 closed vocabulary.

Idea: datasets already say what each image is, just not in MMBU's words. A dataset has a few metadata columns
(DICOM tags, CSV/JSON columns, HF columns and ClassLabel names, BIDS entities, folder names), and each column has a
small number of distinct values ("MR", "T2", "Liver", "H&E", "PA"). So we label *distinct values*, once, and join
them back to every sample. 100k images with 6 distinct organ values need 6 mappings, not 100k labels.

  profile   per dataset, build samples.parquet (one row per media file or HF row; one column per metadata source)
            and profile.json (per column: non-null share, distinct values with counts). Pixels are never decoded;
            DICOM is read header-only.
  map       map every distinct value of every low-cardinality column to (field, class) with the closed-vocabulary
            matcher (bio_waila/data/rules.py match_value). A column is assigned to a field only when ≥ COLUMN_MIN of its rows
            map consistently, so a stray folder called "eye" in a diagnosis column cannot label anything.
            Writes value_map.csv (what was mapped), and review.csv (frequent unmapped values in columns that look like
            Pass-2 fields). Fill review.csv's `class` column — or add rows to overrides.csv — and re-run `map`;
            overrides always win.
  apply     join the value maps back to every sample, reconcile with the registry's dataset-level label
            (never leave the registry's candidate set; a disagreeing registry-hard label is kept and logged as a
            conflict), and write labels/<id>.parquet keyed by relative path. prepare.py reads these.
  remote    for Hugging Face routes, profile *before* downloading: the datasets-server API returns the features
            (ClassLabel names = the distinct values) and first rows as JSON. No images are fetched.

  uv run waila-metalabel profile --raw $RAW --out $PREP --shard 0/8
  uv run waila-metalabel map     --out $PREP
  uv run waila-metalabel apply   --out $PREP
  uv run waila-metalabel remote  --only nih-chestxray14,cystods
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from bio_waila.data.files import kind
from bio_waila.data.rules import DICOM_MODALITY, field_of_column, match_value, resolve
from bio_waila.data.vocab import FIELDS

from bio_waila.paths import PLAN as PLAN_DIR, PREP, RAW
COLUMN_MIN = 0.6
NULLISH = {"other", "others", "unknown", "none", "n/a", "na", "nan", "misc", "mixed", "-", "?", "", "not applicable"}         # share of a column's rows that must map to one field before the column is trusted
MAX_DISTINCT = 400       # columns with more distinct values are IDs or free text; skipped
DICOM_HEADERS = 3000     # header reads per dataset
GENERIC_DIRS = {"files", "extracted", "zip_members", "tar_members", "archives", "hf_images", "images", "image", "img",
                "imgs", "data", "dataset", "train", "test", "val", "valid", "validation", "training", "testing", "all",
                "raw", "png", "jpg", "jpeg", "tif", "tiff", "dicom", "dcm", "nifti", "volumes"}
SOURCE_RANK = {"dicom": 0, "csv": 1, "json": 1, "hf": 1, "bids": 2, "path": 3}  # lower = more trusted
SOURCE_CONF = {"dicom": 0.95, "csv": 0.9, "json": 0.9, "hf": 0.9, "bids": 0.9, "path": 0.85}


def classes():
    sp = json.loads((PLAN_DIR / "label_space.json").read_text())
    return {f: set(v["classes"]) for f, v in sp.items()}


PLAN = PLAN_DIR / "selection.jsonl"


def plan_rows():
    return {r["id"]: r for r in (json.loads(l) for l in open(PLAN))}


# ── profile ─────────────────────────────────────────────────────────────────────────────────────────────────────
def path_columns(rel):
    """Folder names (minus generic ones) as columns d1..dk, plus BIDS entities from the file name."""
    parts = list(Path(rel).parts[:-1])
    if parts and parts[0] in ("extracted", "zip_members", "tar_members"):
        parts = parts[2:]  # the next folder is the archive's own name (often the dataset slug), not sample metadata
    parts = [p for p in parts if p.lower() not in GENERIC_DIRS and not re.fullmatch(r"[\d_\-]+", p)
             and not re.fullmatch(r"[^/\s]+__[^/\s]+", p)]  # owner__slug folders from Kaggle zips
    cols = {f"path:d{i + 1}": p for i, p in enumerate(parts[-4:])}
    stem = Path(rel).name.split(".")[0]
    for k, v in re.findall(r"(sub|ses|acq|ce|rec|run|mod|desc)-([A-Za-z0-9]+)", stem):
        cols[f"bids:{k}"] = v
    if (m := re.search(r"_([A-Za-z0-9]+)$", stem)) and "-" in stem:
        cols["bids:suffix"] = m.group(1)
    # the file name itself (extension and digits removed): "Laser_11 D7 IR.tif" -> "Laser_ D IR", "Human_Heart_3" -> ...
    name = re.sub(r"\.(nii\.gz|tar\.gz|[A-Za-z0-9]{1,5})$", "", Path(rel).name)
    name = re.sub(r"\s+", " ", re.sub(r"(?<![A-Za-z])\d+|\d{3,}", "", name)).strip(" _-.")  # keep T1, T2, D7
    if name and len(name) <= 80:
        cols["path:name"] = name
    return cols


def key_candidates(rel):
    p = Path(rel)
    stem = p.name.split(".")[0]
    keys = {p.name, stem, str(rel)}
    if any(ch.isdigit() for ch in p.parent.name):  # patient/case folder ids are often the join key
        keys |= {p.parent.name}
    return keys


def load_tables(root: Path, media_keys, n_media=None):
    """Every small csv/tsv/json/xlsx/parquet whose key column matches media file names/stems/folders.

    A column qualifies if ≥ 20% of its rows name a downloaded file, OR if it names ≥ 50% of the downloaded files: with a
    sampled download (280 of 3,746 images) only ~7% of a full-dataset table's rows can match, yet it covers every sample."""
    tables = []
    for p in sorted(root.rglob("*")):
        if (not p.is_file() or kind(p.name) != "meta" or p.stat().st_size > 50 * 2**20 or p.name.startswith("_")
                or p.name == "hf_rows.jsonl"):  # our own HF row dump is joined separately as hf:* columns
            continue
        try:
            suf = p.suffix.lower()
            if suf in (".csv", ".tsv", ".txt"):
                df = pd.read_csv(p, sep=None, engine="python", dtype=str, nrows=500_000)
            elif suf == ".jsonl":
                df = pd.read_json(p, lines=True, dtype=str)
            elif suf == ".json":
                j = json.loads(p.read_text())
                df = pd.json_normalize(j if isinstance(j, list) else next((v for v in j.values() if isinstance(v, list)), [j]))
            elif suf in (".xlsx", ".xls"):
                df = pd.read_excel(p, dtype=str)
            elif suf == ".parquet":
                df = pd.read_parquet(p).astype(str)
            else:
                continue
        except Exception:
            continue
        hdr = [str(c) for c in df.columns]
        if (any(re.search(r"[/\\]|\.(png|jpe?g|tiff?|bmp|dcm|nii)$", h, re.I) for h in hdr)  # headerless list: data row as header
                or all(re.fullmatch(r"-?[\d.]+", h) for h in hdr) or "ImageSets" in p.parts):
            continue
        if df.empty or df.shape[1] > 300:
            continue
        df.columns = [str(c) for c in df.columns]
        best, score = None, 0.0
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(2000)  # pandas 3 keeps NaN through astype(str)
            if vals.empty:
                continue
            hit = vals.map(lambda v: Path(v).name in media_keys or Path(v).name.split(".")[0] in media_keys).astype(bool)
            s = hit.mean()
            if n_media and (hit.any() or df[c].dropna().astype(str).head(20000).map(
                    lambda v: Path(v).name in media_keys or Path(v).name.split(".")[0] in media_keys).any()):
                allv = df[c].dropna().astype(str)  # coverage of the downloaded files, judged on all rows
                matched = {Path(v).name.split(".")[0] for v in allv if Path(v).name in media_keys or Path(v).name.split(".")[0] in media_keys}
                s = max(s, min(1.0, len(matched) / n_media))
            if s > score:
                best, score = c, s
        if best is None or score < 0.2:
            continue
        df = df.drop_duplicates(best)
        df = df[df[best].notna()]
        df["_key"] = df[best].map(lambda v: Path(str(v)).name.split(".")[0])
        tables.append((p.relative_to(root).as_posix(), df.set_index("_key")))
    return tables


def cmd_profile(args):
    from bio_waila.data.prepare import extract_archives, find_media  # same archive handling and media discovery as prepare.py
    raw, out = Path(args.raw), Path(args.out)
    rows = list(plan_rows().values())
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only.split(","))]
    i, n = map(int, args.shard.split("/"))
    for row in rows[i::n]:
        root = raw / row["id"]
        st = root / "_status.json"
        if not st.exists() or json.loads(st.read_text())["state"] not in ("done", "partial"):
            continue
        extract_archives(root)
        media = find_media(root)
        recs = []
        hf_rows = {}
        if (root / "hf_rows.jsonl").exists():
            for line in open(root / "hf_rows.jsonl"):
                j = json.loads(line)
                hf_rows[j["file"]] = j
        keys = set()
        for _, rel, _ in media:
            keys |= key_candidates(rel)
        tables = load_tables(root, keys, n_media=len(media))
        n_dicom = 0
        for p, rel, k in media:
            rec = {"rel": rel, "kind": k, **path_columns(rel)}
            if (h := hf_rows.get(Path(rel).name)):
                rec.update({f"hf:{c}": v for c, v in h.items() if c not in ("file", "image_column")})
            for tname, df in tables:
                for key in key_candidates(rel):
                    key = Path(key).name.split(".")[0]
                    if key in df.index:
                        r = df.loc[key]
                        rec.update({f"csv:{tname}:{c}": v for c, v in r.items() if isinstance(v, str) and len(v) <= 200})
                        break
            if k == "volume" and n_dicom < DICOM_HEADERS and (p.suffix.lower() == ".dcm" or kind(p.name) == "other"):
                try:
                    import pydicom
                    ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                    for tag in ("Modality", "BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName",
                                "ViewPosition", "ImageType", "ContrastBolusAgent"):
                        v = getattr(ds, tag, None)
                        if v not in (None, ""):
                            rec[f"dicom:{tag}"] = "\\".join(v) if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue" else str(v)
                    n_dicom += 1
                except Exception:
                    pass
            recs.append(rec)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        d = out / "metalabel" / row["id"]
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "samples.parquet", index=False)
        prof = {}
        for c in df.columns:
            if c in ("rel", "kind"):
                continue
            s = df[c].dropna().astype(str)
            vc = s.value_counts()
            prof[c] = {"non_null": round(len(s) / len(df), 3), "n_distinct": int(len(vc)),
                       "top": {k: int(v) for k, v in vc.head(60).items()},
                       "skipped": "high cardinality" if len(vc) > MAX_DISTINCT or (len(vc) > 30 and len(vc) > 0.5 * len(s)) else ""}
        (d / "profile.json").write_text(json.dumps({"id": row["id"], "n_samples": len(df), "tables": [t for t, _ in tables],
                                                    "columns": prof}, indent=1, ensure_ascii=False))
        print(f"{row['id']:40s} {len(df):>7} samples  {len(prof):>3} columns  tables={len(tables)}", flush=True)


# ── map ─────────────────────────────────────────────────────────────────────────────────────────────────────────
_CONTAINER = re.compile(r"^(csv|json):(?:(?:extracted|zip_members|tar_members)/[^/]+/|(?:files|archives)/)")


def norm_col(col: str) -> str:
    """A metadata column id without the download container: csv:extracted/BTXRD/x.xlsx:hand and
    csv:zip_members/BTXRD/x.xlsx:hand are the same column (whole-archive vs ranged download)."""
    return _CONTAINER.sub(r"\1:", col)


def load_overrides():
    """{(dataset_id, column, value): [(field, class), ...]}; several fields per value are allowed."""
    p = PLAN_DIR / "overrides.csv"
    ov = {}
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("class", "").strip():
                ov.setdefault((r["dataset_id"], norm_col(r["column"]), r["value"].strip().lower()), []).append(
                    (r["field"], r["class"].strip().lower()))
    return ov


# 0/1 columns named after a measurement of a structure (cervix_area, lesion_size, liver_volume) are not class indicators
MEASURE_COLUMN = re.compile(r"(?i)(?:^|[_\W])(area|size|volume|vol|length|len|width|height|diameter|thickness|count|"
                            r"score|ratio|percent|pct|fraction|density|mm|cm|px|pixels?)(?:$|[_\W\d])")


def _override(ov, did, col, val):
    v, col = str(val).lower(), norm_col(col)
    return next((ov[k] for k in ((did, col, v), (did, "*", v), ("*", col, v), ("*", "*", v)) if k in ov), None)


def cmd_map(args):
    out = Path(args.out)
    cls = classes()
    ov = load_overrides()
    vm_rows, review = [], []
    for prof_p in sorted((out / "metalabel").glob("*/profile.json")):
        prof = json.loads(prof_p.read_text())
        did, N = prof["id"], prof["n_samples"]
        df = pd.read_parquet(prof_p.parent / "samples.parquet")
        colmaps = {}
        for col, info in prof["columns"].items():
            if info["skipped"]:
                continue
            vc = df[col].dropna().astype(str).value_counts()
            if vc.empty:
                continue
            per_field = defaultdict(dict)   # field → {value: (class, method)}
            cover = Counter()
            colname = re.sub(r"^(csv|json|hf|dicom|bids|path):([^:]*:)?", "", col)
            if (set(map(str.lower, vc.index.astype(str))) <= {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "y", "n"}
                    and not MEASURE_COLUMN.search(colname)):
                pos = [v for v in vc.index if str(v).lower() in ("1", "1.0", "true", "yes", "y")]
                ovf = set()
                for val in pos:  # hand overrides for "this column is 1" win over the column-name match (btxrd hand → upper limbs)
                    for fld, cls_ in _override(ov, did, col, val) or []:
                        per_field[fld][val] = (cls_, "override")
                        cover[fld] += int(vc[val])
                        ovf.add(fld)
                for f, (c, strong) in match_value(colname, "indicator", cls).items():
                    if strong and f in ("body_part", "stain", "specimen", "submodality") and f not in ovf:
                        for val in pos:
                            per_field[f][val] = (c, "indicator")
                            cover[f] += int(vc[val])
                for f, vals in per_field.items():  # indicator columns: positive rows are labelled, the rest say nothing
                    colmaps.setdefault(col, {})[f] = vals
                    for val, (c, meth) in vals.items():
                        vm_rows.append({"dataset_id": did, "column": col, "value": val, "count": int(vc[val]), "field": f,
                                        "class": c, "method": meth, "column_share": 1.0, "used": True})
                continue
            for val, cnt in vc.items():
                ovf = set()  # fields set by a hand override; the other fields of this value are still matched automatically
                for fld, cls_ in _override(ov, did, col, val) or []:
                    per_field[fld][val] = (cls_, "override")
                    cover[fld] += cnt
                    ovf.add(fld)
                if col.startswith("dicom:Modality") and val.upper() in DICOM_MODALITY:
                    m, s = DICOM_MODALITY[val.upper()]
                    if "modality" not in ovf:
                        per_field["modality"][val] = (m, "dicom-code")
                        cover["modality"] += cnt
                    if s and "submodality" not in ovf:
                        per_field["submodality"][val] = (s, "dicom-code")
                        cover["submodality"] += cnt
                    continue
                for f, (c, strong) in match_value(val, col, cls).items():
                    if strong and f not in ovf:
                        per_field[f][val] = (c, "auto")
                        cover[f] += cnt
            nn = int(sum(c for v, c in vc.items() if str(v).strip().lower() not in NULLISH)) or int(vc.sum())
            ctx = field_of_column(col)
            for f, vals in per_field.items():
                share = cover[f] / nn
                trusted = share >= COLUMN_MIN or (f == ctx and share >= 0.3)
                if not trusted:  # explicit human overrides are always used, even for minority values
                    vals = {v: x for v, x in vals.items() if x[1] == "override"}
                    trusted = bool(vals)
                if trusted:
                    colmaps.setdefault(col, {})[f] = vals
                for val, (c, meth) in vals.items():
                    vm_rows.append({"dataset_id": did, "column": col, "value": val, "count": int(vc[val]), "field": f,
                                    "class": c, "method": meth, "column_share": round(share, 3), "used": trusted})
            # review: columns that look like a Pass-2 field (by name) or are partly mapped, with unmapped frequent values
            if ctx or any(cover[f] / nn >= 0.3 for f in cover):
                fld = ctx or max(cover, key=cover.get)
                mapped = set(per_field.get(fld, {}))
                for val, cnt in vc.head(40).items():
                    if val not in mapped and cnt >= max(3, 0.005 * nn):
                        review.append({"dataset_id": did, "column": col, "value": val, "count": int(cnt), "field": fld, "class": ""})
        (prof_p.parent / "colmaps.json").write_text(json.dumps(colmaps, indent=1, ensure_ascii=False))
    (out / "metalabel").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vm_rows).to_csv(out / "metalabel" / "value_map.csv", index=False)
    pd.DataFrame(review, columns=["dataset_id", "column", "value", "count", "field", "class"]).to_csv(
        out / "metalabel" / "review.csv", index=False)
    used = [r for r in vm_rows if r["used"]]
    print(f"{len(vm_rows):,} value mappings ({len(used):,} in trusted columns) · {len(review):,} values to review → "
          f"{out / 'metalabel' / 'review.csv'} (fill `class`, copy rows into plan/overrides.csv, re-run map)")


# ── cross-field consistency ─────────────────────────────────────────────────────────────────────────────────────
def _submods_of():
    from bio_waila.data.vocab import TAXONOMY, canon_submodality
    m = {}
    for mod, subs in TAXONOMY["submodalities"].items():
        m[mod] = {c for c in (canon_submodality(x) for x in subs) if c}
    return m


SUBMODS_OF = _submods_of()
PARENTS = defaultdict(set)
for _m, _ss in SUBMODS_OF.items():
    for _s in _ss:
        PARENTS[_s].add(_m)


def cross_field(rec):
    """Use one field to narrow another: CT ⇒ conventional/spiral CT; x-ray + chest ⇒ chest x-ray; a known
    submodality ⇒ its modality. Only ever narrows a candidate set; never invents a value outside it."""
    def narrow(f, allowed, why):
        if rec[f"{f}_mode"] != "candidates":
            return
        keep = [v for v in rec[f"{f}_values"] if v in allowed]
        if len(keep) == 1:
            rec[f"{f}_mode"], rec[f"{f}_values"], rec[f"{f}_source"], rec[f"{f}_conf"] = "hard", keep, why, 0.85
        elif 1 < len(keep) < len(rec[f"{f}_values"]):
            rec[f"{f}_values"] = keep
    for _ in range(2):
        if rec["modality_mode"] == "hard":
            mod = rec["modality_values"][0]
            allowed = set(SUBMODS_OF.get(mod, set()))
            if mod == "x-ray imaging" and "chest" in rec["body_part_values"] and rec["body_part_mode"] == "hard":
                allowed = {"chest x-ray"} if "chest x-ray" in rec["submodality_values"] else allowed
            narrow("submodality", allowed, "cross-field")
        if rec["submodality_mode"] == "hard":
            narrow("modality", PARENTS.get(rec["submodality_values"][0], set()), "cross-field")


# ── apply ───────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_apply(args):
    out = Path(args.out)
    plan = plan_rows()
    summary = []
    for cm_p in sorted((out / "metalabel").glob("*/colmaps.json")):
        did = cm_p.parent.name
        if did not in plan:
            continue
        colmaps = json.loads(cm_p.read_text())
        df = pd.read_parquet(cm_p.parent / "samples.parquet")
        reg = plan[did]["labels"]
        from bio_waila.data.vocab import STAINED_MODALITIES
        stained = bool(set(reg["modality"]["values"]) & STAINED_MODALITIES)  # stain words elsewhere are noise ("ZN_PAIR")
        recs, stats = [], Counter()
        for _, s in df.iterrows():
            rec = {"rel": s["rel"]}
            ov_hits = {}  # field → classes given by hand overrides for this sample
            for f in FIELDS:
                ev = []
                if f == "stain" and not stained:
                    colmaps_f = {}
                else:
                    colmaps_f = colmaps
                for col, fmap in sorted(colmaps_f.items(), key=lambda kv: SOURCE_RANK.get(kv[0].split(":")[0], 9)):
                    v = s.get(col)
                    if f in fmap and v is not None and not (isinstance(v, float) and v != v) and str(v) in fmap[f]:
                        src = col.split(":")[0]
                        ev.append((src, fmap[f][str(v)][0], col))
                        if len(fmap[f][str(v)]) > 1 and fmap[f][str(v)][1] == "override":
                            ov_hits.setdefault(f, set()).add(fmap[f][str(v)][0])
                hits = {}
                for src, c, col in ev:
                    hits.setdefault(src, {})[c] = True
                r = resolve(reg[f], [("dicom" if k == "dicom" else "meta" if k in ("csv", "json", "hf", "bids") else "path", v)
                                     for k, v in hits.items()], f, labels=reg)
                if f in ("body_part", "domain") and r["mode"] in ("hard", "multi") and len({c for _, c, _ in ev}) > 1 and reg[f]["mode"] != "hard":
                    # several metadata columns name different regions (e.g. chest + lung): MMBU treats these as multi-label
                    allowed = set(reg[f]["values"]) if reg[f]["mode"] == "candidates" else None
                    vals = sorted({c for _, c, _ in ev if allowed is None or c in allowed})
                    if len(vals) > 1:
                        r = {"mode": "multi", "values": vals, "source": "meta", "confidence": 0.85}
                if ov_hits.get(f) and reg[f]["mode"] != "hard":
                    # a hand override is the most specific evidence there is: it is not limited to the registry's
                    # candidate list (wbc-hr-bodzas: per-folder cell types outside the dataset-level candidates)
                    vals = sorted(ov_hits[f])
                    r = {"mode": "hard" if len(vals) == 1 else "multi", "values": vals, "source": "override", "confidence": 0.95}
                rec[f"{f}_mode"], rec[f"{f}_values"] = r["mode"], r["values"]
                rec[f"{f}_source"], rec[f"{f}_conf"] = r["source"], r["confidence"]
                rec[f"{f}_conflict"] = r.get("conflict", "")
                rec[f"{f}_evidence"] = ";".join(f"{col}={s[col]}" for _, _, col in ev)[:300]
            cross_field(rec)
            for f in FIELDS:
                stats[(f, "auto" if rec[f"{f}_source"] != "registry" and rec[f"{f}_mode"] in ("hard", "multi") else rec[f"{f}_mode"])] += 1
            recs.append(rec)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(out / "labels" / f"{did}.parquet", index=False)
        summary.append({"dataset_id": did, "n": len(recs), **{f"{f}:{m}": v for (f, m), v in stats.items()}})
    if not summary:
        print("nothing to apply: no profiled datasets with colmaps (run profile and map first)")
        return
    sm = pd.DataFrame(summary).fillna(0)
    sm.to_csv(out / "metalabel" / "apply_summary.csv", index=False)
    tot = sm.drop(columns=["dataset_id"]).sum()
    print(f"labelled {int(tot['n']):,} samples across {len(sm)} datasets")
    for f in FIELDS:
        print(f"  {f:12s} " + " · ".join(f"{m} {int(tot.get(f'{f}:{m}', 0)):,}" for m in ("hard", "multi", "auto", "candidates", "masked")))


# ── remote (Hugging Face, before download) ──────────────────────────────────────────────────────────────────────
def cmd_remote(args):
    """Profile HF datasets through datasets-server: features + first rows, JSON only (no images transferred)."""
    import requests
    cls = classes()
    rows = [r for r in plan_rows().values() if r["method"] == "hf"]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only.split(","))]
    hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    res = []
    for r in rows:
        base = "https://datasets-server.huggingface.co"
        info = requests.get(f"{base}/info", params={"dataset": r["ref"]}, headers=hdr, timeout=60)
        if info.status_code != 200:
            res.append({"id": r["id"], "status": f"info HTTP {info.status_code}"})
            continue
        found = []
        for cfg, ci in info.json().get("dataset_info", {}).items():
            for col, feat in (ci.get("features") or {}).items():
                names = feat.get("names") if isinstance(feat, dict) else None
                if names:  # ClassLabel: the distinct values are right here
                    for v in names:
                        for f, (c, strong) in match_value(v, f"hf:{col}", cls).items():
                            if strong:
                                found.append((cfg, col, v, f, c))
            fr = requests.get(f"{base}/first-rows", params={"dataset": r["ref"], "config": cfg,
                              "split": next(iter(ci.get("splits", {"train": 0})))}, headers=hdr, timeout=60)
            if fr.status_code == 200:
                for row in fr.json().get("rows", [])[:100]:
                    for col, v in row["row"].items():
                        if isinstance(v, str) and len(v) < 120:
                            for f, (c, strong) in match_value(v, f"hf:{col}", cls).items():
                                if strong:
                                    found.append((cfg, col, v, f, c))
        uniq = sorted(set(found))
        res.append({"id": r["id"], "status": "ok", "mappable": [f"{col}={v}→{f}:{c}" for _, col, v, f, c in uniq][:40]})
        print(f"{r['id']:40s} {len(uniq):>3} mappable values  " + "; ".join(f"{col}={v}→{c}" for _, col, v, f, c in uniq[:6]), flush=True)
    (PLAN_DIR / "hf_remote_profile.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    global PLAN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["profile", "map", "apply", "remote"])
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(PREP))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="")
    ap.add_argument("--plan", default=str(PLAN))
    args = ap.parse_args()
    PLAN = Path(args.plan)
    {"profile": cmd_profile, "map": cmd_map, "apply": cmd_apply, "remote": cmd_remote}[args.cmd](args)


if __name__ == "__main__":
    main()
