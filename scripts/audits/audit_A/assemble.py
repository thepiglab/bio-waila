import json, sys, glob, re, collections
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lib
for f in sorted(HERE.glob("ov*.py"), key=lambda p: int(re.sub(r"\D", "", p.stem))):
    exec(compile(f.read_text(), str(f), "exec"), {"__name__": "ov"} | {k: getattr(lib, k) for k in ("F", "S", "HARD", "o")})
OV = lib.OV
ROOT = HERE.parent.parent
TMP = Path("/private/tmp/claude-501/-Users-mkieffer-programming-uf-mmbu/927afb66-111c-460c-9b28-07fb6d07d3ea/scratchpad/audit_tmp_A")
rows = [json.loads(l) for l in open(ROOT / "plan/selection.jsonl")][:197]
FIELDS = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]
STAINED = {"light microscopy", "fluorescence microscopy", "electron microscopy", "digital pathology"}
EXTRA_CAT = {"ispy2": "ok_array_format", "harris-2015-hippocampal-neuropil": "ok_array_format", "2p-ivm-leukocyte-cell-death": "ok_array_format",
             "m4raw-abdomen": "ok", "mitoem": "ok_with_caveats", "popsicle-cryoet-benchmark": "no_images_in_record", "rodent-estrous-cytology-compendium-hf": "no_images_in_record",
             "biotisr": "no_images_in_record", "tryptag-tbrucei-localisation": "no_images_in_record", "galar": "wrong_source", "vertebral-bodies-ct-labels": "masks_only",
             "crlm-ct-seg-refined-flr": "masks_only", "mitoem-2": "moved_source", "embryonet-zebrafish": "no_images_in_record",
             "harvard-fairfedmed-oph": "ok_with_caveats", "intrapartum-us-video-dataset": "ok_nested_archive", "deepspot-smfish-simulated": "ok_with_caveats"}


def category(rid, acc):
    if rid in EXTRA_CAT:
        return EXTRA_CAT[rid]
    if not acc:
        return "ok"
    a = acc.upper()
    if a.startswith("RESTRICTED"): return "restricted"
    if a.startswith("EMBARGOED"): return "embargoed"
    if "KAGGLE" in a: return "needs_credentials"
    if "MASKS ONLY" in a: return "masks_only"
    if re.search(r"NO DATA|NO IMAGES|IMAGES MISSING|WRONG|EMPTY", a): return "no_images_in_record"
    if re.search(r"\.RAR|TAR\.GZ|SINGLE \.TAR|MULTI-VOLUME|SPLIT", a): return "ok_unlistable_archive"
    return "ok_with_caveats"


def auto_sources(o):
    out = []
    li = o.get("listing") or {}
    levels = {}
    for d, c in (li.get("depth") or {}).items():
        names = [k for k in c if k != "__n_distinct__"]
        if names and d in ("d1", "d2", "d3"):
            levels[d] = names[:6]
    for z in o.get("zips", [])[:2]:
        for d, c in (z.get("depth") or {}).items():
            names = [k for k in c if k != "__n_distinct__"]
            if names and d in ("d1", "d2", "d3", "d4") and f"{z['name']}:{d}" not in levels:
                levels[f"zip({z['name'][:30]}):{d}"] = names[:6]
    if levels:
        out.append(lib.S("folder", "file/zip member paths", list(levels), [f"{k}: {', '.join(v[:4])}" for k, v in list(levels.items())[:4]]))
    return out


records = []
missing = []
for r in rows:
    rid = r["id"]
    o = json.loads((TMP / f"{rid}.json").read_text())
    ov = OV.get(rid)
    if ov is None:
        missing.append(rid); ov = dict(acc=None, src=[], fields={}, prob=[], note="")
    L = r["labels"]
    acc_detail = ov["acc"] or o.get("access")
    status = category(rid, ov["acc"])
    if o.get("access") not in ("ok", None) and not ov["acc"]:
        acc_detail = o["access"]
    srcs = ov["src"] or auto_sources(o)
    pf = {}
    mod_reg = L["modality"]["values"][0] if L["modality"]["values"] else None
    for f in FIELDS:
        m, vals = L[f]["mode"], L[f]["values"]
        spec = ov["fields"].get(f)
        if spec is None:
            if f == "domain":
                spec = lib.F("registry multi (dataset-level topic)" if m != "masked" else "none (masked)", {}, "high" if m != "masked" else "low")
            elif m == "hard":
                spec = lib.HARD
            elif f == "stain" and mod_reg not in STAINED:
                spec = lib.F("not applicable (non-microscopy modality)", {}, "high")
            else:
                spec = lib.F("no per-sample metadata found in file listing/heads; stays %s (zero-shot fallback only)" % m, {}, "low")
        pf[f] = {"mode_registry": m, "values_registry": vals, "resolvable_by": spec["by"], "mapping_examples": spec["map"], "confidence": spec["conf"]}
    li = o.get("listing") or {}
    exts_all = set((li.get("ext") or {}))
    for z in o.get("zips", []):
        exts_all |= set(z.get("ext") or {})
    note = ov["note"]
    if ".dcm" in exts_all:
        note = (note + " " if note else "") + "DICOM files: BodyPartExamined/Modality/SeriesDescription are readable header-only after download (metalabel dicom_hits), not auditable remotely."
    rec = {"id": rid, "access_status": status, "access_detail": acc_detail, "method": r["method"], "ref": r["ref"],
           "metadata_sources": srcs, "per_field": pf, "problems": ov["prob"], "notes": note,
           "listing_summary": {"n_files": li.get("n"), "n_media": li.get("n_media"), "n_masklike": li.get("n_masklike"), "ext": li.get("ext"),
                               "zips_listed": [{"name": z["name"], "n_media": z.get("n_media"), "n_masklike": z.get("n_masklike")} for z in o.get("zips", []) if "error" not in z][:6]}}
    records.append(rec)
print("missing overrides:", missing)
(ROOT / "research").mkdir(exist_ok=True)
with open(ROOT / "docs/data/audits/audit_A.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(len(records), collections.Counter(x["access_status"] for x in records))
