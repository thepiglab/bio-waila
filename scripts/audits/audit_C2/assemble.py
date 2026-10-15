import json, sys, re, collections
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lib
for f in sorted(HERE.glob("ov_c2*.py")):
    exec(compile(f.read_text(), str(f), "exec"), {"__name__": "ov"} | {k: getattr(lib, k) for k in ("F", "S", "HARD", "o")})
OV = lib.OV
ROOT = HERE.parent.parent
TMP = Path("/private/tmp/claude-501/-Users-mkieffer-programming-uf-mmbu/927afb66-111c-460c-9b28-07fb6d07d3ea/scratchpad/audit_tmp_C2")
ids = json.load(open(ROOT / "docs/data/audits/audit_todo_C2.json"))
sel = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "plan/selection.jsonl")}
FIELDS = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]
STAINED = {"light microscopy", "fluorescence microscopy", "electron microscopy", "digital pathology"}
STATUS = {"biad1137-hht1-single-molecule-tracking-spombe": "unsupported_host_biostudies",
          "color-doppler-twinkling-biopsy-markers": "unsupported_host_ieee_dataport_login",
          "five-d-us-ablation-deshmukh": "ok_unlistable_archive",
          "oocytor-oocyte-segmentation": "ok_unlistable_archive",
          "lightweight-instrument-seg-ush": "ok_blocked_by_archive_filter",
          "medsegmnist-derm2d": "ok_array_format",
          "aml-cytomorphology-lmu": "ok_kaggle", "tbx11k": "ok_kaggle", "bcss": "ok_kaggle"}
records, missing = [], []
for rid in ids:
    r = sel[rid]
    ov = OV.get(rid)
    if ov is None:
        missing.append(rid); ov = dict(acc=None, src=[], fields={}, prob=[], note="")
    o_ = json.loads((TMP / f"{rid}.json").read_text()) if (TMP / f"{rid}.json").exists() else {}
    L = r["labels"]
    mod_reg = L["modality"]["values"][0] if L["modality"]["values"] else None
    pf = {}
    for f in FIELDS:
        m, vals = L[f]["mode"], L[f]["values"]
        spec = ov["fields"].get(f)
        if spec is None:
            if f == "domain":
                spec = lib.F("registry multi (dataset-level topic)" if m != "masked" else "none (masked; no dentistry class)" if rid in ("alphadent", "intraoral-caries-ahmed-2025") else "none (masked)", {}, "high")
            elif m == "hard":
                spec = lib.HARD
            elif f == "stain" and mod_reg not in STAINED:
                spec = lib.F("not applicable (non-microscopy modality)", {}, "high")
            else:
                spec = lib.F("no per-sample metadata found; stays %s" % m, {}, "low")
        pf[f] = {"mode_registry": m, "values_registry": vals, "resolvable_by": spec["by"], "mapping_examples": spec["map"], "confidence": spec["conf"]}
    li = o_.get("listing") or {}
    exts = li.get("ext")
    status = STATUS.get(rid, "ok")
    acc = ov["acc"] or o_.get("access") or "ok"
    rec = {"id": rid, "access_status": status, "access_detail": acc, "method": r["method"], "ref": r["ref"],
           "metadata_sources": ov["src"], "per_field": pf, "problems": ov["prob"], "notes": ov["note"],
           "listing_summary": {"n_files": li.get("n"), "n_media": li.get("n_media"), "n_masklike": li.get("n_masklike"), "ext": exts,
                               "zips_listed": [{"name": z["name"], "n_media": z.get("n_media"), "n_masklike": z.get("n_masklike")} for z in o_.get("zips", []) if "error" not in z][:6]}}
    records.append(rec)
print("missing overrides:", missing)
with open(ROOT / "docs/data/audits/audit_C2.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(len(records), collections.Counter(x["access_status"] for x in records))
