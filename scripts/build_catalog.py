#!/usr/bin/env python3
"""Build the dataset catalog: the datasets bio-waila trains on, in one standard file, plus a readable table.

    uv run python scripts/build_catalog.py            # writes catalog/datasets.yaml and catalog/DATA.md

Sources, merged in this order (later wins):
  plan/selection.jsonl            what we train on: access route, licence group, split, labels, planned images
  ../data_discovery registry TOML name, links, paper, description, counts, organisations (optional: skipped if absent)
  data/cache/reports/<id>.json    local pilot validation status (optional; gitignored, so only on the machine that ran it)
  catalog/notes*.yaml             hand-curated fields: short description, verified sample count, link fixes

Re-run after `waila-select` or after editing catalog/notes*.yaml. Never edit datasets.yaml or DATA.md by hand.
"""
import json
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from bio_waila.paths import CACHE, PLAN, REGISTRY  # noqa: E402

CAT = REPO / "catalog"
REG_ROOT = REGISTRY.parent.parent  # .../data_discovery
FIELDS = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]


def source_url(method, ref):
    ref = str(ref)
    return {
        "hf": f"https://huggingface.co/datasets/{ref}",
        "kaggle": f"https://www.kaggle.com/datasets/{ref}",
        "zenodo": f"https://zenodo.org/records/{ref}",
        "figshare": f"https://figshare.com/articles/dataset/_/{ref}",
        "mendeley": f"https://data.mendeley.com/datasets/{ref}",
        "openneuro": f"https://openneuro.org/datasets/{ref}",
    }.get(method, ref if ref.startswith("http") else "")


def load_notes():
    notes = {}
    for p in sorted(CAT.glob("notes*.yaml")):
        for k, v in (yaml.safe_load(p.read_text()) or {}).items():
            notes.setdefault(k, {}).update(v or {})
    return notes


def report(i):
    p = CACHE / "reports" / f"{i}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    st = r.get("status", "")
    st = "validated" if st == "validated" else st.split(":")[0]
    return {"status": st, "pilot_images": r.get("n_images", 0)}


def build():
    rows = [json.loads(l) for l in open(PLAN / "selection.jsonl")]
    notes = load_notes()
    out = []
    for r in rows:
        reg = {}
        rf = REG_ROOT / r.get("registry_file", "")
        if r.get("registry_file") and rf.exists():
            reg = tomllib.loads(rf.read_text())
        acc, lic, con = reg.get("access", {}), reg.get("license", {}), reg.get("content", {})
        labels = r["labels"] if isinstance(r["labels"], dict) else {}
        n = notes.get(r["id"], {})
        e = {
            "id": r["id"],
            "name": n.get("name") or r.get("name") or reg.get("name", r["id"]),
            "description": n.get("description") or (reg.get("description", "")[:300] + ("…" if len(reg.get("description", "")) > 300 else "")),
            "links": {k: v for k, v in {
                "homepage": n.get("homepage") or r.get("homepage") or acc.get("homepage", ""),
                "source": n.get("source") or source_url(r["method"], r["ref"]),
                "paper": reg.get("paper_url", ""),
                "doi": reg.get("doi", ""),
            }.items() if v},
            "access": {"method": r["method"], "ref": str(r["ref"]), "type": r.get("access", acc.get("access_type", "")),
                       "manual_download": r["method"] == "mendeley" or n.get("manual_download", False)},
            "license": {"spdx": r.get("license") or lic.get("spdx", ""), "group": r.get("license_group", "permissive"),
                        "tier": r.get("license_tier", "")},
            "samples": {k: v for k, v in {
                "images": n.get("num_images") if n.get("num_images") is not None else reg.get("content", {}).get("num_images"),
                "volumes": reg.get("content", {}).get("num_volumes") or None,
                "videos": reg.get("content", {}).get("num_videos") or None,
                "note": n.get("samples_note"),
                "planned_for_training": r.get("plan_images"),
                "download_gb": r.get("full_size_gb"),
            }.items() if v not in (None, "", 0)},
            "content": {"modalities": con.get("modalities", []), "submodalities": [s for s in con.get("submodalities", []) if s != "other"]
                        + con.get("other_submodalities", []), "body_parts": con.get("body_parts", []),
                        "specimens": con.get("specimens", []), "stains": con.get("stains", []),
                        "domains": con.get("medical_domains", []), "dimensionality": r.get("dimensionality", [])},
            "labels": {f: {"mode": labels[f]["mode"], "values": labels[f]["values"]} for f in FIELDS if f in labels},
            "split": r["split"],
            "group": r["group"],
            "mmbu": {"overlap": r.get("mmbu_overlap", ""), "public_rows": r.get("mmbu_public_rows", 0)},
            "provenance": {k: v for k, v in {"year": reg.get("year"), "organizations": reg.get("organizations", []),
                                             "countries": reg.get("countries", [])}.items() if v},
            "selected_for": r.get("selected_for", []),
        }
        v = report(r["id"])
        if v:
            e["validation"] = v
        out.append(e)
    return out


def md_link(text, url):
    return f"[{text}]({url})" if url else text


def write_md(cat):
    by_mod = defaultdict(list)
    for e in cat:
        mods = e["labels"].get("modality", {}).get("values") or e["content"]["modalities"] or ["unlabelled"]
        by_mod[mods[0]].append(e)
    lic = Counter(e["license"]["group"] for e in cat)
    split = Counter(e["split"] for e in cat)
    val = Counter((e.get("validation") or {}).get("status", "not validated here") for e in cat)
    L = ["# Pass-2 training datasets", "",
         f"Generated by `scripts/build_catalog.py` from `plan/selection.jsonl`, the dataset registry and `catalog/notes*.yaml`. "
         f"Machine-readable version: [`datasets.yaml`](datasets.yaml); open issues per dataset: [`FLAGS.md`](FLAGS.md). Do not edit by hand.", "",
         f"**{len(cat)} datasets**, about **{sum(e['samples'].get('planned_for_training', 0) for e in cat):,} planned training images**. "
         f"Licence groups: {', '.join(f'{k} {v}' for k, v in lic.most_common())}. Splits: {', '.join(f'{k} {v}' for k, v in split.most_common())}. "
         f"Pilot validation: {', '.join(f'{k} {v}' for k, v in val.most_common())}.", "",
         "Columns: *Samples* is the size of the full dataset (images, plus volumes or videos where those are the unit); *Planned* is "
         "how many images we sample from it for training. *nc* marks the non-commercial layer (off by default: `include_nc`).", ""]
    L.append("| Modality | Datasets | Planned images |\n|---|---:|---:|")
    for m, es in sorted(by_mod.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {m} | {len(es)} | {sum(e['samples'].get('planned_for_training', 0) for e in es):,} |")
    for m, es in sorted(by_mod.items(), key=lambda kv: -len(kv[1])):
        L += ["", f"## {m} ({len(es)})", "", "| Dataset | Samples | Planned | Submodality · body part | Description |", "|---|---:|---:|---|---|"]
        for e in sorted(es, key=lambda e: e["name"].lower()):
            s = e["samples"]
            samples = " + ".join(f"{s[k]:,} {k}" for k in ("images", "volumes", "videos") if isinstance(s.get(k), int)) or "?"
            if s.get("note"):
                samples += f" ({str(s['note']).replace('|', '/')})"
            sub = ", ".join(e["labels"].get("submodality", {}).get("values", [])[:2] or e["content"]["submodalities"][:2])
            bp = ", ".join(e["labels"].get("body_part", {}).get("values", [])[:3] or e["content"]["body_parts"][:3])
            tags = (" · nc" if e["license"]["group"] == "nc" else "") + (" · manual download" if e["access"]["manual_download"] else "")
            link = e["links"].get("homepage") or e["links"].get("source", "")
            desc = e["description"].replace("|", "/").replace("\n", " ")
            L.append(f"| {md_link(e['name'], link)}{tags} | {samples} | {s.get('planned_for_training', 0):,} | "
                     f"{sub}{' · ' + bp if bp else ''} | {desc} |")
    (CAT / "DATA.md").write_text("\n".join(L) + "\n")


def main():
    CAT.mkdir(exist_ok=True)
    cat = build()
    header = ("# Pass-2 training datasets: one entry per dataset in plan/selection.jsonl.\n"
              "# Generated by scripts/build_catalog.py; edit catalog/notes*.yaml (curated fields) instead of this file.\n")
    (CAT / "datasets.yaml").write_text(header + yaml.safe_dump({"datasets": cat}, sort_keys=False, allow_unicode=True, width=120))
    write_md(cat)
    print(f"{len(cat)} datasets → {CAT / 'datasets.yaml'}, {CAT / 'DATA.md'}; "
          f"{sum(1 for e in cat if e['id'] in load_notes())} with curated notes")


if __name__ == "__main__":
    main()
