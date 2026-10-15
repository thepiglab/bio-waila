#!/usr/bin/env python3
"""Pick registry datasets for the Pass-2 metadata classifier and write a download plan.

Offline: reads the registry (paths.REGISTRY, default ../data_discovery/build/registry.json). Writes to plan/:
  selection.jsonl     one line per chosen dataset: download route, image budget, split, per-field label targets
  label_space.json    class lists per head, with how many selected datasets support each class
  report.md           coverage per submodality vs the MMBU full benchmark, label quality, manual-download list
  disclosure.csv      selected datasets with medium MMBU-overlap risk (disclose per challenge rule 6)

Selection, in order:
  1. eligible = train_ok (tier-A license, open/registration access, MMBU overlap not high), in an MMBU modality,
     image data (not VQA/caption corpora that recycle other datasets' images), a scriptable download route,
     not synthetic unless it is synthetic fluorescence microscopy, ≤ 2 modalities and ≤ 4 submodalities (drops
     multi-source collections), ≥ 30 images, one registry entry per download source.
  2. per MMBU submodality, greedily take the purest, best-scored datasets (≤ MAX_PER_ORG per organization) until the
     submodality's image target is met or MAX_DATASETS is reached. Targets follow the full benchmark's mix with a floor
     so rare classes still get data.
  3. top up stains and body parts that the public split uses but the selection under-covers.
  4. split by dataset group (related_ids linked) so validation measures transfer to unseen datasets.
Run: uv run waila-select [--total-images 600000] [--per-dataset 2000]
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from bio_waila.paths import PLAN, REGISTRY, REPO
from bio_waila.data.vocab import (FIG3, FIELDS, MODALITIES, PUBLIC_VOCAB, STAINED_MODALITIES, SUBMODALITIES, canon_body_part,
                             canon_stain, canon_submodality, target)

OUT = PLAN
SKIP_FORMATS = {"vqa", "instruction", "reasoning", "image_text"}
IMAGE_DIMS = {"2d", "3d", "wsi", "multichannel-2d", "2d+t", "3d+t"}


# ── download routes ──────────────────────────────────────────────────────────────────────────────────────────────
def _zenodo_id(u):
    m = re.search(r"zenodo\.(\d+)", u) or re.search(r"zenodo\.org/(?:records?|deposit)/(\d+)", u)
    return m.group(1) if m else None


def _figshare_id(u):
    """Article/collection id: the first standalone run of >= 6 digits after 'figshare' (never a truncated suffix:
    the old regex let a title-slug wildcard eat the leading digits of 8-digit ids)."""
    i = u.lower().find("figshare")
    if i < 0:
        return None
    m = re.search(r"(?<!\d)(\d{6,})(?!\d)", u[i:])
    return m.group(1) if m else None


def route(r):
    """Best scriptable way to fetch the dataset, as (method, ref). Order = most reliable first."""
    if r["id"] in FIXES["route"]:
        return FIXES["route"][r["id"]]["method"], FIXES["route"][r["id"]]["ref"]
    a = r["access"]
    urls = a["download_urls"] + a["other_mirrors"] + ([a["homepage"]] if a["homepage"] else [])
    hf = [h for h in a["huggingface"] if "/" in h] + [u.split("huggingface.co/datasets/")[1].strip("/") for u in urls if "huggingface.co/datasets/" in u]
    if hf and not a["hf_gated"]:
        return "hf", hf[0]
    for u in a["zenodo"] + urls:
        if (z := _zenodo_id(u)):
            return "zenodo", z
    for u in urls:
        if "figshare" in u and (f := _figshare_id(u)):
            return "figshare", f
    for k in a["kaggle"] + [u for u in urls if "kaggle.com/" in u]:
        if "/competitions/" in k or "/c/" in k:
            return "manual", "kaggle competition (accept rules first): " + k
        slug = re.sub(r"^https?://(www\.)?kaggle\.com/(datasets/)?", "", k).strip("/")
        if slug.count("/") >= 1:
            return "kaggle", "/".join(slug.split("/")[:2])
    for u in urls:
        if "data.mendeley.com/datasets/" in u:
            return "mendeley", re.search(r"datasets/([a-z0-9]+)", u).group(1)
        if "openneuro.org" in u and (m := re.search(r"(ds\d{6})", u)):
            return "openneuro", m.group(1)
        if "dataverse" in u and "doi:" in u.lower():
            return "dataverse", u
    for u in urls:
        if re.search(r"\.(zip|tar|tar\.gz|tgz|7z|png|jpe?g|tiff?|nii(\.gz)?|h5|npz)(\?.*)?$", u, re.I):
            return "direct", u
    for u in urls:
        if "doi.org/" in u:
            return "doi", u  # resolved at download time; becomes zenodo/figshare/mendeley/dataverse or manual
    return None, None


# ── selection ─────────────────────────────────────────────────────────────────────────────────────────────────────
from bio_waila.data.vocab import TAXONOMY
SUBMODS_OF = {m: {c for c in (canon_submodality(x) for x in v) if c} for m, v in TAXONOMY["submodalities"].items()}
FIXES = json.loads((PLAN / "registry_fixes.json").read_text())
AUDIT_DROP = {"no_images_in_record", "masks_only", "restricted", "embargoed", "wrong_source", "no_public_files",
              "ok_but_no_images_in_record"}


def audit_drops():
    """Datasets the metadata audits (docs/data/audits/audit_*.jsonl) found unusable as selected."""
    out = {}
    for p in sorted((REPO / "docs" / "data" / "audits").glob("audit_*.jsonl")):
        for line in open(p):
            j = json.loads(line)
            if j.get("access_status") in AUDIT_DROP:
                out[j["id"]] = j["access_status"]
    return out


ALLOW_HIGH_OVERLAP = True  # the Pass-2 classifier is a labeler, not the submission: MMBU-source datasets are fine
DROPS = {k: v for k, v in {**audit_drops(), **FIXES["drop"]}.items() if k not in FIXES.get("reaudit", {})}


# licence groups: "permissive" (tier A: CC0/CC-BY/CC-BY-SA/permissive) is the default training set; "nc" (tier B:
# non-commercial) is an optional layer on top, tagged in every plan/manifest row so training can switch it off
LICENSE_GROUPS = {"A": "permissive", "B": "nc"}
SOURCES = json.loads((PLAN / "mmbu_sources.json").read_text()) if (PLAN / "mmbu_sources.json").exists() else {}


def train_ok(r, tiers=("A",)):
    """Licence tier in `tiers` and open/registration access (MMBU-overlap risk allowed when ALLOW_HIGH_OVERLAP)."""
    if not ALLOW_HIGH_OVERLAP and tiers == ("A",):
        return r["derived"]["train_ok"]
    return r["license"]["tier"] in tiers and r["access"]["access_type"] in ("open", "registration")


def eligible(r, tiers=("A",)):
    c = r["content"]
    if r["id"] in DROPS or not train_ok(r, tiers) or r["io"]["data_format"] in SKIP_FORMATS:
        return False
    if not any(m in MODALITIES for m in c["modalities"]) or not IMAGE_DIMS & set(c["dimensionality"]):
        return False
    if c["synthetic"] and "synthetic fluorescence microscopy" not in c["submodalities"]:
        return False
    # multi-source collections (MedMNIST, MedIMeta, VLM training mixes) repeat other datasets' images with vague labels
    if len({m for m in c["modalities"] if m in MODALITIES}) > 2 or len({canon_submodality(x) for x in c["submodalities"]} - {None}) > 4:
        return False
    # "clinical photography" must show people; the registry also files biodiversity photo corpora under it
    if set(c["submodalities"]) == {"clinical photography"} and "human" not in c["specimens"]:
        return False
    if 0 <= r["derived"]["effective_images"] < 30:
        return False
    return route(r)[0] is not None


def purity(r):
    """1.0 when the dataset has exactly one modality and one MMBU submodality, lower when labels are ambiguous."""
    c = r["content"]
    mods = [m for m in c["modalities"] if m in MODALITIES]
    subs = {canon_submodality(s) for s in c["submodalities"]} - {None}
    return 1.0 / (max(1, len(mods)) * max(1, len(subs)))


def labels(r):
    c = r["content"]
    mods = [m for m in c["modalities"] if m in MODALITIES]
    lab = {
        "modality": target(mods, "modality"),
        "submodality": target(c["submodalities"], "submodality"),
        "body_part": target(c["body_parts"], "body_part"),
        "stain": target(c["stains"], "stain") if set(mods) & STAINED_MODALITIES else {"mode": "masked", "values": []},
        "specimen": target(c["specimens"], "specimen"),
        "domain": target(c["medical_domains"], "domain", multilabel=True),
    }
    # body parts: datasets list several regions when images differ by region; >4 is too vague to learn from
    if lab["body_part"]["mode"] == "candidates" and len(lab["body_part"]["values"]) > 4:
        lab["body_part"] = {"mode": "masked", "values": []}
    lab = implication_rules(lab, mods)
    lab.update(FIXES["labels"].get(r["id"], {}))  # audited corrections win over everything
    return lab


# Deterministic dataset-level rules from docs/data/label_design.md (IDs match its training_rules table).
B1_BODY = {"fundus photography": "eye", "optical coherence tomography": "eye", "optical coherence tomography angiography": "eye",
           "fluorescein angiography": "eye", "dermoscopy": "skin", "mammography": "breast", "chest x-ray": "chest",
           "colposcopy": "cervix"}
CLINICAL = {"x-ray imaging", "computed tomography", "magnetic resonance imaging", "ultrasound imaging", "endoscopy",
            "optical imaging", "clinical photography"}
SPECIALTIES_RADIOLOGIC = {"x-ray imaging", "computed tomography", "magnetic resonance imaging", "ultrasound imaging"}
NOT_SPECIES = {"cell line", "phantom", "cells"}


def implication_rules(lab, mods):
    hard = lambda f: lab[f]["values"][0] if lab[f]["mode"] == "hard" else None
    def set_hard(f, v):
        lab[f] = {"mode": "hard", "values": [v]}
    sub = hard("submodality")
    # M1/M2: a hard submodality legal under exactly one candidate modality; {digital pathology, light microscopy} -> LM
    if lab["modality"]["mode"] == "candidates":
        cands = lab["modality"]["values"]
        if sub:
            legal = [m for m in cands if sub in SUBMODS_OF.get(m, ())]
            if len(legal) == 1:
                set_hard("modality", legal[0])
        if set(cands) == {"digital pathology", "light microscopy"}:
            set_hard("modality", "light microscopy")
    # S2/S3: specific endoscopy term vs generic white-light
    if lab["submodality"]["mode"] == "candidates":
        c = set(lab["submodality"]["values"])
        if c == {"laparoscopy", "white-light endoscopy"}:
            set_hard("submodality", "laparoscopy")
        elif "white-light endoscopy" in c and c - {"white-light endoscopy"} <= {"colonoscopy", "gastroscopy"}:
            set_hard("submodality", "white-light endoscopy")
    sub = hard("submodality")
    # B1: strong submodality implies body part (only when consistent with the registry's candidates, or none given)
    if sub in B1_BODY and lab["body_part"]["mode"] != "hard":
        v = B1_BODY[sub]
        if lab["body_part"]["mode"] == "masked" or v in lab["body_part"]["values"] or (v == "eye" and "retina" in lab["body_part"]["values"]):
            set_hard("body_part", v if v in lab["body_part"]["values"] or lab["body_part"]["mode"] == "masked" else v)
    # B2: laparoscopy -> abdomen when unspecified
    if sub == "laparoscopy" and lab["body_part"]["mode"] == "masked":
        set_hard("body_part", "abdomen")
    # B3: drop modifiers when a region is listed; 'head' is a superset tag of 'brain' in CT/MRI
    if lab["body_part"]["mode"] == "candidates":
        v = [x for x in lab["body_part"]["values"] if x not in ("blood vessels", "arteries")] or lab["body_part"]["values"]
        if set(mods) & {"computed tomography", "magnetic resonance imaging"} and "brain" in v:
            v = [x for x in v if x != "head"]
        lab["body_part"] = {"mode": "hard" if len(v) == 1 else "candidates", "values": v}
    if lab["body_part"]["mode"] == "candidates" and "sperm cells" in lab["body_part"]["values"]:
        set_hard("body_part", "sperm cells")  # MMBU uses 'sperm cells' (93 public rows), not the GU system
    # ST2/ST3: stain conventions MMBU uses
    if sub in ("cryo-electron tomography", "darkfield microscopy"):
        set_hard("stain", "unstained")
    elif sub == "synthetic fluorescence microscopy":
        set_hard("stain", "synthetic")
    elif sub in ("phase contrast microscopy", "contrast microscopy", "confocal microscopy", "differential fluorescent microscopy", "multiphoton microscopy"):
        lab["stain"] = {"mode": "masked", "values": []}
    # SP1/SP2/SP3: specimen
    if sub == "synthetic fluorescence microscopy":
        set_hard("specimen", "synthetic")
    elif lab["specimen"]["mode"] == "hard" and lab["specimen"]["values"] == ["cell line"]:
        set_hard("specimen", "human")  # MMBU labels most cell-line microscopy 'human' (167 of 199 public cell rows)
    elif lab["specimen"]["mode"] == "candidates":
        sp = [x for x in lab["specimen"]["values"] if x not in NOT_SPECIES]
        if len(sp) == 1:
            set_hard("specimen", sp[0])
        elif "human" in sp and set(mods) and set(mods) <= CLINICAL:
            set_hard("specimen", "human")
    # D1: registry tags nearly every radiologic dataset 'radiology'; MMBU uses the specialty instead
    if lab["domain"]["mode"] == "multi" and set(mods) & SPECIALTIES_RADIOLOGIC:
        d = [x for x in lab["domain"]["values"] if x != "radiology"]
        if d:
            lab["domain"] = {"mode": "multi", "values": d[:3]}
    return lab


def groups(recs):
    """Union-find over related_ids so near-duplicate or derived datasets land in the same split."""
    parent = {r["id"]: r["id"] for r in recs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in recs:
        for o in r.get("related_ids", []):
            if o in parent:
                parent[find(o)] = find(r["id"])
    return {r["id"]: find(r["id"]) for r in recs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-images", type=int, default=600_000, help="overall image budget before rare-class floors")
    ap.add_argument("--per-dataset", type=int, default=2_000, help="default image cap per dataset")
    ap.add_argument("--rare-per-dataset", type=int, default=6_000, help="cap for datasets of rare submodalities")
    ap.add_argument("--max-per-dataset", type=int, default=12_000, help="cap after raising caps for under-filled classes")
    ap.add_argument("--floor", type=int, default=8_000, help="minimum image target per submodality")
    ap.add_argument("--max-datasets", type=int, default=40, help="max datasets chosen per submodality")
    ap.add_argument("--max-per-org", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--no-nc", dest="nc", action="store_false", help="skip the non-commercial (tier B) layer")
    ap.add_argument("--nc-per-sub", type=int, default=3, help="NC datasets added per thin submodality")
    ap.add_argument("--nc-min-datasets", type=int, default=5, help="a submodality with fewer permissive datasets is thin")
    ap.add_argument("--source-max", type=int, default=20_000, help="image cap for MMBU source datasets weighted by public rows")
    args = ap.parse_args()

    R = json.loads(REGISTRY.read_text())
    pool, seen = [], set()
    for r in sorted((r for r in R if eligible(r)), key=lambda r: -r["derived"]["score"]):
        if route(r) in seen:  # several registry entries can point at one download (e.g. MedMNIST v1/v2/plus)
            continue
        seen.add(route(r))
        pool.append(r)
    by_id = {r["id"]: r for r in pool}
    print(f"registry {len(R)} → eligible {len(pool)} (dropped by audit/fixes: {len(DROPS)}; high MMBU overlap allowed: {ALLOW_HIGH_OVERLAP})")

    # targets per submodality: full-benchmark share of the budget, floored for rare classes
    fig = FIG3["submodality_images"]
    tot_fig = sum(fig.values())
    share = {s: fig.get(s, 0) / tot_fig for s in SUBMODALITIES}
    pub = PUBLIC_VOCAB["vocab"]["submodality"]
    for s in SUBMODALITIES:  # public-only submodalities (e.g. digital radiography) get their public share
        share[s] = share[s] or pub.get(s, 0) / PUBLIC_VOCAB["n_rows"]
    tgt = {s: max(args.floor, int(args.total_images * share[s])) for s in SUBMODALITIES}

    cand = defaultdict(list)
    for r in pool:
        for s in {canon_submodality(x) for x in r["content"]["submodalities"]} - {None}:
            cand[s].append(r)
    rare = {s for s in SUBMODALITIES if len(cand[s]) <= 10}

    def plan_images(r):
        cap = args.rare_per_dataset if any(canon_submodality(s) in rare for s in r["content"]["submodalities"]) else args.per_dataset
        eff = r["derived"]["effective_images"]
        return int(min(cap, eff if eff > 0 else cap // 2))

    def rank_key(r):
        method = route(r)[0]
        ease = {"hf": 1.0, "zenodo": 0.9, "figshare": 0.9, "kaggle": 0.8, "mendeley": 0.3, "openneuro": 0.8, "dataverse": 0.75, "direct": 0.7, "doi": 0.6}.get(method, 0.3)
        return purity(r) * 2 + r["derived"]["approach_scores"]["sft_metadata"] / 100 + ease + r["derived"]["features"]["quality"] * 0.5

    chosen, reason = {}, defaultdict(set)
    planned = Counter()
    for s in sorted(SUBMODALITIES, key=lambda s: len(cand[s])):  # rare classes pick first
        orgs = Counter()
        n_here = 0
        for r in sorted(cand[s], key=rank_key, reverse=True):
            if planned[s] >= tgt[s] or n_here >= args.max_datasets:
                break
            org = (r["organizations"] or ["?"])[0]
            if orgs[org] >= args.max_per_org:
                continue
            orgs[org] += 1
            n_here += 1
            reason[r["id"]].add("submodality:" + s)
            if r["id"] not in chosen:
                chosen[r["id"]] = r
            planned[s] += plan_images(r)

    # force_include (registry_fixes.json): datasets the audits/discovery decided we must have (e.g. MMBU sources)
    for fid in FIXES.get("force_include", {}):
        r = next((x for x in pool if x["id"] == fid), None)
        if r is not None and fid not in chosen:
            chosen[fid] = r
            reason[fid].add("force_include")

    # fill: classes still short of target get more images from their purest selected datasets
    plan = {i: plan_images(r) for i, r in chosen.items()}
    subs_of = lambda r: {canon_submodality(x) for x in r["content"]["submodalities"]} - {None}
    for s in SUBMODALITIES:
        sel = [r for r in chosen.values() if s in subs_of(r)]
        short = tgt[s] - sum(plan[r["id"]] for r in sel)
        for r in sorted([r for r in sel if purity(r) == 1.0], key=rank_key, reverse=True):
            if short <= 0:
                break
            room = int(min(args.max_per_dataset, max(r["derived"]["effective_images"], 0))) - plan[r["id"]]
            if room > 0:
                add = min(room, short)
                plan[r["id"]] += add
                short -= add
                reason[r["id"]].add("fill:" + s)

    # top-up: stains and body parts the public split uses that the selection barely covers
    pub_stains = {canon_stain(x) for x in PUBLIC_VOCAB["vocab"]["stain"]} - {None}
    pub_body = {canon_body_part(x) for x in PUBLIC_VOCAB["vocab"]["body_part"]} - {None}
    for field, wanted, key in [("stain", pub_stains, "stains"), ("body_part", pub_body, "body_parts")]:
        canon = canon_stain if field == "stain" else canon_body_part
        have = Counter(v for r in chosen.values() for v in {canon(x) for x in r["content"][key]} - {None})
        for v in sorted(wanted):
            if have[v] >= 3:
                continue
            extra = sorted([r for r in pool if r["id"] not in chosen and v in {canon(x) for x in r["content"][key]}], key=rank_key, reverse=True)
            for r in extra[: 3 - have[v]]:
                chosen[r["id"]] = r
                plan[r["id"]] = plan_images(r)
                reason[r["id"]].add(f"{field}:{v}")
                have[v] += 1

    # ── NC layer: non-commercial (tier B) datasets on top of the permissive plan (never replacing a permissive pick) ──
    group_of = {i: "permissive" for i in chosen}
    if args.nc:
        nc_pool, seen_nc = [], set(seen)
        for r in sorted((r for r in R if eligible(r, tiers=("B",))), key=lambda r: -r["derived"]["score"]):
            if route(r) in seen_nc:
                continue
            seen_nc.add(route(r))
            nc_pool.append(r)
        nc_chosen = {}
        for r in nc_pool:  # (a) MMBU sources published under NC licences
            if SOURCES.get(r["id"], {}).get("conf") in ("H", "M"):
                nc_chosen[r["id"]] = r
                reason[r["id"]].add("nc:mmbu-source")
        perm_ds = Counter(s for r in chosen.values() for s in subs_of(r))
        nc_cand = defaultdict(list)
        for r in nc_pool:
            for s in subs_of(r):
                nc_cand[s].append(r)
        for s in SUBMODALITIES:  # (b) submodalities that are short of target or have few permissive datasets
            short = tgt[s] - sum(plan[r["id"]] for r in chosen.values() if s in subs_of(r))
            if short <= 0 and perm_ds[s] >= args.nc_min_datasets:
                continue
            for r in sorted(nc_cand[s], key=rank_key, reverse=True)[: args.nc_per_sub]:
                if r["id"] not in nc_chosen:
                    nc_chosen[r["id"]] = r
                    reason[r["id"]].add("nc:submodality:" + s)
        for i, r in nc_chosen.items():
            plan[i] = plan_images(r)
            group_of[i] = "nc"
        chosen.update(nc_chosen)

    # MMBU sources (from plan/mmbu_sources.json) get images in proportion to their share of the public split
    for i, r in chosen.items():
        src = SOURCES.get(i)
        if src and src.get("conf") in ("H", "M"):
            want = round(src["public_rows"] / PUBLIC_VOCAB["n_rows"] * args.total_images)
            eff = r["derived"]["effective_images"]
            cap = min(args.source_max, eff if eff > 0 else args.source_max)
            new = int(min(cap, max(plan[i], want)))
            if new > plan[i]:
                plan[i] = new
                reason[i].add(f"mmbu-source:{src['public_rows']}-public-rows")

    # splits by dataset group, stratified so every submodality with ≥ 4 groups has validation data
    grp = groups(list(chosen.values()))
    gsubs = defaultdict(set)
    for i, r in chosen.items():
        gsubs[grp[i]] |= {canon_submodality(s) for s in r["content"]["submodalities"]} - {None}
    h = lambda g: int(hashlib.sha1(g.encode()).hexdigest(), 16) % 10_000 / 10_000
    val_groups = {g for g in gsubs if h(g) < args.val_frac}
    sub_groups = defaultdict(list)
    for g, ss in gsubs.items():
        for s in ss:
            sub_groups[s].append(g)
    for s, gs in sub_groups.items():
        if len(gs) >= 4 and not set(gs) & val_groups:
            val_groups.add(min(gs, key=h))

    OUT.mkdir(exist_ok=True)
    rows = []
    for i, r in sorted(chosen.items(), key=lambda kv: -rank_key(kv[1])):
        m, ref = route(r)
        a, c = r["access"], r["content"]
        rows.append({
            "id": i, "name": r["name"], "registry_file": r["_file"], "method": m, "ref": ref,
            "homepage": a["homepage"], "fallback_urls": (a["download_urls"] + a["other_mirrors"])[:6],
            "license": r["license"]["spdx"], "license_tier": r["license"]["tier"], "license_group": group_of.get(i, "permissive"),
            "mmbu_public_rows": SOURCES.get(i, {}).get("public_rows", 0), "access": a["access_type"],
            "mmbu_overlap": r["derived"]["overlap_effective"], "in_medvlm_training": r["derived"]["in_medvlm_training"],
            "dimensionality": c["dimensionality"], "file_formats": a["file_formats"], "full_size_gb": a["download_size_gb"],
            "effective_images": r["derived"]["effective_images"], "plan_images": plan[i],
            "split": "val" if grp[i] in val_groups else "train", "group": grp[i],
            "purity": round(purity(r), 3), "selected_for": sorted(reason[i]), "labels": labels(r),
            "exclude_paths": FIXES["exclude_paths"].get(i, ""),
        })
    with open(OUT / "selection.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # label space: every value with ≥ 1 selected dataset, plus all public/benchmark values (eval needs them)
    support = {fld: Counter() for fld in FIELDS}
    for row in rows:
        for fld, t in row["labels"].items():
            for v in t["values"]:
                support[fld][v] += 1
    space = {}
    for fld in FIELDS:
        base = set(PUBLIC_VOCAB["vocab"].get(fld, {})) if fld not in ("modality", "submodality") else set(MODALITIES if fld == "modality" else SUBMODALITIES)
        canon = {"stain": canon_stain, "body_part": canon_body_part}.get(fld)
        if canon:
            base = {canon(x) for x in base} - {None}
        if fld == "specimen":
            from bio_waila.data.vocab import canon_specimen
            base = {canon_specimen(x) for x in base} - {None}
        classes = sorted(base | set(support[fld]))
        space[fld] = {"classes": classes, "train_datasets": {k: support[fld][k] for k in classes},
                      "multilabel": fld in ("domain", "body_part")}
    (OUT / "label_space.json").write_text(json.dumps(space, indent=1, ensure_ascii=False))

    with open(OUT / "disclosure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "mmbu_overlap", "license", "homepage"])
        for row in rows:
            if row["mmbu_overlap"] != "low":
                w.writerow([row["id"], row["name"], row["mmbu_overlap"], row["license"], row["homepage"]])

    write_report(rows, space, tgt, cand, pool, args)
    n_img = sum(r["plan_images"] for r in rows)
    print(f"selected {len(rows)} datasets, ~{n_img:,} planned images → {OUT}")


def write_report(rows, space, tgt, cand, pool, args):
    L = []
    w = L.append
    n_img = sum(r["plan_images"] for r in rows)
    meth = Counter(r["method"] for r in rows)
    split = Counter(r["split"] for r in rows)
    w("# Pass-2 classifier dataset plan\n")
    w(f"Generated by `bio_waila/data/select.py` from {len(pool):,} eligible registry datasets "
      f"(budget {args.total_images:,} images, {args.per_dataset:,} per dataset, {args.rare_per_dataset:,} for rare classes).\n")
    w(f"- **{len(rows)} datasets**, about **{n_img:,} images** planned "
      f"({split['train']} train / {split['val']} validation datasets, split by dataset group).")
    lg = Counter(r["license_group"] for r in rows)
    lgi = Counter()
    for r in rows:
        lgi[r["license_group"]] += r["plan_images"]
    w("- Licence groups: " + ", ".join(f"**{k}** {lg[k]} datasets / {lgi[k]:,} images" for k in sorted(lg)) +
      ". `nc` = non-commercial (tier B), an optional layer: training excludes it unless `include_nc` is on.")
    w(f"- MMBU source datasets (plan/mmbu_sources.json) in the plan: {sum(1 for r in rows if r['mmbu_public_rows'])}, "
      f"covering {sum(r['mmbu_public_rows'] for r in rows):,} of {PUBLIC_VOCAB['n_rows']:,} public rows.")
    w("- Download routes: " + ", ".join(f"{k} {v}" for k, v in meth.most_common()) + ".")
    ov = Counter(r["mmbu_overlap"] for r in rows)
    w(f"- MMBU overlap: {ov['low']} low, {ov['medium']} medium, {ov['high']} high, {ov['unknown']} unknown "
      f"(non-low listed in disclosure.csv). High-overlap datasets are allowed because this classifier only labels data "
      f"for training and is never submitted; mmbu-public stays eval-only and near-duplicates of it are removed in prepare.py.\n")
    w("## Coverage per submodality\n")
    w("Target = share of the full benchmark (paper Fig. 3) times the budget, floored. Planned = sum of per-dataset caps.\n")
    w("| Submodality | Benchmark share | Eligible datasets | Selected | Planned images | Target | Hard-labelled datasets |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    fig = FIG3["submodality_images"]
    tot = sum(fig.values())
    for s in sorted(SUBMODALITIES, key=lambda s: -fig.get(s, 0)):
        sel = [r for r in rows if s in r["labels"]["submodality"]["values"]]
        hard = sum(r["labels"]["submodality"]["mode"] == "hard" for r in sel)
        w(f"| {s} | {100 * fig.get(s, 0) / tot:.1f}% | {len(cand[s])} | {len(sel)} | "
          f"{sum(r['plan_images'] for r in sel):,} | {tgt[s]:,} | {hard} |")
    w("\n## Non-commercial (nc) layer\n")
    w("| Dataset | Licence | Why selected | MMBU public rows | Planned images |\n|---|---|---|---:|---:|")
    for r in sorted((r for r in rows if r["license_group"] == "nc"), key=lambda r: -r["plan_images"]):
        w(f"| {r['id']} | {r['license']} | {', '.join(x for x in r['selected_for'] if x.startswith('nc'))} | {r['mmbu_public_rows']} | {r['plan_images']:,} |")
    w("\n## Label quality per head\n")
    w("Labels are dataset-level. *hard* = one value for every image; *candidates* = one of several, train with a "
      "partial-label loss; *multi* = all values apply; *masked* = no usable label (excluded from that head's loss).\n")
    w("| Head | Classes | hard | candidates | multi | masked |\n|---|---:|---:|---:|---:|---:|")
    for fld in FIELDS:
        c = Counter(r["labels"][fld]["mode"] for r in rows)
        w(f"| {fld} | {len(space[fld]['classes'])} | {c['hard']} | {c['candidates']} | {c['multi']} | {c['masked']} |")
    thin = {fld: [k for k, v in space[fld]["train_datasets"].items() if v == 0] for fld in FIELDS}
    w("\n## Classes with no training dataset\n")
    w("These appear in MMBU's vocabulary but no selected dataset carries them. Add data by hand or expect the "
      "model to fall back on related classes.\n")
    for fld, ks in thin.items():
        if ks:
            w(f"- **{fld}**: {', '.join(ks)}")
    man = [r for r in rows if r["method"] in ("doi", "direct", "manual")]
    w(f"\n## Routes that may need a manual step ({len(man)})\n")
    w("`doi` links are resolved at download time; if they land somewhere without an API (institutional repositories, "
      "Synapse, TCIA), `download.py` marks them `manual` in its status file.\n")
    w("| Dataset | Route | Ref |\n|---|---|---|")
    for r in man[:80]:
        w(f"| {r['id']} | {r['method']} | {r['ref']} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
