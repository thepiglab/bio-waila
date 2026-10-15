#!/usr/bin/env python3
"""Pass-2 label design analysis: regenerates label_design.md and label_rules.json in this folder.

Inputs (read-only): mmbu-public/metadata.jsonl, data_discovery/{taxonomy.toml,build/registry.json},
bio-waila/plan/selection.jsonl, src/bio_waila/data/vocab.py.  Needs pandas.  Run:  python label_design_analysis.py
The public split is used for EVALUATION-side statistics only (applicability, conventions, entropy); nothing here trains on it.
"""
import ast
import collections
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
MMBU = P2.parent
sys.path.insert(0, str(P2))
from bio_waila.data import vocab as V  # noqa: E402

FIELDS = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]
MICRO = ["light microscopy", "fluorescence microscopy", "electron microscopy", "digital pathology"]
RAD = {"x-ray imaging", "computed tomography", "magnetic resonance imaging", "ultrasound imaging"}
CLIN = RAD | {"endoscopy", "optical imaging", "clinical photography", "digital pathology"}
DISCIPLINE = {"radiology", "pathology", "surgery", "biology", "cytology"}

# --------------------------------------------------------------------------------------------------------------------
# anatomical families (coarse body-part head); atoms not listed fall into "other"
FAMILY = {
    "eye": ["eye", "retina", "optic disc", "cornea", "anterior segment", "eyelid", "macula", "pupil"],
    "head and neck": ["head", "head and neck", "neck", "face", "ear", "nose", "larynx", "oral cavity", "teeth", "thyroid"],
    "brain": ["brain", "spinal cord", "nerve", "cerebrum"],
    "chest": ["chest", "lung", "thorax", "lungs", "lung tissue", "lower respiratory tract"],
    "heart and vessels": ["heart", "coronary arteries", "arteries", "carotid artery", "blood vessels"],
    "breast": ["breast", "breast tissue"],
    "abdomen": ["abdomen", "liver", "pancreas", "gallbladder", "kidney", "spleen", "abdominal organs", "retroperitoneum",
                "thoracoabdominal region"],
    "gastrointestinal tract": ["gastrointestinal tract", "esophagus", "stomach", "small intestine", "colon", "rectum", "appendix",
                               "colon tissue", "upper gastrointestinal tract"],
    "pelvis": ["pelvis", "prostate", "bladder", "uterus", "ovary", "male genitourinary system", "cervix", "fetus",
               "genitourinary system", "reproductive system"],
    "skeleton and limbs": ["bone", "appendicular skeleton", "axial skeleton", "skeletal system", "knee", "shoulder", "elbow", "hip",
                           "wrist", "hand", "foot", "upper limbs", "lower limbs", "spine", "muscle", "soft tissue",
                           "musculoskeletal system"],
    "skin": ["skin", "nail", "hair", "wound"],
    "blood and lymph": ["blood", "bone marrow", "lymph nodes", "blood cells", "lymphatic system / bone marrow", "white blood cells",
                        "lymphocytes", "neutrophils", "eosinophils", "monocytes"],
    "cells and tissue": ["cells", "epithelial cells", "sperm cells", "tissue", "adipose tissue", "embryo", "cell",
                         "glandular epithelium"],
    "whole body": ["whole body"],
}
PARENT = {c: f for f, cs in FAMILY.items() for c in cs}

# submodality -> body part atom that is (near-)deterministic. strong = >=99% in public with n>=20, or taxonomy-defined.
STRONG_BP = {"fundus photography": "eye", "dermoscopy": "skin", "mammography": "breast", "chest x-ray": "chest",
             "optical coherence tomography": "eye", "optical coherence tomography angiography": "eye",
             "fluorescein angiography": "eye", "colposcopy": "cervix"}
NO_STAIN_SUBS = {"phase contrast microscopy", "contrast microscopy", "confocal microscopy", "differential fluorescent microscopy",
                 "multiphoton microscopy"}


# --------------------------------------------------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def split_pipe(v):
    return [norm(x) for x in re.split(r"\|", str(v)) if norm(x)]


def parse_domain_raw(v):
    return ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else (v or [])


def parse_domain(v):
    """THE domain parsing rule: literal_eval the list string, split every item on commas, lower/strip, alias-map, dedupe, unordered."""
    out = []
    for it in parse_domain_raw(v):
        for x in re.split(r"\s*,\s*", it):
            c = V.canon_domain(x)
            if c is None:
                raise ValueError(f"unmapped domain {x!r}")
            if c not in out:
                out.append(c)
    return sorted(out)


def H(vals):
    c = collections.Counter(vals)
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values())


def condH(a, b):
    g = collections.defaultdict(list)
    for x, y in zip(a, b):
        g[x].append(y)
    n = len(a)
    return sum(len(v) / n * H(v) for v in g.values())


def acc(a, b, grp=None):
    """Majority-vote accuracy of b given a. grp None: leave-one-out; else leave-one-group-out (group = metadata_topic)."""
    a, b = list(a), list(b)
    g = collections.defaultdict(collections.Counter)
    gg = collections.defaultdict(collections.Counter)
    for i, (x, y) in enumerate(zip(a, b)):
        g[x][y] += 1
        if grp is not None:
            gg[(x, grp[i])][y] += 1
    ok = 0
    for i, (x, y) in enumerate(zip(a, b)):
        c = collections.Counter(g[x])
        if grp is None:
            c[y] -= 1
        else:
            c.subtract(gg[(x, grp[i])])
        c = +c
        ok += bool(c) and c.most_common(1)[0][0] == y
    return ok / len(a)


def detmass(a, b, th=0.95):
    g = collections.defaultdict(collections.Counter)
    for x, y in zip(a, b):
        g[x][y] += 1
    tot = sum(sum(c.values()) for c in g.values() if c.most_common(1)[0][1] / sum(c.values()) >= th)
    return tot / len(a)


def md(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    def numeric(c):
        return all(isinstance(v, (int, float)) or re.fullmatch(r"[-+\d.,%x ]+", str(v)) for v in df[c])

    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---:" if numeric(c) else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}%"


# --------------------------------------------------------------------------------------------------------------------
# 0. load public rows
d = pd.read_json(MMBU / "mmbu-public" / "metadata.jsonl", lines=True)
N = len(d)
d["mod_l"] = d.metadata_modality.map(split_pipe)
d["sub_l"] = d.metadata_submodality.map(lambda v: [V.canon_submodality(x) or x for x in split_pipe(v)])
d["mod"] = d.mod_l.map(lambda l: "|".join(sorted(set(l))))
d["sub"] = d.sub_l.map(lambda l: "|".join(sorted(set(l))))
d["dom_raw"] = d.metadata_medical_domain
d["dom"] = d.metadata_medical_domain.map(parse_domain)
d["bp_l"] = d.metadata_body_part.map(split_pipe)
d["has_bp"] = d.bp_l.map(len) > 0
d["has_stain"] = d.metadata_stain.astype(str).str.strip() != ""
d["spec_raw"] = d.metadata_specimen.map(norm)
d["spec"] = d.metadata_specimen.map(V.canon_specimen)
d["stain_c"] = d.metadata_stain.map(lambda s: V.canon_stain(s) if str(s).strip() else None)
d["topic"] = d.metadata_topic


def bpc(l):
    c = [V.canon_body_part(x) for x in l]
    c = [x for x in c if x]
    return "|".join(c) if c else "∅"


d["F_modality"] = d["mod"]
d["F_submodality"] = d["sub"]
d["F_body_part"] = d.bp_l.map(bpc)
d["F_bp_fam"] = d.bp_l.map(lambda l: "|".join(sorted({PARENT.get(V.canon_body_part(x) or x, "other") for x in l})) if l else "∅")
d["F_stain"] = d.stain_c.fillna("∅")
d.loc[d.has_stain & d.stain_c.isna(), "F_stain"] = "<non-answer>"
d["F_specimen"] = d.spec.fillna("<non-answer>")
d["F_domain"] = d.dom.map("|".join)
d["is_micro"] = d["mod"].isin(MICRO)
R = {}  # results for json

# ============================ 1. applicability ======================================================================
by_mod = d.groupby("mod").agg(n=("mod", "size"), body_part=("has_bp", "mean"), stain=("has_stain", "mean"),
                              specimen_human=("F_specimen", lambda s: (s == "human").mean())).reset_index()
by_mod = by_mod.sort_values("n", ascending=False)
by_sub = d.groupby("sub").agg(n=("sub", "size"), body_part=("has_bp", "mean"), stain=("has_stain", "mean")).reset_index()
by_sub = by_sub.sort_values("n", ascending=False)
combo = d.apply(lambda r: ("bp" if r.has_bp else "-") + ("+stain" if r.has_stain else ""), axis=1)
combo_tab = pd.crosstab(d["mod"], combo)
micro = d[d.is_micro]
n_micro = len(micro)
stain_micro = int(micro.has_stain.sum())
both = int((micro.has_bp & micro.has_stain).sum())
bp_only = int((micro.has_bp & ~micro.has_stain).sum())
st_only = int((~micro.has_bp & micro.has_stain).sum())
none_micro = int((~micro.has_bp & ~micro.has_stain).sum())
no_bp_non_micro = int((~d.has_bp & ~d.is_micro).sum())
assert d.has_stain[~d.is_micro].sum() == 0

# stain applicability per submodality (microscopy rows)
st_sub = d[d.is_micro].groupby("sub").agg(n=("sub", "size"), stain=("has_stain", "mean"), body_part=("has_bp", "mean")).reset_index()
st_sub = st_sub.sort_values("n", ascending=False)

# non-answers
nonans_spec = d[d.F_specimen == "<non-answer>"].spec_raw.value_counts()
vague_spec = d[d.spec_raw.isin(["tissue", "animal", "mixed", "synthetic", "breast tissue", "liver tissue", "human cells"])].spec_raw.value_counts()
nonans_stain = d[d.F_stain == "<non-answer>"].metadata_stain.str.lower().value_counts()

# ============================ 2. multi-label structure ==============================================================
raw_dom = d.dom_raw.unique()
parsed = {r: parse_domain(r) for r in raw_dom}  # raises on any unmapped atom
assert len(raw_dom) == 76
n_items = d.dom_raw.map(lambda r: len(parse_domain_raw(r)))
packed = collections.Counter(r for r in d.dom_raw if any("," in it for it in parse_domain_raw(r)))
alias_dups = collections.Counter(r for r in d.dom_raw
                                 if len({V.canon_domain(x) for it in parse_domain_raw(r) for x in re.split(r",\s*", it)})
                                 < sum(len(re.split(r",\s*", it)) for it in parse_domain_raw(r)))
dom_size = d.dom.map(len).value_counts().sort_index()
pair_counts = collections.Counter()
for l in d.dom:
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            pair_counts[(l[i], l[j])] += 1
order_flip = {}
for r in d.dom_raw:
    items = parse_domain_raw(r)
    if len(items) == 2 and "," not in "".join(items):
        order_flip[tuple(sorted(norm(x) for x in items))] = order_flip.get(tuple(sorted(norm(x) for x in items)), set()) | {tuple(norm(x) for x in items)}
flipped = {k: v for k, v in order_flip.items() if len(v) > 1}
dom_comb = d.dom[d.dom.map(len) > 1].map(lambda l: sum(x in DISCIPLINE for x in l))
dom_single_share = float((d.dom.map(len) == 1).mean())
rad_rows = d[d["mod"].isin(RAD | {"computed tomography|x-ray imaging"})]
rad_has_radiology = float(rad_rows.dom.map(lambda l: "radiology" in l).mean())
rad_by_mod = rad_rows.groupby("mod").apply(lambda g: g.dom.map(lambda l: "radiology" in l).mean())
dom_atoms = sorted({x for l in d.dom for x in l})
dom_mod = pd.DataFrame({a: d.dom.map(lambda l, a=a: a in l) for a in dom_atoms}).groupby(d["mod"]).mean().T

# body part pipe structure
pipe_rows = d[d.metadata_body_part.str.contains(r"\|")]
pipe_counts = pipe_rows.metadata_body_part.str.lower().value_counts()
PIPE_TYPE = {"chest|lung": "hierarchy", "eye|retina": "hierarchy", "brain|cerebrum": "hierarchy", "lung|lung tissue": "hierarchy",
             "arteries|blood vessels": "synonym", "esophagus|colon|rectum": "multi-region (dataset inventory)",
             "stomach|colon|esophagus": "multi-region (dataset inventory)",
             "stomach|colon|gastrointestinal tract": "multi-region (dataset inventory)",
             "colon tissue|lung tissue": "multi-region (two datasets pooled)",
             "white blood cells|lymphocytes|neutrophils|eosinophils|monocytes": "cell-type list"}
assert set(pipe_counts.index) == set(PIPE_TYPE), set(pipe_counts.index) ^ set(PIPE_TYPE)
pipe_tab = pd.DataFrame({"raw string": pipe_counts.index, "rows": pipe_counts.values,
                         "type": [PIPE_TYPE[k] for k in pipe_counts.index]})
pipe_type_rows = pipe_tab.groupby("type").rows.sum()
bp_nat = d.bp_l.map(len)
bp_size = bp_nat.value_counts().sort_index()
bp_has = int(d.has_bp.sum())
bp_single_share = float((bp_nat == 1).sum() / bp_has)

# body-part conventions per submodality (what string MMBU uses)
conv = {}
for s, g in d[d.has_bp].groupby("sub"):
    if len(g) < 5:
        continue
    vc = g.metadata_body_part.str.lower().str.strip().value_counts(normalize=True)
    conv[s] = [(k, round(float(v), 3)) for k, v in vc.head(4).items()]
chest_rows = d[d["sub"] == "chest x-ray"]
chest_vc = chest_rows.metadata_body_part.str.lower().value_counts()

# ============================ 3. hierarchy / constraints ============================================================
pub_pairs = collections.defaultdict(collections.Counter)
for ms, ss in zip(d.mod_l, d.sub_l):
    for a in ms:
        for b in ss:
            pub_pairs[a][b] += 1
tax_sub = V.TAXONOMY["submodalities"]
mods_all = list(V.MODALITIES)
legal = {m: sorted(set(pub_pairs[m]) | set(tax_sub.get(m, []))) for m in mods_all}
pub_only = {m: sorted(set(pub_pairs[m]) - set(tax_sub.get(m, []))) for m in mods_all}
tax_only = {m: sorted(set(tax_sub.get(m, [])) - set(pub_pairs[m])) for m in mods_all}
sub2mod = collections.defaultdict(dict)
for m in mods_all:
    for s_, c in pub_pairs[m].items():
        sub2mod[s_][m] = c
legal_mod = collections.defaultdict(set)
for m, ss in legal.items():
    for s_ in ss:
        legal_mod[s_].add(m)

compat_rows = []
for m in mods_all:
    compat_rows.append({"modality": m, "public submodalities (rows)": "; ".join(f"{s_} ({c})" for s_, c in pub_pairs[m].most_common()),
                        "taxonomy-only": "; ".join(tax_only[m]) or "-", "public-only": "; ".join(pub_only[m]) or "-",
                        "legal k": len(legal[m])})
compat_df = pd.DataFrame(compat_rows)
ambig_sub = {s_: dict(m) for s_, m in sub2mod.items() if len(m) > 1}

# typical per submodality
typ = []
for s_, g in d.groupby("sub"):
    if len(g) < 8:
        continue
    vc = lambda col: g[col].value_counts(normalize=True)  # noqa: E731
    row = {"submodality": s_, "n": len(g)}
    for col, nm in [("F_bp_fam", "body-part family"), ("F_specimen", "specimen"), ("F_domain", "domain set"), ("F_stain", "stain")]:
        v = vc(col)
        row[nm] = f"{v.index[0]} ({pct(v.iloc[0])})"
    typ.append(row)
typ_df = pd.DataFrame(typ).sort_values("n", ascending=False)

# pairwise predictability
grp = d.topic.tolist()
pair_rows = []
for a in FIELDS:
    for b in FIELDS:
        if a == b:
            continue
        A, B = d["F_" + a], d["F_" + b]
        pair_rows.append({"A": a, "B": b, "H(B)": H(B), "H(B|A)": condH(A, B),
                          "base": B.value_counts(normalize=True).iloc[0], "acc_loo": acc(A, B), "acc_group": acc(A, B, grp),
                          "det_mass": detmass(A, B)})
pairs = pd.DataFrame(pair_rows)
chain = {}
d["F_ms"] = d.F_modality + "/" + d.F_submodality
d["F_sb"] = d.F_submodality + "/" + d.F_body_part
d["F_sbf"] = d.F_submodality + "/" + d.F_bp_fam
for a, b in [("ms", "body_part"), ("ms", "stain"), ("ms", "specimen"), ("ms", "domain"), ("sb", "domain"), ("sb", "specimen"),
             ("sb", "stain"), ("sbf", "domain")]:
    A, B = d["F_" + a], d["F_" + b]
    chain[f"{a}->{b}"] = {"H(B|A)": condH(A, B), "acc_loo": acc(A, B), "acc_group": acc(A, B, grp)}
n_unique_topics = d.topic.nunique()

# ============================ 4. training-plan rules ================================================================
S = [json.loads(l) for l in open(P2 / "plan" / "selection.jsonl")]
REG = {r["id"]: r for r in json.load(open(MMBU / "data_discovery" / "build" / "registry.json"))}
for s in S:
    s["reg"] = REG[s["id"]]["content"]
TOT_IMG = sum(s["plan_images"] for s in S)


def mode_table(lab_of):
    rows = []
    for f in FIELDS:
        c = collections.Counter()
        w = collections.Counter()
        for s in S:
            m = lab_of(s)[f]["mode"]
            c[m] += 1
            w[m] += s["plan_images"]
        rows.append((f, c, w))
    return rows


log = collections.defaultdict(list)
OUT = {}
for s in S:
    L = {f: dict(s["labels"][f]) for f in FIELDS}
    sid, w = s["id"], s["plan_images"]

    def rec(f, rule, after, L=L, sid=sid, w=w):
        log[(f, rule)].append((sid, L[f]["mode"], after, w))

    # ---- modality
    if L["modality"]["mode"] == "candidates":
        sub = L["submodality"]
        if sub["mode"] == "hard":
            inter = set(L["modality"]["values"]) & legal_mod.get(sub["values"][0], set())
            if len(inter) == 1:
                rec("modality", "M1", "hard")
                L["modality"] = {"mode": "hard", "values": sorted(inter)}
        if L["modality"]["mode"] == "candidates" and set(L["modality"]["values"]) == {"digital pathology", "light microscopy"}:
            rec("modality", "M2", "hard")
            L["modality"] = {"mode": "hard", "values": ["light microscopy"]}
    # ---- submodality
    if L["submodality"]["mode"] == "candidates":
        mv = set(L["modality"]["values"]) if L["modality"]["mode"] == "hard" else None
        v = set(L["submodality"]["values"])
        if mv:
            inter = v & set().union(*[set(legal[m]) for m in mv])
            if len(inter) == 1 and inter != v:
                rec("submodality", "S1", "hard")
                L["submodality"] = {"mode": "hard", "values": sorted(inter)}
        v = set(L["submodality"]["values"]) if L["submodality"]["mode"] == "candidates" else set()
        if v == {"laparoscopy", "white-light endoscopy"}:
            rec("submodality", "S2", "hard")
            L["submodality"] = {"mode": "hard", "values": ["laparoscopy"]}
        elif v in ({"colonoscopy", "white-light endoscopy"}, {"gastroscopy", "white-light endoscopy"}):
            rec("submodality", "S3", "hard")
            L["submodality"] = {"mode": "hard", "values": ["white-light endoscopy"]}
    sub = L["submodality"]["values"][0] if L["submodality"]["mode"] == "hard" else None
    mods = set(L["modality"]["values"])
    # ---- body part
    bp = L["body_part"]
    if bp["mode"] != "hard":
        done = False
        if sub in STRONG_BP:
            fam = PARENT[STRONG_BP[sub]]
            if bp["mode"] == "masked" or any(PARENT.get(v) == fam for v in bp["values"]):
                rec("body_part", "B1", "hard")
                L["body_part"] = {"mode": "hard", "values": [STRONG_BP[sub]]}
                done = True
        if not done and sub == "laparoscopy" and (bp["mode"] == "masked" or all(PARENT.get(v) == "abdomen" for v in bp["values"])):
            rec("body_part", "B2", "hard")
            L["body_part"] = {"mode": "hard", "values": ["abdomen"]}
            done = True
        if not done and bp["values"]:
            vals = set(bp["values"])
            v2 = set(vals)
            if len(v2) > 1 and (v2 - {"blood vessels", "arteries"}):
                v2 -= {"blood vessels", "arteries"}
            if "head" in v2 and "brain" in v2 and mods & {"magnetic resonance imaging", "computed tomography"}:
                v2.discard("head")
            if len(v2) == 1 and v2 != vals:
                rec("body_part", "B3", "hard")
                L["body_part"] = {"mode": "hard", "values": sorted(v2)}
                done = True
            if not done:
                fams = {PARENT.get(x, "other") for x in v2}
                if len(fams) == 1:
                    rec("body_part", "B4", "family")
                    L["body_part"] = {"mode": "family", "values": sorted(v2)}
                else:
                    rec("body_part", "B5", "multi-family")
        elif not done:
            regs = {c for c in (V.canon_body_part(x) for x in s["reg"]["body_parts"]) if c}
            fams = {PARENT.get(x, "other") for x in regs}
            if len(regs) > 4 and len(fams) == 1:
                rec("body_part", "B7", "family")
                L["body_part"] = {"mode": "family", "values": sorted(regs)}
            else:
                rec("body_part", "B6", "masked")
    # ---- specimen
    sp = L["specimen"]
    if sp["mode"] in ("candidates", "masked") and sub == "synthetic fluorescence microscopy":
        rec("specimen", "SP3", "hard")
        L["specimen"] = {"mode": "hard", "values": ["synthetic"]}
    elif sp["mode"] == "candidates":
        v = set(sp["values"])
        v2 = v - {"cell line", "phantom", "cells"}
        if len(v2) == 1 and v2 != v:
            rec("specimen", "SP1", "hard")
            L["specimen"] = {"mode": "hard", "values": sorted(v2)}
        elif mods and mods <= CLIN and "human" in v:
            rec("specimen", "SP2", "hard")
            L["specimen"] = {"mode": "hard", "values": ["human"]}
        elif v2 <= {"bacteria", "e. coli"} and len(v2) == 2:
            rec("specimen", "SP4", "family")
        else:
            rec("specimen", "SP5", "cand")
    elif sp["mode"] == "masked":
        if mods and mods <= CLIN and s["reg"]["specimens"] in (["unknown"], []):
            rec("specimen", "SP6", "soft")
        else:
            rec("specimen", "SP7", "masked")
    # ---- stain
    st = L["stain"]
    if st["mode"] == "masked" and mods & V.STAINED_MODALITIES:
        if sub in ("cryo-electron tomography", "darkfield microscopy"):
            rec("stain", "ST2a", "hard")
            L["stain"] = {"mode": "hard", "values": ["unstained"]}
        elif sub == "synthetic fluorescence microscopy":
            rec("stain", "ST2b", "hard")
            L["stain"] = {"mode": "hard", "values": ["synthetic"]}
        elif sub in NO_STAIN_SUBS:
            rec("stain", "ST3", "mask-confirmed")
        else:
            rec("stain", "ST4", "masked")
    elif st["mode"] != "masked" and sub in NO_STAIN_SUBS:
        rec("stain", "ST5", "warn")
    elif st["mode"] == "masked":
        rec("stain", "ST1", "mask-confirmed")
    # ---- domain
    dm = L["domain"]
    if mods & RAD and "radiology" in dm["values"]:
        if [x for x in dm["values"] if x != "radiology"]:
            rec("domain", "D1", "multi")
            L["domain"] = {"mode": "multi", "values": [x for x in dm["values"] if x != "radiology"]}
        else:
            rec("domain", "D1b", "keep")
    if len(L["domain"]["values"]) > 3:
        rec("domain", "D2", "cap")
    OUT[sid] = L

RULE_DESC = {
    ("modality", "M1"): "submodality hard and exactly one candidate modality is legal for it",
    ("modality", "M2"): "{digital pathology, light microscopy} candidates -> light microscopy (public: 92% of WSI, 96% of brightfield rows are LM)",
    ("submodality", "S1"): "modality hard, exactly one candidate submodality is legal under it",
    ("submodality", "S2"): "{laparoscopy, white-light endoscopy} -> laparoscopy (the specific term)",
    ("submodality", "S3"): "{colonoscopy|gastroscopy, white-light endoscopy} -> white-light endoscopy (public: 163 WLE vs 5 colonoscopy rows)",
    ("body_part", "B1"): "strong submodality implies body part (fundus/OCT/OCTA/FA -> eye, dermoscopy -> skin, mammography -> breast, chest x-ray -> chest, colposcopy -> cervix)",
    ("body_part", "B2"): "laparoscopy -> abdomen when registry says 'not applicable' or lists only abdominal organs (public: 93% abdomen family)",
    ("body_part", "B3"): "prune modifiers: drop blood vessels/arteries when a region is listed; head subset of brain for MRI/CT",
    ("body_part", "B4"): "remaining candidates all in one anatomical family -> train the family head hard, atom head partial-label",
    ("body_part", "B5"): "UNRESOLVED: candidates span several families (true multi-region dataset; needs per-image metadata)",
    ("body_part", "B6"): "UNRESOLVED: masked (registry 'not applicable'/'multiple', or >4 regions across several families)",
    ("body_part", "B7"): "masked only because >4 regions were listed, all in one family (galar, rhode-island-vce: GI tract) -> family-hard",
    ("specimen", "SP1"): "drop non-species tokens (cell line, phantom, cells) when exactly one species is listed",
    ("specimen", "SP2"): "clinical (non-microscopy-lab) modality and 'human' among candidates -> human (public: >=96% human)",
    ("specimen", "SP3"): "submodality = synthetic fluorescence microscopy -> specimen 'synthetic' (public: 38/38)",
    ("specimen", "SP4"): "{bacteria, e. coli}: nested, keep as family",
    ("specimen", "SP5"): "UNRESOLVED candidates (multi-species datasets)",
    ("specimen", "SP6"): "SOFT: clinical modality with registry specimen 'unknown' -> human (not applied automatically)",
    ("specimen", "SP7"): "UNRESOLVED masked ('mixed', plankton, unknown)",
    ("stain", "ST1"): "non-microscopy modality: stain not asked -> mask (already done by select_datasets.py)",
    ("stain", "ST2a"): "cryo-ET or darkfield submodality -> 'unstained' (public: 15/15 rows)",
    ("stain", "ST2b"): "synthetic fluorescence -> stain 'synthetic' (public: 38/38 rows; currently dropped by canon_stain)",
    ("stain", "ST3"): "submodality for which MMBU never asks stain (phase contrast, contrast, confocal, differential, multiphoton): mask confirmed",
    ("stain", "ST4"): "UNRESOLVED: stained modality, registry has no stain (registry text mining finds a stain keyword in only 4/35)",
    ("stain", "ST5"): "WARN: registry lists a stain for a submodality MMBU never asks stain for (harmless, but weight down)",
    ("domain", "D1"): "radiologic modality: drop 'radiology' when a specialty is present (registry 95-100% vs public 10-49%); also brings the 3 four-domain datasets under the public cap of 3",
    ("domain", "D1b"): "radiology is the only registry domain: keep",
    ("domain", "D2"): "more than 3 domains remain after D1: cap at 3 (public max is 3)",
}
rules_rows = []
for (f, rid), v in sorted(log.items(), key=lambda kv: (FIELDS.index(kv[0][0]), kv[0][1])):
    rules_rows.append({"field": f, "id": rid, "rule": RULE_DESC[(f, rid)], "datasets": len(v), "images": sum(x[3] for x in v)})
rules_df = pd.DataFrame(rules_rows)
mt_before, mt_after = mode_table(lambda s: s["labels"]), mode_table(lambda s: OUT[s["id"]])

# masked body-part reasons
bp_masked_reason = collections.Counter()
for s in S:
    if s["labels"]["body_part"]["mode"] == "masked":
        bps = s["reg"]["body_parts"]
        if any(b in ("not applicable",) for b in bps):
            bp_masked_reason["registry says 'not applicable' (simulated/instrument/tracking data)"] += 1
        elif "multiple" in bps:
            bp_masked_reason["registry says 'multiple'"] += 1
        elif len(bps) > 4 or len(s["labels"]["body_part"]["values"]) == 0 and len(bps) > 4:
            bp_masked_reason["more than 4 regions listed (select_datasets.py masks them)"] += 1
        else:
            bp_masked_reason["value outside the vocabulary (leaf epidermis, urine)"] += 1
bp_cand_after_fam = collections.Counter(len({PARENT.get(v, "other") for v in s["labels"]["body_part"]["values"]}) == 1
                                        for s in S if s["labels"]["body_part"]["mode"] == "candidates")

# cause tables for non-hard cells before the rules
def pair_counter(f):
    c = collections.Counter()
    for s in S:
        l = s["labels"][f]
        if l["mode"] == "candidates":
            c[tuple(l["values"])] += 1
    return c

mod_pairs, sub_pairs, spec_pairs = pair_counter("modality"), pair_counter("submodality"), pair_counter("specimen")

# registry vs public domain conventions per modality
conv_rows = []
for m in ["x-ray imaging", "computed tomography", "magnetic resonance imaging", "ultrasound imaging", "light microscopy", "endoscopy"]:
    reg_c, n = collections.Counter(), 0
    for s in S:
        if m in s["labels"]["modality"]["values"]:
            n += 1
            reg_c.update(s["labels"]["domain"]["values"])
    pub_c, Np = collections.Counter(), 0
    for mm, dd in zip(d["mod"], d.dom):
        if m in mm.split("|"):
            Np += 1
            pub_c.update(dd)
    for k in sorted(set(reg_c) | set(pub_c), key=lambda k: -(pub_c[k] / Np + reg_c[k] / n))[:5]:
        conv_rows.append({"modality": m, "domain": k, "registry (share of datasets)": reg_c[k] / n, "public (share of rows)": pub_c[k] / Np})
conv_df = pd.DataFrame(conv_rows)
n_cells_hard = sum(1 for s in S if s["labels"]["body_part"]["mode"] == "hard" and s["labels"]["body_part"]["values"] == ["cells"])
n_cell_line = sum(1 for s in S if "cell line" in s["labels"]["specimen"]["values"])
n_phantom = sum(1 for s in S if "phantom" in s["labels"]["specimen"]["values"])
n_dp_hard = sum(1 for s in S if s["labels"]["modality"]["values"] == ["digital pathology"])
n_rad_dom = sum(1 for s in S if set(s["labels"]["modality"]["values"]) & RAD and "radiology" in s["labels"]["domain"]["values"])
sole_rad = sum(1 for s in S if s["labels"]["domain"]["values"] == ["radiology"])
n_rad_ds = sum(1 for s in S if set(s["labels"]["modality"]["values"]) & RAD)
dom_masked = [(s["id"], s["reg"]["medical_domains"]) for s in S if s["labels"]["domain"]["mode"] == "masked"]

# ============================ write JSON ===========================================================================
bycat = lambda df: {r["mod" if "mod" in df.columns else "sub"]: r for r in df.to_dict("records")}  # noqa: E731
rules_json = {
    "meta": {"generated_by": "bio-waila/docs/label_design_analysis.py", "public_rows": N,
             "note": "public statistics are evaluation-side; do not train on mmbu-public"},
    "applicability": {
        "requested_fields_are_given_in_prompt": True,
        "field_presence_public": {
            "modality": 1.0, "submodality": 1.0, "specimen": 1.0, "domain": 1.0,
            "body_part": bp_has / N, "stain": float(d.has_stain.mean())},
        "by_modality": {r["mod"]: {"n": r["n"], "body_part": round(r["body_part"], 4), "stain": round(r["stain"], 4),
                                   "specimen_human": round(r["specimen_human"], 4)} for r in by_mod.to_dict("records")},
        "by_submodality": {r["sub"]: {"n": r["n"], "body_part": round(r["body_part"], 4), "stain": round(r["stain"], 4)}
                           for r in by_sub.to_dict("records")},
        "field_combos_by_modality": {m: {c: int(v) for c, v in row.items() if v} for m, row in combo_tab.iterrows()},
        "stain_modalities": MICRO,
        "stain_never_asked_submodalities": sorted(s_ for s_ in set(st_sub["sub"]) if st_sub.set_index("sub").stain[s_] == 0),
        "stain_always_asked_submodalities": sorted(s_ for s_ in set(st_sub["sub"]) if st_sub.set_index("sub").stain[s_] == 1),
        "body_part_never_asked": {"modalities_all_rows": ["clinical photography"], "submodalities_all_rows": sorted(
            r["sub"] for r in by_sub.to_dict("records") if r["body_part"] == 0)},
        "rules": [
            {"id": "A1", "fields": ["modality", "submodality", "specimen", "domain"], "applies": "always", "evidence": f"{N}/{N} public rows"},
            {"id": "A2", "field": "stain", "applies": "modality in " + str(MICRO),
             "evidence": f"{stain_micro}/{n_micro} microscopy rows annotated ({pct(stain_micro / n_micro, 1)}); 0/{N - n_micro} non-microscopy rows"},
            {"id": "A3", "field": "body_part", "applies": "all non-microscopy rows except modality 'clinical photography'; "
             "microscopy rows only sometimes (see by_modality)",
             "evidence": f"non-microscopy rows without body part: {no_bp_non_micro} (all 'clinical photography' modality)"},
            {"id": "A4", "applies": "every microscopy row has body_part or stain or both",
             "evidence": f"both {both}, body-part only {bp_only}, stain only {st_only}, neither {none_micro}"},
            {"id": "A5", "field": "specimen", "applies": "always asked; value 'human' for clinical modalities (see implication rules)"},
        ],
    },
    "implication_rules": {
        "hard_submodality_to_modality": {s_: sorted(m) for s_, m in legal_mod.items() if len(m) == 1},
        "ambiguous_submodality_modalities": ambig_sub,
        "strong_submodality_to_body_part": STRONG_BP,
        "soft_submodality_to_body_part": {"laparoscopy": "abdomen", "chest x-ray": "chest|lung", "white-light endoscopy": "gastrointestinal tract"},
        "submodality_to_stain": {"cryo-electron tomography": "unstained", "darkfield microscopy": "unstained",
                                 "synthetic fluorescence microscopy": "synthetic", "whole slide imaging": "h&e (82%, soft prior)"},
        "submodality_to_specimen": {"synthetic fluorescence microscopy": "synthetic"},
        "modality_to_specimen_human_default": {m: round(float((g.F_specimen == "human").mean()), 3) for m, g in d.groupby("mod")
                                               if m in CLIN or m == "computed tomography|x-ray imaging"},
        "stain_implies_modality": MICRO,
        "submodality_to_domain_soft": {"fundus photography": "ophthalmology", "optical coherence tomography": "ophthalmology",
                                       "dermoscopy": "dermatology", "chest x-ray": "pulmonology", "laparoscopy": "surgery",
                                       "colposcopy": "obstetrics and gynecology", "color doppler": "cardiology"},
        "modality_to_field_decode_order": ["modality", "submodality", "body_part", "stain", "specimen", "domain"],
        "body_part_conventions_public": conv,
    },
    "domain_parsing": {
        "steps": ["ast.literal_eval the cell (it is a python-list string)",
                  "for each list item split on /\\s*,\\s*/ (packed items like 'gastroenterology, radiology')",
                  "lower-case, collapse whitespace, strip",
                  "alias map: otorhinolaryngology and 'otolaryngology (ent)' -> otolaryngology; infectious diseases -> infectious disease; "
                  "gynecology -> obstetrics and gynecology",
                  "dedupe; treat as an UNORDERED set (the same pair appears in both orders)",
                  "never split on ' and ' (only 'obstetrics and gynecology' contains it)"],
        "verified_raw_strings": len(raw_dom), "verified_rows": N, "unmapped_atoms": 0,
        "raw_to_set": {k: v for k, v in parsed.items()},
        "size_distribution": {int(k): int(v) for k, v in dom_size.items()},
        "packed_raw_strings": dict(packed), "alias_duplicate_raw_strings": dict(alias_dups),
        "pair_counts": {"|".join(k): v for k, v in pair_counts.most_common()},
        "discipline_atoms": sorted(DISCIPLINE),
    },
    "body_part_parsing": {
        "split": "|", "lower_and_strip": True, "alias": "lungs -> lung; male genitournary system -> male genitourinary system (see vocab.py)",
        "pipe_strings": {r["raw string"]: {"rows": int(r["rows"]), "type": r["type"]} for r in pipe_tab.to_dict("records")},
        "size_distribution": {int(k): int(v) for k, v in bp_size.items()},
        "hierarchy_pairs": [["chest", "lung"], ["eye", "retina"], ["brain", "cerebrum"], ["lung", "lung tissue"]],
        "families": FAMILY,
    },
    "compatibility": {
        "modality_to_submodality_legal": legal,
        "modality_to_submodality_public_counts": {m: dict(pub_pairs[m]) for m in mods_all},
        "taxonomy_only": tax_only, "public_only": pub_only,
        "submodality_to_modality_public_counts": {s_: m for s_, m in sub2mod.items()},
        "submodality_to_modality_legal": {s_: sorted(m) for s_, m in legal_mod.items()},
    },
    "predictability": {"pairs": pairs.round(4).to_dict("records"), "chains": chain, "group_for_cv": "metadata_topic",
                       "n_groups": int(n_unique_topics)},
    "training_rules": rules_rows,
    "training_mode_counts": {f: {"before": dict(b), "after": dict(a)} for (f, b, _), (_, a, _) in zip(mt_before, mt_after)},
    "label_schema": {
        "modality": {"classes": "taxonomy.toml [modalities].core (11)", "type": "single", "head": "softmax",
                     "constraint": "none; root of the decode order", "unknown": "no unknown class"},
        "submodality": {"classes": "public 35 + fluoroscopy + gastroscopy = 37", "type": "single", "head": "softmax masked to legal(modality)",
                        "unknown": "no unknown class"},
        "body_part": {"classes": "public atoms (66 after aliases) + registry atoms with >=3 datasets; plus a 14-way family head",
                      "type": "multi-label atoms (BCE) + single family; partial-label loss for candidates", "applicable": "mask when "
                      "dataset label is masked; at inference ask only when requested", "decode": "family -> atoms p>=tau (cap 3), general first"},
        "stain": {"classes": "public canonical stains + 'synthetic' + 'unstained'", "type": "single", "applicable": MICRO,
                  "mask": "all non-microscopy images and NO_STAIN submodalities", "unknown": "no unknown class"},
        "specimen": {"classes": "human + ~20 species/cell classes; non-answers removed", "type": "single", "applicable": "all images",
                     "mask": "non-answers (x-ray images, unidentified, biological specimen, mixed)"},
        "domain": {"classes": "23 public domains", "type": "multi-label (BCE) max 3", "parse": "see domain_parsing",
                   "decode": "top-1 always, add 2nd/3rd only if p>=tau", "train_fix": "drop radiology when another specialty present"},
    },
}
(HERE / "label_rules.json").write_text(json.dumps(rules_json, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))

# ============================ write MD ==============================================================================
L_ = []
w = L_.append
w("# Pass-2 label design: which fields apply to which images, and how to represent each as a label\n")
w(f"Generated by `label_design_analysis.py` from the {N:,} public rows (evaluation-side statistics only), the 393-dataset plan, and "
  "`taxonomy.toml`. Machine-readable rules: `label_rules.json`.\n")

w("## 0. Summary of decisions\n")
w(f"""1. **Applicability is a conditioning input, not something to predict.** The Pass-2 prompt ends with "Requested metadata fields for this example: ...", so the requested set R is known at inference. In the public split modality, submodality, specimen and domain are annotated for {N}/{N} rows; **stain only ever appears on microscopy** ({stain_micro}/{n_micro} microscopy rows = {pct(stain_micro / n_micro, 1)}, 0 of {N - n_micro} others); body part is missing on {N - bp_has} rows (278 stain-only microscopy rows + 31 clinical-photography-modality rows).
2. Train all six heads on every image where the label exists; use R only to decide which keys to emit. The rules' own example prompt requests only "body part, modality, stain, submodality" (no specimen, domain), so do not assume specimen/domain are always requested in the hidden set even though they are 100% in the public split.
3. **Domain** is a set of 1-3 atoms ({pct(dom_single_share)} of rows have one). Parse with: literal_eval, split items on commas, lower/alias/dedupe, unordered set. "obstetrics and gynecology" is one label; "radiology, urology" is two. Verified on all 76 raw strings with zero unmapped atoms.
4. **Body part** is a `|`-joined set only on {len(pipe_rows)} rows ({pct(len(pipe_rows) / bp_has, 1)} of annotated): 280 are hierarchy/synonym pairs (chest|lung), 79 are dataset-level multi-region inventories (esophagus|colon|rectum, colon tissue|lung tissue), 20 are one cell-type list. Train as multi-hot atoms + a 14-way family head; decode general-then-specific.
5. **Modality is the easy root** (submodality determines it for {pct(detmass(d.F_submodality, d.F_modality))} of rows at >=95% purity; H(modality|submodality)=0.10 bits). Recommended decode order: modality -> submodality (masked to legal set) -> body part -> stain (microscopy only) -> specimen -> domain. Domain goes last: it is the least visual and the most annotator-dependent.
6. Deterministic training rules (section 4) resolve {int(rules_df[rules_df.id.isin(['M1','M2','S1','S2','S3','B1','B2','B3','B4','B7','SP1','SP2','SP3','ST2a','ST2b'])].datasets.sum())} non-hard dataset-field cells to hard or family-hard without any model (body-part hard datasets 144 -> {dict(mt_after[2][1])['hard']}, plus {dict(mt_after[2][1]).get('family', 0)} family-hard) and edit {int(rules_df[rules_df.id=='D1'].datasets.sum())} domain sets (rule D1).
7. **Registry domain labels are systematically off from MMBU convention** (radiology on 95-100% of x-ray/CT/MRI datasets vs 10-49% of public rows; oncology on 73% of CT datasets vs 10%). Fix with rule D1 before training.
8. No `unknown` class: the public references never contain it for a requested field, so abstaining can only lose. Keep abstention as an off-by-default threshold.
""")

# ---------------- section 1
w("## 1. Applicability matrix (public split)\n")
w("### 1.1 Field coverage\n")
cov = pd.DataFrame([{"field": f, "rows annotated": (N if f in ("modality", "submodality", "specimen", "domain") else bp_has if f == "body_part" else int(d.has_stain.sum())),
                     "share": (1.0 if f in ("modality", "submodality", "specimen", "domain") else bp_has / N if f == "body_part" else float(d.has_stain.mean()))} for f in FIELDS])
cov["share"] = cov.share.map(lambda x: pct(x, 1))
w(md(cov) + "\n")
w("### 1.2 By modality\n")
t = by_mod.copy()
t["body_part"] = t.body_part.map(pct)
t["stain"] = t.stain.map(pct)
t["specimen_human"] = t.specimen_human.map(pct)
t.columns = ["modality", "n", "body part annotated", "stain annotated", "specimen = human"]
w(md(t) + "\n")
w("Modality, submodality, specimen and domain are 100% in every row of every modality (not shown).\n")
w("### 1.3 By submodality\n")
t = by_sub.copy()
t["body_part"] = t.body_part.map(pct)
t["stain"] = t.stain.map(pct)
t.columns = ["submodality", "n", "body part annotated", "stain annotated"]
w(md(t) + "\n")
w("### 1.4 Which fields are asked together\n")
ct = combo_tab.copy()
ct.columns = [{"-": "neither", "-+stain": "stain only", "bp": "body part only", "bp+stain": "both"}.get(c, c) for c in ct.columns]
ct = ct.reset_index().rename(columns={"mod": "modality"})
w("Row counts (body part / stain combination, on top of the four always-asked fields):\n")
w(md(ct) + "\n")
w(f"""Microscopy rows ({n_micro}): both {both}, body part only {bp_only}, stain only {st_only}, neither {none_micro}. So **every microscopy row has at least one of body part or stain**, and {pct((n_micro - stain_micro) / n_micro, 1)} of microscopy rows lack stain (all of them have a body part). Outside microscopy the only combination is "body part" except the 31 rows of modality `clinical photography`, which carry neither.
""")
w("### 1.5 Stain applicability inside microscopy, by submodality\n")
t = st_sub.copy()
t["stain"] = t.stain.map(pct)
t["body_part"] = t.body_part.map(pct)
t.columns = ["submodality (microscopy rows)", "n", "stain annotated", "body part annotated"]
w(md(t) + "\n")
w("""Reading: stain is **always** asked for whole slide imaging, darkfield, cryo-ET, synthetic fluorescence and TIRF; **never** for confocal, contrast, phase-contrast and differential fluorescence (these rows carry a body part instead); and 43-74% for SEM, epifluorescence and brightfield. The brightfield/epifluorescence/SEM split is dataset-level in the annotation (whole datasets have a stain or not), not predictable from the image.
""")
w("### 1.6 Explicit applicability rules\n")
w(f"""| id | rule | evidence |
|---|---|---|
| A1 | modality, submodality, specimen, domain: always annotated | {N}/{N} |
| A2 | stain: only when modality is one of light/fluorescence/electron microscopy or digital pathology | {stain_micro}/{n_micro} microscopy ({pct(stain_micro / n_micro, 1)}); 0/{N - n_micro} elsewhere |
| A3 | stain inside microscopy: always for WSI, darkfield, cryo-ET, synthetic fluorescence, TIRF; never for confocal, contrast, phase contrast, differential fluorescence | table 1.5 |
| A4 | body part: all non-microscopy rows except modality `clinical photography` (0/31); microscopy {pct(micro.has_bp.mean())} overall (light {pct(d[d['mod']=='light microscopy'].has_bp.mean())}, fluorescence {pct(d[d['mod']=='fluorescence microscopy'].has_bp.mean())}, EM {pct(d[d['mod']=='electron microscopy'].has_bp.mean())}, digital pathology 100%) | table 1.2 |
| A5 | every microscopy row has body part or stain or both | {both} + {bp_only} + {st_only} = {n_micro} |
| A6 | body part never asked for synthetic fluorescence, cryo-ET, TIRF | 0/38, 0/8, 0/4 |
| A7 | the requested set R is part of the prompt; use it to select which keys to emit | rules PDF, Pass-2 prompt |
""")
w("""Practical consequences: the stain head needs no `none` class (a microscopy image with stain requested always has an answer; a not-requested image is simply never scored). For training, **mask** stain for every non-microscopy image and for the never-asked submodalities, and mask it for microscopy datasets whose registry gives no stain (35 datasets, section 4). Do not train a `none` target from absence: absence in MMBU means "not requested", not "no stain" (the exception is the explicit `unstained` class: 15 public rows, all in darkfield or cryo-ET).
""")
w("### 1.7 Non-answers in the references\n")
w(f"""Specimen non-answers ({int(nonans_spec.sum())} rows = {pct(nonans_spec.sum() / N, 1)}): {', '.join(f'`{k}` ({v})' for k, v in nonans_spec.items())}. All 62 `x-ray images` / `biological specimen` rows are chest x-rays from two datasets (topics 'Pneumonia' 52, 'Tuberculosis' 10), so they cannot be predicted from pixels; the right visual answer (`human`) will often be judged wrong against them. Vague-but-real values: {', '.join(f'`{k}` ({v})' for k, v in vague_spec.items())}. Stain non-answers ({int(nonans_stain.sum())} rows): {', '.join(f'`{k}` ({v})' for k, v in nonans_stain.items())}; `synthetic` (38 of 38 synthetic-fluorescence rows) is a consistent, learnable answer and should be a class, whereas `uniform test slide` (8) is a one-off.
""")

# ---------------- section 2
w("## 2. Multi-label structure\n")
w("### 2.1 Medical domain\n")
w(f"""Raw cell format: a python-list string, e.g. `['gastroenterology, radiology', 'surgery', 'radiology']`. Items per list: {dict(sorted(n_items.value_counts().items()))}; atoms per row after the parsing rule: {dict(sorted(dom_size.items()))}.

**Parsing rule** (implemented in `parse_domain` of the analysis script and recorded in `label_rules.json`):

1. `ast.literal_eval` the cell.
2. For each item, split on `\\s*,\\s*` (item-internal packing: {sum(packed.values())} rows in {len(packed)} raw strings, e.g. `'urology, radiology'`).
3. Lower-case, collapse whitespace, strip.
4. Alias: `otorhinolaryngology`, `otolaryngology (ent)` -> `otolaryngology`; `infectious diseases` -> `infectious disease`; `gynecology` -> `obstetrics and gynecology`.
5. Dedupe and treat as an **unordered set**: the same pair appears in both orders in raw data ({', '.join(' / '.join('+'.join(o) for o in sorted(v)) for v in flipped.values())}), so order carries no "primary" signal.
6. Never split on ` and `: `obstetrics and gynecology` is the only atom containing it and it is one domain. "radiology, urology" is two labels.

Verification: all {len(raw_dom)} distinct raw strings parse; every atom maps to the 23-class vocabulary (0 unmapped, `parse_domain` raises otherwise). Two raw strings contain alias duplicates that collapse after step 4 (`['surgery', 'otolaryngology', 'otorhinolaryngology']` 32 rows; `['gastroenterology, radiology', 'surgery', 'radiology']` 14 rows, where `radiology` appears both packed and standalone). The 3-item lists are 48 rows after dedupe, never more than 3.
""")
pc = pd.DataFrame([{"pair": " + ".join(k), "rows": v} for k, v in pair_counts.most_common(14)])
w("Most frequent co-occurring pairs (all pairs in `label_rules.json`):\n")
w(md(pc) + "\n")
w(f"""Structure: domains behave like **specialty (+ optional discipline)**. Of {len(dom_comb)} multi-domain rows, {int((dom_comb == 1).sum())} pair a specialty with exactly one discipline atom (radiology, pathology, surgery, biology, cytology), {int((dom_comb == 0).sum())} have none (e.g. andrology + urology: 93 rows, all brightfield `sperm cells` images from one dataset) and {int((dom_comb == 2).sum())} have two. Radiology is an *add-on*, not the default: it appears on {pct(rad_has_radiology)} of x-ray/CT/MRI/ultrasound rows (CT {pct(rad_by_mod['computed tomography'])}, MRI {pct(rad_by_mod['magnetic resonance imaging'])}, ultrasound {pct(rad_by_mod['ultrasound imaging'])}, x-ray {pct(rad_by_mod['x-ray imaging'])}). Chest x-ray rows are `pulmonology` alone 93% of the time. Domain is largely **a dataset-level annotator choice**: group-held-out accuracy of predicting domain from (submodality, body part) is {pct(chain['sb->domain']['acc_group'])} vs {pct(chain['sb->domain']['acc_loo'])} leave-one-out.

**Recommendation**: multi-label (BCE) over the 23 atoms, cap 3, decoded as: always the argmax atom; add further atoms only if p >= tau (tau ~ 0.5, tune on the validation datasets, not on the public split). Under a semantic judge, one correct atom of two likely earns partial or full credit; a wrong extra atom costs little compared with missing the right one. Do not make "radiology" a default for radiologic modalities (see D1, section 4).
""")
w("### 2.2 Body part\n")
w(f"""{bp_has} rows annotated; atoms per row: {dict(sorted(bp_size.items()))[1]} one ({pct(bp_single_share, 1)} of annotated), the rest multi. Only {len(pipe_tab)} distinct pipe strings exist, in three kinds:
""")
w(md(pipe_tab) + "\n")
w(f"""Totals by type: {', '.join(f'{k} {v}' for k, v in pipe_type_rows.items())}.

* **Hierarchy/synonym pairs** are written general-to-specific and are *annotation variants of the same thing*. Chest x-ray rows: `chest|lung` {chest_vc.get('chest|lung', 0)}, `chest` {chest_vc.get('chest', 0)}, `thorax` {chest_vc.get('thorax', 0)}, `lungs` {chest_vc.get('lungs', 0)}, `heart` {chest_vc.get('heart', 0)}. A semantic judge will likely accept `chest` for `chest|lung` and vice versa (judge behavior is not documented, so verify on a few examples); the safest output for a chest x-ray is `chest|lung` (exact in 62%, a semantic superset in 30%).
* **Multi-region inventories** (GI endoscopy `esophagus|colon|rectum`, `stomach|colon|esophagus`; LC25000 `colon tissue|lung tissue`) are dataset-level lists; each image shows only one of the listed organs. The best visual answer is the organ actually seen, which is a member of the reference set. Train multi-hot so mass spreads across listed atoms; decode the top atom plus any atom with p >= tau.
* **Cell-type list** (20 rows): five blood-cell names; treat as one class family `blood and lymph`/`white blood cells`.

**Train/evaluate**: pick-one has an exact-match ceiling of {pct(bp_single_share, 1)} of annotated rows, but a ceiling of 100% under any-overlap. Because the judge is semantic, use three proxies on the public split: *exact set match*, *Jaccard*, *any-overlap (hit@1 in set)*. Training target = multi-hot over atoms (BCE for hard/multi datasets; `-log sum p(candidates)` for candidates) plus a coarse **family** head (14 anatomical families, table in `label_rules.json`, hard-labelled whenever all registry atoms share a family). Decode: argmax family -> atoms with p >= tau under that family (cap 3), general atom first (`chest|lung`, `eye|retina`).

MMBU conventions the decoder should reproduce (top strings per submodality, share of annotated rows):
""")
ctab = pd.DataFrame([{"submodality": s_, "top body-part strings": "; ".join(f"{k} ({pct(v)})" for k, v in vv)} for s_, vv in sorted(conv.items(), key=lambda kv: -len(d[(d['sub'] == kv[0]) & d.has_bp]))][:16])
w(md(ctab) + "\n")

# ---------------- section 3
w("## 3. Hierarchy and constraints\n")
w("### 3.1 Modality -> submodality compatibility\n")
w("Public counts and `taxonomy.toml [submodalities]` side by side. `legal` = union. 'taxonomy-only' pairs never occur in the public rows (may appear in the hidden split); 'public-only' pairs are in the public data but missing from taxonomy.\n")
w(md(compat_df) + "\n")
w(f"""Ambiguous submodalities (legal under several modalities): {'; '.join(f'`{s_}` -> ' + ', '.join(f'{m} ({c})' for m, c in v.items()) for s_, v in ambig_sub.items())}. Everything else maps to exactly one modality. The first five are structural ambiguities; `chest x-ray`/`conventional ct` are the 8-row CT/x-ray artifact. In practice **submodality -> modality is almost a function**: 33 of 36 values are >=95% pure (only confocal, clinical photography and whole slide imaging are not) and they cover {pct(detmass(d.F_submodality, d.F_modality))} of rows. Noted annotation artifacts: 8 CT rows carry both modalities (`computed tomography|x-ray imaging`, submodality `chest x-ray|conventional ct`); 22 brightfield + 8 WSI rows are labelled `digital pathology`, indistinguishable from light-microscopy histology (treat as soft: train `light microscopy` primary, `digital pathology` alternate). Taxonomy lists confocal and multiphoton under light microscopy and colposcopy under endoscopy, whereas the public data files them under optical imaging: keep both as legal and let the data decide.
""")
w("### 3.2 What a submodality implies (public, submodalities with n >= 8)\n")
w(md(typ_df) + "\n")
w("Deterministic enough to use as implication rules (>= 99% and n >= 20, or taxonomy-defined): fundus -> eye family (91% `eye`, 9% `retina`), dermoscopy -> skin, mammography -> breast, chest x-ray -> chest family (99%), OCT/OCTA/fluorescein angiography -> eye, cryo-ET and darkfield -> `unstained`, synthetic fluorescence -> stain and specimen `synthetic`. Soft (80-95%): laparoscopy -> abdomen family (93%) and `surgery` (88%), chest x-ray -> `pulmonology` (93%), WSI -> `h&e` (82%). Everything else (MRI, CT, brightfield, epifluorescence, ultrasound body parts; MRI/CT/US domains) is dataset-driven and must be learned, not ruled.\n")
w("### 3.3 Predictability between fields\n")
w(f"""Entropy and majority-vote accuracy of field B given field A on the public rows. H in bits. `acc_loo` is leave-one-out (optimistic: rows of one dataset predict each other); `acc_group` leaves out the whole `metadata_topic` group ({n_unique_topics} groups, a dataset proxy) and is the honest number. `base` is the majority-class rate of B. `det` (share of rows whose A-value predicts B at >= 95% purity), `acc_loo` and `base` are in the `predictability` block of `label_rules.json`. Stain is scored with `∅` (not annotated) as a class, so its baseline is high.
""")
H_ = {f: H(d["F_" + f]) for f in FIELDS}
ent_df = pd.DataFrame([{"A \\ B (H(B|A) bits)": a, **{b: (condH(d["F_" + a], d["F_" + b]) if a != b else float("nan")) for b in FIELDS}} for a in FIELDS])
ent_df = pd.concat([pd.DataFrame([{"A \\ B (H(B|A) bits)": "H(B)", **{b: H_[b] for b in FIELDS}}]), ent_df], ignore_index=True)
w(md(ent_df.fillna("-")) + "\n")
acc_df = pd.DataFrame([{"A \\ B (group acc)": a, **{b: (float(pairs[(pairs.A == a) & (pairs.B == b)].acc_group.iloc[0]) if a != b else float("nan")) for b in FIELDS}} for a in FIELDS])
base_row = pd.DataFrame([{"A \\ B (group acc)": "baseline", **{b: float(d["F_" + b].value_counts(normalize=True).iloc[0]) for b in FIELDS}}])
w(md(pd.concat([base_row, acc_df], ignore_index=True).fillna("-")) + "\n")
dm_ = pairs.set_index(["A", "B"]).det_mass
w(f"""Findings:

* **submodality -> modality**: H drops 2.88 -> 0.10 bits; {pct(float(dm_['submodality', 'modality']))} of rows are deterministic; group accuracy {pct(float(pairs[(pairs.A=='submodality')&(pairs.B=='modality')].acc_group.iloc[0]))}. The reverse (modality -> submodality) leaves 1.05 bits ({pct(float(pairs[(pairs.A=='modality')&(pairs.B=='submodality')].acc_group.iloc[0]))}). Submodality is the finer variable and carries modality inside it.
* **stain** is predictable only through applicability: given modality, `∅` vs stained is exact (0 outside microscopy); given submodality, {pct(float(dm_['submodality', 'stain']))} of rows are deterministic. The stain *value* is not predictable from other fields (given microscopy, H(stain) stays high): it must come from pixels.
* **body part | submodality** removes 3.0 of 4.6 bits in-sample but only {pct(float(pairs[(pairs.A=='submodality')&(pairs.B=='body_part')].acc_group.iloc[0]))} group accuracy vs {pct(float(d.F_body_part.value_counts(normalize=True).iloc[0]))} baseline: the family is predictable (strong rules above), the atom is dataset-driven.
* **domain | (submodality, body part)**: H 4.21 -> {chain['sb->domain']['H(B|A)']:.2f} bits in-sample (a dataset fingerprint) but only {pct(chain['sb->domain']['acc_group'])} group accuracy. Domain is the **least transferable** field; predict it last, conditioned on everything else, and do not over-invest in it.
* **specimen** is ~89% `human` and {pct(float(pairs[(pairs.A=='modality')&(pairs.B=='specimen')].acc_group.iloc[0]))} predictable from modality alone (group); the non-human minority sits in fluorescence/EM/light microscopy and ultrasound (`tissue`, `animal`).
* Stain given specimen, domain or body part are all weak for the value; they matter only via applicability.
""")
w("### 3.4 Decode order\n")
w("""Chain rule makes the joint order-invariant; order matters because of (a) legal-set masking and (b) error propagation. Recommended progressive order:

1. **modality** (11-way; visually the easiest root).
2. **submodality**, logits masked to `legal(modality)` (mean legal-set size is 4.5 (range 1-12) vs 37 unconstrained; only brightfield/WSI/confocal/clinical photography/white-light endoscopy leave modality ambiguous, so decode submodality with a joint (modality, submodality) beam, not a hard commit).
3. **body part**: family first, then atoms (the strong submodality rules fix the family for 8 submodalities).
4. **stain**, only if modality is microscopy (or R contains stain), masked off for never-asked submodalities.
5. **specimen** (89% human prior; non-human only in microscopy/US).
6. **domain**, last, conditioned on modality, submodality and body part.

Domain-first (coarse-to-fine by specialty) is *not* recommended: domain is the noisiest, most annotator-dependent field and predicts modality worse (H(modality|domain)=0.53) than submodality does (0.10). With a plain multi-head classifier the same constraints can be applied post hoc (joint argmax over legal (modality, submodality) pairs, then conditional heads). With a text-contrastive model, score label strings in this order and prune by the compatibility table.
""")

# ---------------- section 4
w("## 4. Deterministic labeling rules for the training plan\n")
w(f"`plan/selection.jsonl`: {len(S)} datasets, {TOT_IMG:,} planned images. Modes are dataset-level targets from `select_datasets.py`.\n")
w("### 4.1 Mode counts (datasets / % of planned images)\n")
mrows = []
for (f, b, bw), (_, a, aw) in zip(mt_before, mt_after):
    for mode in ["hard", "family", "multi", "candidates", "masked"]:
        if b.get(mode) or a.get(mode):
            mrows.append({"field": f, "mode": mode, "datasets before": b.get(mode, 0), "images before": pct(bw.get(mode, 0) / TOT_IMG, 1),
                          "datasets after rules": a.get(mode, 0), "images after": pct(aw.get(mode, 0) / TOT_IMG, 1)})
w(md(pd.DataFrame(mrows)) + "\n")
w("`family` = body-part family is hard, atom is a candidate set. `multi` = domain (all values apply). Domain stays `multi` but its value sets change under D1.\n")
w("### 4.2 Systematic causes of non-hard labels\n")
w(f"""* **modality (16 candidate datasets)**: 8 are {{digital pathology, light microscopy}} (annotation convention, not a real mixture); 8 are registry entries listing two real modalities ({', '.join('+'.join(k) for k, v in mod_pairs.items() if set(k) != {'digital pathology', 'light microscopy'})}), i.e. multi-source datasets that need per-image labels.
* **submodality (28 candidate + 2 masked)**: 15 are {{colonoscopy|gastroscopy|laparoscopy, white-light endoscopy}} (the registry lists the procedure and the generic technique; MMBU files GI endoscopy under `white-light endoscopy` 163 rows vs 5 `colonoscopy`); 4 are {{fundus, OCT}} true mixtures; 9 other true mixtures; 2 masked (`other`: no MMBU equivalent).
* **body part (222 candidate + 27 masked)**: (i) hierarchical lists inside one family (`eye, retina, optic disc`, `chest, lung`, `brain, head`); (ii) modifiers listed as regions (`blood vessels`, `arteries`) on top of a region; (iii) genuinely multi-region datasets (`abdomen, chest, lymph nodes`); (iv) masked: {', '.join(f'{v} x {k}' for k, v in bp_masked_reason.items())}. After the ontology, {bp_cand_after_fam[True]} candidate datasets are single-family and {bp_cand_after_fam[False]} span several families.
* **stain (16 candidate, 325 masked)**: 264 masked because the modality is not microscopy (correct: not asked), 61 masked in stained modalities, of which 13 are never-asked submodalities, 13 are cryo-ET/darkfield/synthetic (implied), and 35 have no registry stain at all; 16 candidates are multi-stain datasets.
* **specimen (47 candidate, 13 masked)**: 16+3 are {{cell line, human|mouse}} (cell line is not a species), 3 {{bacteria, e. coli}}, human + a second species/phantom, plus 7-species collections; masked ones are `mixed` (plankton, cryo-ET collections) or `unknown`.
* **domain (7 masked)**: registry domains outside MMBU's 23 (`microbiology`, `dentistry`, `vascular medicine`, `general medicine`, `neuroscience`). The other 386 are `multi`.
""")
w("### 4.3 Proposed rules and how many dataset-field cells each resolves\n")
w("`datasets` = number of selected datasets whose cell the rule changes; `images` = their planned images. 'Resolve' means non-hard -> hard (or family-hard).\n")
w(md(rules_df) + "\n")
w(f"""Totals of resolved cells (excluding the UNRESOLVED/WARN/confirm rows): modality {int(rules_df[rules_df.id.isin(['M1','M2'])].datasets.sum())}, submodality {int(rules_df[rules_df.id.isin(['S1','S2','S3'])].datasets.sum())}, body part {int(rules_df[rules_df.id.isin(['B1','B2','B3'])].datasets.sum())} hard + {int(rules_df[rules_df.id.isin(['B4','B7'])].datasets.sum())} family-hard, specimen {int(rules_df[rules_df.id.isin(['SP1','SP2','SP3'])].datasets.sum())}, stain {int(rules_df[rules_df.id.isin(['ST2a','ST2b'])].datasets.sum())}. Body-part hardness rises from 144 to {dict(mt_after[2][1])['hard']} datasets (37% -> {pct(dict(mt_after[2][1])['hard'] / len(S))}); with family-hard it is {pct((dict(mt_after[2][1])['hard'] + dict(mt_after[2][1])['family']) / len(S))}.

What still needs per-image labels (metalabel.py metadata, DICOM tags, or the autolabel model): the {dict(mt_after[2][1]).get('candidates', 0)} multi-family body-part datasets, the 16 multi-stain datasets, 35 stained-modality datasets with no stain information (registry text search finds a stain keyword in only 4), 5 two-modality datasets, 13 submodality mixtures, 21 multi-species datasets.
""")
w("### 4.4 Registry labels that look wrong or off-convention\n")
w(md(conv_df) + "\n")
w(f"""1. **Domain `radiology` is over-assigned**: {n_rad_dom} of {n_rad_ds} x-ray/CT/MRI/US datasets carry it ({sole_rad} as the only domain) vs {pct(rad_has_radiology)} of public rows. As a BCE positive it teaches "every scan is radiology", which would cost the specialty atom (the part the judge can match). Rule D1. Similarly `oncology` on 73% of CT and 53% of MRI datasets (public 10% / 15%), `neurology` 65% vs 18% for MRI, `biology` on 31% of light-microscopy datasets (public 7%), while `pathology` is under-assigned (17% vs 56%) and `surgery` on endoscopy 56% vs 77%. These are share-of-dataset vs share-of-row comparisons, so treat them as direction, not exact rates. Suggest a registry pass: remove `oncology`/`radiology` unless the paper is about tumors/radiology reads.
2. **Body part `cells` is hard for {n_cells_hard} datasets** (10% of the plan): in MMBU, cell-microscopy rows mostly have *no* body part (stain-only rows) and only 23 of 90 fluorescence body-part rows say `cells`. Use low weight, or mask `cells` when a stain is the asked field.
3. **`head` (29 candidate datasets), `optic disc`, `kidney`, `pancreas`, `spine` and about 30 other registry atoms are not MMBU body-part values** (36 of 80 candidate atoms are outside the public vocabulary); the public vocabulary has `brain`, `eye`, `retina`. Fold per B3/B1, and treat out-of-vocabulary atoms as training-only classes mapped to their family at decode time.
4. **Specimen `cell line` ({n_cell_line} datasets) and `phantom` ({n_phantom}) are not MMBU values** (MMBU uses `human`/`Human Cells`, `Kc167 cells`, `Vero cells`, `synthetic`). Map `cell line + species` -> species (SP1) and `phantom` -> `synthetic` only if you want the class.
5. **Modality `digital pathology` is hard for {n_dp_hard} datasets** though only 30/3147 public rows (1%) use it and the same slide images are `light microscopy` elsewhere. Rule M2 for ambiguous cases; consider merging for the modality head and keeping `digital pathology` as an accepted alternate.
6. `cystods` is hard `white-light endoscopy` but its registry also lists `other` (cystoscopy, 450 blue-light frames): small label noise, not worth a rule.
7. **Stain `synthetic` is dropped by `canon_stain`** (it filters any string containing "synthetic"), yet it is the reference value for 38/38 public synthetic-fluorescence rows. Keep it as a class. Likewise `uniform test slide`, `fluorescent beads` appear but are one-offs.
8. `select_datasets.py` masks body part when more than 4 candidates remain (11 datasets). Rule B7 recovers the single-family ones (`galar`, `rhode-island-vce`: all GI organs); whole-body CT sets (`totalsegmentator-v3`, `dap-atlas`, `magic-ct`) span several families and stay masked (or take `whole body` if you want a coarse label).
9. Domains outside MMBU (`dentistry`, `microbiology`, ...) mask 7 datasets; acceptable, but `microbiology` datasets could map to MMBU `biology`/`infectious disease` (3 datasets) if you want the cells.
""")

# ---------------- section 5
w("## 5. Final label schema per field\n")
w(f"""| field | class list source | prediction type | applicability / training mask | unknown / abstain |
|---|---|---|---|---|
| modality | `taxonomy.toml [modalities].core` (11); `digital pathology` kept but decoded only if p>0.5 | single, softmax | always asked; train on all images | no `unknown` class |
| submodality | public 35 + `fluoroscopy`, `gastroscopy` (37) from `V.SUBMODALITIES`; legal set from section 3.1 | single, softmax, logits masked to legal(modality) via joint (modality, submodality) scoring | always asked; mask the 2 'other' datasets | no `unknown` class |
| body part | atoms: public aliases-merged 66 + registry atoms with >=3 datasets; plus 14-way family head (`label_rules.json`) | multi-label atoms (BCE; partial-label `-log sum p(cand)` for candidates) + single-label family | asked unless modality is clinical photography or stain-only microscopy; mask registry `not applicable`/`multiple`; down-weight generic `cells`/`tissue` | none; decode general-then-specific, cap 3 |
| stain | public canonical stains (h&e, ihc, giemsa, wright-giemsa, papanicolaou, hoechst 33342, gfp, phalloidin, oil red o, uranyl acetate, unstained, ...) + `synthetic` | single, softmax over ~20; optional coarse family head (h&e / ihc / romanowsky / pap / fluorophore / EM heavy metal / unstained / synthetic) | applicable only when modality in {{LM, FM, EM, DP}}; mask all other images, never-asked submodalities, and microscopy datasets with no registry stain | no `none` class: absence = not requested |
| specimen | `human` + species/cell classes that have >=3 training datasets (mouse, rat, zebrafish, c. elegans, drosophila, bacteria, e. coli, yeast, plant, pig, dog, cat, non-human primate, vero cells, embryo, synthetic, phantom->synthetic) + `other animal` | single, softmax | always asked; mask non-answers (`x-ray images`, `unidentified`, `biological specimen`, `mixed`, `unknown`) | no `unknown` class; default `human` (89%) |
| domain | 23 public atoms (`V.DOMAINS`) | multi-label BCE, max 3; decode top-1 + extras if p>=tau | always possibly asked; mask 7 out-of-vocab datasets; apply D1 (D2 cap of 3 is then moot) | no `unknown` class |

Handling of MMBU non-answers: never include them as classes. In the eval set report scores *with and without* rows whose reference is a non-answer (specimen: 79 rows `x-ray images`/`unidentified`/`biological specimen`, 8 `mixed`; stain: 8 `uniform test slide`, 4 `n/a (not specified)` which `canon_stain` already maps to `unstained`). Predicting the true visual answer costs at most 2.5% of the specimen score and cannot be learned from pixels.

Abstention: references never read `unknown` for a requested field, so `unknown` can only lose points. Provide an off-by-default confidence gate (emit `unknown` only when max p < 0.15) and evaluate it on held-out validation datasets before enabling.

Applicability mask at training time (per image, per field): `mask_f = 0` if the dataset label for f is masked, or if f = stain and the modality is not microscopy or the submodality is never-asked. At inference, emit key f only if f is in the requested set R; all heads still run (cheap), and the stain head is simply not consulted for non-microscopy images.
""")
(HERE / "label_design.md").write_text("\n".join(L_))
print("wrote", HERE / "label_design.md", HERE / "label_rules.json")
