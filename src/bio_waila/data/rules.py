"""Auto-labelling layer 2: free per-image evidence → closed-vocabulary labels.

Evidence sources, most to least trustworthy:
  dicom   DICOM header (Modality, BodyPartExamined, StudyDescription)              confidence 0.95
  meta    a metadata row (csv/json/HF columns) matched to the image by file name   confidence 0.90
  path    folder and file names, BIDS suffixes (_T1w, _FLAIR …)                     confidence 0.85

Every keyword is *strong* (unambiguous on its own, e.g. "dermoscopy", "flair") or *weak* (short or overloaded
tokens like "ct", "us", "he", "pas"). Weak hits may only pick between values the registry already lists for the
dataset; strong hits may also fill a field the registry left empty. Registry hard labels are never overridden;
a disagreeing strong hit is recorded as a conflict instead, so bad registry entries can be found and fixed.
"""
import re

from bio_waila.data.vocab import canon_body_part, canon_stain

# field → [(regex on normalized text, value, strong?)]
_R = {
    "modality": [
        (r"\b(ct|cect|ncct|cta)\b", "computed tomography", False),
        (r"\b(computed tomography|ct scan|ct_scan|cbct)\b", "computed tomography", True),
        (r"\b(mri|mr|mra|t1|t2|pdw?|dwi|adc|swi)\b", "magnetic resonance imaging", False),
        (r"\b(magnetic resonance|t1 weighted|t2 weighted|t1w|t2w|t1ce|t1c|t1gd|t1n|t2f|t1wce|flair)\b", "magnetic resonance imaging", True),
        (r"\b(x ?ray|xray|cxr|radiograph\w*|mammogra\w*|mammo|fluoroscop\w*|angiogra\w*)\b", "x-ray imaging", True),
        (r"\b(dx|cr|xr)\b", "x-ray imaging", False),
        (r"\b(ultrasound|sonograph\w*|echocardiogra\w*|b ?mode|doppler)\b", "ultrasound imaging", True),
        (r"\b(us|echo)\b", "ultrasound imaging", False),
        (r"\b(endoscop\w*|colonoscop\w*|gastroscop\w*|laparoscop\w*|cystoscop\w*)\b", "endoscopy", True),
        (r"\b(fundus|dermoscop\w*|dermatoscop\w*|oct|octa|retinal photo\w*|colposcop(y|ic|e)|enface|en face)\b", "optical imaging", True),
        (r"\b(whole slide|wsi|svs|ndpi|histopatholog\w*)\b", "digital pathology", False),
        (r"\b(brightfield|bright field|phase contrast|darkfield|dark field|light microscop\w*)\b", "light microscopy", True),
        (r"\b(fluorescen\w*|epifluorescen\w*|confocal|tirf|multiphoton|two photon|dapi|hoechst|gfp)\b", "fluorescence microscopy", False),
        (r"\b(electron microscop\w*|fib ?sem|sbf ?sem|sssem|sstem|serial section em|cryo ?em)\b", "electron microscopy", True),
        (r"\b(sem|tem|cryo ?et|tomogram\w*)\b", "electron microscopy", False),
        (r"\b(gi ?sim|3d ?sim|nl ?sim|tirf ?sim|structured illumination)\b", "fluorescence microscopy", True),
    ],
    "submodality": [
        (r"\b(chest x ?ray|cxr|chest radiograph\w*)\b", "chest x-ray", True),
        (r"\b(mammogra\w*|mammo|cbis|inbreast)\b", "mammography", True),
        (r"\b(fluoroscop\w*|angiogra\w*|ercp)\b", "fluoroscopy", True),
        (r"\b(radiograph\w*|x ?ray|xray)\b", "digital radiography", False),
        (r"\b(spiral|helical)\b", "spiral ct", False),
        (r"\b(ct|cect|ncct|computed tomography)\b", "conventional ct", False),
        (r"\b(t1w|t2w|t1ce|t1c|t1gd|t1n|t2f|t1wce|flair|mprage)\b", "structural mri", True),
        (r"\b(t1|t2|pdw?|structural|anat)\b", "structural mri", False),
        (r"\b(color doppler|colour doppler|doppler)\b", "color doppler", True),
        (r"\b(3d ultrasound|3d us|volumetric ultrasound)\b", "3d ultrasound", True),
        (r"\b(b ?mode|ultrasound|sonograph\w*)\b", "b-mode ultrasound", False),
        (r"\b(fundus|cfp|retinal photo\w*)\b", "fundus photography", True),
        (r"\b(dic|hmc|hoffman)\b", "contrast microscopy", True),
        (r"\b(phase contrast|phc|zernike)\b", "phase contrast microscopy", True),
        (r"\b(qpi|quantitative phase)\b", "phase contrast microscopy", False),
        (r"\b(octa|oct angiography)\b", "optical coherence tomography angiography", True),
        (r"\b(oct|as ?oct|optical coherence tomography|enface|en face)\b", "optical coherence tomography", True),
        (r"\b(dermoscop\w*|dermatoscop\w*|isic)\b", "dermoscopy", True),
        (r"\b(near infrared|nir|nirf)\b", "near-infrared imaging", False),
        (r"\b(fluorescein angiogra\w*|ffa)\b", "fluorescein angiography", True),
        (r"\b(clinical photo\w*|macroscopic|smartphone)\b", "clinical photography", False),
        (r"\b(laparoscop\w*|cholec\w*)\b", "laparoscopy", True),
        (r"\b(colonoscop\w*)\b", "colonoscopy", True),
        (r"\b(gastroscop\w*|egd|esophagogastroduodenoscop\w*)\b", "gastroscopy", True),
        (r"\b(colposcop(y|ic|e))\b", "colposcopy", True),
        (r"\b(endomicroscop\w*|cle)\b", "confocal laser endomicroscopy", True),
        (r"\b(white light|wli|wlc)\b", "white-light endoscopy", True),
        (r"\b(whole slide|wsi|svs|ndpi|mrxs)\b", "whole slide imaging", True),
        (r"\b(brightfield|bright field|bf)\b", "brightfield microscopy", False),
        (r"\b(phase contrast|phc)\b", "phase contrast microscopy", True),
        (r"\b(darkfield|dark field)\b", "darkfield microscopy", True),
        (r"\b(dic|differential interference)\b", "contrast microscopy", False),
        (r"\b(confocal)\b", "confocal microscopy", True),
        (r"\b(tirf)\b", "total internal reflection fluorescence microscopy (tirf)", True),
        (r"\b(multiphoton|two photon|2 ?photon)\b", "multiphoton microscopy", True),
        (r"\b(widefield|epifluorescen\w*)\b", "epifluorescence microscopy", True),
        (r"\b(specular)\b", "specular microscopy", True),
        (r"\b(scanning electron|fib ?sem|sbf ?sem|sssem)\b", "scanning electron microscopy", True),
        (r"\b(sem)\b", "scanning electron microscopy", False),
        (r"\b(cryo ?et|cryo electron tomograph\w*|tomogram\w*)\b", "cryo-electron tomography", True),
    ],
    "stain": [
        (r"\b(h ?& ?e|hne|h e|hematoxylin eosin|haematoxylin eosin)\b", "h&e", True),
        (r"\b(he)\b", "h&e", False),
        (r"\b(ihc|immunohisto\w*|dab|her2|ki ?67|pd ?l1|er|pr)\b", "ihc", False),
        (r"\b(giemsa|mgg)\b", "giemsa", True),
        (r"\b(wright)\b", "wright-giemsa", True),
        (r"\b(papanicolaou|pap smear)\b", "papanicolaou", True),
        (r"\b(pap)\b", "papanicolaou", False),
        (r"\b(masson|trichrome)\b", "masson's trichrome", True),
        (r"\b(ziehl|zn|kinyoun|acid fast)\b", "ziehl-neelsen", True),
        (r"\b(gram)\b", "gram", False),
        (r"\b(dapi)\b", "dapi", True),
        (r"\b(hoechst)\b", "hoechst 33342", True),
        (r"\b(phalloidin)\b", "phalloidin", True),
        (r"\b(e?gfp)\b", "gfp", True),
        (r"\b(unstained|label free)\b", "unstained", True),
    ],
    "specimen": [
        (r"\b(mouse|mice|murine|mus musculus|c57bl\w*|balb ?c|jrc mus)\b", "mouse", True),
        (r"\b(mus)\b", "mouse", False),
        (r"\b(sprague dawley|wistar|rattus)\b", "rat", True),
        (r"\b(rat|rats)\b", "rat", False),
        (r"\b(zebrafish|danio)\b", "zebrafish", True),
        (r"\b(zf)\b", "zebrafish", False),
        (r"\b(jurkat|hela|cos ?7|u2os|l929|mcf ?7|sum ?159|hek ?293\w*|cell line)\b", "cell line", True),
        (r"\b(phantom)\b", "phantom", True),
        (r"\b(synthetic|simulated|simcep)\b", "synthetic", True),
        (r"\b(ex ?vivo (porcine|pig)|porcine|swine)\b", "pig", True),
        (r"\b(u ?2 ?os|a549|mda ?mb ?231|bt ?20|ht ?1080|rpe ?1|hacat|huvec)\b", "cell line", True),
        (r"\b(cos ?7|vero)\b", "non-human primate", True),
        (r"\b(eph4|nih ?3t3)\b", "mouse", True),
        (r"\b(c ?elegans|caenorhabditis)\b", "c. elegans", True),
        (r"\b(drosophila|fruit fly)\b", "drosophila", True),
        (r"\b(fly)\b", "drosophila", False),
        (r"\b(yeast|saccharomyces)\b", "yeast", True),
        (r"\b(e ?coli|escherichia)\b", "e. coli", True),
        (r"\b(ecol)\b", "e. coli", False),
        (r"\b(canine|dogs?)\b", "dog", False),
        (r"\b(pig)\b", "pig", False),
        (r"\b(human|patient|homo sapiens)\b", "human", False),
    ],
}
_RC = {f: [(re.compile(rx), v, s) for rx, v, s in rules] for f, rules in _R.items()}

# body parts: every class name is its own keyword; plus synonyms. Matched on whole words.
_BODY_SYN = {"lca": "coronary arteries", "rca": "coronary arteries", "lad": "coronary arteries", "cecum": "colon", "caecum": "colon", "sigmoid": "colon", "hepatic flexure": "colon", "splenic flexure": "colon",
             "anal canal": "rectum", "ileum": "small intestine", "terminal ileum": "small intestine", "small bowel": "small intestine",
             "duodenum": "small intestine", "hearts": "heart", "hands": "hand", "arms": "upper limbs", "legs": "lower limbs",
             "upper extremity": "upper limbs", "lower extremity": "lower limbs", "scalp": "head", "torso": "thorax", "trunk": "thorax",
             "mouth": "oral cavity", "peritoneum": "abdomen", "lymphadenectomy": "lymph nodes", "lymph node": "lymph nodes",
             "abd": "abdomen", "abdominal": "abdomen", "hepatic": "liver", "renal": "kidney", "kidneys": "kidney",
             "cardiac": "heart", "pulmonary": "lung", "lungs": "lung", "thoracic": "chest", "cerebral": "brain",
             "cranial": "head", "colorectal": "colon", "gastric": "stomach", "oesophagus": "esophagus",
             "esophageal": "esophagus", "retinal": "retina", "ocular": "eye", "breasts": "breast", "mammary": "breast",
             "dental": "teeth", "tooth": "teeth", "knees": "knee", "hips": "hip", "lumbar": "spine", "spinal": "spine",
             "vertebra": "spine", "vertebrae": "spine", "prostatic": "prostate", "pancreatic": "pancreas",
             "thyroidal": "thyroid", "dermal": "skin", "cutaneous": "skin", "skin": "skin", "wrist": "wrist",
             "hand": "hand", "foot": "foot", "feet": "foot", "pelvic": "pelvis", "lymph node": "lymph nodes"}
_BODY_WEAK = {"head", "hand", "back", "cell", "cells", "tissue", "bone", "blood", "nerve", "ear", "face", "foot", "nail", "hair", "root"}

# DICOM Modality tag → (modality, submodality or None)
DICOM_MODALITY = {"CT": ("computed tomography", None),  # "CT" says nothing about conventional vs spiral vs CTA "MR": ("magnetic resonance imaging", None),
                  "CR": ("x-ray imaging", None), "DX": ("x-ray imaging", None), "MG": ("x-ray imaging", "mammography"),
                  "XA": ("x-ray imaging", "fluoroscopy"), "RF": ("x-ray imaging", "fluoroscopy"),
                  "US": ("ultrasound imaging", "b-mode ultrasound"), "OP": ("optical imaging", "fundus photography"),
                  "OPT": ("optical imaging", "optical coherence tomography"), "ES": ("endoscopy", None),
                  "SM": ("digital pathology", "whole slide imaging"), "XC": ("optical imaging", "clinical photography"),
                  "GM": ("light microscopy", None)}
# BIDS suffixes in OpenNeuro-style names
BIDS = {s_: "structural mri" for s_ in ("t1w", "t2w", "flair", "pdw", "t2starw", "inplanet1", "inplanet2", "t1", "t1c",
                                        "t1ce", "t1gd", "t1n", "t2", "t2f", "t1wce", "mprage")}
BIDS_MRI_ONLY = {"mra", "dwi", "adc", "trace", "swi", "cbf", "cbv", "ttp", "tmax", "asl", "bold", "perf", "dsc"}  # MRI, not structural


def normalize_text(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))  # LymphNode -> Lymph Node, OlfactoryBulb -> Olfactory Bulb
    s = s.lower()
    return " " + re.sub(r"[^a-z0-9&]+", " ", s).strip() + " "


_STOP = re.compile(r" (asc us|asc h|lsil|hsil) ")  # Bethesda cytology classes, not ultrasound
_CTMR = re.compile(r"\b(ct|cta|mr|mra|mri|computed tomography|magnetic resonance)\b")
_XRAY_WORD = re.compile(r"\b(x ?ray|xray|radiograph\w*|fluoroscop\w*|mammogra\w*|dsa)\b")
# diagnosis words: a value naming a disease mentions an organ incidentally ("retinal hemorrhage")
_DIAG = re.compile(r"(disorder|hemorrhage|haemorrhage|occlusion|retinopathy|oedema|edema|carcinoma|tumou?r|lesion|"
                   r"disease|syndrome|infection|cancer|neoplasm|adenoma|glioma|nodule|fracture|pneumonia|itis\b|opacity|stone|failure|"
                   r"insufficien\w*|infarct\w*|stenosis|aneurysm|cyst\b|lesion)")


_NEG = re.compile(r"\b(not|non|no|without) (a |an |the )?([a-z]+)")
_GASTRIC_FUNDUS = re.compile(r"\b(fundus (cardia|gastric|stomach)|(cardia|gastric|stomach|corpus|antrum) fundus|fundic gland)\b")
_NBI = re.compile(r"\b(nbi|narrow band|narrowband|blc|blue light)\b")


def text_hits(text, body_classes=()):
    """{field: {value: strong?}} from free text (paths, metadata values, descriptions)."""
    t = _STOP.sub(" ", normalize_text(text))
    t = _NEG.sub(" ", t)  # "Not A Brain" names a negative class, not the brain
    gastric = bool(_GASTRIC_FUNDUS.search(t))
    out = {}
    for f, rules in _RC.items():
        for rx, v, strong in rules:
            if rx.search(t):
                out.setdefault(f, {})
                out[f][v] = out[f].get(v, False) or strong
    words = set(t.split())
    bigrams = {" ".join(p) for p in zip(t.split(), t.split()[1:])}
    if "hepatic flexure" in bigrams or "splenic flexure" in bigrams:
        words -= {"hepatic", "splenic"}
    if "renal pelvis" in bigrams:
        words -= {"pelvis"}
    for w in words | bigrams:
        v = _BODY_SYN.get(w) or (w if w in body_classes else None)
        v = canon_body_part(v) if v else None
        if v:
            out.setdefault("body_part", {})
            out["body_part"][v] = out["body_part"].get(v, False) or (w not in _BODY_WEAK)
    if "stain" in out:  # canonical spelling via the shared normalizer
        out["stain"] = {canon_stain(k) or k: s for k, s in out["stain"].items()}
    if _CTMR.search(t) and not _XRAY_WORD.search(t):  # "CT angiography" is CT, not catheter angiography
        out.get("modality", {}).pop("x-ray imaging", None)
        out.get("submodality", {}).pop("fluoroscopy", None)
        out = {f: v for f, v in out.items() if v}
    if gastric:  # stomach fundus, not an eye photograph
        for f in ("modality", "submodality"):
            out.get(f, {}).pop("fundus photography", None)
            out.get(f, {}).pop("optical imaging", None)
        out.setdefault("body_part", {})["stomach"] = True
        out = {f: v for f, v in out.items() if v}
    if _NBI.search(t):  # narrow-band / blue-light endoscopy has no MMBU class: never call it white-light
        out.get("submodality", {}).pop("white-light endoscopy", None)
        out = {f: v for f, v in out.items() if v}
    if "body_part" in out and _DIAG.search(t):
        out["body_part"] = {k: False for k in out["body_part"]}  # demote: only usable inside a body-part column
    return out


def dicom_hits(tags):
    """tags: dict with Modality, BodyPartExamined, StudyDescription, SeriesDescription (strings)."""
    out = {}
    m = DICOM_MODALITY.get(str(tags.get("Modality", "")).upper())
    if m:
        out["modality"] = {m[0]: True}
        if m[1]:
            out["submodality"] = {m[1]: True}
    bp = canon_body_part(str(tags.get("BodyPartExamined", "")).lower().replace("_", " "))
    if bp:
        out.setdefault("body_part", {})[bp] = True
        if m and m[0] == "x-ray imaging" and bp == "chest" and not m[1]:
            out["submodality"] = {"chest x-ray": True}
    desc = " ".join(str(tags.get(k, "")) for k in ("StudyDescription", "SeriesDescription"))
    for f, vals in text_hits(desc).items():
        for v, s in vals.items():
            out.setdefault(f, {}).setdefault(v, s)
    return out


def bids_hits(path):
    m = re.search(r"_([A-Za-z0-9]+)\.nii(\.gz)?$", str(path))
    suf = m.group(1).lower() if m else ""
    if suf in BIDS:
        return {"modality": {"magnetic resonance imaging": True}, "submodality": {BIDS[suf]: True}}
    if suf in BIDS_MRI_ONLY:
        return {"modality": {"magnetic resonance imaging": True}}
    return {}


SOURCE_CONF = {"dicom": 0.95, "meta": 0.9, "path": 0.85}


def resolve(dataset_label, evidence, field):
    """Combine the registry's dataset-level label with per-image evidence for one field.

    evidence: list of (source, {value: strong?}). Returns
      {"mode", "values", "source", "confidence", "conflict"?}
    """
    mode, vals = dataset_label["mode"], dataset_label["values"]
    out = {"mode": mode, "values": vals, "source": "registry", "confidence": 1.0 if mode in ("hard", "multi") else 0.0}
    if mode == "multi":
        return out
    ranked = sorted(evidence, key=lambda e: -SOURCE_CONF[e[0]])
    if mode == "hard":
        for src, hits in ranked:
            strong_other = [v for v, s in hits.items() if s and v not in vals]
            if strong_other and not set(hits) & set(vals):
                out["conflict"] = f"{src}:{strong_other[0]}"
                break
        return out
    for src, hits in ranked:
        if mode == "candidates":
            inter = [v for v in hits if v in vals]
            if len(inter) == 1:
                return {"mode": "hard", "values": inter, "source": src, "confidence": SOURCE_CONF[src]}
            if 1 < len(inter) < len(vals):
                out.update(values=sorted(inter), source=src)  # narrowed, still ambiguous
        elif mode == "masked":
            strong = [v for v, s in hits.items() if s]
            if len(strong) == 1:
                return {"mode": "hard", "values": strong, "source": src, "confidence": SOURCE_CONF[src] - 0.15}
    return out


# ── metadata-value matcher (used by metalabel.py) ───────────────────────────────────────────────────────────────
# A column whose *name* says which Pass-2 field it holds lets short/ambiguous values count ("MR", "US", "HE").
FIELD_COLUMNS = {
    "modality": r"modalit|imaging|image.?type|technique",
    "body_part": r"(?<![a-z])(organ|body|anatom\w*|region|site|location|part|structure|tissue|bodypart)(?![a-z])",
    "stain": r"stain|dye|marker|antibod|chromogen|label.?type",
    "specimen": r"specie|organism|specimen|animal|host|strain|model.?organism",
    "submodality": r"sequence|protocol|series.?desc|contrast|weighting|view|acqui|exam|submodal|scan.?type",
}  # order matters: first match wins ("BodyPartExamined" is body_part, not an exam type)
_FIELD_COL_RX = {f: re.compile(rx, re.I) for f, rx in FIELD_COLUMNS.items()}
# exact short codes that are only trusted inside a column of the right field (or a DICOM tag)
CODES = {
    "modality": {"ct": "computed tomography", "mr": "magnetic resonance imaging", "mri": "magnetic resonance imaging",
                 "us": "ultrasound imaging", "cr": "x-ray imaging", "dx": "x-ray imaging", "xr": "x-ray imaging",
                 "mg": "x-ray imaging", "xa": "x-ray imaging", "rf": "x-ray imaging", "op": "optical imaging",
                 "opt": "optical imaging", "es": "endoscopy", "sm": "digital pathology", "xc": "optical imaging",
                 "em": "electron microscopy", "lm": "light microscopy", "fm": "fluorescence microscopy"},
    "submodality": {"wlc": "white-light endoscopy", "sem": "scanning electron microscopy", "t1n": "structural mri",
                    "t2f": "structural mri", "t1wce": "structural mri", "t1": "structural mri", "t2": "structural mri", "t1w": "structural mri", "t2w": "structural mri",
                    "flair": "structural mri", "pd": "structural mri", "pdw": "structural mri", "t1ce": "structural mri",
                    "t1c": "structural mri", "t1gd": "structural mri", "mg": "mammography", "mlo": "mammography",
                    "cc": "mammography", "xa": "fluoroscopy", "rf": "fluoroscopy",
                    "bf": "brightfield microscopy", "phc": "phase contrast microscopy", "dic": "contrast microscopy",
                    "oct": "optical coherence tomography", "octa": "optical coherence tomography angiography",
                    "cfp": "fundus photography", "wli": "white-light endoscopy"},
    "stain": {"her2neu": "ihc", "er": "ihc", "pr": "ihc", "ki67": "ihc", "he": "h&e", "h&e": "h&e", "hne": "h&e", "ihc": "ihc", "dab": "ihc", "pap": "papanicolaou",
              "mgg": "may-grünwald-giemsa", "zn": "ziehl-neelsen", "pas": "pas", "gfp": "gfp", "dapi": "dapi"},
    "specimen": {"hs": "human", "mm": "mouse", "homo sapiens": "human", "mus musculus": "mouse", "rattus norvegicus": "rat"},
}


# folder names that equal one of these codes are unambiguous on their own ("CT/", "MRI/", "IHC/" …)
PATH_WORDS = {"fluorescence": ("modality", "fluorescence microscopy"), "fluorescent": ("modality", "fluorescence microscopy"),
              "brightfield": ("submodality", "brightfield microscopy"), "bright field": ("submodality", "brightfield microscopy"),
              "fa": ("submodality", "fluorescein angiography"), "ffa": ("submodality", "fluorescein angiography"),
              "phase contrast": ("submodality", "phase contrast microscopy")}
PATH_CODES = {k: (f, CODES[f][k]) for f, k in [("modality", "mr"), ("modality", "mri"),
              ("modality", "us"), ("modality", "cr"), ("modality", "dx"), ("modality", "xr"),
              ("submodality", "oct"), ("submodality", "octa"), ("submodality", "cfp"), ("submodality", "t1"),
              ("submodality", "t2"), ("submodality", "t1w"), ("submodality", "t2w"), ("submodality", "flair"),
              ("submodality", "t1ce"), ("stain", "ihc"), ("stain", "h&e"), ("stain", "dapi"), ("stain", "gfp")]}


def field_of_column(col):
    """Pass-2 field a column name points at, or None (e.g. 'BodyPartExamined' → body_part)."""
    name = re.sub(r"^(csv|json|hf|dicom|bids|path):([^:]*:)?", "", str(col))  # csv:<file>:<col> or hf:<col>
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)  # BodyPartExamined -> Body Part Examined (whole-word matching)
    for f, rx in _FIELD_COL_RX.items():
        if rx.search(name):
            return f
    return None


# columns that never carry Pass-2 metadata, however their values read (dates, ids, diagnoses, free text)
SKIP_COLUMN = re.compile(r"(date|time|year|_id$|^id$|uuid|hash|path|file|url|note|comment|caption|diagnos|disease|"
                         r"etiolog|lineage|histolog|grade|stage|finding|report|question|answer|prompt|common.?name|"
                         r"scientific|taxon|genus|family|order|origin|primary|metast|habit|smok|tobacco|device|model|manufacturer|vendor|dx.?type|"
                         r"folder|\bdir\b|directory)", re.I)
_DATE = re.compile(r"\b\d{1,4}[-/ .](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2})[a-z]*[-/ .]\d{2,4}\b", re.I)
_IDLIKE = re.compile(r"^[a-z]{0,8}[_-]?\d{3,}([_-]\w+)?$", re.I)


def match_value(value, column, classes):
    """{field: (class, strong?)} for one metadata value. `classes` = label_space classes per field."""
    v = re.sub(r"\s+", " ", str(value).strip().lower())
    if not v or v in ("nan", "none", "null", "unknown", "n/a", "na", "-", "?") or len(v) > 200:
        return {}
    col_name = re.sub(r"^(csv|json|hf|dicom|bids|path):([^:]*:)?", "", str(column))
    if SKIP_COLUMN.search(col_name) and not str(column).startswith("dicom:") or _DATE.search(v) or _IDLIKE.match(v):
        return {}
    ctx = field_of_column(column)
    free_text = len(v.split()) > 5
    out = {}
    # 1. exact class name (any field), e.g. a column of organ names
    for f, cls in classes.items():
        if v in cls and (f != "specimen" or ctx == "specimen"):  # "Tobacco" in a habit column is not a specimen
            out[f] = (v, True)
    # 2. short codes, trusted only in a column of that field — or as a whole folder name if unambiguous
    if str(column).startswith("bids:") and (v in BIDS or v in BIDS_MRI_ONLY):
        out["modality"] = ("magnetic resonance imaging", True)
        if v in BIDS:
            out["submodality"] = (BIDS[v], True)
    if str(column).startswith("path:") and v in PATH_CODES:
        f, c = PATH_CODES[v]
        out[f] = (c, True)
    if str(column).startswith("path:") and re.sub(r"[_\-]+", " ", v) in PATH_WORDS:
        f, c = PATH_WORDS[re.sub(r"[_\-]+", " ", v)]
        out[f] = (c, True)
    if ctx and v in CODES.get(ctx, {}):
        out[ctx] = (CODES[ctx][v], True)
        if ctx == "modality" and v == "mg":
            out["submodality"] = ("mammography", True)
    # 3. keyword rules over the value text; weak hits are promoted inside a matching column
    for f, hits in text_hits(str(value), classes.get("body_part", ())).items():  # original case: camelCase splitting
        if f in out:
            continue
        in_ctx = f == ctx or (ctx in ("modality", "submodality") and f in ("modality", "submodality"))
        strong = [k for k, s in hits.items() if s or in_ctx]
        if len(strong) == 1:
            out[f] = (strong[0], True)
        elif len(hits) == 1 and ctx is None:
            out[f] = (next(iter(hits)), False)
    if free_text:  # long phrases mention organs/cells incidentally ("basal cell carcinoma"): only trust in-context
        out = {f: x for f, x in out.items() if f in ("modality", "submodality", "stain") or f == ctx}
    # canonical spellings
    if "stain" in out:
        out["stain"] = (canon_stain(out["stain"][0]) or out["stain"][0], out["stain"][1])
    if "body_part" in out:
        bp = canon_body_part(out["body_part"][0])
        out = {**out, "body_part": (bp, out["body_part"][1])} if bp else {k: x for k, x in out.items() if k != "body_part"}
    return out
