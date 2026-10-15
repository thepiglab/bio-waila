"""Pass-2 label space and normalizers shared by selection, preparation and eval.

The canonical vocabulary is MMBU's own (data_discovery/pass2_vocabulary.json, extracted from mmbu-public) plus the
submodalities that only appear in the paper's Fig. 3 (fluoroscopy, gastroscopy). Registry values that MMBU never uses
are either mapped onto an MMBU value, kept as an extra class (body part, stain, specimen) or dropped (domain).
"""
import json
import re
import tomllib
from pathlib import Path

from bio_waila.paths import PUBLIC, RESOURCES

PUBLIC_VOCAB = json.loads((RESOURCES / "pass2_vocabulary.json").read_text())
FIG3 = json.loads((RESOURCES / "mmbu_fig3_digitized.json").read_text())
TAXONOMY = tomllib.loads((RESOURCES / "taxonomy.toml").read_text())
# alias maps used to normalise the public split (copied from data_discovery/scripts/extract_pass2_vocab.py), so train
# and eval labels normalise identically
PUBLIC_ALIASES = json.loads((RESOURCES / "public_aliases.json").read_text())
MMBU = PUBLIC.parent  # kept for callers that resolve mmbu-public relative paths

FIELDS = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]
MODALITIES = list(TAXONOMY["modalities"]["core"])
SUBMODALITIES = sorted(set(PUBLIC_VOCAB["vocab"]["submodality"]) | set(FIG3["submodality_images"]))
DOMAINS = sorted(set(PUBLIC_VOCAB["vocab"]["domain"]))
# stains only make sense for microscopy-type images; Pass 2 does not ask for them elsewhere
STAINED_MODALITIES = {"light microscopy", "fluorescence microscopy", "electron microscopy", "digital pathology"}

# values that carry no label information
NULL_VALUES = {"", "multiple", "not applicable", "unknown", "mixed", "various", "other", "n/a", "none"}


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def canon_submodality(s):
    s = norm(s)
    s = PUBLIC_ALIASES.get("submodality", {}).get(s, s)
    return s if s in SUBMODALITIES else None


def canon_modality(s):
    s = norm(s)
    return s if s in MODALITIES else None


_DOMAIN_MAP = {"otorhinolaryngology": "otolaryngology", "infectious diseases": "infectious disease",
               "gynecology": "obstetrics and gynecology", "parasitology": "infectious disease"}


def canon_domain(s):
    s = norm(s)
    s = PUBLIC_ALIASES["domain"].get(s, _DOMAIN_MAP.get(s, s))
    return s if s in DOMAINS else None  # registry-only domains (neuroscience, dentistry, …) are dropped


_BODY_MAP = {"lungs": "lung", "urinary bladder": "bladder", "intestine": "gastrointestinal tract",
             "lumbar spine": "spine", "limbs": None, "back": None, "root": None, "leaf epidermis": None, "urine": None}


def canon_body_part(s):
    s = norm(s)
    s = PUBLIC_ALIASES["body_part"].get(s, s)
    s = _BODY_MAP.get(s, s)
    return None if s is None or s in NULL_VALUES else s


_STAIN_RULES = [  # (regex, canonical); first match wins. Order matters (wright before giemsa).
    (r"^(h&e|he|hes|hematoxylin and eosin.*|haematoxylin and eosin.*)$", "h&e"),
    (r"(ihc|immunohisto|^dab$|^her2$|^ki-?67$|^p53$|^cd\d+|^pr$|^er$)", "ihc"),
    (r"wright", "wright-giemsa"),
    (r"(may-gr[uü]nwald|jenner|leishman|pappenheim|field stain)", "may-grünwald-giemsa"),
    (r"giemsa", "giemsa"),
    (r"papanicolaou|^pap$|shorr", "papanicolaou"),
    (r"^(unstained|no stain|label-free|none)$", "unstained"),
    (r"hoechst", "hoechst 33342"),
    (r"phalloidin|^phal$|actin", "phalloidin"),
    (r"^(e?gfp|.*-gfp.*|gfp .*|soluble gfp)$", "gfp"),
    (r"masson", "masson's trichrome"),
    (r"ziehl|kinyoun|acid-fast", "ziehl-neelsen"),
    (r"^gram", "gram"),
    (r"immunofluorescence|codex|multiplex", "immunofluorescence"),
    (r"cell painting", "cell painting dyes"),
    (r"^(fish|smfish|ish|dual-ish|in situ hybridization).*", "in situ hybridization"),
]


def canon_stain(s):
    s = norm(s)
    if s == "synthetic":  # MMBU's own reference stain for every synthetic-fluorescence row
        return "synthetic"
    if s in NULL_VALUES or "not specified" in s or "simulated" in s or "synthetic" in s:
        return None
    s = PUBLIC_ALIASES["stain"].get(s, s)
    for rx, c in _STAIN_RULES:
        if re.search(rx, s):
            return c
    return s


_SPECIMEN_MAP = {"homo sapiens": "human", "human cells": "human", "danio rerio": "zebrafish", "mus musculus": "mouse",
                 "escherichia coli": "e. coli", "bacterial specimens": "bacteria", "chlorocebus aethiops": "vero cells",
                 "vero cells (chlorocebus aethiops)": "vero cells", "cavia porcellus": "guinea pig", "x-ray images": None,
                 "unidentified": None, "biological specimen": None, "animal": None}


def canon_specimen(s):
    s = norm(s)
    s = _SPECIMEN_MAP.get(s, s)
    return None if s is None or s in NULL_VALUES else s


CANON = {"modality": canon_modality, "submodality": canon_submodality, "body_part": canon_body_part,
         "stain": canon_stain, "specimen": canon_specimen, "domain": canon_domain}


def target(values, field, multilabel=False):
    """Dataset-level label for one field.

    mode = "hard"       one value: every image gets it
           "multi"      every value applies to every image (domains)
           "candidates" one (or more) of these applies per image, unknown which: train with a partial-label loss
           "masked"     no usable label: exclude from that head's loss
    """
    vals = sorted({c for c in (CANON[field](v) for v in values) if c})
    if not vals:
        return {"mode": "masked", "values": []}
    if multilabel:
        return {"mode": "multi", "values": vals}
    return {"mode": "hard" if len(vals) == 1 else "candidates", "values": vals}
