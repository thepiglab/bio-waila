"""Model zoo: score off-the-shelf models zero-shot on mmbu-public Pass 2 (eval only), exactly like the trained teachers.

    waila-zoo list [--porcelain]                      the registry (configs/zoo.yaml) with enabled flags
    waila-zoo prefetch [--models a,b] [--strict]      CPU: download weights into HF_HOME, build plan/eval_public.parquet
    waila-zoo run --model SLUG [--limit N] [--batch B] [--base-url URL] [--enable]    GPU: score one model → runs/zoo/<slug>/
    waila-zoo leaderboard                             aggregate runs/zoo/*/public_metrics.json → leaderboard.{csv,md}

Two scoring modes (docs/experiments/zoo.md):
  generative  image + the official Pass-2 prompt (only the fields annotated for that image), greedy decode, parse the JSON,
              map each string onto the label space (canonical normalizer → alias table → fuzzy match; else unmapped = wrong)
  decision    per field, the class names as lettered options and the next-token letter probabilities (Jev style); more than
              k_max options use a two-round tournament; Liquid d1 uses its own `choice` API; `backend: systemone` entries
              (respanai/span-02) call a System One HTTP server, one request per image with every field as a choice question
Predictions become logits (±8 one-hot for generative, log-probs for decision) so `public.score_public` computes C and the
field metrics exactly as for trained models.
"""
import argparse
import difflib
import json
import os
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from bio_waila.data.vocab import CANON, FIELDS, PUBLIC_ALIASES
from bio_waila.paths import REPO, RUNS

REGISTRY_PATH = REPO / "configs" / "zoo.yaml"
MODES = ("generative", "decision")
READOUTS = ("letters", "d1")
BACKENDS = ("", "systemone")  # "" = the in-process transformers backends; systemone = HTTP client (models/zoo_models.SystemOneScorer)
DTYPES = ("bfloat16", "float16", "float32")
# policy: no Chinese-origin / Qwen-based checkpoints in the zoo
BANNED = re.compile(r"qwen|deepseek|internvl|chatglm|glm-|cogvlm|minicpm|baichuan|yi-vl|hunyuan|ernie|kimi|moonshot", re.I)


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Entry:
    slug: str
    hf_id: str
    mode: str
    readout: str = "letters"
    base: str | None = None
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    gpus: int = 1
    mem_gb: int = 64
    max_new_tokens: int = 384
    k_max: int | None = None
    enabled: bool = True
    reason: str = ""
    notes: str = ""
    backend: str = ""
    options: dict | None = None  # backend settings (systemone: base_url, endpoint, api_key_env, reasoning, concurrency, ...)

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.slug or ""):
            raise ValueError(f"bad slug {self.slug!r} (letters, digits, _ . - only)")
        for k, allowed in (("mode", MODES), ("readout", READOUTS), ("dtype", DTYPES), ("backend", BACKENDS)):
            if getattr(self, k) not in allowed:
                raise ValueError(f"{self.slug}: {k}={getattr(self, k)!r} not in {allowed}")
        if not self.hf_id or "/" not in self.hf_id:
            raise ValueError(f"{self.slug}: hf_id must look like org/name")
        if self.mode == "generative" and (self.base or self.readout != "letters"):
            raise ValueError(f"{self.slug}: base/readout only apply to decision entries")
        if self.readout == "d1" and self.base:
            raise ValueError(f"{self.slug}: a d1 model has no base")
        if self.backend == "systemone" and (self.mode != "decision" or self.base or self.k_max):
            raise ValueError(f"{self.slug}: a systemone entry is mode=decision without base / k_max")
        if self.options and not self.backend:
            raise ValueError(f"{self.slug}: options only apply to entries with a backend")
        if self.gpus < 1 or self.mem_gb < 1 or self.max_new_tokens < 1 or (self.k_max is not None and self.k_max < 2):
            raise ValueError(f"{self.slug}: gpus, mem_gb, max_new_tokens must be positive and k_max >= 2")
        if self.k_max and self.k_max > 26:
            raise ValueError(f"{self.slug}: k_max > 26 (only letters A-Z can be read)")
        if not self.enabled and not self.reason:
            raise ValueError(f"{self.slug}: a disabled entry needs a reason")
        for ref in (self.hf_id, self.base or ""):
            if BANNED.search(ref):
                raise ValueError(f"{self.slug}: {ref} is a Chinese-origin / Qwen-family model (not allowed)")
        return self


def load_registry(path=None) -> list[Entry]:
    import yaml
    raw = yaml.safe_load(Path(path or REGISTRY_PATH).read_text()) or {}
    known = {f.name for f in fields(Entry)}
    entries, seen = [], set()
    for i, r in enumerate(raw.get("models") or []):
        extra = set(r) - known
        if extra:
            raise ValueError(f"zoo entry {i} ({r.get('slug')}): unknown keys {sorted(extra)}")
        try:
            e = Entry(**r).validate()
        except TypeError as ex:
            raise ValueError(f"zoo entry {i} ({r.get('slug')}): {ex}") from ex
        if e.slug in seen:
            raise ValueError(f"duplicate slug {e.slug}")
        seen.add(e.slug)
        entries.append(e)
    return entries


def get_entry(slug, path=None) -> Entry:
    for e in load_registry(path):
        if e.slug == slug:
            return e
    raise SystemExit(f"unknown model {slug!r}; `waila-zoo list` shows the registry")


# ── the official Pass-2 prompt ───────────────────────────────────────────────────────────────────────────────────
DISPLAY = {"body_part": "body part", "modality": "modality", "stain": "stain", "submodality": "submodality",
           "specimen": "specimen", "domain": "medical domain"}
PASS2_ORDER = ["body_part", "modality", "stain", "submodality", "specimen", "domain"]
PASS2_PROMPT = (
    "Look at the image provided and identify the following metadata: {fields}. Predict each field from visual evidence in the "
    "image only. If a field genuinely cannot be determined, return 'unknown'. Also give a short **perception rationale** "
    "describing the visible evidence supporting your metadata predictions. Keep this concise and focused on observable image "
    "properties. Return machine-readable JSON only: "
    '{{"predicted_metadata": {{"<field>": "..."}}, "perception_rationale": "..."}} '
    "The keys inside predicted_metadata must correspond exactly to the requested metadata fields named below. "
    "Do not include additional text outside this JSON structure. Requested metadata fields for this example: {fields}")


def pass2_prompt(field_names) -> str:
    """Prompt listing only the given fields (in the official order), by their display names."""
    names = [DISPLAY[f] for f in PASS2_ORDER if f in set(field_names)]
    return PASS2_PROMPT.format(fields=", ".join(names))


# ── robust JSON parsing ──────────────────────────────────────────────────────────────────────────────────────────
_KEY_ALIASES = {"medical domain": "domain", "medical specialty": "domain", "specialty": "domain", "bodypart": "body_part"}


def field_key(k) -> str | None:
    """'Body Part' / 'body_part' / 'medical domain' → the field name; None if it is not a metadata field."""
    s = re.sub(r"[\s_-]+", " ", str(k).strip().lower())
    s = _KEY_ALIASES.get(s, s).replace(" ", "_")
    return s if s in FIELDS else None


def _balanced_objects(text):
    """Yield every top-level {...} substring (string-aware brace matching), in order."""
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    i = j + 1
                    break
        else:
            return  # unbalanced (truncated) from here on


def _loads(s):
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s.replace("“", '"').replace("”", '"').replace("’", "'"))):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    if '"' not in s:
        try:
            v = json.loads(s.replace("'", '"'))
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    return None


def parse_output(text) -> tuple[dict, str, bool]:
    """(raw values by field, rationale, json_ok) from a model's reply. Tolerates code fences, text around the JSON, trailing
    commas, smart quotes, a flat object without `predicted_metadata`, and (last resort) a truncated object, in which case
    `"key": "value"` pairs are scraped by regex (json_ok False)."""
    text = (text or "").replace("```json", "```").replace("```JSON", "```")
    text = re.sub(r"```", "\n", text)
    for obj in _balanced_objects(text):
        d = _loads(obj)
        if d is None:
            continue
        meta = d.get("predicted_metadata", d.get("metadata", d))
        if isinstance(meta, dict):
            vals = {field_key(k): v for k, v in meta.items() if field_key(k)}
            if vals:
                return vals, str(d.get("perception_rationale", "") or ""), True
    vals = {}
    for m in re.finditer(r'"([A-Za-z _-]+)"\s*:\s*(\[[^\]]*\]|"(?:[^"\\]|\\.)*"|null)', text):
        k = field_key(m.group(1))
        if k and k not in vals:
            v = m.group(2)
            vals[k] = None if v == "null" else _loads_value(v)
    return vals, "", False


def _loads_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v.strip('"[]')


# ── mapping strings onto the label space ─────────────────────────────────────────────────────────────────────────
UNKNOWN = {"unknown", "n/a", "n a", "na", "not applicable", "none of the above", "cannot be determined", "undetermined", "unclear",
           "not specified", "unspecified", "not visible", "null", ""}
# hand-written synonyms (key → class); keys and values go through key() before lookup. Heuristics: extend freely.
EXTRA_ALIASES = {
    "modality": {"ct": "computed tomography", "ct scan": "computed tomography", "cat scan": "computed tomography",
                 "mri": "magnetic resonance imaging", "mr": "magnetic resonance imaging", "mri scan": "magnetic resonance imaging",
                 "magnetic resonance": "magnetic resonance imaging", "x ray": "x-ray imaging", "xray": "x-ray imaging",
                 "radiograph": "x-ray imaging", "radiography": "x-ray imaging", "plain film": "x-ray imaging",
                 "ultrasound": "ultrasound imaging", "sonography": "ultrasound imaging", "ultrasonography": "ultrasound imaging",
                 "histopathology": "digital pathology", "histology": "digital pathology", "pathology": "digital pathology",
                 "whole slide imaging": "digital pathology", "photography": "clinical photography",
                 "photograph": "clinical photography", "clinical photograph": "clinical photography",
                 "fluorescence": "fluorescence microscopy", "fluorescent microscopy": "fluorescence microscopy",
                 "confocal microscopy": "fluorescence microscopy", "microscopy": "light microscopy",
                 "optical microscopy": "light microscopy", "brightfield microscopy": "light microscopy",
                 "bright field microscopy": "light microscopy", "transmission electron microscopy": "electron microscopy",
                 "scanning electron microscopy": "electron microscopy", "sem": "electron microscopy", "tem": "electron microscopy",
                 "endoscopic": "endoscopy", "colonoscopy": "endoscopy", "oct": "optical imaging",
                 "fundus photography": "optical imaging", "optical coherence tomography": "optical imaging"},
    "submodality": {"ct": "conventional ct", "ct scan": "conventional ct", "computed tomography": "conventional ct",
                    "mri": "structural mri", "mri scan": "structural mri", "magnetic resonance imaging": "structural mri",
                    "x ray": "chest x-ray", "chest radiograph": "chest x-ray", "cxr": "chest x-ray",
                    "radiography": "digital radiography", "oct": "optical coherence tomography",
                    "octa": "optical coherence tomography angiography", "oct angiography": "optical coherence tomography angiography",
                    "fundus": "fundus photography", "fundus photograph": "fundus photography",
                    "retinal photography": "fundus photography", "ultrasound": "b-mode ultrasound",
                    "2d ultrasound": "b-mode ultrasound", "b mode": "b-mode ultrasound", "doppler": "color doppler",
                    "color doppler ultrasound": "color doppler", "bright field microscopy": "brightfield microscopy",
                    "bright field": "brightfield microscopy", "brightfield": "brightfield microscopy",
                    "histology": "brightfield microscopy", "wsi": "whole slide imaging", "whole slide image": "whole slide imaging",
                    "sem": "scanning electron microscopy", "tirf": "total internal reflection fluorescence microscopy (tirf)",
                    "tirf microscopy": "total internal reflection fluorescence microscopy (tirf)",
                    "phase contrast": "phase contrast microscopy", "widefield fluorescence": "epifluorescence microscopy",
                    "widefield fluorescence microscopy": "epifluorescence microscopy", "confocal": "confocal microscopy",
                    "endoscopy": "white-light endoscopy", "cryo et": "cryo-electron tomography",
                    "photography": "clinical photography", "clinical photograph": "clinical photography"},
    "stain": {"hematoxylin and eosin": "h&e", "haematoxylin and eosin": "h&e", "he": "h&e", "h and e": "h&e",
              "h&e stain": "h&e", "hematoxylin eosin": "h&e", "immunohistochemistry": "ihc", "dab": "ihc",
              "none": "unstained", "no stain": "unstained", "not stained": "unstained", "label free": "unstained",
              "without stain": "unstained", "giemsa stain": "giemsa", "wright giemsa": "wright-giemsa",
              "may grunwald giemsa": "may-grünwald-giemsa", "pap": "papanicolaou", "pap stain": "papanicolaou",
              "papanicolaou stain": "papanicolaou", "hoechst": "hoechst 33342", "fluorescent": "immunofluorescence",
              "diff quik": "diff-quik", "ziehl neelsen": "ziehl-neelsen"},
    "specimen": {"homo sapiens": "human", "human patient": "human", "patient": "human", "mus musculus": "mouse",
                 "murine": "mouse", "mice": "mouse", "rattus norvegicus": "rat", "danio rerio": "zebrafish",
                 "escherichia coli": "e. coli", "caenorhabditis elegans": "c. elegans", "drosophila melanogaster": "drosophila",
                 "fruit fly": "drosophila", "sus scrofa": "pig", "porcine": "pig", "bovine": "cow", "canine": "dog",
                 "feline": "cat", "cultured cells": "cell line", "cell culture": "cell line", "cells": "cell line",
                 "monkey": "non-human primate", "macaque": "non-human primate", "primate": "non-human primate",
                 "bird": "avian", "chicken": "avian", "ovine": "sheep", "cavia porcellus": "guinea pig"},
    "domain": {"ent": "otolaryngology", "otorhinolaryngology": "otolaryngology", "gynecology": "obstetrics and gynecology",
               "obstetrics": "obstetrics and gynecology", "ob gyn": "obstetrics and gynecology", "obgyn": "obstetrics and gynecology",
               "infectious diseases": "infectious disease", "microbiology": "infectious disease",
               "parasitology": "infectious disease", "pulmonary": "pulmonology", "respiratory": "pulmonology",
               "pneumology": "pulmonology", "orthopaedics": "orthopedics", "orthopedic surgery": "orthopedics",
               "musculoskeletal": "orthopedics", "histopathology": "pathology", "anatomic pathology": "pathology",
               "hematopathology": "hematology", "neuroradiology": "radiology", "cell biology": "biology"},
    "body_part": {"lungs": "lung", "thorax": "chest", "gi tract": "gastrointestinal tract", "digestive tract": "gastrointestinal tract",
                  "intestine": "gastrointestinal tract", "bowel": "gastrointestinal tract", "abdominal": "abdomen",
                  "knee joint": "knee", "vertebral column": "spine", "blood cell": "blood cells", "whole body": "whole body",
                  "eyes": "eye", "white blood cell": "white blood cells", "red blood cells": "blood cells"},
}
FUZZY_CUTOFF = 0.88


def key(s) -> str:
    """Case/punctuation-insensitive form: 'H&E stain' → 'h&e stain', 'Wright–Giemsa' → 'wright giemsa'."""
    s = re.sub(r"[^a-z0-9&]+", " ", str(s).lower()).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def _variants(s):
    """The string and cleaned forms of it: without a parenthetical, the parenthetical itself, with generic suffixes toggled."""
    out = [key(s)]
    s = str(s)
    for m in re.finditer(r"\(([^)]*)\)", s):
        out.append(key(m.group(1)))
    out.append(key(re.sub(r"\([^)]*\)", " ", s)))
    for v in list(out):
        for suf in (" imaging", " image", " images", " scan", " stain", " staining", " stained"):
            if v.endswith(suf):
                out.append(v[: -len(suf)].strip())
        out += [v + " imaging", v + " microscopy"]
        if v.endswith("s"):
            out.append(v[:-1])
    return [v for i, v in enumerate(out) if v and v not in out[:i]]


class Mapper:
    """String → class of one field's label space: exact (canonical normalizer or class name) → alias → fuzzy → unmapped."""

    def __init__(self, space):
        self.space = space
        self.exact, self.alias = {}, {}
        for f in space.fields:
            ex = {}
            for c in space.classes[f]:
                for v in _variants(c)[:3]:     # class name, and without / inside its parenthetical (e.g. "(tirf)")
                    ex.setdefault(v, c)
                ex[key(c)] = c
            al = {}
            for a, t in {**PUBLIC_ALIASES.get(f, {}), **EXTRA_ALIASES.get(f, {})}.items():
                c = t if t in space.index[f] else ex.get(key(t))
                if c:
                    al[key(a)] = c
            self.exact[f], self.alias[f] = ex, al

    def map_one(self, f, s):
        s = str(s).strip()
        if key(s) in UNKNOWN or s.lower() in UNKNOWN:
            return None, "unknown"
        canon = CANON[f](s) if f in CANON else None
        if canon in self.space.index[f]:
            return canon, "exact"
        vs = _variants(s)
        for v in vs:
            if v in self.exact[f]:
                return self.exact[f][v], "exact"
        for v in vs:
            if v in self.alias[f]:
                return self.alias[f][v], "alias"
        keys = list(self.exact[f]) + list(self.alias[f])
        for v in vs[:2]:
            m = difflib.get_close_matches(v, keys, n=1, cutoff=FUZZY_CUTOFF)
            if m:
                return self.exact[f].get(m[0]) or self.alias[f][m[0]], "fuzzy"
        return None, "unmapped"

    def map(self, f, value):
        """(class | None, how) for a raw model value. A list, or a 'a|b' / 'a; b' / 'a, b' string, yields its first mappable item
        (the model's primary answer); how ∈ exact | alias | fuzzy | unknown | unmapped | missing."""
        if value is None:
            return None, "missing"
        items = [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
        if not isinstance(value, (list, tuple)) and re.search(r"[|;,]", items[0]):
            c, how = self.map_one(f, items[0])               # the whole string first (class names can contain commas)
            if c:
                return c, how
            items = [x for x in re.split(r"\s*[|;,]\s*", items[0]) if x.strip()]
        hows = []
        for it in items:
            c, how = self.map_one(f, it)
            if c:
                return c, how
            hows.append(how)
        return None, ("unknown" if hows and all(h == "unknown" for h in hows) else "unmapped" if hows else "missing")


# ── turning predictions into the tensors score_public expects ────────────────────────────────────────────────────
HI, LO = 8.0, -8.0


def onehot_logits(pred_idx, tgt, n_classes):
    """[N, C] logits: +8 at the predicted class, -8 elsewhere. No prediction (unknown / unmapped / missing): -8 everywhere with
    the first class that is NOT a target nudged to -7, so top-1 is guaranteed wrong (argmax of a flat row would be class 0,
    which could be correct by accident) while sigmoid(-7) stays far below any multi-label threshold."""
    import torch
    L = torch.full((len(pred_idx), n_classes), LO)
    t = tgt.cpu().numpy() > 0
    for i, p in enumerate(pred_idx):
        if p is not None:
            L[i, p] = HI
        else:
            wrong = np.nonzero(~t[i])[0]
            if len(wrong):
                L[i, int(wrong[0])] = LO + 1
    return L


def encode_rows(df, space):
    """(modes {field: [N]}, tgts {field: [N, C]}) tensors from the eval manifest."""
    import torch
    enc = [space.encode(r) for _, r in df.iterrows()]
    return ({f: torch.tensor([e[f][0] for e in enc]) for f in space.fields},
            {f: torch.from_numpy(np.stack([e[f][1] for e in enc])) for f in space.fields})


def load_image(root, rel, max_side):
    from PIL import Image
    im = Image.open(Path(root) / rel).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


# ── running a model ──────────────────────────────────────────────────────────────────────────────────────────────
def load_backend(entry):
    from bio_waila.models.zoo_models import load_backend as lb
    return lb(entry)


def _generative(backend, df, root, space, modes, tgts, batch, max_side):
    from bio_waila.train.labels import MASKED
    mapper = Mapper(space)
    N = len(df)
    ann = {f: modes[f].numpy() != MASKED for f in space.fields}
    pred = {f: [None] * N for f in space.fields}
    how = {f: ["missing"] * N for f in space.fields}
    raw = []
    todo = [i for i in range(N) if any(ann[f][i] for f in space.fields)]
    t0 = time.perf_counter()
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        asked = [[f for f in space.fields if ann[f][i]] for i in idx]
        texts = backend.generate([load_image(root, df["image"].iloc[i], max_side) for i in idx], [pass2_prompt(a) for a in asked])
        for i, a, text in zip(idx, asked, texts):
            vals, rationale, ok = parse_output(text)
            rec = {"unique_id": df["unique_id"].iloc[i], "image": df["image"].iloc[i], "fields": a, "raw": text,
                   "json_ok": ok, "rationale": rationale, "values": {}, "mapped": {}}
            for f in a:
                c, h = mapper.map(f, vals.get(f))
                pred[f][i] = space.index[f][c] if c else None
                how[f][i] = h if ok or h != "missing" else "parse_failed"
                rec["values"][f], rec["mapped"][f] = vals.get(f), {"class": c, "how": how[f][i]}
            raw.append(rec)
        print(f"[zoo] generative {min(s + batch, len(todo))}/{len(todo)}", flush=True)
    secs = time.perf_counter() - t0
    logits = {f: onehot_logits(pred[f], tgts[f], space.n(f)) for f in space.fields}
    asked_n = {f: int(ann[f].sum()) for f in space.fields}
    cnt = {f: {h: sum(1 for i in range(N) if ann[f][i] and how[f][i] == h)
               for h in ("exact", "alias", "fuzzy", "unknown", "unmapped", "missing", "parse_failed")} for f in space.fields}
    tot = max(sum(asked_n.values()), 1)
    bad = sum(c["unmapped"] + c["missing"] + c["parse_failed"] for c in cnt.values())
    extra = {"unmapped_rate": bad / tot, "unknown_rate": sum(c["unknown"] for c in cnt.values()) / tot,
             "json_ok_rate": float(np.mean([r["json_ok"] for r in raw])) if raw else float("nan"),
             "mapping": cnt, "n_asked": asked_n}
    cols = {f"{f}_raw": [json.dumps(r["values"].get(f)) if f in r["values"] else None for r in _by_row(raw, df)] for f in space.fields}
    cols.update({f"{f}_map": [how[f][i] if ann[f][i] else None for i in range(N)] for f in space.fields})
    return logits, secs, extra, cols, raw


def _by_row(raw, df):
    by = {r["unique_id"]: r for r in raw}
    return [by.get(u, {"values": {}}) for u in df["unique_id"]]


def _decision_rows(backend, df, root, space, modes, batch, max_side):
    """Backends with `score_rows` (System One) answer all annotated fields of an image in one request."""
    import torch
    from bio_waila.train.labels import MASKED
    N = len(df)
    logits = {f: torch.full((N, space.n(f)), -30.0) for f in space.fields}
    todo = [(i, [f for f in space.fields if int(modes[f][i]) != MASKED]) for i in range(N)]
    todo = [t for t in todo if t[1]]
    t0 = time.perf_counter()
    for s in range(0, len(todo), batch):
        sel = todo[s:s + batch]
        rows = backend.score_rows([load_image(root, df["image"].iloc[i], max_side) for i, _ in sel], [fs for _, fs in sel], space)
        for (i, _), r in zip(sel, rows):
            for f, lp in r.items():
                logits[f][i] = lp.float()
        print(f"[zoo] decision rows {min(s + batch, len(todo))}/{len(todo)}", flush=True)
    return logits, time.perf_counter() - t0


def _decision(backend, df, root, space, modes, tgts, batch, max_side, k_max):
    import torch
    from bio_waila.train.labels import MASKED
    N = len(df)
    if hasattr(backend, "score_rows"):
        logits, secs = _decision_rows(backend, df, root, space, modes, batch, max_side)
        return logits, secs, {"unmapped_rate": None}, _decision_cols(logits, modes, space), None
    logits = {f: torch.full((N, space.n(f)), -30.0) for f in space.fields}
    t0 = time.perf_counter()
    for f in space.fields:
        idx = [i for i in range(N) if int(modes[f][i]) != MASKED]
        for s in range(0, len(idx), batch):
            sel = idx[s:s + batch]
            lp = backend.logprobs([load_image(root, df["image"].iloc[i], max_side) for i in sel], f, space, k_max)
            logits[f][sel] = lp.float()
        print(f"[zoo] decision {f}: {len(idx)} images", flush=True)
    secs = time.perf_counter() - t0
    return logits, secs, {"unmapped_rate": None}, _decision_cols(logits, modes, space), None


def _decision_cols(logits, modes, space):
    from bio_waila.train.labels import MASKED
    N = len(next(iter(logits.values())))
    ann = {f: modes[f].numpy() != MASKED for f in space.fields}
    return {f"{f}_prob": [float(logits[f][i].softmax(-1).max()) if ann[f][i] else None for i in range(N)] for f in space.fields}


def run_model(entry, limit=None, batch=None, out_root=None, backend=None, public=None, seed=0):
    """Score one zoo entry on mmbu-public and write runs/zoo/<slug>/ (public_metrics.json, public_predictions.parquet,
    raw_outputs.jsonl for generative) plus a line in plan/public_looks.jsonl. Returns the result dict."""
    from bio_waila.eval import public as pub
    from bio_waila.train.labels import LabelSpace
    got = public or pub.try_load_public({}, limit=None)
    if got is None:
        raise SystemExit("mmbu-public is not available (set WAILA_PUBLIC; `waila-zoo prefetch` builds the eval set)")
    df, root = got
    if limit:
        df = pub.subsample(df, limit, seed)
    df = df.reset_index(drop=True)
    space = LabelSpace.load()
    modes, tgts = encode_rows(df, space)
    backend = backend or load_backend(entry)
    bs = batch or (8 if entry.mode == "generative" else 16)
    if entry.mode == "generative":
        logits, secs, extra, cols, raw = _generative(backend, df, root, space, modes, tgts, bs, 1024)
    else:
        logits, secs, extra, cols, raw = _decision(backend, df, root, space, modes, tgts, bs, 512, entry.k_max or 26)
    res = pub.score_public(logits, modes, tgts, df, space)
    res["zoo"] = {"slug": entry.slug, "hf_id": entry.hf_id, "base": entry.base, "mode": entry.mode, "readout": entry.readout,
                  "params_b": float(getattr(backend, "n_params", float("nan"))), "sec_per_image": secs / max(len(df), 1),
                  "total_sec": secs, "limit": limit, "batch": bs,
                  "backend": entry.backend or None, "max_new_tokens": entry.max_new_tokens if entry.mode == "generative" else None,
                  "k_max": (entry.k_max or 26) if entry.mode == "decision" and entry.readout == "letters" else entry.k_max,
                  **extra}
    table = pub.predictions_table(logits, modes, tgts, df, space)
    for k, v in cols.items():
        table[k] = v
    out = Path(out_root or RUNS / "zoo") / entry.slug
    pub.write_public_report(out, f"zoo:{entry.slug}", res, table)
    if raw is not None:
        (out / "raw_outputs.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in raw) + "\n")
    print(f"[zoo] {entry.slug}: C={res['C']:.4f} mean_field_acc={res['mean_field_acc']:.4f} n={res['n_images']} "
          f"{res['zoo']['sec_per_image']:.2f}s/img → {out}", flush=True)
    return res


# ── prefetch ─────────────────────────────────────────────────────────────────────────────────────────────────────
IGNORE = ["*.gguf", "*.mp4", "*.png", "*.svg", "*.jpg", "docs/*", "videos/*", "assets/*", "reports/*", "serving/*", "evaluation/*",
          "adapter_vllm/*", "patches/*", "*.tar.gz", "original/*", "consolidated*.safetensors"]


def prefetch_one(repo, token=None, ignore=IGNORE):
    """('ok'|'gated'|'missing'|'error', detail) for one snapshot_download."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    try:
        snapshot_download(repo, token=token, ignore_patterns=ignore)
        return "ok", ""
    except GatedRepoError as e:
        return "gated", f"gated: accept the terms at https://huggingface.co/{repo} with the account that owns HF_TOKEN ({e})"
    except RepositoryNotFoundError as e:
        return "missing", f"not found / private (401): check the id and HF_TOKEN ({e})"
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            return "gated", f"HTTP {code}: accept the terms at https://huggingface.co/{repo} and/or set HF_TOKEN"
        return "error", f"HTTP {code}: {str(e)[:200]}"
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {str(e)[:200]}"


def prefetch(slugs=None, path=None, strict=False, fetch=prefetch_one):
    from bio_waila.eval import public as pub
    entries = load_registry(path)
    if slugs:
        unknown = set(slugs) - {e.slug for e in entries}
        if unknown:
            raise SystemExit(f"unknown models: {sorted(unknown)}")
        entries = [e for e in entries if e.slug in slugs]
    else:
        entries = [e for e in entries if e.enabled]
    try:
        df, _ = pub.load_public()
        print(f"[prefetch] eval set: {len(df)} rows ({pub.PLAN / 'eval_public.parquet'})", flush=True)
    except pub.PublicMissing as e:
        print(f"[prefetch] eval set NOT built: {e}", flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    status, done = {}, {}
    for e in entries:
        for repo in dict.fromkeys([e.base, e.hf_id] if e.base else [e.hf_id]):
            if repo not in done:
                print(f"[prefetch] {repo} …", flush=True)
                # a systemone repo is the whole engine + weights: nothing is filtered out
                done[repo] = fetch(repo, token, ignore=[]) if e.backend == "systemone" else fetch(repo, token)
            status[(e.slug, repo)] = done[repo]
    bad = {k: v for k, v in status.items() if v[0] != "ok"}
    print("\n[prefetch] summary")
    for (slug, repo), (st, detail) in status.items():
        print(f"  {st:8s} {slug:34s} {repo}" + (f"\n           {detail}" if detail else ""))
    print(f"[prefetch] {len(status) - len(bad)}/{len(status)} ok" + (f"; {len(bad)} FAILED (gated models need the terms accepted on HF)" if bad else ""))
    return 1 if (bad and strict) else 0


# ── leaderboard ──────────────────────────────────────────────────────────────────────────────────────────────────
def leaderboard(root=None):
    """Aggregate <root>/*/public_metrics.json (root default runs/zoo) → leaderboard.csv + leaderboard.md; returns the DataFrame."""
    root = Path(root or RUNS / "zoo")
    rows = []
    for p in sorted(root.glob("*/public_metrics.json")):
        m = json.loads(p.read_text())
        z = m.get("zoo")
        if not z:
            continue
        r = {"slug": z["slug"], "mode": z["mode"], "hf_id": z["hf_id"], "params_b": z.get("params_b"), "n": m.get("n_images"),
             "C": m.get("C"), "mean_field_acc": m.get("mean_field_acc"), "sec_per_image": z.get("sec_per_image"),
             "unmapped_rate": z.get("unmapped_rate"), "limit": z.get("limit")}
        for f in FIELDS:
            r[f"acc_{f}"] = (m.get("field_acc", {}).get(f) or {}).get("acc")
        rows.append(r)
    cols = ["slug", "mode", "hf_id", "params_b", "n", "C", "mean_field_acc"] + [f"acc_{f}" for f in FIELDS] + \
           ["sec_per_image", "unmapped_rate", "limit"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values("C", ascending=False, na_position="last").reset_index(drop=True)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "leaderboard.csv", index=False)
    (root / "leaderboard.md").write_text(_markdown(df))
    print(f"[leaderboard] {len(df)} runs → {root / 'leaderboard.csv'}, {root / 'leaderboard.md'}", flush=True)
    return df


def _markdown(df):
    def fmt(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return "–"
        if c in ("params_b", "sec_per_image"):
            return f"{v:.2f}"
        if c == "n":
            return str(int(v))
        if c == "limit":
            return str(int(v))
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = {"params_b": "params (B)", "sec_per_image": "s/img", "mean_field_acc": "mean acc", "unmapped_rate": "unmapped"}
    cols = [c for c in df.columns if c not in ("hf_id", "limit")]
    lines = ["# Model zoo leaderboard (mmbu-public, Pass 2)", "",
             "C = context score (mean over images of the fraction of annotated fields whose top-1 is correct); acc_* = per-field top-1 accuracy. "
             "`unmapped` = share of requested fields whose generated answer could not be mapped to a class (generative only).", "",
             "| " + " | ".join(head.get(c, c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=None, help=f"registry YAML (default {REGISTRY_PATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="show the registry")
    p.add_argument("--porcelain", action="store_true", help="enabled in-process entries only (no backend: systemone), as slug<TAB>gpus<TAB>mem_gb (for Slurm scripts)")
    p = sub.add_parser("prefetch", help="CPU: download weights and build the eval set")
    p.add_argument("--models", default=None, help="comma-separated slugs (default: all enabled)")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any download failed")
    p = sub.add_parser("run", help="GPU: score one model")
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=None, help="random subsample of N public images (seed 0)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--out", default=None, help="output root (default runs/zoo)")
    p.add_argument("--base-url", default=None, help="systemone entries: server URL (overrides options.base_url; implies --enable)")
    p.add_argument("--enable", action="store_true", help="run an entry even though it is `enabled: false` in the registry")
    sub.add_parser("leaderboard", help="aggregate runs/zoo/*/public_metrics.json")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        es = load_registry(a.registry)
        if a.porcelain:
            for e in es:
                if e.enabled and not e.backend:  # backend entries (systemone) need a server, not a Slurm GPU job per model
                    print(f"{e.slug}\t{e.gpus}\t{e.mem_gb}")
            return 0
        for e in es:
            print(f"{'on ' if e.enabled else 'OFF'}  {e.slug:34s} {e.mode:10s} {e.mem_gb:4d}GB  {e.hf_id}"
                  + (f"  (+LoRA on {e.base})" if e.base else "") + ("" if e.enabled else f"\n       disabled: {e.reason}"))
        print(f"{sum(e.enabled for e in es)}/{len(es)} enabled")
        return 0
    if a.cmd == "prefetch":
        return prefetch(a.models.split(",") if a.models else None, a.registry, a.strict)
    if a.cmd == "run":
        e = get_entry(a.model, a.registry)
        if a.base_url:
            if e.backend != "systemone":
                raise SystemExit("--base-url only applies to backend: systemone entries")
            e.options = {**(e.options or {}), "base_url": a.base_url}
        if not e.enabled and not (a.enable or a.base_url):
            raise SystemExit(f"{e.slug} is disabled: {e.reason}  (pass --enable, or --base-url for a running server, to run it anyway)")
        run_model(e, a.limit, a.batch, a.out)
        return 0
    leaderboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
