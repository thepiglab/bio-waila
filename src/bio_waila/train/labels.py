"""Label space and per-row targets shared by every trainer (CLM-vision, CLM-LLM, Jev).

Each manifest row carries, per field, `<field>_mode` ∈ {hard, multi, candidates, masked} and `<field>_values`.
`LabelSpace.encode` turns that into a mode code plus a 0/1 vector over the field's classes:

    MASKED 0  no loss for this field
    HARD   1  exactly one class (single-label field) / the given classes (multi-label field)
    MULTI  2  several classes that are all true (body part, domain; or a "multi" set in a single-label field,
              which we treat as a candidate set because only one can be the answer there)
    CAND   3  the true class is one of these (partial label)

Values that are not in the label space (vocabulary drift) are dropped; if nothing is left the field is masked.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bio_waila.data.vocab import FIELDS
from bio_waila.paths import PLAN

MASKED, HARD, MULTI, CAND = 0, 1, 2, 3
MODE_CODE = {"masked": MASKED, "hard": HARD, "multi": MULTI, "candidates": CAND}

# Natural-language templates for label prompts (text-tower / LLM label embeddings and Jev option text).
TEMPLATES = {
    "modality": ["a medical image acquired with {}.", "{} image.", "an image from {}.", "this image was obtained by {}."],
    "submodality": ["a {} image.", "an image acquired by {}.", "{}.", "a biomedical image showing {}."],
    "body_part": ["an image of the {}.", "{} shown in a medical image.", "the {}.", "anatomy: {}."],
    "stain": ["a microscopy image stained with {}.", "{} staining.", "a slide prepared with {} stain.", "stain: {}."],
    "specimen": ["an image of a {} specimen.", "specimen: {}.", "a sample from {}.", "{} tissue or cells."],
    "domain": ["a {} image.", "an image used in {}.", "medical specialty: {}.", "a case from {}."],
}


@dataclass
class LabelSpace:
    fields: list
    classes: dict                       # field → [class names]
    multilabel: dict                    # field → bool
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {f: {c: i for i, c in enumerate(cs)} for f, cs in self.classes.items()}

    @classmethod
    def load(cls, path: Path | str | None = None, fields=FIELDS):
        j = json.loads(Path(path or PLAN / "label_space.json").read_text())
        return cls(list(fields), {f: list(j[f]["classes"]) for f in fields},
                   {f: bool(j[f].get("multilabel", f in ("body_part", "domain"))) for f in fields})

    def n(self, f):
        return len(self.classes[f])

    def encode(self, row) -> dict:
        """{field: (mode_code, float32 vector of length n(field))} for one manifest row (dict or pandas Series)."""
        out = {}
        for f in self.fields:
            mode = MODE_CODE.get(str(row.get(f"{f}_mode", "masked")), MASKED)
            vals = row.get(f"{f}_values")
            vals = [] if vals is None else list(vals)
            vec = np.zeros(self.n(f), np.float32)
            for v in vals:
                i = self.index[f].get(str(v))
                if i is not None:
                    vec[i] = 1.0
            k = int(vec.sum())
            if mode == MASKED or k == 0:
                mode = MASKED
                vec[:] = 0
            elif not self.multilabel[f] and k > 1 and mode in (HARD, MULTI):
                mode = CAND  # several "true" classes in a single-label field: only one can be the answer
            elif not self.multilabel[f] and mode == CAND and k == 1:
                mode = HARD  # a candidate set narrowed to one
            out[f] = (mode, vec)
        return out

    def prompts(self, f, templates=None) -> list[list[str]]:
        """[class][k] prompt strings for field f."""
        ts = (templates or TEMPLATES)[f]
        return [[t.format(c) for t in ts] for c in self.classes[f]]
