"""Graded label similarity, for soft contrastive targets and hierarchy-aware label smoothing.

Binary targets treat "chest x-ray vs digital radiography" (same modality) as wrong as "chest x-ray vs H&E", and give no
credit when two images share 3 of 4 body parts. The similarity S_f[a, b] ∈ [0, 1] between classes of a field fixes that:

  submodality  1 same class · `sibling` if both sit under one parent modality (resources/taxonomy.toml)
  modality     1 same class · `sibling` if the two modalities share a submodality (e.g. light microscopy / pathology)
  body_part    1 same class · `family` if both are in one anatomical family (docs/data/label_rules.json, 14 families)
  stain, specimen, domain   identity

Soft targets between two samples i, j of one field: z = normalise(y @ S) (y = their 0/1 label vectors; works for one
label, a multi-label set or a candidate set), T_ij = z_i · z_j ∈ [0, 1], T_ii = 1. Across fields, the image-image
similarity is the weighted mean of T_f over the fields both samples have labels for.
"""
import json
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

from bio_waila.paths import REPO, RESOURCES


def _taxonomy():
    return tomllib.loads((RESOURCES / "taxonomy.toml").read_text())


def _families():
    p = REPO / "docs" / "data" / "label_rules.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("body_part_parsing", {}).get("families", {})


def label_similarity(space, sibling=0.3, family=0.5) -> dict:
    """{field: [C, C] float tensor} for every field of the label space."""
    tax = _taxonomy().get("submodalities", {})
    parents = {}                                        # submodality → set of modalities
    for mod, subs in tax.items():
        for s in subs:
            parents.setdefault(s, set()).add(mod)
    fam = {}
    for f, members in _families().items():
        for m in members:
            fam.setdefault(m, set()).add(f)
    out = {}
    for f in space.fields:
        cls = space.classes[f]
        C = len(cls)
        S = torch.eye(C)
        for a in range(C):
            for b in range(a + 1, C):
                x, y = cls[a], cls[b]
                s = 0.0
                if f == "submodality" and parents.get(x, set()) & parents.get(y, set()):
                    s = sibling
                elif f == "modality" and any(x in p and y in p for p in parents.values()):
                    s = sibling
                elif f == "body_part" and fam.get(x, set()) & fam.get(y, set()):
                    s = family
                S[a, b] = S[b, a] = s
        out[f] = S
    return out


def soft_targets(y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """[B, B] target similarities for one field; rows with no labels are all zero."""
    z = y.float() @ S.to(y.device)
    z = F.normalize(z, dim=-1)                          # zero rows stay zero
    return (z @ z.t()).clamp(0, 1)


def smoothing_distribution(S: torch.Tensor) -> torch.Tensor:
    """[C, C]: row c = where label-smoothing mass goes for true class c (∝ similarity to c, uniform floor, not c)."""
    C = S.shape[0]
    D = S.clone().fill_diagonal_(0) + 1e-3              # small uniform floor so every class keeps some mass
    D.fill_diagonal_(0)
    return D / D.sum(-1, keepdim=True).clamp(min=1e-8) if C > 1 else torch.ones(1, 1)
