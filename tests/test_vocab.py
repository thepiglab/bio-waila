"""bio_waila.data.vocab: canonical-value normalisers shared by training labels and the eval set."""
import pytest

from bio_waila.data.eval_set import split_vals
from bio_waila.data.vocab import (canon_body_part, canon_domain, canon_modality, canon_specimen, canon_stain,
                              canon_submodality, target)


@pytest.mark.parametrize("raw", ["H&E", "he", "HE", "HES", "Hematoxylin and Eosin", "hematoxylin and eosin stain",
                                 "Haematoxylin and Eosin (H&E)"])
def test_h_and_e_variants(raw):
    assert canon_stain(raw) == "h&e"


@pytest.mark.parametrize("raw", ["IHC", "immunohistochemistry", "DAB", "Ki-67", "HER2", "CD3", "IHC-DAB"])
def test_ihc_variants(raw):
    assert canon_stain(raw) == "ihc"


def test_synthetic_stain_is_kept_but_simulated_is_dropped():
    assert canon_stain("synthetic") == "synthetic"          # MMBU's reference stain for synthetic fluorescence
    assert canon_stain("Synthetic") == "synthetic"
    assert canon_stain("simulated") is None
    assert canon_stain("synthetic fluorescence") is None    # only the exact word is MMBU's class


@pytest.mark.parametrize("raw,expected", [("wright", "wright-giemsa"), ("Wright-Giemsa", "wright-giemsa"),
                                          ("Giemsa", "giemsa"), ("PAP", "papanicolaou"), ("unstained", "unstained")])
def test_other_stains(raw, expected):
    assert canon_stain(raw) == expected


@pytest.mark.parametrize("raw", ["", "unknown", "not specified", "N/A", "none", "mixed"])
def test_null_stain_values(raw):
    assert canon_stain(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("Homo sapiens", "human"), ("Mus musculus", "mouse"), ("Danio rerio", "zebrafish"), ("Escherichia coli", "e. coli"),
    ("Mouse", "mouse"), ("human", "human"), ("unknown", None), ("X-ray images", None), ("animal", None)])
def test_specimen_aliases(raw, expected):
    assert canon_specimen(raw) == expected


@pytest.mark.parametrize("raw,expected", [("Lungs", "lung"), ("Urinary bladder", "bladder"),
                                          ("intestine", "gastrointestinal tract"), ("back", None), ("N/A", None)])
def test_body_part_aliases(raw, expected):
    assert canon_body_part(raw) == expected


def test_modality_and_submodality_are_closed():
    assert canon_modality("Computed Tomography") == "computed tomography"
    assert canon_modality("CT") is None                      # abbreviations are the matcher's job, not the normaliser's
    assert canon_submodality("fundus photography") == "fundus photography"
    assert canon_submodality("Chest X-Ray") == "chest x-ray"
    assert canon_submodality("bogus") is None


def test_domain_aliases_and_registry_only_domains():
    assert canon_domain("Otorhinolaryngology") == "otolaryngology"
    assert canon_domain("gynecology") == "obstetrics and gynecology"
    assert canon_domain("neuroscience") is None              # registry-only domain: dropped


def test_domain_splitting():
    """MMBU domain cells are comma lists; 'obstetrics and gynecology' is ONE domain even though it contains 'and'."""
    assert split_vals("domain", "radiology, urology") == ["radiology", "urology"]
    assert split_vals("domain", "obstetrics and gynecology") == ["obstetrics and gynecology"]
    assert split_vals("domain", "['radiology', 'urology']") == ["radiology", "urology"]
    assert split_vals("body_part", "Chest|Lung") == ["Chest", "Lung"]
    vals = lambda s: sorted({c for c in map(canon_domain, split_vals("domain", s)) if c})
    assert len(vals("radiology, urology")) == 2
    assert len(vals("obstetrics and gynecology")) == 1


def test_target_modes():
    assert target(["computed tomography"], "modality") == {"mode": "hard", "values": ["computed tomography"]}
    assert target(["computed tomography", "magnetic resonance imaging"], "modality")["mode"] == "candidates"
    assert target([], "modality") == {"mode": "masked", "values": []}
    assert target(["bogus"], "modality") == {"mode": "masked", "values": []}
    assert target(["radiology", "urology"], "domain", multilabel=True) == {"mode": "multi", "values": ["radiology", "urology"]}
    assert target(["obstetrics and gynecology"], "domain", multilabel=True)["values"] == ["obstetrics and gynecology"]
