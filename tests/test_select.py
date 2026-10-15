"""bio_waila.data.select helpers that need no registry: id extraction, eligibility, routing and implication rules."""
import pytest

from bio_waila.data import select
from bio_waila.data.select import _figshare_id, _zenodo_id, eligible, implication_rules, route


@pytest.mark.parametrize("url,expected", [
    ("https://figshare.com/articles/dataset/Some_Title/32646378", "32646378"),
    ("https://figshare.com/articles/dataset/32646378", "32646378"),
    ("https://figshare.com/articles/dataset/PAPILA/14798004", "14798004"),
    ("https://doi.org/10.6084/m9.figshare.14798004", "14798004"),
    ("https://doi.org/10.6084/m9.figshare.24007515.v1", "24007515"),         # version suffix is not part of the id
    ("https://springernature.figshare.com/collections/Title/6071216", "6071216"),
    ("https://doi.org/10.6084/m9.figshare.c.6071216", "6071216"),            # collection doi: c.<id>
    ("https://figshare.com/articles/dataset/Title/12345", None),              # < 6 digits: not an id
    ("https://zenodo.org/records/1234567", None),
])
def test_figshare_id(url, expected):
    assert _figshare_id(url) == expected


def test_figshare_id_not_truncated_for_8_digit_ids():
    assert _figshare_id("https://figshare.com/articles/dataset/Brain_MRI_scans/21345678") == "21345678"


@pytest.mark.parametrize("url,expected", [
    ("https://zenodo.org/records/7890123", "7890123"), ("https://zenodo.org/record/7890123", "7890123"),
    ("https://doi.org/10.5281/zenodo.7890123", "7890123"), ("https://example.org/x", None)])
def test_zenodo_id(url, expected):
    assert _zenodo_id(url) == expected


def reg(**access):
    a = {"download_urls": [], "other_mirrors": [], "homepage": "", "huggingface": [], "hf_gated": False,
         "zenodo": [], "kaggle": [], "access_type": "open"}
    a.update(access)
    return {"id": "toy-dataset-xyz", "access": a}


def test_route_prefers_hf_then_zenodo_then_figshare_then_kaggle():
    assert route(reg(huggingface=["org/name"], zenodo=["https://zenodo.org/records/1234567"])) == ("hf", "org/name")
    assert route(reg(huggingface=["org/name"], hf_gated=True, zenodo=["https://zenodo.org/records/1234567"])) == ("zenodo", "1234567")
    assert route(reg(download_urls=["https://figshare.com/articles/dataset/T/14798004"])) == ("figshare", "14798004")
    assert route(reg(kaggle=["https://www.kaggle.com/datasets/owner/slug"])) == ("kaggle", "owner/slug")


def test_route_special_cases():
    assert route(reg(homepage="https://data.mendeley.com/datasets/abc123xyz/1")) == ("mendeley", "abc123xyz")
    assert route(reg(homepage="https://openneuro.org/datasets/ds004567")) == ("openneuro", "ds004567")
    assert route(reg(download_urls=["https://host.org/data.zip"])) == ("direct", "https://host.org/data.zip")
    assert route(reg(kaggle=["https://www.kaggle.com/competitions/some-comp"]))[0] == "manual"
    assert route(reg()) == (None, None)


def full_record(**over):
    r = {"id": "toy-dataset-xyz",
         "license": {"tier": "A"},
         "access": {"download_urls": ["https://figshare.com/articles/dataset/T/14798004"], "other_mirrors": [], "homepage": "",
                    "huggingface": [], "hf_gated": False, "zenodo": [], "kaggle": [], "access_type": "open"},
         "io": {"data_format": "images"},
         "content": {"modalities": ["optical imaging"], "submodalities": ["fundus photography"], "dimensionality": ["2d"],
                     "synthetic": False, "specimens": ["human"], "body_parts": ["eye"], "stains": [], "medical_domains": ["ophthalmology"]},
         "derived": {"effective_images": 500, "train_ok": True}}
    for k, v in over.items():
        sect, _, key = k.partition("__")
        r[sect] = {**r[sect], key: v} if key else v
    return r


def test_eligible_baseline():
    assert eligible(full_record())


@pytest.mark.parametrize("over", [
    {"license__tier": "C"},                          # license tier A only
    {"access__access_type": "restricted"},
    {"io__data_format": "vqa"},                      # VQA/caption corpora recycle other datasets' images
    {"content__modalities": ["genomics"]},           # outside MMBU modalities
    {"content__dimensionality": ["tabular"]},
    {"content__synthetic": True},                    # synthetic only OK for synthetic fluorescence microscopy
    {"derived__effective_images": 10},               # < 30 images
    {"content__modalities": ["optical imaging", "endoscopy", "x-ray imaging"]},      # > 2 modalities
    {"content__submodalities": ["fundus photography", "dermoscopy", "colposcopy", "clinical photography", "mammography"]},
    {"access__download_urls": []},                   # no scriptable route
])
def test_ineligible(over):
    assert not eligible(full_record(**over))


def test_synthetic_fluorescence_is_allowed():
    r = full_record(content__modalities=["fluorescence microscopy"], content__submodalities=["synthetic fluorescence microscopy"],
                    content__synthetic=True)
    assert eligible(r)


def test_clinical_photography_needs_humans():
    assert not eligible(full_record(content__submodalities=["clinical photography"], content__specimens=["bird"]))
    assert eligible(full_record(content__submodalities=["clinical photography"], content__specimens=["human"]))


def test_audited_drops_are_ineligible(monkeypatch):
    monkeypatch.setitem(select.DROPS, "toy-dataset-xyz", "masks_only")
    assert not eligible(full_record())


def _lab(**over):
    lab = {f: {"mode": "masked", "values": []} for f in ("modality", "submodality", "body_part", "stain", "specimen", "domain")}
    lab.update(over)
    return lab


def test_implication_strong_submodality_sets_modality_and_body_part():
    lab = _lab(modality={"mode": "candidates", "values": ["optical imaging", "endoscopy"]},
               submodality={"mode": "hard", "values": ["fundus photography"]})
    out = implication_rules(lab, ["optical imaging", "endoscopy"])
    assert out["modality"] == {"mode": "hard", "values": ["optical imaging"]}   # fundus photography only legal under optical imaging
    assert out["body_part"] == {"mode": "hard", "values": ["eye"]}              # B1


def test_implication_radiology_domain_dropped_when_specialty_known():
    lab = _lab(domain={"mode": "multi", "values": ["radiology", "urology"]})
    out = implication_rules(lab, ["computed tomography"])
    assert out["domain"]["values"] == ["urology"]


def test_implication_cell_line_is_human():
    lab = _lab(specimen={"mode": "hard", "values": ["cell line"]})
    assert implication_rules(lab, ["fluorescence microscopy"])["specimen"]["values"] == ["human"]


def test_implication_synthetic_fluorescence_stain_and_specimen():
    lab = _lab(submodality={"mode": "hard", "values": ["synthetic fluorescence microscopy"]})
    out = implication_rules(lab, ["fluorescence microscopy"])
    assert out["stain"] == {"mode": "hard", "values": ["synthetic"]}
    assert out["specimen"] == {"mode": "hard", "values": ["synthetic"]}
