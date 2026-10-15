"""Regression tests for the metadata → closed-vocabulary matcher (bio_waila.data.rules).

Every case was verified in the dataset audits (docs/data/audits/audit_*.md, "matcher" sections): each one is a
false positive or false negative that was found on real metadata values and fixed in rules.py.
"""
import pytest

from bio_waila.data.rules import bids_hits, field_of_column, match_value, text_hits


def mv(classes, value, column="csv:meta.csv:x"):
    return match_value(value, column, classes)


# ── false positives that must stay fixed ────────────────────────────────────────────────────────────────────────
def test_negated_class_is_not_a_body_part(classes):
    assert mv(classes, "Not A Brain", "csv:a.csv:label") == {}
    assert "body_part" not in text_hits("Not A Brain", classes["body_part"])


@pytest.mark.parametrize("value", ["Fundus_Cardia", "fundus cardia", "Stomach Fundus"])
def test_gastric_fundus_is_stomach_not_fundus_photography(classes, value):
    hits = text_hits(value, classes["body_part"])
    assert hits.get("body_part") == {"stomach": True}
    assert "fundus photography" not in hits.get("submodality", {})
    assert "optical imaging" not in hits.get("modality", {})
    assert mv(classes, value, "path:d1") == {"body_part": ("stomach", True)}


def test_retinal_fundus_is_still_fundus_photography():
    hits = text_hits("fundus photograph")
    assert hits["submodality"] == {"fundus photography": True}
    assert hits["modality"] == {"optical imaging": True}


@pytest.mark.parametrize("value", ["ct angiography abdomen-pelvis-", "CT Angiography", "MR angiography head", "cta chest"])
def test_ct_mr_angiography_is_not_xray(classes, value):
    hits = text_hits(value)
    assert "x-ray imaging" not in hits.get("modality", {})
    assert "fluoroscopy" not in hits.get("submodality", {})
    out = mv(classes, value, "csv:meta.csv:study_type")
    assert out.get("modality", ("", 0))[0] in ("computed tomography", "magnetic resonance imaging")


def test_plain_angiography_is_xray_fluoroscopy():
    hits = text_hits("coronary angiography")
    assert hits["modality"] == {"x-ray imaging": True}
    assert hits["submodality"] == {"fluoroscopy": True}


@pytest.mark.parametrize("value", ["Mammosomatotroph", "mammosomatotroph adenoma"])
def test_mammosomatotroph_is_not_mammography(classes, value):
    assert text_hits(value) == {}
    assert mv(classes, value, "csv:m:cell_type") == {}


def test_real_mammography_still_matches():
    hits = text_hits("Mammography CC view")
    assert hits["submodality"] == {"mammography": True}
    assert hits["modality"] == {"x-ray imaging": True}


@pytest.mark.parametrize("value", ["2021-03-04", "12/Mar/2020", "ID_00123", "patient_0045", "123456"])
def test_dates_and_ids_map_to_nothing(classes, value):
    assert mv(classes, value, "csv:m:x") == {}
    assert mv(classes, value, "csv:m:modality") == {}


@pytest.mark.parametrize("col", ["csv:m:StudyDate", "csv:m:patient_id", "csv:m:diagnosis", "csv:m:primary_origin", "csv:m:caption"])
def test_non_metadata_columns_are_skipped(classes, col):
    assert mv(classes, "Liver", col) == {}


def test_bethesda_asc_us_is_not_ultrasound(classes):
    assert mv(classes, "ASC-US", "path:d1") == {}
    assert text_hits("ASC-US") == {} and text_hits("LSIL HSIL") == {}


def test_nbi_is_never_white_light():
    for v in ("NBI white light", "narrow band imaging white light", "BLC"):
        assert "white-light endoscopy" not in text_hits(v).get("submodality", {})
    assert text_hits("white light endoscopy")["submodality"] == {"white-light endoscopy": True}


# ── positive mappings ───────────────────────────────────────────────────────────────────────────────────────────
def test_mr_code_needs_a_modality_column(classes):
    assert mv(classes, "MR", "csv:m:Modality") == {"modality": ("magnetic resonance imaging", True)}
    assert mv(classes, "MR", "dicom:Modality") == {"modality": ("magnetic resonance imaging", True)}
    weak = mv(classes, "MR", "csv:m:foo")  # outside a modality column "mr" is only a weak hit
    assert weak["modality"][1] is False


def test_t2w_in_view_column_is_structural_mri(classes):
    out = mv(classes, "T2W", "hf:view")
    assert out["submodality"] == ("structural mri", True)
    assert out["modality"] == ("magnetic resonance imaging", True)


@pytest.mark.parametrize("value", ["LymphNode", "Lymph Node", "lymph nodes"])
def test_lymph_node_camelcase(classes, value):
    assert mv(classes, value, "csv:m:organ") == {"body_part": ("lymph nodes", True)}


@pytest.mark.parametrize("value", ["hepatic flexure", "Splenic Flexure", "cecum", "sigmoid"])
def test_colon_synonyms(classes, value):
    assert mv(classes, value, "csv:m:organ") == {"body_part": ("colon", True)}


def test_flexure_does_not_leak_organ_words():
    assert text_hits("hepatic flexure")["body_part"] == {"colon": True}  # not liver


def test_dic_is_contrast_microscopy(classes):
    assert mv(classes, "DIC", "csv:m:technique")["submodality"] == ("contrast microscopy", True)
    assert text_hits("DIC")["submodality"] == {"contrast microscopy": True}


def test_exact_class_name_in_any_column(classes):
    assert mv(classes, "Liver", "csv:m:organ") == {"body_part": ("liver", True)}
    assert mv(classes, "Lung", "csv:m:whatever") == {"body_part": ("lung", True)}


def test_diagnosis_words_demote_body_part():
    hits = text_hits("retinal hemorrhage")
    assert hits["body_part"] == {"retina": False}  # weak: only usable inside a body-part column


def test_colposcopy_is_optical_imaging(classes):
    out = mv(classes, "colposcopy", "csv:m:modality")
    assert out["modality"] == ("optical imaging", True)
    assert out["submodality"] == ("colposcopy", True)


# ── BIDS suffixes and column → field routing ────────────────────────────────────────────────────────────────────
def test_bids_hits():
    assert bids_hits("sub-01/anat/sub-01_T1w.nii.gz") == {
        "modality": {"magnetic resonance imaging": True}, "submodality": {"structural mri": True}}
    assert bids_hits("sub-01_dwi.nii") == {"modality": {"magnetic resonance imaging": True}}  # MRI, not structural
    assert bids_hits("sub-01/volume.nii.gz") == {}
    assert bids_hits("sub-01_T1w.png") == {}


@pytest.mark.parametrize("col,field", [
    ("BodyPartExamined", "body_part"), ("dicom:BodyPartExamined", "body_part"), ("csv:a.csv:Modality", "modality"),
    ("dicom:Modality", "modality"), ("hf:view", "submodality"), ("dicom:SeriesDescription", "submodality"),
    ("csv:s.csv:organ", "body_part"), ("csv:s.csv:Stain", "stain"), ("hf:species", "specimen"),
    ("path:d1", None), ("bids:suffix", None), ("csv:f:primary_origin", None)])
def test_field_of_column(col, field):
    assert field_of_column(col) == field


def test_bids_column_values(classes):
    out = match_value("t2w", "bids:suffix", classes)
    assert out["submodality"] == ("structural mri", True)
    assert out["modality"] == ("magnetic resonance imaging", True)
    assert match_value("dwi", "bids:suffix", classes) == {"modality": ("magnetic resonance imaging", True)}
