"""Shared fixtures. Everything here runs offline against tiny synthetic inputs."""
import pytest

# A small, stable label space (class lists per field) for the matcher tests, so they do not depend on the
# registry-derived plan/label_space.json changing when the selection is re-run.
CLASSES = {
    "modality": ["computed tomography", "magnetic resonance imaging", "x-ray imaging", "ultrasound imaging", "endoscopy",
                 "optical imaging", "light microscopy", "fluorescence microscopy", "electron microscopy", "digital pathology"],
    "submodality": ["conventional ct", "structural mri", "chest x-ray", "mammography", "fundus photography",
                    "contrast microscopy", "white-light endoscopy"],
    "body_part": ["brain", "liver", "lung", "chest", "heart", "kidney", "stomach", "colon", "lymph nodes", "eye", "skin",
                  "breast", "abdomen", "bladder"],
    "stain": ["h&e", "ihc", "giemsa"],
    "specimen": ["human", "mouse", "rat", "zebrafish"],
    "domain": ["radiology", "urology"],
}


@pytest.fixture(scope="session")
def classes():
    return {f: set(v) for f, v in CLASSES.items()}


@pytest.fixture(autouse=True)
def _no_real_public(tmp_path_factory, monkeypatch):
    """Tests must never touch the real mmbu-public or log a look in the real plan/public_looks.jsonl."""
    try:
        from bio_waila.eval import public
    except ImportError:  # torch not installed
        return
    d = tmp_path_factory.mktemp("nopublic")
    monkeypatch.setattr(public, "PUBLIC", d / "mmbu-public")
    monkeypatch.setattr(public, "PLAN", d / "plan")
