"""Regressions from the cluster CPU pilot (23,038 images, 401/420 datasets): download, archive and rule bugs."""
import io
import struct
import zipfile

import pytest

from bio_waila.data import download as dl
from bio_waila.data import zipcompat
from bio_waila.data.rules import is_skin_dataset, match_value, normalize_text, resolve, text_hits


# ── download: HTTP 416 on the resume request ────────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Client Error: Requested range not satisfiable for url: http://x")

    def iter_content(self, n):
        yield self._body

    def __enter__(self): return self
    def __exit__(self, *a): pass


def test_416_after_a_complete_body_is_success(tmp_path, monkeypatch):
    """Kaggle: Entry.size is the dataset's totalBytes, not the bundle zip's size, so a finished download looked short,
    the loop resumed with `Range: bytes=<size>-` and storage.googleapis.com answered 416."""
    body = b"PK-not-a-real-zip" * 100
    calls = []

    def fake_get(url, headers=None, **k):
        calls.append(dict(headers or {}))
        if "Range" in (headers or {}):
            return _Resp(416, headers={"Content-Range": f"bytes */{len(body)}"})
        return _Resp(200, body)  # no Content-Length: completeness cannot be judged from the response

    monkeypatch.setattr(dl.S, "get", fake_get)
    out = tmp_path / "a.bin"
    dl.get_file(dl.Entry("a.bin", len(body) * 3, "http://x"), out)  # registry size does not match the bundle
    assert out.read_bytes() == body
    assert any("Range" in c for c in calls)  # the 416 path was exercised


def test_416_with_a_different_total_restarts(tmp_path, monkeypatch):
    body = b"x" * 1000
    state = {"n": 0}

    def fake_get(url, headers=None, **k):
        state["n"] += 1
        if "Range" in (headers or {}):
            return _Resp(416, headers={"Content-Range": f"bytes */{len(body)}"})
        return _Resp(200, body, {"Content-Length": str(len(body))})

    monkeypatch.setattr(dl.S, "get", fake_get)
    out = tmp_path / "b.bin"
    (tmp_path / "b.bin.part").write_bytes(b"y" * 5000)  # stale partial file larger than the real one
    dl.get_file(dl.Entry("b.bin", len(body), "http://x"), out)
    assert out.read_bytes() == body


def test_truncated_body_is_resumed(tmp_path, monkeypatch):
    body = bytes(range(256)) * 8
    def fake_get(url, headers=None, **k):
        rng = (headers or {}).get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _Resp(206, body[start:], {"Content-Length": str(len(body) - start)})
        return _Resp(200, body[:700], {"Content-Length": str(len(body))})  # connection closed early
    monkeypatch.setattr(dl.S, "get", fake_get)
    out = tmp_path / "c.bin"
    dl.get_file(dl.Entry("c.bin", len(body), "http://x"), out)
    assert out.read_bytes() == body


# ── Deflate64 zips and the 7-Zip fallback ───────────────────────────────────────────────────────────────────────
def test_deflate64_member_is_readable():
    inflate64 = pytest.importorskip("inflate64")
    payload = bytes(range(251)) * 2000
    d = inflate64.Deflater()
    comp = d.deflate(payload) + d.flush()
    # hand-built single-member zip, method 9
    import zlib
    name = b"a.bin"
    crc = zlib.crc32(payload)
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 21, 0, 9, 0, 0, crc, len(comp), len(payload), len(name), 0) + name + comp
    cd = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 21, 21, 0, 9, 0, 0, crc, len(comp), len(payload), len(name), 0, 0, 0, 0, 0, 0) + name
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cd), len(local), 0)
    assert zipcompat.enable_deflate64()
    with zipfile.ZipFile(io.BytesIO(local + cd + eocd)) as z:
        assert z.read("a.bin") == payload


def test_7z_binary_lookup_and_command(tmp_path, monkeypatch):
    monkeypatch.setattr(zipcompat.shutil, "which", lambda n: "/opt/bin/7zz" if n == "7zz" else None)
    assert zipcompat.find_7z() == "/opt/bin/7zz"
    seen = {}

    class P:
        returncode, stdout, stderr = 0, "", ""

    zipcompat.extract_7z(tmp_path / "a.zip", tmp_path / "out", ["x/1.png", "x/2.png"], run=lambda cmd, **k: seen.update(cmd=cmd) or P())
    assert seen["cmd"][0] == "/opt/bin/7zz" and seen["cmd"][1] == "x"
    assert f"-o{tmp_path / 'out'}" in seen["cmd"] and seen["cmd"][-3:] == ["--", "x/1.png", "x/2.png"] or seen["cmd"][-2:] == ["x/1.png", "x/2.png"]


def test_7z_prefers_7zz_then_7z(monkeypatch):
    monkeypatch.setattr(zipcompat.shutil, "which", lambda n: f"/bin/{n}" if n in ("7z", "7za") else None)
    assert zipcompat.find_7z() == "/bin/7z"


def test_no_7z_binary_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(zipcompat.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="no 7-Zip binary"):
        zipcompat.extract_7z(tmp_path / "a.zip", tmp_path / "o")


def test_extract_archives_falls_back_to_7z(tmp_path, monkeypatch):
    from bio_waila.data import prepare
    root = tmp_path / "ds"
    (root / "archives").mkdir(parents=True)
    with zipfile.ZipFile(root / "archives" / "a.zip", "w") as z:
        z.writestr("im/1.png", b"x")
    calls = []
    monkeypatch.setattr(zipfile.ZipFile, "extract", lambda self, *a, **k: (_ for _ in ()).throw(NotImplementedError("compression type 99")))
    monkeypatch.setattr(prepare.zipcompat, "extract_7z", lambda a, d, members=None, run=None: calls.append((a.name, members)))
    prepare.extract_archives(root)
    assert calls == [("a.zip", ["im/1.png"])]
    assert (root / "extracted" / "a.zip.ok").exists()


# ── HDF5 images (cetus-2014, zea format) ────────────────────────────────────────────────────────────────────────
def test_hdf5_image_array_is_read_and_segmentation_skipped(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    from bio_waila.data.files import kind
    from bio_waila.data.prepare import read_array
    p = tmp_path / "patient1_ED.hdf5"
    with h5py.File(p, "w") as f:
        f["tracks/track_0/data/image/values"] = np.random.default_rng(0).random((1, 12, 40, 40), dtype="float32")
        f["tracks/track_0/data/segmentation/values"] = np.ones((1, 12, 40, 40, 1), dtype=bool)
        f["tracks/track_0/data/image/coordinates"] = np.zeros((12, 40, 40, 3), dtype="float32")
    assert kind(p.name) == "volume" or kind(p.name) == "array"
    ims = list(read_array(p))
    assert ims and all(im.size == (40, 40) for im, _, _ in ims)


# ── rules ───────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v", ["MemBrain_seg_training_data", "membrain-seg", "Mem_Brain_seg", "MemBrain"])
def test_membrain_is_not_brain(classes, v):
    assert "body_part" not in text_hits(v, classes["body_part"])
    assert match_value(v, "path:d1", classes) == {}


def test_real_brain_words_still_match(classes):
    assert text_hits("Brain/T1", classes["body_part"])["body_part"]["brain"]
    assert text_hits("cerebral_ct", classes["body_part"])["body_part"]["brain"]
    assert "brain" not in normalize_text("MemBrain").split()


def test_column_keywords_need_a_word_start(classes):
    from bio_waila.data.rules import field_of_column
    assert field_of_column("hf:preview_slice") is None  # "view" inside "preview"
    assert field_of_column("csv:x.csv:View") == "submodality"
    assert match_value("2021", "csv:x.csv:profile", classes) == {}


H = lambda field, **k: {"mode": "hard", "values": k["v"]}


@pytest.mark.parametrize("reg,hit", [("eye", "retina"), ("abdomen", "liver"), ("head and neck", "brain")])
def test_child_body_part_is_refinement_not_conflict(reg, hit):
    r = resolve({"mode": "hard", "values": [reg]}, [("meta", {hit: True})], "body_part")
    assert "conflict" not in r and r["mode"] == "multi" and r["values"] == sorted([reg, hit])


def test_unrelated_body_part_is_still_a_conflict():
    r = resolve({"mode": "hard", "values": ["eye"]}, [("meta", {"liver": True})], "body_part")
    assert r["conflict"] == "meta:liver" and r["mode"] == "hard"
    r = resolve({"mode": "hard", "values": ["retina"]}, [("meta", {"eye": True})], "body_part")  # parent of the registry value
    assert r["conflict"] == "meta:eye"


def test_hit_matching_registry_value_stays_hard():
    r = resolve({"mode": "hard", "values": ["eye"]}, [("meta", {"eye": True, "retina": True})], "body_part")
    assert r["mode"] == "hard" and "conflict" not in r


@pytest.mark.parametrize("text", ["M1_U2OS", "HUVEC", "HeLa cells", "MCF7"])
def test_named_human_cell_lines_are_human(classes, text):
    assert text_hits(text, classes["body_part"])["specimen"] == {"human": True}


def test_generic_cell_line_does_not_override_species():
    assert text_hits("cell line")["specimen"] == {"cell line": True}
    r = resolve({"mode": "hard", "values": ["human"]}, [("path", {"cell line": True})], "specimen")
    assert "conflict" not in r and r["values"] == ["human"]
    r = resolve({"mode": "hard", "values": ["human"]}, [("path", {"mouse": True})], "specimen")
    assert r["conflict"] == "path:mouse"


def test_generic_ct_does_not_contradict_spiral_ct(classes):
    assert match_value("CT", "csv:x.csv:modality", classes).get("submodality") is None
    r = resolve({"mode": "hard", "values": ["spiral ct"]}, [("meta", {"conventional ct": True})], "submodality")
    assert "conflict" not in r
    r = resolve({"mode": "hard", "values": ["structural mri"]}, [("meta", {"conventional ct": True})], "submodality")
    assert r["conflict"] == "meta:conventional ct"


def test_skin_datasets_ignore_anatomical_site():
    skin = {"body_part": {"mode": "hard", "values": ["skin"]}, "submodality": {"mode": "hard", "values": ["dermoscopy"]}}
    assert is_skin_dataset(skin)
    for site in ("face", "thorax", "lower limbs"):
        r = resolve(skin["body_part"], [("meta", {site: True})], "body_part", labels=skin)
        assert "conflict" not in r and r["values"] == ["skin"]
    other = {"body_part": {"mode": "hard", "values": ["liver"]}}
    assert not is_skin_dataset(other)
    assert resolve(other["body_part"], [("meta", {"face": True})], "body_part", labels=other)["conflict"] == "meta:face"
    unset = {"body_part": {"mode": "masked", "values": []}, "submodality": {"mode": "hard", "values": ["clinical photography"]}}
    assert is_skin_dataset(unset)
