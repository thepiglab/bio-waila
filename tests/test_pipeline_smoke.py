"""End-to-end smoke test on a tiny synthetic dataset, offline: metalabel profile → map → apply, then prepare_dataset.

Layout of the fake raw dataset (WAILA_RAW/<id>/):
    files/CT/ct_001.png ct_002.png            labelled by labels.csv (modality=CT, organ=Liver)
    files/MRI/mr_001.png mr_002.png           labelled by the MRI/ folder name and labels.csv (organ=Brain)
    files/lesions/mask/ct_001_mask.png        a mask: must be skipped
    files/__MACOSX/._ct_001.png               junk: must be skipped
    labels.csv                                file, modality, organ (dataset root, joined by file stem)
    files/sub-01_T1w.nii.gz                   BIDS suffix _T1w -> MRI / structural mri
    files/chest.dcm                           DICOM Modality=DX, BodyPartExamined=CHEST -> x-ray / chest x-ray
The registry label for the dataset is deliberately loose (candidate sets), so everything below is auto-labelled.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bio_waila.data import metalabel, prepare
from bio_waila.data.vocab import FIELDS

DID = "toy"
MODS = ["computed tomography", "magnetic resonance imaging", "x-ray imaging"]
BODY = ["liver", "brain", "chest"]


def noise(seed, shape=(64, 64)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def write_dicom(path, modality, body_part):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.BodyPartExamined = modality, body_part
    ds.Rows = ds.Columns = 64
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = (noise(7).astype(np.uint16) * 200).tobytes()
    ds.save_as(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    plan, raw, prep, public = tmp_path / "plan", tmp_path / "raw", tmp_path / "prepared", tmp_path / "mmbu-public"
    root = raw / DID / "files"
    for d in ("CT", "MRI", "lesions/mask", "__MACOSX"):
        (root / d).mkdir(parents=True)
    (public / "images").mkdir(parents=True)
    plan.mkdir()
    imgs = {"CT/ct_001.png": 1, "CT/ct_002.png": 2, "MRI/mr_001.png": 3, "MRI/mr_002.png": 4}
    for rel, seed in imgs.items():
        Image.fromarray(noise(seed)).save(root / rel)
    Image.fromarray(noise(5)).save(root / "lesions/mask/ct_001_mask.png")
    Image.fromarray(noise(6)).save(root / "__MACOSX/._ct_001.png")
    with open(raw / DID / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "modality", "organ"])
        w.writerows([["ct_001.png", "CT", "Liver"], ["ct_002.png", "CT", "Liver"],
                     ["mr_001.png", "MR", "Brain"], ["mr_002.png", "MR", "Brain"]])
    import nibabel as nib
    nib.save(nib.Nifti1Image(noise(8, (48, 48, 10)), np.eye(4)), str(root / "sub-01_T1w.nii.gz"))
    write_dicom(root / "chest.dcm", "DX", "CHEST")
    # a public-split image identical to ct_002.png: the contamination filter must drop it from training
    Image.fromarray(noise(2)).save(public / "images" / "pub_0001.png")
    (raw / DID / "_status.json").write_text(json.dumps({"state": "done"}))

    none = {"mode": "masked", "values": []}
    row = {"id": DID, "plan_images": 50, "split": "train", "group": DID, "exclude_paths": "",
           "labels": {"modality": {"mode": "candidates", "values": MODS}, "submodality": none,
                      "body_part": {"mode": "candidates", "values": BODY}, "stain": none,
                      "specimen": {"mode": "hard", "values": ["human"]}, "domain": {"mode": "multi", "values": ["radiology"]}}}
    (plan / "selection.jsonl").write_text(json.dumps(row) + "\n")
    space = {f: {"classes": c, "train_datasets": {}, "multilabel": f in ("body_part", "domain")}
             for f, c in {"modality": MODS, "submodality": ["structural mri", "chest x-ray", "conventional ct"], "body_part": BODY,
                          "stain": [], "specimen": ["human"], "domain": ["radiology"]}.items()}
    (plan / "label_space.json").write_text(json.dumps(space))
    monkeypatch.setattr(metalabel, "PLAN_DIR", plan)
    monkeypatch.setattr(metalabel, "PLAN", plan / "selection.jsonl")
    monkeypatch.setattr(prepare, "PLAN", plan)
    return dict(plan=plan, raw=raw, prep=prep, public=public, row=row)


def run_metalabel(monkeypatch, world, cmd):
    monkeypatch.setattr(sys, "argv", ["waila-metalabel", cmd, "--raw", str(world["raw"]), "--out", str(world["prep"]),
                                      "--plan", str(world["plan"] / "selection.jsonl")])
    metalabel.main()


def run_all(monkeypatch, world):
    for cmd in ("profile", "map", "apply"):
        run_metalabel(monkeypatch, world, cmd)
    pub = prepare.public_hashes(world["public"], world["prep"] / "_cache" / "dhash.npy")
    return prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(BODY), pub, seed=0)


def vals(r, f):
    return list(r[f"{f}_values"])


def test_profile_skips_masks_and_junk(monkeypatch, world):
    run_metalabel(monkeypatch, world, "profile")
    s = pd.read_parquet(world["prep"] / "metalabel" / DID / "samples.parquet")
    rels = set(s.rel)
    assert not any("mask" in r.lower() or "__MACOSX" in r for r in rels)
    assert len(rels) == 6                                   # 4 PNG + 1 NIfTI + 1 DICOM
    assert {"dicom:Modality", "dicom:BodyPartExamined"} <= set(s.columns)
    assert any(c.startswith("csv:labels.csv:") for c in s.columns)
    assert s.loc[s.rel.str.endswith("chest.dcm"), "dicom:BodyPartExamined"].iloc[0] == "CHEST"


def test_map_uses_columns_and_override_wins(monkeypatch, world):
    run_metalabel(monkeypatch, world, "profile")
    run_metalabel(monkeypatch, world, "map")
    vm = pd.read_csv(world["prep"] / "metalabel" / "value_map.csv")
    got = {(r.column, r.value): (r.field, r["class"]) for _, r in vm.iterrows() if r.used}
    assert got[("csv:labels.csv:modality", "CT")] == ("modality", "computed tomography")
    assert got[("csv:labels.csv:modality", "MR")] == ("modality", "magnetic resonance imaging")
    assert got[("csv:labels.csv:organ", "Liver")] == ("body_part", "liver")
    assert got[("dicom:Modality", "DX")] == ("modality", "x-ray imaging")

    (world["plan"] / "overrides.csv").write_text("dataset_id,column,value,field,class\n"
                                                  f"{DID},csv:labels.csv:organ,Liver,body_part,chest\n")
    run_metalabel(monkeypatch, world, "map")
    vm = pd.read_csv(world["prep"] / "metalabel" / "value_map.csv")
    row = vm[(vm.column == "csv:labels.csv:organ") & (vm.value == "Liver")].iloc[0]
    assert (row["class"], row.method) == ("chest", "override")


def test_apply_labels_every_sample(monkeypatch, world):
    for cmd in ("profile", "map", "apply"):
        run_metalabel(monkeypatch, world, cmd)
    lab = pd.read_parquet(world["prep"] / "labels" / f"{DID}.parquet").set_index("rel")
    ct = lab[lab.index.str.contains("ct_00")]
    mr = lab[lab.index.str.contains("mr_00")]
    assert len(lab) == 6 and len(ct) == 2 and len(mr) == 2
    for _, r in ct.iterrows():
        assert (r.modality_mode, vals(r, "modality")) == ("hard", ["computed tomography"])
        assert vals(r, "body_part") == ["liver"] and r.body_part_mode == "hard"
        assert r.modality_source == "meta" and r.modality_conf == pytest.approx(0.9)
    for _, r in mr.iterrows():
        assert vals(r, "modality") == ["magnetic resonance imaging"] and vals(r, "body_part") == ["brain"]
    nii = lab[lab.index.str.endswith("T1w.nii.gz")].iloc[0]
    assert vals(nii, "modality") == ["magnetic resonance imaging"] and nii.modality_mode == "hard"
    assert vals(nii, "submodality") == ["structural mri"] and nii.submodality_source == "meta"   # bids:suffix
    dcm = lab[lab.index.str.endswith("chest.dcm")].iloc[0]
    assert vals(dcm, "modality") == ["x-ray imaging"] and dcm.modality_source == "dicom"
    assert vals(dcm, "body_part") == ["chest"] and dcm.body_part_source == "dicom"
    assert dcm.submodality_mode == "masked"                 # DX alone does not name a submodality; registry has none either
    # registry-level hard/multi labels pass through untouched
    assert set(lab.specimen_mode) == {"hard"} and set(lab.specimen_source) == {"registry"}
    assert set(lab.domain_mode) == {"multi"}


def test_prepare_dataset_end_to_end(monkeypatch, world):
    res = run_all(monkeypatch, world)
    assert res["state"] == "ok" and res["media_files"] == 6
    assert res["near_mmbu_public"] == 1                      # ct_002.png equals the planted public image
    man = pd.read_parquet(world["prep"] / "manifest" / f"{DID}.parquet")
    assert not any("mask" in s or "MACOSX" in s for s in man.source)
    assert "CT/ct_002.png" not in set(man.source)
    for im in man.image:                                      # JPEGs on disk, <= 512 px
        assert Image.open(world["prep"] / im).size[0] <= prepare.LONG_SIDE
    assert set(man.dataset_id) == {DID} and set(man.split) == {"train"}
    assert set(FIELDS) <= {c.removesuffix("_mode") for c in man.columns if c.endswith("_mode")}

    by = {k: g for k, g in man.groupby(man.source.map(lambda s: Path(s).name.split(".")[0]))}
    ct, mr = by["ct_001"].iloc[0], by["mr_001"].iloc[0]
    assert vals(ct, "modality") == ["computed tomography"] and vals(ct, "body_part") == ["liver"]
    assert vals(mr, "modality") == ["magnetic resonance imaging"] and vals(mr, "body_part") == ["brain"]
    vol = by["sub-01_T1w"]                                    # volumes yield several slices, all with the BIDS label
    assert len(vol) == prepare.SLICES_PER_VOLUME
    assert all(vals(r, "submodality") == ["structural mri"] for _, r in vol.iterrows())
    dcm = by["chest"].iloc[0]
    assert vals(dcm, "modality") == ["x-ray imaging"] and vals(dcm, "body_part") == ["chest"]
    assert dcm.modality_source == "dicom"


def test_prepare_without_metalabel_falls_back_to_header_and_path_rules(monkeypatch, world):
    pub = prepare.public_hashes(world["public"], world["prep"] / "_cache" / "dhash.npy")
    res = prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(BODY), pub, seed=0)
    assert res["state"] == "ok"
    man = pd.read_parquet(world["prep"] / "manifest" / f"{DID}.parquet")
    dcm = man[man.source.str.endswith("chest.dcm")].iloc[0]
    assert vals(dcm, "modality") == ["x-ray imaging"] and dcm.modality_source == "dicom"
    nii = man[man.source.str.endswith("T1w.nii.gz")].iloc[0]
    assert vals(nii, "modality") == ["magnetic resonance imaging"]


def test_merge_writes_manifest_and_stats(monkeypatch, world):
    run_all(monkeypatch, world)
    prepare.merge(world["prep"])
    df = pd.read_parquet(world["prep"] / "manifest.parquet")
    assert df.dataset_id.nunique() == 1 and len(df) >= 6
    assert "Prepared Pass-2 training set" in (world["prep"] / "stats.md").read_text()


def test_not_downloaded_dataset_is_reported_not_crashed(world):
    (world["raw"] / DID / "_status.json").write_text(json.dumps({"state": "failed"}))
    res = prepare.prepare_dataset(world["row"], world["raw"], world["prep"], set(BODY), np.zeros(0, dtype=np.uint64), seed=0)
    assert res == {"id": DID, "state": "not downloaded"}


def test_apply_with_nothing_profiled_is_a_noop(monkeypatch, world):
    (world["prep"] / "metalabel").mkdir(parents=True)
    run_metalabel(monkeypatch, world, "apply")


def test_merge_with_no_manifests_is_a_noop(world):
    (world["prep"] / "manifest").mkdir(parents=True)
    prepare.merge(world["prep"])
