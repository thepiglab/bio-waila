"""bio_waila.eval.public + waila-eval: mmbu-public scoring (offline; a tiny fake public split, random tiny models)."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bio_waila.eval import public as pub  # noqa: E402
from bio_waila.train.labels import LabelSpace  # noqa: E402
from test_train import KINDS, SPACE, tiny_siglip_cfg, world  # noqa: E402,F401


@pytest.fixture
def fake_public(tmp_path, monkeypatch):
    from PIL import Image
    d = tmp_path / "mmbu-public"
    (d / "images").mkdir(parents=True)
    rows = []
    for i in range(4):
        Image.new("RGB", (32, 32), (60 * i, 40, 40)).save(d / "images" / f"{i}.jpg")
        rows.append({"file_name": f"images/{i}.jpg", "task": "vqa|x" if i < 2 else "report|y", "unique_id": f"u{i}",
                     "metadata_modality": "computed tomography", "metadata_submodality": "conventional ct",
                     "metadata_body_part": "Chest|Lung", "metadata_stain": "", "metadata_specimen": "Homo sapiens",
                     "metadata_medical_domain": "radiology"})
    (d / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(pub, "PUBLIC", d)
    monkeypatch.setattr(pub, "PLAN", tmp_path / "plan_pub")
    return d


def test_load_public_builds_parquet(fake_public):
    df, root = pub.load_public()
    assert len(df) == 4 and root == fake_public and (pub.PLAN / "eval_public.parquet").exists()
    assert pub.load_public(build_if_missing=False)[0].shape == df.shape      # now cached on disk


def test_load_public_missing_message(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PUBLIC", tmp_path / "nope")
    monkeypatch.setattr(pub, "PLAN", tmp_path / "plan_x")
    with pytest.raises(pub.PublicMissing, match="WAILA_PUBLIC"):
        pub.load_public()
    assert pub.try_load_public({}) is None                                   # trainers only warn
    assert pub.try_load_public({"eval": {"public": False}}) is None


def _hand_made():
    sp = LabelSpace(["modality", "body_part"], {"modality": ["a", "b", "c"], "body_part": ["x", "y", "z"]},
                    {"modality": False, "body_part": True})
    oh = lambda *i, n=3: torch.eye(n)[list(i)].sum(0) if i else torch.zeros(n)  # noqa: E731
    # img0: modality right, body_part (multi {x,y}) top-1 = y ∈ set → C 1.0
    # img1: modality wrong, body_part right → 0.5;  img2: body_part masked, modality right → 1.0;  img3: nothing annotated
    logits = {"modality": torch.tensor([[5., 0, 0], [0, 5., 0], [0, 0, 5.], [5., 0, 0]]),
              "body_part": torch.tensor([[0, 4., 0], [0, 0, 4.], [4., 0, 0], [4., 0, 0]])}
    modes = {"modality": torch.tensor([1, 1, 1, 0]), "body_part": torch.tensor([2, 1, 0, 0])}
    tgts = {"modality": torch.stack([oh(0), oh(0), oh(2), oh()]), "body_part": torch.stack([oh(0, 1), oh(2), oh(), oh()])}
    df = pd.DataFrame({"task": ["t1", "t1", "t2", "t2"], "unique_id": list("abcd"), "image": list("wxyz"),
                       "modality_values": [["a"], ["a"], ["c"], []]})
    return sp, logits, modes, tgts, df


def test_score_public_context_score():
    sp, logits, modes, tgts, df = _hand_made()
    r = pub.score_public(logits, modes, tgts, df, sp)
    assert r["C"] == pytest.approx((1.0 + 0.5 + 1.0) / 3) and r["n_scored"] == 3 and r["n_images"] == 4
    assert r["C_by_task"]["t1"]["C"] == pytest.approx(0.75) and r["C_by_task"]["t2"]["C"] == 1.0
    assert r["C_by_modality"]["a"]["C"] == pytest.approx(0.75) and r["C_by_modality"]["c"]["C"] == 1.0
    assert "unannotated" not in r["C_by_modality"]                           # never scored
    assert r["field_acc"]["modality"]["acc"] == pytest.approx(2 / 3) and r["field_acc"]["modality"]["n"] == 3
    assert set(r["metrics_by_task"]) >= {"modality", "body_part", "mean_acc_macro_ds"} and "modality" in r["metrics_pooled"]


def test_write_public_report(tmp_path, fake_public):
    sp, logits, modes, tgts, df = _hand_made()
    res = pub.score_public(logits, modes, tgts, df, sp)
    tab = pub.predictions_table(logits, modes, tgts, df, sp)
    p = pub.write_public_report(tmp_path / "run", "runA", res, tab)
    assert json.loads(p.read_text())["C"] == pytest.approx(res["C"])
    back = pd.read_parquet(tmp_path / "run" / "public_predictions.parquet")
    assert list(back.modality_pred) == ["a", "b", "c", "a"] and list(back.modality_correct[:3]) == [True, False, True]
    assert pd.isna(back.body_part_correct[2]) and list(back.body_part_target[1]) == ["z"]
    pub.write_public_report(tmp_path / "run", "runB", res, tab)
    lines = (pub.PLAN / "public_looks.jsonl").read_text().splitlines()
    assert [json.loads(x)["run"] for x in lines] == ["runA", "runB"] and "C" in json.loads(lines[0])


def _tiny_public_manifest(world, fake_public):
    """eval_public.parquet in the tiny test label space (the real vocabulary is irrelevant here)."""
    rows = []
    for i in range(4):
        mod, sub, bps, stain, dom, _ = KINDS[i % 3]
        r = {"image": f"images/{i}.jpg", "dataset_id": "mmbu-public", "split": "eval", "task": "vqa" if i < 2 else "report",
             "unique_id": f"u{i}"}
        for f, v, m in (("modality", [mod], "hard"), ("submodality", [sub], "hard"), ("body_part", bps, "multi"),
                        ("stain", [stain] if stain else [], "hard" if stain else "masked"),
                        ("specimen", [], "masked"), ("domain", dom, "multi")):
            r[f"{f}_mode"], r[f"{f}_values"] = m, v
        rows.append(r)
    pub.PLAN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(pub.PLAN / "eval_public.parquet")


def test_clm_trainer_scores_public_and_waila_eval(world, fake_public):
    import yaml
    from bio_waila.eval import run
    from bio_waila.train import train_clm
    _tiny_public_manifest(world, fake_public)
    cfg = tiny_siglip_cfg(world)
    p = world / "clm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    rep = train_clm.main(["--config", str(p), "--fold", "0"])
    out = world / "runs" / "fold0"
    assert 0 <= rep["public"]["C"] <= 1 and (out / "public_metrics.json").exists()
    assert len(pd.read_parquet(out / "public_predictions.parquet")) == 4
    # standalone CLI on the saved checkpoint reproduces the trainer's number
    res = run.main(["--run", str(out), "--out", str(world / "evalout")])
    assert res["C"] == pytest.approx(rep["public"]["C"], abs=1e-4)
    assert (world / "evalout" / "public_metrics.json").exists()
    assert run.main(["--run", str(out), "--out", str(world / "evalout2"), "--limit", "2"])["n_images"] == 2
    assert len((pub.PLAN / "public_looks.jsonl").read_text().splitlines()) == 3


def test_trainer_skips_when_public_missing(world, tmp_path):
    import yaml
    from bio_waila.train import train_clm
    p = world / "clm.yaml"
    p.write_text(yaml.safe_dump(tiny_siglip_cfg(world)))
    assert "public" not in train_clm.main(["--config", str(p), "--fold", "0"])   # warns, does not crash


def test_waila_eval_unfinished_run(tmp_path, capsys):
    from bio_waila.eval import run
    (tmp_path / "r").mkdir()
    assert run.main(["--run", str(tmp_path / "r")]) is None
    assert "not supported yet" in capsys.readouterr().out


def test_jev_trainer_and_waila_eval_fake_backend(world, fake_public):
    import yaml
    from bio_waila.eval import run
    from bio_waila.train import train_jev
    _tiny_public_manifest(world, fake_public)
    cfg = {"out": str(world / "runs_jev"),
           "data": {"manifest": str(world / "prep" / "manifest.parquet"), "image_root": str(world / "prep"),
                    "label_space": str(world / "plan" / "label_space.json"), "folds_path": str(world / "plan" / "folds.json"),
                    "folds_k": 3},
           "model": {"kind": "fake"},
           "train": {"batch": 8, "workers": 0, "epochs": 1, "examples_per_epoch": 64, "lr": 5e-2, "k_max": 3,
                     "eval_per_dataset": 2, "log_every": 5}}
    p = world / "jev.yaml"
    p.write_text(yaml.safe_dump(cfg))
    rep = train_jev.main(["--config", str(p), "--fold", "0"])
    out = world / "runs_jev" / "F2_fold0"
    assert rep["public"]["n_images"] == 4 and (out / "public_predictions.parquet").exists()
    res = run.main(["--run", str(out), "--out", str(world / "jev_eval")])
    assert res["C"] == pytest.approx(rep["public"]["C"], abs=1e-4)


def test_clm_llm_public_from_cached_embeddings(world, fake_public):
    import yaml
    from bio_waila.eval import run
    from bio_waila.train import train_clm_llm
    _tiny_public_manifest(world, fake_public)
    cfg = {"out": str(world / "runs_llm"), "cache": str(world / "cache_llm"),
           "data": {"manifest": str(world / "prep" / "manifest.parquet"), "image_root": str(world / "prep"),
                    "label_space": str(world / "plan" / "label_space.json"), "folds_path": str(world / "plan" / "folds.json"),
                    "folds_k": 3, "eval_public": str(pub.PLAN / "eval_public.parquet"), "public_root": str(fake_public)},
           "encoder": {"kind": "fake"}, "heads": {"width": 64, "d_out": 32},
           "train": {"batch": 16, "epochs": 3, "epoch_size": 64, "lr": 3e-3}}
    p = world / "llm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    train_clm_llm.main(["embed", "--config", str(p)])
    rep = train_clm_llm.main(["train", "--config", str(p), "--fold", "0"])
    assert rep["public"]["n_images"] == 4
    res = run.main(["--run", str(world / "runs_llm" / "fold0"), "--out", str(world / "llm_eval")])
    assert res["C"] == pytest.approx(rep["public"]["C"], abs=1e-4)
