"""bio_waila.data.eval_set: mmbu-public metadata → eval manifest in the training label format."""
import json

import pandas as pd
import pytest

from bio_waila.data import eval_set


@pytest.fixture
def eval_df(tmp_path, monkeypatch):
    pub, plan = tmp_path / "public", tmp_path / "plan"
    pub.mkdir(), plan.mkdir()
    rows = [{"file_name": "images/a.jpg", "task": "vqa|x", "unique_id": "u1", "metadata_modality": "computed tomography",
             "metadata_submodality": "conventional ct", "metadata_body_part": "Chest|Lung", "metadata_stain": "",
             "metadata_specimen": "Homo sapiens", "metadata_medical_domain": "radiology, urology"}]
    (pub / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(eval_set, "PUBLIC", pub)
    monkeypatch.setattr(eval_set, "PLAN", plan)
    eval_set.main()
    return pd.read_parquet(plan / "eval_public.parquet")


def test_eval_labels(eval_df):
    r = eval_df.iloc[0]
    assert r.modality_mode == "hard" and list(r.modality_values) == ["computed tomography"]
    assert r.body_part_mode == "multi" and list(r.body_part_values) == ["chest", "lung"]
    assert r.stain_mode == "masked"                      # empty cell = not annotated = skipped when scoring
    assert list(r.specimen_values) == ["human"]
    assert r.domain_mode == "multi" and len(r.domain_values) == 2


def test_eval_rows_carry_dataset_id_and_split(eval_df):
    assert set(eval_df.dataset_id) == {"mmbu-public"} and set(eval_df.split) == {"eval"}
