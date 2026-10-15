import pandas as pd

from bio_waila.data.manifest import load_manifest


def test_nc_layer_toggle(tmp_path):
    df = pd.DataFrame({"image": ["a", "b", "c"], "dataset_id": ["p1", "n1", "p2"], "split": ["train", "train", "val"],
                       "license_group": ["permissive", "nc", "permissive"]})
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    assert load_manifest(p, splits=("train",)).dataset_id.tolist() == ["p1"]
    assert load_manifest(p, include_nc=True, splits=("train",)).dataset_id.tolist() == ["p1", "n1"]
    assert len(load_manifest(p, include_nc=True)) == 3
