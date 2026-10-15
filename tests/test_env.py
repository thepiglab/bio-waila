import os

from bio_waila import load_env


def test_load_env_parses_and_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\n\nHF_TOKEN=hf_abc\nexport KAGGLE_API_TOKEN='KGAT_x'\nEMPTY=\nWAILA_T1=keep # note\nWAILA_T2=new\n")
    for k in ("HF_TOKEN", "KAGGLE_API_TOKEN", "EMPTY", "WAILA_T1"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WAILA_T2", "shell-wins")
    load_env(f)
    assert os.environ["HF_TOKEN"] == "hf_abc"
    assert os.environ["KAGGLE_API_TOKEN"] == "KGAT_x"
    assert "EMPTY" not in os.environ          # blank values are skipped, so the example file sets nothing
    assert os.environ["WAILA_T1"] == "keep"   # inline comment stripped
    assert os.environ["WAILA_T2"] == "shell-wins"


def test_missing_env_file_is_fine(tmp_path):
    load_env(tmp_path / "nope.env")
