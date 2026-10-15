"""bio-waila: labels the MMBU Pass-2 biomedical-context fields ("What Am I Looking At")."""
import os
from pathlib import Path


def load_env(path=None):
    """Load KEY=VALUE lines from the repo's .env into os.environ. Variables already set in the environment win.
    No dependency on python-dotenv; supports comments, blank lines, `export KEY=...` and quoted values."""
    p = Path(path or os.environ.get("WAILA_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env")
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if key and val and key not in os.environ:
            os.environ[key] = val


load_env()  # every entry point imports bio_waila.*, so secrets are available before paths.py and download.py read them
