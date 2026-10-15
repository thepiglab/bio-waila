"""Every filesystem location the package uses, in one place. Override any of them with environment variables
(set in slurm/env.sh on the cluster)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]                  # .../bio-waila  (src/bio_waila/paths.py → repo root)
RESOURCES = REPO / "resources"                               # vocabulary, taxonomy, alias maps, paper Fig. 3 numbers
PLAN = Path(os.environ.get("WAILA_PLAN_DIR", REPO / "plan"))  # selection.jsonl, label_space.json, registry_fixes.json …
DATA = Path(os.environ.get("WAILA_DATA", REPO / "data"))       # gitignored working data
RAW = Path(os.environ.get("WAILA_RAW", DATA / "raw"))           # download.py output
PREP = Path(os.environ.get("WAILA_PREP", DATA / "prepared"))    # metalabel/prepare output (images + manifests)
CACHE = Path(os.environ.get("WAILA_CACHE", DATA / "cache"))     # local validation cache (size-capped, see data/cache.py)
# the public MMBU split (eval only) and the dataset registry live next to this repo in the mmbu/ workspace
PUBLIC = Path(os.environ.get("WAILA_PUBLIC", REPO.parent / "mmbu-public"))
REGISTRY = Path(os.environ.get("WAILA_REGISTRY", REPO.parent / "data_discovery" / "build" / "registry.json"))
