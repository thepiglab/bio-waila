"""Size-capped local working cache, for validating the pipeline on a laptop before using the cluster.

The cache lives at paths.CACHE (default <repo>/data/cache, gitignored):
  raw/<id>/        downloaded pilot sample (deleted after validation unless --keep-raw)
  work/            metalabel + prepare outputs (small: ≤ pilot-size JPEGs per dataset, manifests)
  reports/<id>.json  per-dataset validation report (written by validate.py)
  manifest.json    what is in the cache: per dataset status, bytes, who processed it, when

Rule: nothing new is downloaded while the cache is over its cap (default 30 GB, env WAILA_CACHE_CAP_GB) or the disk has
less than WAILA_MIN_FREE_GB (default 8 GB) free. Several
validation workers can share one cache: manifest updates take a file lock, and every worker checks the total size
before each download.

  uv run waila-cache status            # size vs cap, per-dataset table
  uv run waila-cache clean-raw         # delete every raw/<id> (keeps reports and small prepared samples)
  uv run waila-cache clean-raw --only a,b
"""
import argparse
import fcntl
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from bio_waila.paths import CACHE

CAP_GB = float(os.environ.get("WAILA_CACHE_CAP_GB", 30))
# never let the cache push the disk itself to the edge: keep at least this much free space on the volume
MIN_FREE_GB = float(os.environ.get("WAILA_MIN_FREE_GB", 8))


def dir_bytes(p: Path) -> int:
    total = 0
    stack = [p]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        pass
        except (FileNotFoundError, NotADirectoryError):
            pass
    return total


class CacheGuard:
    def __init__(self, root: Path = CACHE, cap_gb: float = CAP_GB):
        self.root = Path(root)
        self.cap = int(cap_gb * 2**30)
        for sub in ("raw", "work", "reports"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    # ── size ──
    def size(self) -> int:
        return dir_bytes(self.root)

    def disk_free(self) -> int:
        return shutil.disk_usage(self.root).free

    def has_room(self) -> bool:
        """Under the cache cap AND the disk keeps ≥ MIN_FREE_GB free (a laptop disk can fill before the cap does)."""
        return self.size() < self.cap and self.disk_free() > MIN_FREE_GB * 2**30

    def wait_for_room(self, timeout_s=3600, poll_s=30, log=print) -> bool:
        """Block until the cache is under its cap (another worker may be cleaning up). False on timeout."""
        t0 = time.time()
        while not self.has_room():
            if time.time() - t0 > timeout_s:
                return False
            log(f"cache {self.size() / 2**30:.1f} GB (cap {self.cap / 2**30:.0f} GB), disk free {self.disk_free() / 2**30:.1f} GB "
                f"(min {MIN_FREE_GB:.0f} GB): waiting for cleanup …")
            time.sleep(poll_s)
        return True

    # ── manifest (file-locked; safe across processes) ──
    @contextmanager
    def _locked(self):
        lock = self.root / ".manifest.lock"
        with open(lock, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def read(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def update(self, dataset_id: str, **fields):
        with self._locked():
            m = self.read()
            rec = m.get(dataset_id, {})
            rec.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
            m[dataset_id] = rec
            tmp = self.manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(m, indent=1, sort_keys=True))
            tmp.replace(self.manifest_path)

    # ── cleanup ──
    def remove_raw(self, dataset_id: str) -> int:
        p = self.root / "raw" / dataset_id
        n = dir_bytes(p)
        shutil.rmtree(p, ignore_errors=True)
        return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "clean-raw"])
    ap.add_argument("--only", default="")
    ap.add_argument("--root", default=str(CACHE))
    args = ap.parse_args()
    g = CacheGuard(Path(args.root))
    if args.cmd == "status":
        m = g.read()
        print(f"cache {g.root}: {g.size() / 2**30:.2f} GB of {g.cap / 2**30:.0f} GB cap; disk free {g.disk_free() / 2**30:.1f} GB "
              f"(min {MIN_FREE_GB:.0f} GB); {len(m)} datasets in manifest")
        states = {}
        for k, v in m.items():
            states[v.get("status", "?")] = states.get(v.get("status", "?"), 0) + 1
        print("by status:", states)
        raw = sorted(((dir_bytes(p), p.name) for p in (g.root / "raw").iterdir() if p.is_dir()), reverse=True)[:10]
        if raw:
            print("largest raw folders:", ", ".join(f"{n} {b / 2**30:.2f} GB" for b, n in raw))
    else:
        ids = [x for x in args.only.split(",") if x] or [p.name for p in (g.root / "raw").iterdir() if p.is_dir()]
        freed = sum(g.remove_raw(i) for i in ids)
        print(f"removed raw for {len(ids)} datasets, freed {freed / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
