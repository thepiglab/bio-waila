"""Zip compression methods that Python's zipfile lacks.

Deflate64 (method 9, written by Windows Explorer / .NET for large archives; I-SPY2's 9 GB npz zip) is registered into
`zipfile` here with `inflate64` (already a dependency of py7zr, wheels for every platform). Importing this module is
enough; `enable_deflate64()` is idempotent. Anything still unsupported (PPMd, ...) falls back to the 7-Zip CLI:
`find_7z()` locates `7zz` / `7z` / `7za` / `7zr` on PATH (on the cluster: `module load 7zip/24.09`) and `extract_7z()`
extracts the wanted members with it.
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

DEFLATE64 = 9
SEVENZIP_NAMES = ("7zz", "7z", "7za", "7zr")


class _Deflate64Decompressor:
    """Adapter giving inflate64.Inflater the decompressobj interface zipfile expects."""

    def __init__(self):
        import inflate64
        self._inf = inflate64.Inflater()

    def decompress(self, data):
        return self._inf.inflate(data)

    @property
    def eof(self):
        return self._inf.eof


def enable_deflate64() -> bool:
    """Teach zipfile to read Deflate64 members. Returns False when inflate64 is not importable."""
    if getattr(zipfile, "_waila_deflate64", False):
        return True
    try:
        import inflate64  # noqa: F401
    except ImportError:
        return False
    check, get = zipfile._check_compression, zipfile._get_decompressor

    def _check_compression(ct):
        if ct != DEFLATE64:
            check(ct)

    def _get_decompressor(ct):
        return _Deflate64Decompressor() if ct == DEFLATE64 else get(ct)

    zipfile._check_compression, zipfile._get_decompressor = _check_compression, _get_decompressor
    zipfile._waila_deflate64 = True
    return True


def find_7z() -> str | None:
    for n in SEVENZIP_NAMES:
        if (p := shutil.which(n)):
            return p
    return None


def extract_7z(archive, dest, members=None, run=subprocess.run):
    """Extract `members` (default: everything) of `archive` into `dest` with the 7-Zip CLI."""
    exe = find_7z()
    if exe is None:
        raise RuntimeError(f"{Path(archive).name} uses a compression method Python's zipfile cannot read, and no 7-Zip "
                           f"binary ({'/'.join(SEVENZIP_NAMES)}) is on PATH; install 7-Zip (cluster: `module load 7zip`)")
    Path(dest).mkdir(parents=True, exist_ok=True)
    cmd = [exe, "x", "-y", "-bd", f"-o{dest}", str(archive), *(["--", *members] if members else [])]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(exe).name} failed on {Path(archive).name} (exit {r.returncode}): {(r.stderr or r.stdout)[-400:]}")


enable_deflate64()
