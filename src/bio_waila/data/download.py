#!/usr/bin/env python3
"""Fetch a bounded sample of every dataset in plan/selection.jsonl. Meant for a cluster; resumable and shardable.

Each source handler only *lists* files (name, size, url). One shared sampler then decides what to fetch:
  • loose media files     → a random, folder-spread sample of about plan_images × OVERSAMPLE files
  • archive ≤ budget      → downloaded whole (prepare.py extracts it)
  • zip   > budget        → only the sampled members, read with HTTP range requests (remotezip)
  • tar.* > budget        → streamed, keeping media members until the quota is met, then the connection is closed
  • small metadata files  → csv/json/… ≤ 50 MB, kept for later image-level labels
Hugging Face datasets are read with `datasets` streaming, so only the sampled rows are transferred.

Output per dataset: OUT/<id>/ with the files, `_status.json` (state, counts, bytes, notes) and `_files.jsonl`.
States: done · partial (fewer files than wanted) · manual (needs a human: login, terms, unsupported host) · too_big ·
error · skipped. Re-running skips `done`/`manual`; pass --retry to redo those too.

Credentials come from the environment (the repo's .env is loaded automatically, see .env.example): HF_TOKEN (optional,
raises rate limits) and, for Kaggle routes, KAGGLE_API_TOKEN or ~/.kaggle/access_token (KGAT_… API token, Bearer) or the legacy kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY.
Nothing is ever written into this file.

Examples
  uv run waila-download --dry-run --only cystods,cradi              # list sources and estimate, transfer nothing
  uv run waila-download --out data/raw --shard 3/16
"""
import argparse
import csv
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bio_waila.data.files import MEDIA_EXT, ext, is_mask, kind

from bio_waila.paths import PLAN, RAW
OVERSAMPLE = 1.4          # extra files to absorb unreadable files, masks that slipped through and dedup losses
SLICES_PER_VOLUME = 4     # prepare.py takes this many slices per 3D volume
META_MAX = 50 * 2**20
UA = "bio-waila/0.1 (research; contact via github)"


def session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    retry = Retry(total=8, backoff_factor=2, backoff_max=120, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


S = session()


class Manual(Exception):
    """The dataset needs a human step (login, terms, unsupported host)."""


@dataclass
class Entry:
    name: str
    size: int            # bytes; -1 if unknown
    url: str
    auth: tuple | None = None
    headers: dict = field(default_factory=dict)


# ── listing handlers: return list[Entry] ────────────────────────────────────────────────────────────────────────
def list_zenodo(rec):
    r = S.get(f"https://zenodo.org/api/records/{rec}", timeout=60)
    if r.status_code in (403, 404, 410):
        raise Manual(f"zenodo record {rec}: HTTP {r.status_code} (restricted, deleted or wrong id)")
    r.raise_for_status()
    j = r.json()
    if j.get("metadata", {}).get("access_right") not in (None, "open"):
        raise Manual(f"zenodo record {rec} is {j['metadata']['access_right']}")
    return [Entry(f["key"], int(f.get("size", -1)), f["links"]["self"]) for f in j.get("files", [])]


def list_figshare(aid):
    r = S.get(f"https://api.figshare.com/v2/articles/{aid}", timeout=60)
    if r.status_code == 404:  # maybe a collection
        arts = S.get(f"https://api.figshare.com/v2/collections/{aid}/articles?page_size=100", timeout=60)
        if arts.status_code != 200:
            raise Manual(f"figshare {aid}: not an article or collection")
        out = []
        for a in arts.json():
            out += list_figshare(a["id"])
        return out
    r.raise_for_status()
    files = r.json().get("files", [])
    if not files:
        raise Manual(f"figshare {aid}: no public files (embargoed or link-only)")
    return [Entry(f["name"], int(f.get("size", -1)), f["download_url"]) for f in files]


def list_mendeley(did):
    base = "https://data.mendeley.com/public-api/datasets"
    meta = S.get(f"{base}/{did}", timeout=60)
    if meta.status_code != 200:  # Mendeley Data blocks scripted clients (403); a browser download works
        raise Manual(f"mendeley {did}: HTTP {meta.status_code}. In a browser open https://data.mendeley.com/datasets/{did}, "
                     f"click 'Download All', and put the zip in <raw>/<id>/archives/ (then re-run)")
    ver = meta.json().get("version", 1)
    out = []

    def walk(folder, prefix):
        fs = S.get(f"{base}/{did}/files", params={"folder_id": folder, "version": ver}, timeout=60)
        if fs.status_code == 200:
            for f in fs.json():
                cd = f.get("content_details", {})
                out.append(Entry(prefix + f["filename"], int(cd.get("size", f.get("size", -1)) or -1), cd.get("download_url", "")))
        sub = S.get(f"{base}/{did}/folders/{ver}", timeout=60)
        if sub.status_code == 200 and folder == "root":
            for d in sub.json():
                if d.get("parent_id") in (None, "root") or folder == "root":
                    walk(d["id"], prefix + d.get("name", d["id"]) + "/")
    walk("root", "")
    out = [e for e in out if e.url]
    if not out:
        raise Manual(f"mendeley {did}: no downloadable files via public API")
    return out


def list_dataverse(url):
    m = re.search(r"https?://([^/]+)/.*?(doi:[^&#\s]+)", url, re.I)
    if not m:
        raise Manual("dataverse url without persistentId: " + url)
    host, pid = m.group(1), m.group(2)
    r = S.get(f"https://{host}/api/datasets/:persistentId/", params={"persistentId": pid}, timeout=60)
    if r.status_code != 200:
        raise Manual(f"dataverse {pid}: HTTP {r.status_code}")
    out = []
    for f in r.json()["data"]["latestVersion"]["files"]:
        if f.get("restricted"):
            continue
        df = f["dataFile"]
        name = (f.get("directoryLabel", "") + "/" if f.get("directoryLabel") else "") + df["filename"]
        out.append(Entry(name, int(df.get("filesize", -1)), f"https://{host}/api/access/datafile/{df['id']}"))
    return out


def list_openneuro(ds):
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": f"{ds}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = S.get("https://s3.amazonaws.com/openneuro.org", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        nsx = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", nsx):
            key = c.find("s:Key", nsx).text
            if "/anat/" in key and key.endswith((".nii.gz", ".nii")) and "derivatives" not in key:  # structural scans only
                out.append(Entry(key, int(c.find("s:Size", nsx).text), f"https://s3.amazonaws.com/openneuro.org/{key}"))
        t = root.find("s:NextContinuationToken", nsx)
        if t is None:
            break
        token = t.text
    return out


def kaggle_auth():
    """(auth, headers) for Kaggle. Prefers the newer API token (KGAT_…: env KAGGLE_API_TOKEN or ~/.kaggle/access_token,
    sent as a Bearer header); falls back to the legacy username/key pair (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."""
    tok = os.environ.get("KAGGLE_API_TOKEN")
    p = Path.home() / ".kaggle" / "access_token"
    if not tok and p.exists():
        tok = p.read_text().strip()
    if tok:
        return None, {"Authorization": f"Bearer {tok}"}
    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (u and k):
        p = Path.home() / ".kaggle" / "kaggle.json"
        if p.exists():
            j = json.loads(p.read_text())
            u, k = j.get("username"), j.get("key")
    if not (u and k):
        raise Manual("kaggle credentials missing (~/.kaggle/access_token, KAGGLE_API_TOKEN, or kaggle.json)")
    return (u, k), {}


KAGGLE_WHOLE_MAX = int(float(os.environ.get("WAILA_KAGGLE_WHOLE_GB", 2)) * 2**30)


def list_kaggle(slug, max_files=50_000):
    """Per-file listing (so the sampler fetches only the files it picks), plus the whole-dataset zip as a fallback."""
    from urllib.parse import quote
    auth, hdr = kaggle_auth()
    v = S.get(f"https://www.kaggle.com/api/v1/datasets/view/{slug}", auth=auth, headers=hdr, timeout=60)
    if v.status_code != 200:
        raise Manual(f"kaggle {slug}: HTTP {v.status_code}")
    size = int(v.json().get("totalBytes") or -1)
    whole = Entry(slug.replace("/", "__") + ".zip", size, f"https://www.kaggle.com/api/v1/datasets/download/{slug}", auth=auth, headers=hdr)
    if 0 <= size <= KAGGLE_WHOLE_MAX:  # small dataset: one zip request beats hundreds of rate-limited per-file requests
        return [whole]
    out, token = [], None
    while len(out) < max_files:
        params = {"pageSize": 200, **({"pageToken": token} if token else {})}  # the API caps pages at 200
        for attempt in range(6):  # Kaggle throttles listings (HTTP 429) after a few pages
            r = S.get(f"https://www.kaggle.com/api/v1/datasets/list/{slug}", params=params, auth=auth, headers=hdr, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(min(60, 5 * 2**attempt))
        if r.status_code != 200:
            break
        time.sleep(1.0)
        j = r.json()
        for f in j.get("datasetFiles", []):
            name = f.get("name") or f.get("nameNullable")
            if not name or "__MACOSX" in name:
                continue
            out.append(Entry(name, int(f.get("totalBytes") or f.get("totalBytesNullable") or -1),
                             f"https://www.kaggle.com/api/v1/datasets/download/{slug}/{quote(name, safe='')}", auth=auth, headers=hdr))
        token = j.get("nextPageTokenNullable") or j.get("nextPageToken")
        if not token:
            break
    if len(out) >= max_files:  # listing is capped: sample members of the whole zip instead (range reads over all files)
        return [whole]
    # whole-dataset zip: used only if the per-file listing has no media (e.g. a single archive inside)
    out.append(whole)
    return out


def list_direct(url):
    h = S.head(url, allow_redirects=True, timeout=60)
    size = int(h.headers.get("Content-Length", -1)) if h.ok else -1
    if size < 0:  # pre-signed redirect targets often refuse HEAD: ask for one byte and read Content-Range
        try:
            with S.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as g:
                cr = g.headers.get("Content-Range", "")
                size = int(cr.rsplit("/", 1)[-1]) if "/" in cr and cr.rsplit("/", 1)[-1].isdigit() else -1
        except Exception:
            pass
    name = re.sub(r"[?#].*$", "", url.rstrip("/").split("/")[-1]) or "download"
    return [Entry(name, size, url)]


def resolve_doi(url):
    """Follow a DOI to its landing page and dispatch to a handler we know."""
    r = S.get(url, allow_redirects=True, timeout=60, stream=True)
    final = r.url
    r.close()
    if (m := re.search(r"zenodo\.org/(?:records?|record)/(\d+)", final)):
        return "zenodo", m.group(1)
    if ("figshare" in final or "rdr.ucl.ac.uk" in final) and (m := re.search(r"(?<!\d)(\d{6,})(?!\d)", final.split("figshare")[-1] if "figshare" in final else final)):
        return "figshare", m.group(1)  # UCL RDR is a figshare instance; its article ids work with the figshare API
    if (m := re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)", final)):
        return "mendeley", m.group(1)
    if "persistentId=" in final or "/dataset.xhtml" in final:
        return "dataverse", final
    raise Manual(f"DOI resolves to an unsupported host: {final}")


LISTERS = {"zenodo": list_zenodo, "figshare": list_figshare, "mendeley": list_mendeley, "dataverse": list_dataverse,
           "openneuro": list_openneuro, "kaggle": list_kaggle, "direct": list_direct}


# ── fetching ────────────────────────────────────────────────────────────────────────────────────────────────────
def get_file(e: Entry, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (e.size < 0 or dest.stat().st_size == e.size):
        return dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    for attempt in range(6):  # long downloads drop mid-stream (IncompleteRead): resume with a Range request
        have = tmp.stat().st_size if tmp.exists() else 0
        hdr = {**(e.headers or {}), **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with S.get(e.url, auth=e.auth, headers=hdr, stream=True, timeout=120) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the Range: start over
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if e.size < 0 or tmp.stat().st_size >= e.size:
                break
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as ex:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    tmp.rename(dest)
    if kind(dest.name) != "archive":  # Kaggle sometimes serves a single file wrapped in a zip
        with open(dest, "rb") as fh:
            magic = fh.read(4)
        if magic == b"PK\x03\x04":
            import zipfile
            try:
                with zipfile.ZipFile(dest) as z:
                    members = [m for m in z.infolist() if not m.is_dir()]
                    if len(members) == 1:
                        data = z.read(members[0])
                        dest.write_bytes(data)
            except zipfile.BadZipFile:
                pass
    return dest.stat().st_size


def spread_sample(items, n, key, seed):
    """Random sample that spreads across parent folders (class folders, patients) instead of taking one folder."""
    rng = random.Random(seed)
    by_dir = {}
    for it in items:
        by_dir.setdefault(str(Path(key(it)).parent), []).append(it)
    for v in by_dir.values():
        rng.shuffle(v)
    dirs = list(by_dir.values())
    rng.shuffle(dirs)
    out = []
    while len(out) < n and dirs:
        dirs = [d for d in dirs if d]
        for d in dirs:
            if d and len(out) < n:
                out.append(d.pop())
    return out


WHOLE_ZIP_MAX = 200 * 2**20  # zips above this are sampled member by member when only a small share of them is needed
RANGE_SHARE_MAX = 0.3        # ... i.e. when the sampled members are < 30% of the zip's bytes


def sample_zip_remote(e: Entry, dest: Path, n, seed, log, budget=None, max_share=None):
    """Extract a spread sample of media members through HTTP range reads. With max_share, return None (nothing fetched)
    when the sample would be more than that share of the zip: downloading it whole is then cheaper."""
    from remotezip import RemoteZip
    url = e.url
    if e.auth or e.headers:  # resolve redirect to a signed URL so range requests do not need auth
        r = S.get(url, auth=e.auth, headers=e.headers, allow_redirects=False, timeout=60)
        url = r.headers.get("Location", url)
    for attempt in range(6):  # dataverse.no answers random 403s to range requests, also when opening the zip
        try:
            rz = RemoteZip(url, session=S)
            break
        except Exception as ex:
            if attempt == 5 or not re.search(r"\b(403|429|503)\b", str(ex)):
                raise
            time.sleep(20)
    with rz as z:
        infos = [i for i in z.infolist() if not i.is_dir() and not excluded(i.filename)]
        media = [i for i in infos if kind(i.filename) in ("image", "volume", "video", "wsi", "array")]
        if not media:  # extension-less DICOM members
            media = [i for i in infos if kind(i.filename) == "other" and ext(i.filename) == "" and i.file_size > 10_000]
        meta = sorted((i for i in infos if kind(i.filename) == "meta" and i.file_size <= META_MAX),
                      key=lambda i: (ext(i.filename) not in (".csv", ".tsv", ".tab", ".xlsx", ".xls"), -i.file_size))
        # range reads are cheap: take 2x so blank frames and near-duplicates dropped by prepare still leave n images
        # range reads are cheap: take 2x so blank frames and near-duplicates dropped by prepare still leave n images
        pick = spread_sample(media, min(len(media), 2 * n) if max_share is not None else n, lambda i: i.filename, seed)
        by_name = {i.filename: i for i in infos}

        def unit(i):  # a MetaImage header travels with its data file, and both count toward the budget
            u = [i]
            if i.filename.lower().endswith(".mhd"):
                u += [by_name[i.filename[:-4] + x] for x in (".raw", ".zraw", ".RAW", ".img") if i.filename[:-4] + x in by_name]
            return u
        units = [unit(i) for i in pick if not i.filename.lower().endswith((".raw", ".zraw", ".img"))]
        if budget:
            acc, keep = 0, []
            for u in units:
                sz = sum(i.file_size for i in u)
                if acc + sz > budget:  # budget on extracted size (what lands on disk)
                    break
                keep.append(u)
                acc += sz
            units = keep
        pick = [i for u in units for i in u]
        if max_share is not None and (not pick or sum(i.compress_size for i in pick) > max_share * max(e.size, 1)):
            return None
        for i in pick + meta[:50]:
            for attempt in range(6):  # some hosts (dataverse.no) answer 403 to bursts of range requests: back off
                try:
                    z.extract(i, dest / "zip_members" / Path(e.name).stem)
                    break
                except Exception as ex:
                    if attempt == 5 or not re.search(r"\b(403|429|503)\b", str(ex)):
                        raise
                    time.sleep(20)
            log.append({"src": e.name, "member": i.filename, "bytes": i.compress_size})
        # the 2x spare members are insurance for prepare, not extra quota: count at most n toward the dataset's target
        # bytes = extracted size (what lands on disk; the caller's budget is about disk, and .raw/.nrrd members are
        # often stored uncompressed or expand a lot)
        return min(len(units), n), sum(i.file_size for i in pick)


TAR_SCAN_FACTOR = 10  # tars can only be read in order: reservoir-sample over the first 10x n media members


def sample_tar_stream(e: Entry, dest: Path, n, log, budget=None):
    """Spread sample from a streamed tar: a reservoir of n media members over the first TAR_SCAN_FACTOR * n members
    (or until `budget` bytes have streamed past). Taking simply the first n gives one patient / class / video
    (cross-choleact: 18 clips, all one patient)."""
    rng = random.Random(e.name)
    kept, seen, streamed = [], 0, 0  # kept: [(path, size, log record)]
    base = dest / "tar_members" / Path(e.name).name.split(".")[0]
    with S.get(e.url, auth=e.auth, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=r.raw, mode="r|*") as t:
            for m in t:
                if not m.isfile() or excluded(m.name):
                    continue
                k = kind(m.name)
                is_media = k in ("image", "volume", "video", "wsi", "array")
                if not is_media and not (k == "meta" and m.size <= META_MAX):
                    continue
                slot = None
                if is_media:
                    seen += 1
                    streamed += m.size
                    if len(kept) < n:
                        slot = len(kept)
                    elif (j := rng.randrange(seen)) < n:
                        slot = j
                    if slot is None:
                        if seen >= TAR_SCAN_FACTOR * n or (budget and streamed >= budget):
                            break  # closing the response stops the transfer
                        continue
                out = base / m.name.lstrip("/")
                out.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)
                rec = {"src": e.name, "member": m.name, "bytes": m.size}
                if slot is None:  # metadata: always kept
                    log.append(rec)
                    continue
                if slot < len(kept):  # replace an earlier pick
                    old = kept[slot][0]
                    if old != out:
                        old.unlink(missing_ok=True)
                    kept[slot] = (out, m.size, rec)
                else:
                    kept.append((out, m.size, rec))
                if seen >= TAR_SCAN_FACTOR * n or (budget and streamed >= budget):
                    break
    log += [k_[2] for k_ in kept]
    return len(kept), sum(k_[1] for k_ in kept)


_ARCH_BAD = re.compile(r"(model|weight|ckpt|checkpoint|pretrained|code|src|script|result|plot|fold|pred|output|detection|"
                       r"feature|embedding|mask|\blabels?\b|_labels?\b|seg|annot|doc|supplement|readme|spss|database|xls)", re.I)
_ARCH_GOOD = re.compile(r"(image|img|data|png|jpe?g|dicom|dcm|nii|raw|frame|photo|slide|tile|patch|train|test|scan)", re.I)


def archive_rank(e):
    """Prefer image archives; push model weights, code, results, annotation bundles and tiny (<5 MB) zips to the back."""
    return (bool(_ARCH_BAD.search(e.name)), 0 <= e.size < 5 * 2**20, not _ARCH_GOOD.search(e.name), -max(e.size, 0))


def meta_rank(e):
    """Prefer real tables (csv/tsv/xlsx) over per-image json sidecars, bigger tables first."""
    return (ext(e.name) not in (".csv", ".tsv", ".tab", ".xlsx", ".xls", ".parquet"), -max(e.size, 0))


EXCLUDE = None  # per-dataset regex (row["exclude_paths"]), set by run_one; applied to listings AND archive members


def excluded(name):
    return bool(EXCLUDE and re.search(EXCLUDE, name))


def count_media_in_archive(path: Path, default):
    """Images/volumes (not masks) inside a downloaded zip; tar/rar/7z return `default` (not cheaply countable)."""
    import zipfile
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                return sum(1 for m in z.infolist() if not m.is_dir() and kind(m.filename) in ("image", "volume", "video", "wsi", "array"))
    except Exception:
        pass
    return default


def fetch_entries(entries, dest: Path, want, budget, seed, dry, log):
    """Shared sampler. Returns (n_media, bytes, strategy, notes)."""
    entries = [e for e in entries if not excluded(e.name)]
    media = [e for e in entries if kind(e.name) in ("image", "volume", "video", "wsi", "array")]
    if not media:  # extension-less members are often DICOM (prepare.py checks the DICM magic)
        media = [e for e in entries if kind(e.name) == "other" and ext(e.name) == "" and e.size > 10_000]
    archives = sorted((e for e in entries if kind(e.name) == "archive"), key=archive_rank)
    clean = [e for e in archives if not re.search(r"mask|\blabels?\b|_labels?\b|(^|[_\-])seg", e.name, re.I)]
    archives = clean or archives  # a dataset whose only archive is "xyz_seg.zip" still has its images inside
    meta = sorted((e for e in entries if kind(e.name) == "meta" and 0 <= e.size <= META_MAX), key=meta_rank)
    notes = []
    if media and (len(media) >= max(10, want // 4) or not archives):
        if len(media) < want:
            notes.append(f"only {len(media)} loose media files listed")
        pick = spread_sample(media, want, lambda e: e.name, seed)
        by_name = {e.name: e for e in entries}
        for e in list(pick):  # MetaImage headers need their data file next to them
            if e.name.lower().endswith(".mhd"):
                for ext_ in (".raw", ".zraw", ".RAW"):
                    side = by_name.get(e.name[:-4] + ext_)
                    if side and side not in pick:
                        pick.append(side)
        size = sum(max(e.size, 0) for e in pick)
        if size > budget:  # very large files (WSIs, 4D volumes): keep as many as fit
            pick, acc = [p for p in sorted(pick, key=lambda e: e.size)], 0
            pick = [p for p in pick if (acc := acc + max(p.size, 0)) <= budget]
            notes.append(f"budget limits to {len(pick)} files")
        if dry:
            return len(pick), sum(max(e.size, 0) for e in pick), "files", notes
        workers = 1 if any("kaggle.com" in e.url for e in pick) else 8  # Kaggle rate-limits parallel file requests
        if any(e.size < 0 for e in pick):  # sizes unknown: download one by one and stop at the byte budget
            sizes, total = [], 0
            for e in pick:
                if total >= budget:
                    notes.append(f"stopped at the {budget / 2**30:.1f} GB budget after {len(sizes)} files")
                    break
                sizes.append(get_file(e, dest / "files" / e.name))
                total += sizes[-1]
            pick = pick[: len(sizes)]
        else:
            with ThreadPoolExecutor(workers) as ex:
                sizes = list(ex.map(lambda e: get_file(e, dest / "files" / e.name), pick))
        for e in meta[:50]:
            get_file(e, dest / "files" / e.name)
        log += [{"src": e.name, "bytes": s} for e, s in zip(pick, sizes)]
        return len(pick), sum(sizes), "files", notes
    if not archives:
        if meta:
            raise Manual("record lists only metadata files (" + ", ".join(e.name for e in meta[:5]) + "); images are hosted elsewhere")
        return 0, 0, "none", ["no media files or archives listed"]
    # archives: image-looking archives first (archive_rank), then by size
    got, nbytes, strat, left = 0, 0, [], budget
    # several comparable image archives (often one per class: mll23 has 18 class zips, bicervi 9) are shards: take a
    # share from each instead of filling the whole quota from the first one
    shards = [a for a in archives if not _ARCH_BAD.search(a.name) and a.size >= 5 * 2**20]
    per = math.ceil(want / len(shards)) if len(shards) >= 3 else want
    if per < want:
        notes.append(f"{len(shards)} archive shards: ~{per} images from each")
    for e in archives:
        if got >= want:
            break
        tgt = min(per, want - got) if e in shards else want - got
        e_ext = ext(e.name)
        big = e.size > (20 * 2**20 if per < want else WHOLE_ZIP_MAX)
        if e_ext == ".zip" and big and not dry:  # a few members of a big zip: range reads, not 2 GB
            try:
                r = sample_zip_remote(e, dest, tgt, seed, log, budget=left, max_share=RANGE_SHARE_MAX)
            except Exception as ex:
                r = None
                notes.append(f"zip-range failed on {e.name} ({type(ex).__name__}); trying whole")
            if r is not None and r[0] > 0:
                strat.append("zip-range")
                got, nbytes, left = got + r[0], nbytes + r[1], left - r[1]
                continue
        if 0 <= e.size <= left:
            strat.append("whole")
            n_in = tgt
            if not dry:
                path = get_file(e, dest / "archives" / e.name) and dest / "archives" / e.name
                log.append({"src": e.name, "bytes": e.size})
                n_in = min(tgt, count_media_in_archive(path, tgt))
            got += n_in  # zips are counted (images that are not masks); other archives assumed to fill the quota
            nbytes += max(e.size, 0)
            left -= max(e.size, 0)
        elif e_ext == ".zip":
            strat.append("zip-range")
            if dry:
                got, nbytes = want, min(left, e.size)
                break
            g, b = sample_zip_remote(e, dest, tgt, seed, log, budget=left)
            got, nbytes, left = got + g, nbytes + b, left - b
        elif e_ext.startswith(".tar") or e_ext == ".tgz":
            strat.append("tar-stream")
            if dry:
                got, nbytes = want, min(left, e.size)
                break
            g, b = sample_tar_stream(e, dest, tgt, log, budget=left)
            got, nbytes, left = got + g, nbytes + b, left - b
        else:
            notes.append(f"skipped {e.name} ({e.size / 2**30:.1f} GB, {e_ext} cannot be sampled)")
    if not dry:
        for e in meta[:50]:
            get_file(e, dest / "files" / e.name)
    return got, nbytes, "+".join(strat) or "none", notes


def fetch_hf(ref, dest: Path, want, seed, dry, log, budget=None):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(ref, files_metadata=True)
    if getattr(info, "gated", False):
        raise Manual(f"HF {ref} is gated")
    files = [s for s in info.siblings]
    try:  # siblings is truncated for very large repos; the tree listing is complete
        from huggingface_hub.hf_api import RepoFile
        tree = [f for f in api.list_repo_tree(ref, repo_type="dataset", recursive=True) if isinstance(f, RepoFile)]
        if len(tree) > len(files):
            from types import SimpleNamespace
            files = [SimpleNamespace(rfilename=f.path, size=f.size) for f in tree]
    except Exception:
        pass
    total = sum(s.size or 0 for s in files)
    if dry:
        return want, min(total, want * 300_000), "hf-stream", [f"repo {total / 2**30:.1f} GB"]
    import datasets
    from PIL import Image
    try:
        try:
            dsd = datasets.load_dataset(ref, streaming=True)
        except ValueError as ex:  # multi-config repo: "Config name is missing" -> take the first config with images
            cfgs = datasets.get_dataset_config_names(ref)
            dsd = datasets.load_dataset(ref, cfgs[0], streaming=True)
    except Exception as ex:  # script-based or raw-file repos: fall back to file listing
        entries = [Entry(s.rfilename, s.size or -1, f"https://huggingface.co/datasets/{ref}/resolve/main/{s.rfilename}",
                         headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}) for s in files]
        n, b, strat, notes = fetch_entries(entries, dest, want, budget or 50 * 2**30, seed, False, log)
        return n, b, "hf-files:" + strat, notes + [f"streaming failed: {type(ex).__name__}"]
    try:
        return _stream_hf(dsd, ref, dest, want, seed, log, budget, files)
    except Exception as ex:  # e.g. Arrow offset overflow on huge rows: fall back to plain files
        entries = [Entry(s.rfilename, s.size or -1, f"https://huggingface.co/datasets/{ref}/resolve/main/{s.rfilename}",
                         headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}) for s in files]
        n, b, strat, notes = fetch_entries(entries, dest, want, budget or 50 * 2**30, seed, False, log)
        return n, b, "hf-files:" + strat, notes + [f"streaming failed: {type(ex).__name__}"]


def _parquet_files(ds):
    """hf:// parquet files behind a streaming split (None if the split is not parquet-backed)."""
    ei = getattr(ds, "_ex_iterable", None)
    for _ in range(8):
        fs = (getattr(ei, "kwargs", None) or {}).get("files")
        if isinstance(fs, list) and fs and all(isinstance(f, str) and f.endswith(".parquet") for f in fs):
            return fs
        ei = getattr(ei, "ex_iterable", None)
        if ei is None:
            return None
    return None


def _spread_rows(files, n, seed, max_bytes=2 * 2**30):
    """Rows from evenly spaced parquet row groups across all files: a shuffle buffer only sees the head of the split, which
    is one class when a dataset is stored sorted (PanNuke is sorted by tissue). None if row-group reads are not possible."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs, pfs, groups = HfFileSystem(), {}, []
    for f in files:
        pfs[f] = pf = pq.ParquetFile(fs.open(f))
        groups += [(f, g, pf.metadata.row_group(g).num_rows, pf.metadata.row_group(g).total_byte_size) for g in range(pf.num_row_groups)]
    if not groups:
        return None
    avg = max(1, sum(g[3] for g in groups) / len(groups))
    k = max(1, min(len(groups), n, int(max_bytes // avg)))
    if k == 1 and len(groups) > 1 and avg > max_bytes:
        return None
    pick = [groups[i] for i in sorted(set(np.linspace(0, len(groups) - 1, k).round().astype(int)))]
    per = math.ceil(n / len(pick))
    rng, rows = random.Random(seed), []
    for f, g, nr, _ in pick:
        t = pfs[f].read_row_group(g)
        rows += t.take(sorted(rng.sample(range(nr), min(per, nr)))).to_pylist()
    rng.shuffle(rows)
    return rows


def _split_rows(ds, n, seed):
    try:
        files = _parquet_files(ds)
        rows = _spread_rows(files, n, seed) if files else None
        if rows is not None:
            return rows
    except Exception:
        pass
    return ds.shuffle(seed=seed, buffer_size=100)  # small buffer: big images exhaust memory otherwise


def _stream_hf(dsd, ref, dest, want, seed, log, budget, files):
    from PIL import Image
    out = dest / "hf_images"
    out.mkdir(parents=True, exist_ok=True)
    rows = open(dest / "hf_rows.jsonl", "a")
    got, nbytes = 0, 0
    splits = sorted(dsd.keys(), key=lambda s: (s != "train", s))
    names = {}  # ClassLabel columns: store the class *name*, not its integer
    img_cols = set()  # Image() columns: raw parquet rows hold {"bytes", "path": None}, so the path cannot tell
    for sp in splits:
        try:
            for col, feat in (dsd[sp].features or {}).items():
                if hasattr(feat, "names") and feat.names:
                    names[col] = list(feat.names)
                if type(feat).__name__ == "Image":
                    img_cols.add(col)
        except Exception:
            pass
    for sp in splits:
        quota = want - got if sp == splits[-1] else max(1, int((want - got) * (0.8 if sp == "train" else 0.5)))
        it = _split_rows(dsd[sp], min(2 * quota + 20, quota + 500), seed)  # spare rows for masks / rows without images
        taken = 0
        for i, ex in enumerate(it):
            for k, v in list(ex.items()):  # Image(decode=False) yields {"bytes", "path"}
                if isinstance(v, dict) and v.get("bytes") and (k in img_cols or str(v.get("path") or "").lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"))):
                    try:
                        import io as _io
                        ex[k] = Image.open(_io.BytesIO(v["bytes"]))
                    except Exception:
                        pass
            imgs = [(k, v) for k, v in ex.items() if isinstance(v, Image.Image)]
            imgs = [(k, v) for k, v in imgs if not re.search(r"mask|label|seg|gt", k, re.I)
                    and not is_mask(getattr(v, "filename", "") or "")]
            # rows that describe a mask (e.g. a "type"/"kind" column = "mask") are not images
            if any(isinstance(vv, str) and re.fullmatch(r"(mask|label|segmentation|gt|ground.?truth)s?", vv.strip(), re.I)
                   for kk, vv in ex.items() if re.search(r"type|kind|category|split|role", kk, re.I)):
                imgs = []
            if not imgs:
                if i > 50 and taken == 0:
                    break  # no image column in this split
                continue
            for k, im in imgs[:1]:
                fn = f"{sp}_{i:07d}.png"
                im.save(out / fn)
                nbytes += (out / fn).stat().st_size
                meta = {kk: (names[kk][vv] if kk in names and isinstance(vv, int) and 0 <= vv < len(names[kk]) else vv)
                        for kk, vv in ex.items() if isinstance(vv, (str, int, float, bool)) and len(str(vv)) <= 300}
                if any(isinstance(v, str) and re.fullmatch(r"(mask|label|segmentation|gt|ground.?truth)s?", v.strip(), re.I)
                       for k_, v in meta.items() if k_ in names):
                    continue  # a ClassLabel saying this row is a mask (e.g. raabin-wbc "Mask"/"Original")
                rows.write(json.dumps({"file": fn, "split": sp, "image_column": k, **meta}, ensure_ascii=False, default=str) + "\n")
            got += 1
            taken += 1
            if taken >= quota or got >= want or (budget and nbytes >= budget):
                break
        if got >= want:
            break
    rows.close()
    log.append({"src": f"hf:{ref}", "images": got, "bytes": nbytes})
    if got < max(1, 0.2 * want):  # CSV-only config, preview-only parquet (e.g. 60 slices), or images stored as files
        entries = [Entry(s.rfilename, s.size or -1, f"https://huggingface.co/datasets/{ref}/resolve/main/{s.rfilename}",
                         headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}) for s in files]
        n, b, strat, notes = fetch_entries(entries, dest, want, budget or 50 * 2**30, seed, False, log)
        return n, b, "hf-files:" + strat, notes + ["streaming found no image column"]
    return got, nbytes, "hf-stream", []


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────────────────
def want_files(row):
    n = row["plan_images"]
    dims = set(row["dimensionality"])
    if dims & {"3d", "3d+t"} and not dims & {"2d", "multichannel-2d"}:
        n = math.ceil(n / SLICES_PER_VOLUME)
    elif dims & {"2d+t"} and not dims & {"2d"}:
        n = math.ceil(n / 8)  # prepare.py takes up to 8 frames per video
    elif "wsi" in dims and not dims & {"2d"}:
        n = math.ceil(n / 16)  # 16 tiles per slide
    return max(10, int(n * OVERSAMPLE))


def run_one(row, out_root: Path, args):
    dest = out_root / row["id"]
    st_path = dest / "_status.json"
    if st_path.exists() and not args.retry and not args.dry_run:
        st = json.loads(st_path.read_text())
        # a pilot-sized download counts as done only for runs that want no more files than it fetched
        if st["state"] in ("manual", "too_big") or (st["state"] == "done" and st.get("want", 0) >= want_files(row)):
            return st
    method, ref = row["method"], row["ref"]
    want = want_files(row)
    budget = int(args.max_gb * 2**30)
    log, t0 = [], time.time()
    st = {"id": row["id"], "method": method, "ref": ref, "want": want}
    global EXCLUDE
    EXCLUDE = row.get("exclude_paths") or None
    # manual drop: someone downloaded the data by hand into archives/ or files/ → nothing to fetch
    dropped = [p for sub in ("archives", "files") if (dest / sub).exists() for p in (dest / sub).iterdir() if not p.name.endswith(".part")]
    if dropped and not st_path.exists() and not args.dry_run:
        st.update(state="done", strategy="manual-drop", n_media=len(dropped), bytes=sum(p.stat().st_size for p in dropped if p.is_file()), notes=[])
        st_path.write_text(json.dumps(st, indent=1))
        return st
    try:
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        if method == "doi":
            method, ref = resolve_doi(ref)
            st.update(method=method, ref=ref, via="doi")
        if method == "manual":
            raise Manual(ref)
        if method == "hf":
            n, b, strat, notes = fetch_hf(ref, dest, want, args.seed, args.dry_run, log, budget=budget)
        else:
            entries = LISTERS[method](ref)
            if row.get("exclude_paths"):  # audited: synthetic subsets, model weights, duplicate bundles
                entries = [e for e in entries if not re.search(row["exclude_paths"], e.name)]
            st["listed_files"] = len(entries)
            st["listed_bytes"] = sum(max(e.size, 0) for e in entries)
            n, b, strat, notes = fetch_entries(entries, dest, want, budget, args.seed, args.dry_run, log)
        state = "done" if n >= want * 0.5 else ("partial" if n > 0 else "too_big" if any("skipped" in x for x in notes) else "error")
        st.update(state=state, n_media=n, bytes=b, strategy=strat, notes=notes)
    except Manual as m:
        st.update(state="manual", notes=[str(m)])
    except Exception as ex:
        st.update(state="error", notes=[f"{type(ex).__name__}: {ex}"], trace=traceback.format_exc(limit=3))
    st["seconds"] = round(time.time() - t0, 1)
    if not args.dry_run:
        st_path.write_text(json.dumps(st, indent=1))
        with open(dest / "_files.jsonl", "a") as f:
            for x in log:
                f.write(json.dumps(x) + "\n")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(PLAN / "selection.jsonl"))
    ap.add_argument("--out", default=str(RAW), help="download root (put this on /blue; env WAILA_RAW)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th dataset starting at i (Slurm arrays)")
    ap.add_argument("--only", default="", help="comma-separated dataset ids")
    ap.add_argument("--methods", default="", help="comma-separated routes to run, e.g. hf,zenodo")
    ap.add_argument("--max-gb", type=float, default=20.0, help="per-dataset transfer budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="list sources and estimate transfer; download nothing")
    ap.add_argument("--retry", action="store_true", help="also redo datasets marked done/manual/too_big")
    ap.add_argument("--pilot", type=int, default=0, help="cap images per dataset (e.g. 128) for a fast first pass; "
                    "a later full run tops the same folders up (files already fetched are reused)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.plan)]
    if args.pilot:
        for r in rows:
            r["plan_images"] = min(r["plan_images"], args.pilot)
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if args.methods:
        rows = [r for r in rows if r["method"] in set(args.methods.split(","))]
    i, n = map(int, args.shard.split("/"))
    rows = rows[i::n]
    out_root = Path(args.out)
    tag = "dryrun" if args.dry_run else "status"
    report = (PLAN if args.dry_run else out_root / "_status") / f"{tag}_shard{i}of{n}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "method", "ref", "state", "want", "n_media", "bytes", "strategy", "listed_files", "listed_bytes", "seconds", "notes"]
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k, row in enumerate(rows):
            st = run_one(row, out_root, args)
            st["notes"] = " | ".join(st.get("notes", []))
            w.writerow(st)
            f.flush()
            print(f"[{k + 1}/{len(rows)}] {row['id']:40s} {st['state']:8s} {st.get('n_media', 0):>6} files "
                  f"{(st.get('bytes') or 0) / 2**30:7.2f} GB  {st.get('strategy', '')} {st['notes'][:80]}", flush=True)
    print("report:", report)


if __name__ == "__main__":
    sys.exit(main())
