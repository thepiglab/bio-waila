"""Metadata-only collector for the Pass-2 labeling audit (first half). Writes audit_tmp_A/<id>.json."""
import csv, io, json, os, re, sys, time, traceback, collections, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/Users/mkieffer/programming/uf/mmbu/pass2_dataset")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import requests
import download as D
from bio_waila.data.files import ext, kind

TMP = Path("/private/tmp/claude-501/-Users-mkieffer-programming-uf-mmbu/927afb66-111c-460c-9b28-07fb6d07d3ea/scratchpad/audit_tmp_A")
TMP.mkdir(exist_ok=True)
HEAD = 64 * 1024
TABLE_EXT = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls"}
GOOD = re.compile(r"meta|label|annot|clinic|particip|info|class|diagnos|patient|train|test|val|index|manifest|catalog|summary|data|descr|readme|sheet|table|excel|key|map", re.I)
HDR = {"User-Agent": D.UA}
lock = threading.Lock()


def get(url, **kw):
    time.sleep(0.25)
    return D.S.get(url, timeout=45, **kw)


def range_head(url, n=HEAD, headers=None):
    h = {"Range": f"bytes=0-{n-1}", **(headers or {})}
    r = D.S.get(url, headers=h, timeout=45, stream=True)
    try:
        if r.status_code >= 400:
            return None, r.status_code
        buf = b""
        for ch in r.iter_content(8192):
            buf += ch
            if len(buf) >= n:
                break
        return buf[:n], r.status_code
    finally:
        r.close()
    time.sleep(0.25)


def parse_head(name, raw):
    e = ext(name)
    out = {"name": name, "bytes": len(raw)}
    if e in (".json", ".jsonl"):
        out["text"] = raw[:2500].decode("utf8", "replace")
        return out
    if e == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        out["sheets"] = {}
        for ws in wb.worksheets[:4]:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                rows.append([("" if c is None else str(c))[:80] for c in row])
                if i >= 49:
                    break
            out["sheets"][ws.title] = rows
        return out
    txt = raw.decode("utf8", "replace")
    lines = txt.splitlines()
    if len(raw) >= HEAD and len(lines) > 1:
        lines = lines[:-1]
    if not lines:
        out["text"] = ""
        return out
    delim = ","
    first = lines[0]
    if first.count("\t") > first.count(","):
        delim = "\t"
    elif first.count(";") > first.count(","):
        delim = ";"
    if e == ".txt" and first.count(delim) < 1:
        out["text"] = "\n".join(lines[:15])[:1500]
        return out
    rows = list(csv.reader(lines[:51], delimiter=delim))
    out["delim"] = delim
    out["rows"] = [[c[:80] for c in r] for r in rows]
    return out


def zip_list(url, headers=None):
    import remotezip
    for suffix in (True, False):
        try:
            sess = D.S
            z = remotezip.RemoteZip(url, support_suffix_range=suffix, session=sess, headers=headers)
            infos = [(i.filename, i.file_size, i.compress_size) for i in z.infolist()]
            return z, infos
        except Exception as ex:
            err = repr(ex)[:200]
    return None, err


def zip_member_head(z, name, n=HEAD):
    with z.open(name) as f:
        raw = f.read(n)
    return raw


def struct_summary(names):
    """names: list of paths. returns ext counts, per-depth distinct folder names, sample paths."""
    exts = collections.Counter(ext(n) or "(none)" for n in names)
    depth = collections.defaultdict(collections.Counter)
    for n in names:
        parts = n.strip("/").split("/")[:-1]
        for i, p in enumerate(parts[:5]):
            depth[i][p] += 1
    ds = {f"d{i+1}": dict(c.most_common(40)) | ({"__n_distinct__": len(c)} if len(c) > 40 else {}) for i, c in sorted(depth.items())}
    return {"n": len(names), "ext": dict(exts.most_common(12)), "depth": ds, "sample": names[:8] + names[-3:] if len(names) > 11 else names}


def collect(row):
    rid, m, ref = row["id"], row["method"], row["ref"]
    out = {"id": rid, "method": m, "ref": ref, "access": "ok", "notes": [], "tables": [], "listing": None, "zips": [], "hf": None}
    try:
        entries = []
        if m == "kaggle":
            out["access"] = "needs_credentials (kaggle)"
            return out
        if m == "hf":
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                info = api.dataset_info(ref)
            except Exception as ex:
                out["access"] = f"hf_error {repr(ex)[:150]}"
                return out
            if getattr(info, "gated", False):
                out["access"] = "gated"
            entries = [D.Entry(s.rfilename, -1, f"https://huggingface.co/datasets/{ref}/resolve/main/{s.rfilename}") for s in info.siblings]
            # datasets-server profile
            base = "https://datasets-server.huggingface.co"
            hf = {"configs": {}}
            try:
                ir = get(f"{base}/info", params={"dataset": ref})
                if ir.status_code != 200:
                    hf["info_status"] = ir.status_code
                else:
                    for cfg, ci in list(ir.json().get("dataset_info", {}).items())[:4]:
                        feats = {}
                        for col, f in (ci.get("features") or {}).items():
                            if isinstance(f, dict) and f.get("names"):
                                feats[col] = {"classlabel": f["names"][:80], "n": len(f["names"])}
                            elif isinstance(f, dict):
                                feats[col] = f.get("dtype") or f.get("_type") or str(f)[:40]
                            else:
                                feats[col] = "list/other"
                        splits = ci.get("splits", {})
                        c = {"features": feats, "splits": {k: v.get("num_examples") for k, v in splits.items()}}
                        sp = next(iter(splits), "train")
                        fr = get(f"{base}/first-rows", params={"dataset": ref, "config": cfg, "split": sp})
                        if fr.status_code == 200:
                            rows = []
                            for r in fr.json().get("rows", [])[:30]:
                                rows.append({k: (v if isinstance(v, (str, int, float, bool)) or v is None else "<obj>") for k, v in r["row"].items()})
                            c["first_rows"] = rows
                        else:
                            c["first_rows_status"] = fr.status_code
                        hf["configs"][cfg] = c
            except Exception as ex:
                hf["error"] = repr(ex)[:200]
            out["hf"] = hf
        elif m == "openneuro":
            ds = ref
            allf = []
            token = None
            while True:
                params = {"list-type": "2", "prefix": f"{ds}/", "max-keys": "1000"}
                if token:
                    params["continuation-token"] = token
                r = get("https://s3.amazonaws.com/openneuro.org", params=params)
                r.raise_for_status()
                import xml.etree.ElementTree as ET
                root = ET.fromstring(r.content)
                nsx = {"s": root.tag.split("}")[0].strip("{")}
                for c in root.findall("s:Contents", nsx):
                    key = c.find("s:Key", nsx).text
                    allf.append((key[len(ds) + 1:], int(c.find("s:Size", nsx).text)))
                t = root.find("s:NextContinuationToken", nsx)
                if t is None or len(allf) > 20000:
                    break
                token = t.text
            entries = [D.Entry(n, s, f"https://s3.amazonaws.com/openneuro.org/{ds}/{n}") for n, s in allf if "derivatives/" not in n or n.count("/") < 3]
        elif m == "figshare":
            entries = D.list_figshare(ref)
        elif m == "zenodo":
            entries = D.list_zenodo(ref)
        elif m == "dataverse":
            entries = D.list_dataverse(ref)
        elif m == "direct":
            entries = D.list_direct(ref)
        elif m in ("doi",):
            k, rid2 = D.resolve_doi(ref)
            out["notes"].append(f"doi->{k}:{rid2}")
            entries = D.LISTERS[k](rid2)
    except D.Manual as ex:
        out["access"] = "manual: " + str(ex)[:200]
        return out
    except Exception as ex:
        out["access"] = f"error {repr(ex)[:200]}"
        return out

    names = [e.name for e in entries]
    out["listing"] = struct_summary(names)
    out["listing"]["total_bytes"] = sum(max(e.size, 0) for e in entries)
    out["listing"]["files_top"] = [(e.name, e.size) for e in sorted(entries, key=lambda e: -e.size)[:15]]
    out["listing"]["n_media"] = sum(1 for n in names if kind(n) in ("image", "volume", "video", "wsi", "array"))
    out["listing"]["n_masklike"] = sum(1 for n in names if kind(n) == "mask")
    out["listing"]["nonmedia_small"] = [(e.name, e.size) for e in entries if kind(e.name) in ("meta",)][:40]

    # loose tables
    cand = [e for e in entries if ext(e.name) in TABLE_EXT and (e.size < 0 or e.size <= 50 * 2**20)]
    cand.sort(key=lambda e: (not GOOD.search(e.name), e.name.count("/"), e.name))
    n_t = 0
    for e in cand:
        if n_t >= 5:
            break
        if "readme" in e.name.lower() and ext(e.name) == ".txt":
            pass
        try:
            if ext(e.name) in (".xlsx", ".xls"):
                if ext(e.name) == ".xls" or e.size > 3 * 2**20 or e.size < 0:
                    out["tables"].append({"name": e.name, "skipped": f"xlsx/xls size {e.size}"})
                    continue
                r = get(e.url)
                raw = r.content
            else:
                raw, st = range_head(e.url, headers=e.headers)
                if raw is None:
                    out["tables"].append({"name": e.name, "skipped": f"HTTP {st}"})
                    continue
            t = parse_head(e.name, raw)
            t["source"] = "loose"
            out["tables"].append(t)
            n_t += 1
        except Exception as ex:
            out["tables"].append({"name": e.name, "skipped": repr(ex)[:150]})
    # zips
    zips = [e for e in entries if ext(e.name) == ".zip"]
    zips.sort(key=lambda e: -e.size)
    for e in zips[:6]:
        z, infos = zip_list(e.url, e.headers)
        if z is None:
            out["zips"].append({"name": e.name, "size": e.size, "error": infos})
            continue
        mem = [i[0] for i in infos if not i[0].endswith("/")]
        zs = {"name": e.name, "size": e.size, **struct_summary(mem)}
        zs["n_media"] = sum(1 for n in mem if kind(n) in ("image", "volume", "video", "wsi", "array"))
        zs["n_masklike"] = sum(1 for n in mem if kind(n) == "mask")
        zs["tables"] = []
        tcand = [i for i in infos if ext(i[0]) in (".csv", ".tsv", ".xlsx", ".json", ".txt", ".jsonl") and i[2] <= 30 * 2**20 and not i[0].endswith("/") and "__MACOSX" not in i[0]]
        tcand.sort(key=lambda i: (not GOOD.search(i[0]), i[0].count("/"), i[0]))
        # prefer csv/xlsx over txt
        tcand.sort(key=lambda i: ext(i[0]) in (".txt",))
        zs["table_names"] = [i[0] for i in tcand[:30]]
        for i in tcand[:3]:
            try:
                if ext(i[0]) == ".xlsx" and i[2] > 3 * 2**20:
                    continue
                n = (3 * 2**20) if ext(i[0]) == ".xlsx" else HEAD
                raw = zip_member_head(z, i[0], n)
                t = parse_head(i[0], raw)
                t["source"] = f"zip:{e.name}"
                zs["tables"].append(t)
                time.sleep(0.25)
            except Exception as ex:
                zs["tables"].append({"name": i[0], "skipped": repr(ex)[:150]})
        out["zips"].append(zs)
        time.sleep(0.3)
    other_arch = [e.name for e in entries if kind(e.name) == "archive" and ext(e.name) != ".zip"]
    if other_arch:
        out["notes"].append("non-zip archives (no remote listing): " + ", ".join(other_arch[:6]))
    return out


def run(row):
    rid = row["id"]
    p = TMP / f"{rid}.json"
    if p.exists():
        return rid, "cached"
    t0 = time.time()
    try:
        o = collect(row)
    except Exception as ex:
        o = {"id": rid, "access": "collector_error " + repr(ex)[:200], "trace": traceback.format_exc()[-800:]}
    o["secs"] = round(time.time() - t0, 1)
    p.write_text(json.dumps(o, ensure_ascii=False, default=str))
    return rid, o.get("access")


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(ROOT / "plan/selection.jsonl")][:197]
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    if only:
        rows = [r for r in rows if r["id"] in only]
    with ThreadPoolExecutor(4) as ex:
        for rid, st in ex.map(run, rows):
            print(rid, st, flush=True)
