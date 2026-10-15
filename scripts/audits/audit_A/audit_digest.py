import json, sys, os, collections
from pathlib import Path
ROOT = Path("/Users/mkieffer/programming/uf/mmbu/pass2_dataset")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from bio_waila.data.rules import match_value
import metalabel
TMP = Path("/private/tmp/claude-501/-Users-mkieffer-programming-uf-mmbu/927afb66-111c-460c-9b28-07fb6d07d3ea/scratchpad/audit_tmp_A")
CLS = metalabel.classes()
rows = [json.loads(l) for l in open(ROOT / "plan/selection.jsonl")][:197]
ROW = {r["id"]: r for r in rows}


def automap(col, values, acc):
    for v in values:
        try:
            res = match_value(v, col, CLS)
        except Exception:
            continue
        for f, (c, s) in res.items():
            acc[f][c].append(f"{col}={str(v)[:30]}{'' if s else '(weak)'}")


def table_cols(t):
    rows_ = t.get("rows")
    if rows_ is None and t.get("sheets"):
        sh = next(iter(t["sheets"].values()))
        rows_ = sh
    if not rows_:
        return None, {}
    hdr = rows_[0]
    cols = {}
    for j, h in enumerate(hdr):
        vals = [r[j] for r in rows_[1:] if j < len(r)]
        cols[h or f"col{j}"] = vals
    return hdr, cols


def fmt_vals(vals):
    c = collections.Counter(vals)
    if len(c) <= 12:
        return "{" + ", ".join(f"{k[:28]}:{n}" for k, n in c.most_common()) + "}"
    return f"{len(c)} distinct e.g. " + ", ".join(k[:24] for k in list(c)[:4])


def digest(rid, verbose=True):
    p = TMP / f"{rid}.json"
    r = ROW[rid]
    if not p.exists():
        return f"## {rid}: NOT COLLECTED"
    o = json.loads(p.read_text())
    L = r["labels"]
    reg = " | ".join(f"{f[:4]}={L[f]['mode'][0]}:{','.join(L[f]['values'])[:45]}" for f in ["modality", "submodality", "body_part", "stain", "specimen"])
    out = [f"## {rid} [{r['method']}:{r['ref'][:50]}] access={o.get('access')} size={r['full_size_gb']}GB planimgs={r['plan_images']} dims={r['dimensionality']}", f"REG {reg}"]
    acc = {f: collections.defaultdict(list) for f in ["modality", "submodality", "body_part", "stain", "specimen", "domain"]}
    li = o.get("listing")
    if li:
        out.append(f"LIST n={li['n']} media={li['n_media']} masklike={li['n_masklike']} ext={li['ext']} bytes={li['total_bytes']/2**30:.2f}GB")
        for d, c in li["depth"].items():
            if c and d in ("d1", "d2", "d3"):
                out.append(f"  {d}: " + ", ".join(f"{k[:30]}:{v}" for k, v in list(c.items())[:10]))
                automap("path:" + d, [k for k in c if k != "__n_distinct__"], acc)
        if li["n"] < 25 or li["n_media"] == 0:
            out.append("  files: " + "; ".join(f"{n[:40]}({s/2**20:.0f}M)" for n, s in li["files_top"][:6]))
        out.append("  sample: " + " | ".join(s[:70] for s in li["sample"][:4]))
    ts = o.get("tables", [])
    ts = [t for t in ts if "rows" in t or "sheets" in t or t.get("skipped")] + [t for t in ts if "text" in t][:1]
    for t in ts[:6]:
        out += table_lines(t, acc)
    for zi, z in enumerate(o.get("zips", [])):
        if zi >= 3:
            out.append(f"ZIP+ {z['name'][:50]} n={z.get('n')} media={z.get('n_media')} ext={z.get('ext')}")
            continue
        if "error" in z:
            out.append(f"ZIP {z['name']} ERROR {z['error'][:80]}")
            continue
        out.append(f"ZIP {z['name']} ({z['size']/2**30:.2f}GB) n={z['n']} media={z['n_media']} mask={z['n_masklike']} ext={z['ext']}")
        for d, c in z["depth"].items():
            if c and d in ("d1", "d2", "d3", "d4"):
                out.append(f"   {d}: " + ", ".join(f"{k[:30]}:{v}" for k, v in list(c.items())[:10]))
                automap("path:" + d, [k for k in c if k != "__n_distinct__"], acc)
        out.append("   sample: " + " | ".join(s[:70] for s in z["sample"][:3]))
        if z.get("table_names"):
            out.append("   tables: " + ", ".join(n.split("/")[-1][:30] for n in z["table_names"][:6]))
        zt = z.get("tables", [])
        zt = [t for t in zt if "rows" in t or "sheets" in t] or zt[:1]
        for t in zt[:2]:
            out += table_lines(t, acc)
    hf = o.get("hf")
    if hf:
        if hf.get("info_status") or hf.get("error"):
            out.append(f"HF info_status={hf.get('info_status')} err={hf.get('error')}")
        for cfg, c in hf["configs"].items():
            out.append(f"HF cfg={cfg} splits={c.get('splits')} feats=" + ", ".join(f"{k}:{v if isinstance(v, str) else 'CL'+str(v['n'])}" for k, v in c["features"].items())[:400])
            for k, v in c["features"].items():
                if isinstance(v, dict):
                    out.append(f"   CL {k}: {v['classlabel'][:20]}")
                    automap(f"hf:{k}", v["classlabel"], acc)
            fr = c.get("first_rows") or []
            keys = collections.defaultdict(list)
            for row in fr:
                for k, v in row.items():
                    if isinstance(v, (str, int, float)) and v != "<obj>":
                        keys[k].append(str(v))
            shown = 0
            import re as _re
            from bio_waila.data.rules import FIELD_COLUMNS, SKIP_COLUMN
            fld = _re.compile("|".join(FIELD_COLUMNS.values()), _re.I)
            for k, vals in keys.items():
                cc = collections.Counter(vals)
                info = bool(fld.search(k)) or (2 <= len(cc) <= 12 and not SKIP_COLUMN.search(k))
                if info and shown < 8 and cfg == next(iter(hf["configs"])):
                    out.append(f"   row.{k}: {fmt_vals(vals)}")
                    shown += 1
                automap(f"hf:{k}", list(cc), acc)
            if "first_rows_status" in c:
                out.append(f"   first_rows_status={c['first_rows_status']}")
    for n in o.get("notes", []):
        out.append("NOTE " + n[:200])
    am = []
    for f, d in acc.items():
        if d:
            am.append(f"{f}: " + "; ".join(f"{c}<-{','.join(v[:2])}" for c, v in list(d.items())[:8]))
    out.append("AUTOMAP " + (" || ".join(am) if am else "(nothing)"))
    return "\n".join(l[:300] for l in out)


def table_lines(t, acc):
    import re as _re
    from bio_waila.data.rules import FIELD_COLUMNS, SKIP_COLUMN
    out = []
    if t.get("skipped"):
        return [f"TABLE {t['name'][-45:]} SKIPPED {t['skipped'][:60]}"]
    if "text" in t:
        txt = t["text"]
        if t["name"].lower().endswith(".json") or t["name"].lower().endswith(".jsonl"):
            keys = _re.findall(r'"([A-Za-z_]+)":', txt)[:25]
            return [f"JSON {t['name'][-45:]} keys={keys} :: " + _re.sub(r"\s+", " ", txt)[:160]]
        return [f"TEXT {t['name'][-45:]}: " + _re.sub(r"\s+", " ", txt)[:260]]
    hdr, cols = table_cols(t)
    if hdr is None:
        return [f"TABLE {t['name'][-45:]} empty"]
    out.append(f"TABLE {t['name'][-55:]} [{t.get('source','')[:24]}] nrows~{len(next(iter(cols.values()),[]))} cols={[h[:22] for h in hdr][:28]}")
    shown = 0
    fld = _re.compile("|".join(FIELD_COLUMNS.values()), _re.I)
    for h, vals in cols.items():
        cc = collections.Counter(vals)
        info = bool(fld.search(h or "")) or (2 <= len(cc) <= 12 and not SKIP_COLUMN.search(h or ""))
        if len(cc) <= 15 and len(vals) >= 3 and h and info and shown < 9:
            out.append(f"   {h[:25]}: {fmt_vals(vals)}")
            shown += 1
        if h:
            automap(f"csv:{t['name'].split('/')[-1]}:{h}", list(cc), acc)
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or [r["id"] for r in rows]
    for i in ids:
        print(digest(i))
        print()
