"""Slow, resumable Kaggle probe: view + a few list pages with the nextPageToken carried over. Token read at runtime, never printed.
usage: kaggle_probe.py owner/slug out.json max_pages page_size"""
import requests, time, os, sys, json
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
H = {"Authorization": "Bearer " + tok}
slug, outp, maxp, ps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def get(url, params=None):
    for i in range(10):
        time.sleep(30)
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.status_code != 429:
            return r
        time.sleep(40 + 20 * i)
    return r
out = {"slug": slug, "pages": []}
v = get(f"https://www.kaggle.com/api/v1/datasets/view/{slug}")
out["view_status"] = v.status_code
if v.status_code == 200:
    j = v.json()
    out["view"] = {k: (str(x)[:2500] if x is not None else None) for k, x in j.items() if k in ("titleNullable", "totalBytesNullable", "licenseNameNullable", "subtitleNullable", "descriptionNullable")}
token = None
for pg in range(maxp):
    p = {"pageSize": ps}
    if token: p["pageToken"] = token
    r = get(f"https://www.kaggle.com/api/v1/datasets/list/{slug}", p)
    if r.status_code != 200:
        out["list_status"] = r.status_code; break
    j = r.json()
    out["pages"].append([(f.get("name") or f.get("nameNullable"), int(f.get("totalBytes") or f.get("totalBytesNullable") or -1)) for f in j.get("datasetFiles", [])])
    token = j.get("nextPageToken") or j.get("nextPageTokenNullable")
    json.dump(out, open(outp, "w"))
    if not token: break
json.dump(out, open(outp, "w"))
print(slug, out.get("view_status"), out.get("list_status"), [len(p) for p in out["pages"]])
