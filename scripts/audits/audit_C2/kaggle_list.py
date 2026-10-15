"""Throttled Kaggle lister (metadata only). Reads token from ~/.kaggle/access_token at runtime; never prints it.
usage: kaggle_list.py owner/slug out.json"""
import requests, time, os, sys, json
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
H = {"Authorization": "Bearer " + tok}
def get(url, params=None, tries=8):
    for i in range(tries):
        time.sleep(8)
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(10 + 10 * i); continue
        return r
    return r
slug, outp = sys.argv[1], sys.argv[2]
out = {"slug": slug}
v = get(f"https://www.kaggle.com/api/v1/datasets/view/{slug}")
out["view_status"] = v.status_code
if v.status_code == 200:
    j = v.json()
    out["view"] = {k: (str(x)[:1500] if x is not None else None) for k, x in j.items() if k in ("titleNullable", "totalBytesNullable", "licenseNameNullable", "ownerNameNullable", "subtitleNullable", "descriptionNullable")}
files, token = [], None
while True:
    p = {"pageSize": 200}
    if token: p["pageToken"] = token
    r = get(f"https://www.kaggle.com/api/v1/datasets/list/{slug}", p)
    out["list_status"] = r.status_code
    if r.status_code != 200: break
    j = r.json()
    for f in j.get("datasetFiles", []):
        files.append((f.get("name") or f.get("nameNullable"), int(f.get("totalBytes") or f.get("totalBytesNullable") or -1)))
    token = j.get("nextPageToken") or j.get("nextPageTokenNullable")
    if not token or len(files) > 30000: break
out["files"] = files
json.dump(out, open(outp, "w"))
print(slug, out.get("view_status"), out.get("list_status"), len(files))
