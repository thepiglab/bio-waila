"""mmbu-public prior: python prior.py field=value [field=value]  -> counts of other fields among rows matching."""
import sys, collections, pandas as pd
d = pd.read_parquet("plan/eval_public.parquet")
def has(col, v): return d[col].apply(lambda x: v in list(x))
m = pd.Series(True, index=d.index)
for a in sys.argv[1:]:
    f, v = a.split("=", 1)
    m &= has(f + "_values", v)
s = d[m]
print("n rows", len(s))
for f in ["modality", "submodality", "body_part", "stain", "specimen", "domain"]:
    c = collections.Counter(); mm = 0
    for _, r in s.iterrows():
        if r[f + "_mode"] == "masked": mm += 1
        for x in r[f + "_values"]: c[x] += 1
    print(f, "masked", mm, dict(c.most_common(8)))
