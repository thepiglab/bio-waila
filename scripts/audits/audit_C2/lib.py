"""Helpers for hand-written audit overrides (audit_A)."""
HARD = {"by": "registry hard", "map": {}, "conf": "high"}
def F(by, map=None, conf="high"):
    return {"by": by, "map": map or {}, "conf": conf}
def S(type_, loc, cols, ex=None):
    return {"type": type_, "location": loc, "columns_or_levels": cols, "example_values": ex or []}
OV = {}
def o(id, acc=None, src=None, mod=None, sub=None, bp=None, stain=None, spec=None, dom=None, prob=None, note=""):
    OV[id] = dict(acc=acc, src=src or [], fields=dict(modality=mod, submodality=sub, body_part=bp, stain=stain, specimen=spec, domain=dom),
                  prob=prob or [], note=note)
