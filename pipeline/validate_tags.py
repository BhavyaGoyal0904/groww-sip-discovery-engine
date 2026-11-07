"""Check AI tag output: valid JSON, all fields, enum values legal, evidence_quote verbatim in the post."""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codebook import SCHEMA
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
P = SCHEMA["properties"]
def norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def check(out_file, in_file):
    src = {json.loads(l)["id"]: json.loads(l) for l in open(in_file)}
    errs, seen = [], set()
    for i, line in enumerate(open(out_file), 1):
        if not line.strip(): continue
        try: o = json.loads(line)
        except Exception: errs.append(f"line {i}: bad JSON"); continue
        pid = o.get("id")
        if pid not in src: errs.append(f"line {i}: unknown id {pid}"); continue
        seen.add(pid)
        for k, spec in P.items():
            if k not in o: errs.append(f"{pid}: missing {k}"); continue
            v = o[k]
            if "enum" in spec and v not in spec["enum"]: errs.append(f"{pid}: {k}={v!r} not allowed")
            if spec.get("type") == "array":
                if not isinstance(v, list): errs.append(f"{pid}: {k} must be a list")
                else:
                    bad = [x for x in v if x not in spec["items"]["enum"]]
                    if bad: errs.append(f"{pid}: {k} has {bad}")
        q = o.get("evidence_quote", "")
        if q and norm(q.strip('"…. ')) not in norm(src[pid]["text"] + " " + src[pid].get("title", "")):
            errs.append(f"{pid}: evidence_quote not verbatim")
    missing = set(src) - seen
    if missing: errs.append(f"{len(missing)} ids not tagged yet: {sorted(missing)[:5]}...")
    return errs
if __name__ == "__main__":
    b = sys.argv[1]
    e = check(os.path.join(D, "tag_out", f"batch_{b}.jsonl"), os.path.join(D, "tag_in", f"batch_{b}.jsonl"))
    print("\n".join(e[:60]) if e else "OK — all posts tagged, schema valid, quotes verbatim")
    print(f"{len(e)} problems")
