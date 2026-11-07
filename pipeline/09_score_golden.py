"""Score AI tags against the human golden set. Writes data/golden/golden_report.md."""
import os, json
import pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "golden")
FIELDS = ["is_relevant", "journey_stage", "sip_action", "triggers", "continuation_signal",
          "emotion", "outside_app_help", "income_stability", "influenced_by"]
MULTI = {"triggers", "outside_app_help"}
EMPTY = {"", "none", "none_mentioned", "nan"}

h = pd.read_csv(os.path.join(D, "golden_set_LABEL_ME.csv"), dtype=str, encoding="utf-8-sig").fillna("")
ai = pd.read_json(os.path.join(D, "golden_ai_answers.jsonl"), lines=True)
m = h.merge(ai.drop(columns="item"), on="id", suffixes=("_human", "_ai"))

def norm_multi(v):
    if isinstance(v, list): vals = v
    else: vals = [x.strip() for x in str(v).replace(";", ",").split(",")]
    return {x for x in vals if x and x not in EMPTY}

def agree(f, hv, av):
    if f == "is_relevant":
        return str(hv).strip().upper() == str(bool(av)).upper()
    if f in MULTI:
        H, A = norm_multi(hv), norm_multi(av)
        return (H == A) if not H or not A else len(H & A) > 0   # any overlap = agree (lenient)
    return str(hv).strip() == str(av).strip()

rows, misses = [], []
labelled = m[m["is_relevant_human"].str.strip() != ""]
for f in FIELDS:
    # non-relevance fields only scored where both say relevant
    sub = labelled if f == "is_relevant" else labelled[(labelled.is_relevant_human.str.upper() == "TRUE") & labelled.is_relevant_ai]
    ok = [agree(f, r[f + "_human"], r[f + "_ai"]) for _, r in sub.iterrows()]
    rows.append({"field": f, "n": len(ok), "agreement": (sum(ok) / len(ok)) if ok else float("nan")})
    for (_, r), k in zip(sub.iterrows(), ok):
        if not k: misses.append({"item": r["item"], "field": f, "human": r[f + "_human"], "ai": r[f + "_ai"],
                                 "post": r["post text"][:160].replace("\n", " "), "note": r.get("your note (optional)", "")})
R = pd.DataFrame(rows)
overall = sum(R.agreement * R.n) / R.n.sum()
M = pd.DataFrame(misses)
rep = [f"# Golden set: human vs AI agreement\n", f"Items labelled: {len(labelled)} / {len(m)}  |  Overall field-level agreement: **{overall:.0%}**\n",
       "Multi-value fields (triggers, outside_app_help) count as agreeing if any value overlaps. Fields other than is_relevant are scored only where both said relevant.\n",
       R.assign(agreement=R.agreement.map(lambda x: f"{x:.0%}")).to_markdown(index=False),
       "\n\n## Disagreements (where the AI failed or the codebook is unclear)\n",
       M.to_markdown(index=False) if len(M) else "None"]
open(os.path.join(D, "golden_report.md"), "w").write("\n".join(rep))
print("\n".join(rep))
