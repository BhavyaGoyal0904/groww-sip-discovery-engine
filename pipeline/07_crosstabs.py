"""Answers the discovery questions from the 993 AI-tagged relevant posts."""
import os, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
t = pd.read_parquet(os.path.join(D, "tags_ai.parquet")); r = t[t.is_relevant].copy()
out = []
def vc_list(col, sub=r, drop=("none", "none_mentioned", "not_mentioned")):
    s = pd.Series([x for l in sub[col] for x in l if x not in drop]); return (s.value_counts()/len(sub)).round(3)
def sec(title, obj): out.append(f"\n### {title}\n{obj.to_string()}")
STOPPERS = r.continuation_signal.isin(["dropped", "at_risk"])
sec("Q1 Why start (share of relevant posts)", vc_list("start_motivation"))
sec("Q1 Expectation", r.expectation.value_counts(normalize=True).round(3))
sec("Q2/3 SIP action", r.sip_action.value_counts())
sec("Q3 Triggers among at-risk/dropped posts (share of those posts)", vc_list("triggers", r[STOPPERS]))
sec("Q3 Triggers among likely-continue posts", vc_list("triggers", r[r.continuation_signal=="likely_continue"]))
sec("Q4 Check frequency (mentioned only)", r.check_frequency[r.check_frequency!="not_mentioned"].value_counts())
sec("Q4 What they check", vc_list("what_checked"))
sec("Q5 In-app confusion", vc_list("app_confusion"))
sec("Q6 Outside-app help", vc_list("outside_app_help"))
sec("Q6 Content type (asking strangers = question_seeking_help)", r.content_type.value_counts(normalize=True).round(3))
sec("Q7 Expectation vs continuation (row %)", pd.crosstab(r.expectation, r.continuation_signal, normalize="index").round(2))
sec("Q7 Emotion vs continuation (row %)", pd.crosstab(r.emotion, r.continuation_signal, normalize="index").round(2))
for seg in ["income_stability", "age_band", "money_source", "influenced_by", "investor_experience"]:
    k = r[r[seg] != "unknown"]
    tab = pd.crosstab(k[seg], k.continuation_signal); tab["n"] = tab.sum(1)
    tab["at_risk_or_dropped_%"] = ((tab.get("at_risk", 0) + tab.get("dropped", 0)) / tab.n * 100).round(0)
    sec(f"Q8 Segment: {seg} (n known={len(k)})", tab)
sec("Q2 Journey stage vs continuation", pd.crosstab(r.journey_stage, r.continuation_signal))
open(os.path.join(D, "crosstabs.txt"), "w").write("\n".join(out)); print("\n".join(out))
