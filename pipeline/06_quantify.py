"""Stage 6: quantify and rank themes -> opportunity scores.
Inputs: data/corpus_scored.parquet (full corpus, clusters), data/tags_ai.parquet (AI codebook tags),
        data/themes.json (reviewed cluster->theme mapping, addressability rubric, metric-tree node).
Outputs: data/theme_scorecard.csv, data/theme_details.md
"""
import os, json
import numpy as np, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
corpus = pd.read_parquet(os.path.join(D, "corpus_scored.parquet"))
tags = pd.read_parquet(os.path.join(D, "tags_ai.parquet"))
T = json.load(open(os.path.join(D, "themes.json")))
c2t = {c: t["name"] for t in T["themes"] for c in t["clusters"]}
rel = corpus[corpus.relevant & corpus.cluster.notna()].copy()
rel["theme"] = rel.cluster.astype(int).map(c2t)
rel = rel[rel.theme.notna()]
tg = tags[tags.is_relevant].merge(rel[["id", "theme"]], on="id")
FIRST_YEAR = {"month_0_3", "month_3_6", "month_6_12"}
SEG = {"income_stability": ["variable_income", "no_own_income", "fixed_salary"],
       "age_band": ["under_25", "25_34"], "influenced_by": ["finfluencer", "friends_family"],
       "investor_experience": ["first_time"], "money_source": ["family_or_pocket_money", "salary"]}

def known_share(s, vals, unknown=("unknown", "not_mentioned")):
    k = s[~s.isin(unknown)]
    return (k.isin(vals).mean() if len(k) else np.nan), len(k)

base = {f: tg[f][tg[f] != "unknown"].value_counts(normalize=True) for f in SEG}
rows, details = [], []
for t in T["themes"]:
    g, a = rel[rel.theme == t["name"]], tg[tg.theme == t["name"]]
    early, n_stage = known_share(a.journey_stage, FIRST_YEAR)
    first, n_exp = known_share(a.investor_experience, ["first_time"])
    lifts = []
    for f, vals in SEG.items():
        k = a[f][a[f] != "unknown"]
        for v in vals:
            if len(k) >= 8 and (k == v).sum() >= 3 and base[f].get(v, 0) > 0:
                lifts.append((f"{v}", round((k == v).mean() / base[f][v], 2), int((k == v).sum())))
    lifts = sorted([l for l in lifts if l[1] >= 1.3], key=lambda x: -x[1])[:3]
    trig = pd.Series([x for l in a.triggers for x in l if x != "none"]).value_counts().head(4)
    outside = pd.Series([x for l in a.outside_app_help for x in l if x != "none_mentioned"]).value_counts().head(3)
    rows.append(dict(theme=t["name"], metric_node=t["metric_node"], posts=len(g),
                     share_of_relevant=len(g) / len(rel), stop_pause_share_classifier=g.stop_signal.mean(),
                     ai_tagged_n=len(a), ai_stop_pause_share=a.continuation_signal.isin(["dropped"]).mean() if len(a) else np.nan,
                     at_risk_or_dropped=a.continuation_signal.isin(["dropped", "at_risk"]).mean() if len(a) else np.nan,
                     first_year_share=early, n_stage_known=n_stage, first_timer_share=first,
                     addressability=t["addressability"], segment_skew="; ".join(f"{v} x{l} (n={n})" for v, l, n in lifts),
                     top_triggers=", ".join(trig.index), outside_app=", ".join(outside.index)))
    needs = a[a.unmet_need.str.len() > 0].sample(min(6, (a.unmet_need.str.len() > 0).sum()), random_state=1) if len(a) else a
    details.append(f"\n## {t['name']}\nMetric node: {t['metric_node']}  |  Addressability {t['addressability']}/5 — {t['addressability_why']}\n"
                   + "Unmet needs (AI-extracted):\n" + "".join(f"- {n}\n" for n in needs.unmet_need)
                   + "Quotes:\n" + "".join(f"- \"{q}\"\n" for q in needs.evidence_quote if q))

S = pd.DataFrame(rows)
mm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0
S["opportunity_score"] = (100 * (0.25 * mm(np.sqrt(S.share_of_relevant)) + 0.30 * mm(S.at_risk_or_dropped.fillna(0))
                                 + 0.15 * mm(S.first_year_share.fillna(S.first_year_share.mean()))
                                 + 0.30 * mm(S.addressability.astype(float)))).round(1)
S = S.sort_values("opportunity_score", ascending=False)
S.to_csv(os.path.join(D, "theme_scorecard.csv"), index=False)
open(os.path.join(D, "theme_details.md"), "w").write("# Theme details\n" + "".join(details))
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(S[["theme", "posts", "share_of_relevant", "ai_tagged_n", "at_risk_or_dropped", "first_year_share",
         "first_timer_share", "addressability", "opportunity_score", "segment_skew"]].round(2).to_string(index=False))
