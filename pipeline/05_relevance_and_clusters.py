"""Stage 3b + 5: scale AI labels to the full corpus, then cluster relevant posts into candidate themes.

- Trains a classifier on the AI-tagged sample (embeddings + rule features) to predict
  (a) is_relevant and (b) stop/pause signal, for ALL filtered posts. Reports 5-fold CV quality.
- Clusters predicted-relevant posts (KMeans on embeddings) and writes, per cluster, distinctive
  terms + representative posts for AI naming (data/clusters_for_naming.md).
"""
import os, json, glob, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
K = int(os.environ.get("K", 30))

corpus = pd.read_parquet(os.path.join(D, "corpus_filtered.parquet"))
ids = pd.read_parquet(os.path.join(D, "embeddings_ids.parquet"))["id"]
E = np.load(os.path.join(D, "embeddings.npy")).astype(np.float32)
assert (ids.values == corpus["id"].values).all()
tags = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(os.path.join(D, "tag_out", "*.jsonl")))
                     for l in open(f) if l.strip()]).drop_duplicates("id", keep="last")
tags.to_parquet(os.path.join(D, "tags_ai.parquet"), index=False)
print(f"AI-tagged posts: {len(tags)}  relevant: {tags.is_relevant.sum()}")

src = pd.get_dummies(corpus["source"]).astype(float).values
rule = corpus[["p_action", "p_feel", "p_first", "p_stage"]].astype(float).values
X = np.hstack([E, src, rule, corpus[["priority"]].astype(float).values / 8])
pos = {i: n for n, i in enumerate(corpus["id"])}
ti = tags["id"].map(pos)
tags, ti = tags[ti.notna()], ti.dropna().astype(int).values

def fit_report(y, idx, name):
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    cv = cross_val_predict(clf, X[idx], y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    rep = classification_report(y, cv, output_dict=True, zero_division=0)
    print(f"\n[{name}] 5-fold CV vs AI labels:\n" + classification_report(y, cv, zero_division=0))
    clf.fit(X[idx], y)
    return clf, rep

# (a) relevance
y_rel = tags["is_relevant"].astype(int).values
clf_rel, rep_rel = fit_report(y_rel, ti, "relevance")
corpus["p_relevant"] = clf_rel.predict_proba(X)[:, 1]
# (b) stop/pause signal among relevant posts
STOP = {"paused", "skipped", "reduced", "stopped", "redeemed", "considering_stopping"}
rel_tags = tags[tags.is_relevant]
y_stop = rel_tags["sip_action"].isin(STOP).astype(int).values
clf_stop, rep_stop = fit_report(y_stop, rel_tags["id"].map(pos).astype(int).values, "stop/pause signal")
corpus["p_stop"] = clf_stop.predict_proba(X)[:, 1]
# AI label wins where we have one
lab = tags.set_index("id")
corpus["ai_tagged"] = corpus["id"].isin(lab.index)
corpus["relevant"] = np.where(corpus.ai_tagged, corpus["id"].map(lab["is_relevant"]).fillna(False), corpus.p_relevant >= 0.5)
corpus["relevant"] = corpus["relevant"].astype(bool)
corpus["stop_signal"] = np.where(corpus["id"].map(lab["sip_action"]).isin(STOP), True,
                                 np.where(corpus.ai_tagged, False, corpus.p_stop >= 0.5)).astype(bool)
rel = corpus[corpus.relevant].copy()
print(f"\nRelevant posts (full corpus): {len(rel):,} of {len(corpus):,}")
print(rel.groupby("source").size().to_string())

# clustering
Er = E[rel.index.values]
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Er)
rel["cluster"] = km.labels_
dist = np.linalg.norm(Er - km.cluster_centers_[km.labels_], axis=1)
rel["dist"] = dist
# distinctive terms per cluster (class-based tf-idf)
docs = rel.groupby("cluster")["text"].apply(lambda s: " ".join(s.str.lower().str[:800]))
tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=3,
                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
M = tf.fit_transform(docs.values); vocab = np.array(tf.get_feature_names_out())
with open(os.path.join(D, "clusters_for_naming.md"), "w") as f:
    for c in range(K):
        g = rel[rel.cluster == c].sort_values("dist")
        terms = vocab[np.argsort(-M[c].toarray()[0])[:15]]
        f.write(f"\n## Cluster {c}  (n={len(g):,}, stop/pause share={g.stop_signal.mean():.0%}, "
                f"sources={g.source.value_counts().head(3).to_dict()})\nTerms: {', '.join(terms)}\n")
        for t in g.head(10).itertuples():
            f.write(f"- [{t.source}/{t.group}] {re.sub(r'\s+', ' ', t.text)[:280]}\n")
corpus = corpus.merge(rel[["id", "cluster"]], on="id", how="left")
corpus.drop(columns=[]).to_parquet(os.path.join(D, "corpus_scored.parquet"), index=False)
json.dump({"relevance": rep_rel, "stop_signal": rep_stop}, open(os.path.join(D, "classifier_cv.json"), "w"), indent=1)
print("\nwrote data/clusters_for_naming.md and data/corpus_scored.parquet")
