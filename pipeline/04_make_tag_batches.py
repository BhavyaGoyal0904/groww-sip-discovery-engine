"""Stratified 1,500-post sample for AI tagging (also the training set for the relevance classifier).
70% from high rule-priority (>=5), 30% from the rest so the classifier also sees negatives."""
import os, json, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
df = pd.read_parquet(os.path.join(D, "corpus_filtered.parquet"))
QUOTA = {"Reddit": 700, "YouTube": 300, "Play Store": 250, "App Store": 100, "Forums": 120, "Social": 30}
parts = []
for src, n in QUOTA.items():
    s = df[df.source == src]
    if src in ("Play Store", "App Store"):
        s = pd.concat([s[s.group == "Groww"], s[s.group != "Groww"]])  # Groww first
        hi_pool = s[s.priority >= 4]
    else:
        hi_pool = s[s.priority >= 5]
    n_hi = min(int(n * 0.7), len(hi_pool))
    hi = hi_pool.sample(n_hi, random_state=7) if src not in ("Play Store", "App Store") else hi_pool.head(n_hi)
    rest = s.drop(hi.index)
    lo = rest.sample(min(n - n_hi, len(rest)), random_state=7)
    parts += [hi.assign(stratum="high_priority"), lo.assign(stratum="random")]
sample = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
sample.to_parquet(os.path.join(D, "tag_sample.parquet"), index=False)
print(sample.groupby(["source", "stratum"]).size().to_string(), "\ntotal", len(sample))
B = 10
for b in range(B):
    chunk = sample.iloc[b::B]
    with open(os.path.join(D, "tag_in", f"batch_{b:02d}.jsonl"), "w") as f:
        for r in chunk.itertuples():
            f.write(json.dumps({"id": r.id, "source": r.source, "group": r.group, "rating": r.rating,
                                "thread_title": (r.thread_title or "")[:200], "title": (r.title or "")[:200],
                                "text": r.text[:2500]}, ensure_ascii=False) + "\n")
print("wrote", B, "batches of ~", len(sample)//B)
