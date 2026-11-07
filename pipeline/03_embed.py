"""Embed every filtered post locally (free). Multilingual model handles Hindi/Hinglish."""
import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
df = pd.read_parquet(os.path.join(D, "corpus_filtered.parquet"))
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
txt = (df["thread_title"].fillna("").str[:120] + " | " + df["text"].str[:1500]).tolist()
E = m.encode(txt, batch_size=128, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
np.save(os.path.join(D, "embeddings.npy"), E.astype(np.float16))
df[["id"]].to_parquet(os.path.join(D, "embeddings_ids.parquet"), index=False)
print("done", E.shape)
