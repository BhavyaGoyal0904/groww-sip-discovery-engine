"""Stage 1-2: normalise 6 sources, rule-filter, dedupe, prioritise for LLM tagging."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, glob, re, hashlib, sys, os
SRC = sys.argv[1]; OUT = os.path.join(os.path.dirname(__file__), "..", "data")

MF = re.compile(r"\b(sips?|mutual funds?|mf|mfs|nav|xirr|cagr|autopay|auto[- ]?debit|mandate|instal+ments?|step[- ]?up|redeem|redemption|elss|index funds?|flexi ?cap|small ?cap|mid ?cap|large ?cap|nifty ?50|folio)\b", re.I)
ACTION = re.compile(r"\b(stop+(ed|ing)?|paus(e|ed|ing)|skip+(ed|ing)?|cancel+(ed|ing)?|redeem(ed|ing)?|withdr[ae]w|reduc(e|ed|ing)|discontinu\w*|continu\w*|quit|exit(ed)?|keep going|stay invested|break)\b", re.I)
FEEL = re.compile(r"\b(negative|red|loss|losing|lost|down|worried|scared|panic|confus\w*|doubt|underwhelm\w*|disappoint\w*|expected|expectation|returns?|xirr|worth it|should i)\b", re.I)
FIRSTP = re.compile(r"\b(i|i'm|im|i've|my|me|mera|meri|maine|mujhe)\b", re.I)
STAGE = re.compile(r"\b(\d{1,2}\s*(months?|mnths?|yrs?|years?)|first (sip|time|year|month)|beginner|newbie|new to|just started|started (a |my )?sip)\b", re.I)
TRADING = re.compile(r"\b(intraday|f&o|fno|futures?|options?|option chain|call|put|strike|scalp\w*|stop ?loss|sl hit|margin|brokerage|ipo|gmp|chart|candle\w*|kite|tick|algo|square ?off|lot size|expiry|nifty options|banknifty|swing trad\w*)\b", re.I)
SPAM = re.compile(r"(t\.me/|telegram|whats ?app (me|group|no)|join (my|our) (channel|group)|dm me|refer(ral)? code|use my code|\+91[ -]?\d{10}|https?://\S+\s*$)", re.I)
JUNK = {"[removed]", "[deleted]", "[removed in archive]", ""}

frames = []
for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    df = pd.read_csv(f, dtype=str, keep_default_na=False, on_bad_lines="skip")
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
n0 = len(df)
df["text"] = df["text"].str.strip()
df = df[~df["text"].isin(JUNK)]
df = df[~df["user_flag"].str.contains("staff|Fund House|admin", case=False, na=False)]
df["ctx"] = (df["title"] + " " + df["thread_title"]).str.strip()
full = df["ctx"] + " " + df["text"]
mf_in_text = df["text"].str.contains(MF)
mf_ctx = df["ctx"].str.contains(MF) | df["group"].isin(["r/mutualfunds"])
# app stores need explicit MF/SIP language in the review; discussion sources can inherit from thread context
is_store = df["section"].str.contains("Store|Trustpilot")
trading_only = df["text"].str.contains(TRADING) & ~mf_in_text
spam = df["text"].str.contains(SPAM) | df["text"].str.fullmatch(r"(.)\1{5,}.*") | (df["text"].str.count(r"[A-Za-z\u0900-\u097F]") < 25)
print(f"dropping trading-only: {(trading_only).sum():,}  spam/non-text: {(spam).sum():,}")
keep = ~trading_only & ~spam & (mf_in_text | (~is_store & mf_ctx)) & (df["text"].str.len() >= 60)
df = df[keep].copy()
df["norm"] = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
df = df.drop_duplicates("norm")
df["id"] = df.apply(lambda r: hashlib.md5((r["url"] + r["norm"][:200]).encode()).hexdigest()[:12], axis=1)
# priority score = how likely this is a first-person journey story useful for continuation analysis
t = df["ctx"] + " " + df["text"]
df["p_action"] = t.str.contains(ACTION).astype(int)
df["p_feel"] = df["text"].str.contains(FEEL).astype(int)
df["p_first"] = df["text"].str.contains(FIRSTP).astype(int)
df["p_stage"] = t.str.contains(STAGE).astype(int)
df["p_mftext"] = mf_in_text.loc[df.index].astype(int)
df["priority"] = 2*df.p_action + df.p_feel + df.p_first + 2*df.p_stage + df.p_mftext + (df["text"].str.len() > 200).astype(int)
df["source"] = df["section"].replace({"Investment Forums & Review Sites": "Forums", "Public Social Media": "Social",
                                      "Google Play Store": "Play Store", "Apple App Store": "App Store"})
cols = ["id","source","group","thread_title","title","text","date","rating","score","url","is_reply","priority","p_action","p_feel","p_first","p_stage"]
df[cols].to_parquet(os.path.join(OUT, "corpus_filtered.parquet"), index=False)

print(f"raw rows: {n0:,} -> filtered+deduped: {len(df):,}")
print(df.groupby("source").size().to_string())
print("\npriority distribution:\n", df["priority"].value_counts().sort_index().to_string())
for th in (5, 6, 7):
    s = df[df.priority >= th]
    print(f"\npriority>={th}: {len(s):,}  ", s.groupby("source").size().to_dict())
