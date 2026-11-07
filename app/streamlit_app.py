"""SIP Drop-off Discovery Engine — Groww Growth case study (Part 1).
Dashboard of themes, evidence-cited Q&A over tagged posts, and a live codebook tagger with visible quality checks."""
import json, os
import pandas as pd
import streamlit as st
import llm

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="SIP Drop-off Discovery Engine", page_icon="📉", layout="wide")


@st.cache_data
def load():
    P = json.load(open(os.path.join(HERE, "data", "posts.json")))
    T = json.load(open(os.path.join(HERE, "data", "themes.json")))
    M = json.load(open(os.path.join(HERE, "data", "meta.json")))
    X = open(os.path.join(HERE, "data", "crosstabs.txt")).read()
    return P, T, M, X


@st.cache_resource
def index(_posts):
    return llm.BM25([f"{p['title']} {p['text']} {p['need']}" for p in _posts])


@st.cache_data
def codebook():
    import codebook as cb
    props = cb.SCHEMA["properties"]
    enums = {k: (v["items"]["enum"] if v["type"] == "array" else v.get("enum")) for k, v in props.items()
             if k not in ("is_relevant", "unmet_need", "evidence_quote")}
    multi = [k for k, v in props.items() if v["type"] == "array"]
    return cb.SYSTEM, enums, multi


posts, themes, meta, crosstabs = load()
bm25 = index(posts)
SYSTEM, ENUMS, MULTI = codebook()
theme_names = [t["name"] for t in themes]
HAS_KEY = bool(llm.get_key())

st.title("SIP Drop-off Discovery Engine")
st.caption("Why do first-time mutual-fund investors stop their SIPs in the first 12 months? Built from 430k public posts "
           "(Play Store, App Store, Reddit, TradingQ&A/ValuePickr, X/LinkedIn, YouTube). "
           "**Research tool — not investment advice.** It never recommends funds or stocks.")
if not HAS_KEY:
    st.warning("AI features are off: no MISTRAL_API_KEY is configured. The dashboard and evidence search still work.")

tab_dash, tab_ask, tab_tag, tab_eval = st.tabs(["Dashboard", "Ask the data", "Tag a review (live)", "How it works & quality"])

# ---------------- Dashboard ----------------
with tab_dash:
    c = st.columns(5)
    c[0].metric("Raw posts collected", f"{meta['raw_rows']:,}")
    c[1].metric("After rule filter", f"{meta['filtered']:,}")
    c[2].metric("AI-tagged (codebook)", f"{meta['ai_tagged']:,}")
    c[3].metric("Relevant among tagged", f"{meta['ai_relevant']:,}")
    c[4].metric("Relevant, full corpus (est.)", f"{meta['relevant_full']:,}")

    st.subheader("Ranked opportunity areas")
    st.caption("Opportunity score = 25% reach + 30% severity (share of AI-tagged posts at risk or dropped) + 15% first-year share "
               "+ 30% addressability (how much a Groww product change can fix it, 1–5, human-rated). Each factor min-max normalised.")
    df = pd.DataFrame(themes).sort_values("score", ascending=False)
    st.dataframe(
        df[["name", "score", "posts", "risk", "first_year", "first_timer", "addr", "n", "node", "triggers"]],
        hide_index=True, width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Theme", width="large"),
            "score": st.column_config.ProgressColumn("Opportunity score", min_value=0, max_value=100, format="%.1f"),
            "posts": st.column_config.NumberColumn("Posts (full corpus)", format="%d"),
            "risk": st.column_config.NumberColumn("At risk / dropped", format="percent"),
            "first_year": st.column_config.NumberColumn("In first year", format="percent"),
            "first_timer": st.column_config.NumberColumn("First-time investors", format="percent"),
            "addr": st.column_config.NumberColumn("Addressability (1–5)"),
            "n": st.column_config.NumberColumn("AI-tagged n"),
            "node": st.column_config.TextColumn("Metric-tree node"),
            "triggers": st.column_config.TextColumn("Top triggers"),
        })

    st.subheader("What separates people who continue from those who drop off")
    tagged = pd.DataFrame(posts)
    a, b = st.columns(2)
    with a:
        st.markdown("**Starting expectation → continuation** (row %)")
        k = tagged[tagged.exp != "not_mentioned"]
        st.dataframe((pd.crosstab(k.exp, k.sig, normalize="index") * 100).round(0), width="stretch")
        st.caption(f"n = {len(k)} posts that state an expectation. Those who understood volatility mostly continue.")
    with b:
        st.markdown("**Journey stage → continuation** (counts)")
        st.dataframe(pd.crosstab(tagged.stage, tagged.sig), width="stretch")
        st.caption("Months 6–12 has the highest at-risk + dropped share among posts that state a stage.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Triggers among at-risk or dropped posts**")
        risk = tagged[tagged.sig.isin(["at_risk", "dropped"])]
        s = pd.Series([t for l in risk.trig for t in l if t != "none"]).value_counts().head(10)
        st.bar_chart(s, horizontal=True)
    with b:
        st.markdown("**Where people go when unsure (outside the app)**")
        s = pd.Series([t for l in tagged.out for t in l if t != "none_mentioned"]).value_counts()
        st.bar_chart(s, horizontal=True)

    st.subheader("Browse the evidence")
    f1, f2, f3 = st.columns(3)
    th = f1.selectbox("Theme", ["All"] + theme_names)
    sig = f2.multiselect("Continuation signal", ["at_risk", "dropped", "likely_continue", "unclear"], default=["at_risk", "dropped"])
    src = f3.multiselect("Source", sorted(tagged.src.unique()))
    view = tagged
    if th != "All":
        view = view[view.th == theme_names.index(th)]
    if sig:
        view = view[view.sig.isin(sig)]
    if src:
        view = view[view.src.isin(src)]
    st.caption(f"{len(view)} posts")
    st.dataframe(view[["n", "src", "grp", "stage", "act", "sig", "q", "need", "url"]].rename(columns={
        "n": "P#", "src": "Source", "grp": "App / community", "stage": "Stage", "act": "SIP action", "sig": "Signal",
        "q": "Evidence quote", "need": "Unmet need (AI)", "url": "Link"}), hide_index=True, width="stretch",
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")})
    with st.expander("All cross-tabs (raw)"):
        st.text(crosstabs)

# ---------------- Ask the data ----------------
with tab_ask:
    st.markdown("Ask a question about SIP drop-off. The engine retrieves the most relevant tagged posts, the AI answers **only "
                "from them with [P#] citations**, and the answer is shown only if it passes the checks below.")
    examples = ["Why do people stop their SIP in the first year?",
                "What do investors worry about when XIRR is negative after a few months?",
                "What goes wrong when people try to skip or pause a SIP?",
                "Where do beginners go for advice when they feel unsure?"]
    ex = st.radio("Examples", examples, index=None, horizontal=False)
    q = st.text_input("Your question", value=ex or "", max_chars=300)
    tf = st.selectbox("Limit to theme (optional)", ["All themes"] + theme_names, key="ask_theme")
    if st.button("Ask", type="primary", disabled=not q.strip()):
        if llm.is_advice_request(q):
            st.error("This looks like a request for fund or investment advice. The engine studies investor behaviour and does not "
                     "recommend funds. For scheme information see official sources such as AMFI (amfiindia.com) or the fund's SID.")
        else:
            allowed = None if tf == "All themes" else {i for i, p in enumerate(posts) if p["th"] == theme_names.index(tf)}
            hits = [posts[i] for i in bm25.search(q, k=18, allowed=allowed)]
            if len(hits) < 3:
                st.warning(f"Only {len(hits)} matching posts found, which is too little evidence to answer. Try different words.")
            elif not HAS_KEY:
                st.info("AI is off, so here are the top matching posts (non-AI fallback).")
                for p in hits[:8]:
                    st.markdown(f"**[P{p['n']}]** {p['src']}/{p['grp']} · {p['sig']} — “{p['q'] or p['text'][:200]}”")
            else:
                with st.spinner("Reading the evidence..."):
                    try:
                        out, raw = llm.chat_json(llm.ask_prompt(q, hits), llm.ASK_SYSTEM)
                        checks, good_quotes = llm.validate_answer(out, hits)
                    except RuntimeError as e:
                        out, checks = None, None
                        st.error(str(e))
                if out is not None:
                    passed = all(ok for _, ok, _ in checks)
                    if not out.get("sufficient_evidence", True):
                        st.warning("The AI says the evidence is insufficient: " + out.get("answer", ""))
                    elif passed:
                        st.success("Answer (passed all quality checks)")
                        st.markdown(out.get("answer", ""))
                        for qq in good_quotes:
                            st.markdown(f"> “{qq.get('quote')}” — [P{qq.get('p')}]")
                    else:
                        st.error("The AI answer failed a quality check, so it is not shown. Here is the raw evidence instead.")
                        for p in hits[:6]:
                            st.markdown(f"**[P{p['n']}]** “{p['q'] or p['text'][:200]}”")
                    with st.expander("Quality checks", expanded=not passed):
                        st.table(pd.DataFrame(checks, columns=["Check", "Passed", "Detail"]))
                    with st.expander(f"Retrieved posts ({len(hits)})"):
                        for p in hits:
                            st.markdown(f"**[P{p['n']}]** {p['src']}/{p['grp']} · stage {p['stage']} · {p['act']} · {p['sig']}  \n"
                                        f"{p['text'][:400]}  \n[source]({p['url']})")

# ---------------- Live tagger ----------------
with tab_tag:
    st.markdown("Paste any app review, Reddit post or YouTube comment about SIPs. The AI codes it against the same 24-field "
                "codebook used on the corpus, and every output goes through visible quality checks. If a check fails, the tags are "
                "not shown.")
    samples = {
        "Example: negative XIRR at month 7": "It's been 7-8 months since I started and this is the first time returns has gone less than what I invested. XIRR shows -2.2%. Should I just move to FD?",
        "Example: skip SIP bug": "Groww team, there is a critical bug while trying to skip sip on iPhone. I am taken to the login screen whenever I try to skip. Will move to other platforms if not fixed soon.",
        "Example: irregular income": "I stopped my 3k SIP because I don't have fixed income, freelancing months are uneven. Now I just invest lumpsum when I can.",
        "Example: not relevant": "Great app for intraday trading, charts are smooth.",
    }
    pick = st.selectbox("Load an example (optional)", ["—"] + list(samples))
    review = st.text_area("Post to tag", value=samples.get(pick, ""), height=140, max_chars=3000)
    if st.button("Tag it", type="primary", disabled=len(review.strip()) < 20 or not HAS_KEY):
        with st.spinner("Tagging..."):
            try:
                out, raw = llm.chat_json(llm.tagger_prompt(review, ENUMS, MULTI, theme_names), SYSTEM)
                checks = llm.validate_tags(out, review, ENUMS, MULTI)
            except RuntimeError as e:
                out = None
                st.error(str(e))
        if out is not None:
            passed = all(ok for _, ok, _ in checks)
            if passed and not out.get("is_relevant"):
                st.info("Not relevant to the SIP journey (e.g. trading, generic praise, news). Nothing to tag.")
            elif passed:
                st.success("Tagged — passed all quality checks")
                th_i = out.get("theme", -1)
                c1, c2, c3 = st.columns(3)
                c1.metric("Continuation signal", out.get("continuation_signal"))
                c2.metric("SIP action", out.get("sip_action"))
                c3.metric("Journey stage", out.get("journey_stage"))
                if isinstance(th_i, int) and 0 <= th_i < len(themes):
                    t = themes[th_i]
                    st.markdown(f"**Feeds opportunity theme:** {t['name']} (score {t['score']}, metric node {t['node']})")
                st.markdown(f"**Triggers:** {', '.join(out.get('triggers', [])) or '—'}  \n"
                            f"**Emotion:** {out.get('emotion')}  ·  **Outside-app help:** {', '.join(out.get('outside_app_help', [])) or '—'}  \n"
                            f"**Segment signals:** income {out.get('income_stability')}, age {out.get('age_band')}, "
                            f"money from {out.get('money_source')}, influenced by {out.get('influenced_by')}  \n"
                            f"**Unmet need:** {out.get('unmet_need')}  \n**Evidence:** “{out.get('evidence_quote')}”")
            else:
                st.error("The AI output failed a quality check, so it is not shown as a result. "
                         "Try again, or shorten the post to the part about the SIP.")
            st.table(pd.DataFrame(checks, columns=["Check", "Passed", "Detail"]))
            with st.expander("Raw JSON"):
                st.json(out)
    if not HAS_KEY:
        st.caption("Tagging needs a MISTRAL_API_KEY.")

# ---------------- How it works ----------------
with tab_eval:
    st.subheader("Pipeline")
    st.markdown(f"""
1. **Collect** — {meta['raw_rows']:,} posts from 6 sources, Nov 2024–Nov 2026, normalised to one schema.
2. **Rule filter (free)** — keep SIP/MF posts; drop <60 chars, deleted, staff replies, trading-only, spam, duplicates → **{meta['filtered']:,}**.
3. **Embed** — multilingual MiniLM (handles Hindi/Hinglish), run locally.
4. **AI relevance + codebook tagging** — stratified sample of {meta['ai_tagged']:,} posts tagged on 24 fields by an LLM; a validator
   rejects any output with missing fields, illegal values, or a quote that is not verbatim → **{meta['validator_errors']} errors** after fixes.
5. **Scale to full corpus** — classifier trained on AI labels: relevance accuracy {meta['cv']['rel_acc']:.0%},
   precision {meta['cv']['rel_p']:.0%} (5-fold CV) → **{meta['relevant_full']:,}** relevant posts.
6. **Themes** — k-means (k=30) → AI-named clusters → human review merged them into **10 themes**.
7. **Quantify & rank** — reach, severity, first-year share, segment skew, addressability → opportunity score.
""")
    st.subheader("Quality gates in this app")
    st.markdown("""
- **Ask the data:** answers come only from retrieved posts; must cite ≥2 posts; every [P#] must exist; quoted words must appear in
  the cited post; advice-seeking questions are declined and pointed to official sources (AMFI, scheme documents).
- **Tag a review:** all fields present; only allowed codebook values; evidence quote must be verbatim; action and signal must agree;
  no advice language. Failing outputs are withheld, not shown.
""")
    st.subheader("Known limits")
    st.markdown(f"""
- Stop/pause classifier precision was only {meta['cv']['stop_p']:.0%}, so severity uses AI-tagged posts, not the classifier.
- Reddit dominates the relevant posts; app-store reviews over-represent complaints; people who quietly stop rarely post.
- Segment fields (income stability, money source) are stated in fewer than 100 posts, so segment patterns need primary research.
""")
