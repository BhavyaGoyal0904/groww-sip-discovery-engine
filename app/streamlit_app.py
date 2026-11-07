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

tab_dash, tab_ask, tab_tag, tab_eval, tab_flow = st.tabs(["Dashboard", "Ask the data", "Tag a review (live)", "Quality & limits", "Workflow"])

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

# ---------------- How the engine was built ----------------
with tab_flow:
    st.markdown("How 430k public posts became a ranked list of opportunity areas. Each step is labelled with **who does the work**: "
                "rules (free, deterministic), a local ML model, an LLM, or a human review.")
    st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; nodesep=0.45; ranksep=0.35; bgcolor="transparent"; newrank=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.06", penwidth=0.8, width=1.9];
  edge [color="#6b7280", arrowsize=0.7];
  src [label="1  COLLECT\n6 public sources\n{meta['raw_rows']:,} posts", fillcolor="#e5e7eb"];
  flt [label="2  RULE FILTER [rules]\nMF terms, spam, dupes\n→ {meta['filtered']:,}", fillcolor="#dbeafe"];
  emb [label="3  EMBED [local ML]\nmultilingual MiniLM\n384-dim vectors", fillcolor="#dbeafe"];
  smp [label="4  SAMPLE [rules]\n{meta['ai_tagged']:,} posts\n70% priority, 30% random", fillcolor="#dbeafe"];
  tag [label="5  AI TAGGING [LLM]\nrelevance + 24-field\ncodebook", fillcolor="#fde68a"];
  val [label="VALIDATOR [rules]\nschema · values · quote\n{meta['validator_errors']} errors left", fillcolor="#fecaca", shape=octagon];
  cls [label="6  SCALE UP [local ML]\nclassifier, precision {meta['cv']['rel_p']:.0%}\n→ {meta['relevant_full']:,} relevant", fillcolor="#dbeafe"];
  clu [label="7  CLUSTER [local ML]\nk-means, k = 30", fillcolor="#dbeafe"];
  nam [label="8  NAME [LLM]\nterms + sample posts", fillcolor="#fde68a"];
  hum [label="9  REVIEW [human]\n30 clusters → 10 themes\ncodebook v2", fillcolor="#bbf7d0", penwidth=2];
  scr [label="10  RANK [rules]\nreach · severity ·\nfirst-year · addressability", fillcolor="#dbeafe"];
  out [label="OPPORTUNITY\nAREAS\nmapped to M1–M6", fillcolor="#e5e7eb", shape=doubleoctagon];
  src -> flt -> emb -> smp -> tag -> val;
  val -> cls;
  cls -> clu -> nam -> hum -> scr -> out [constraint=false];
  val -> tag [label=" fail → re-code", fontsize=9, fontcolor="#6b7280", style=dashed, constraint=false];
  {{rank=same; src; out}} {{rank=same; flt; scr}} {{rank=same; emb; hum}}
  {{rank=same; smp; nam}} {{rank=same; tag; clu}} {{rank=same; val; cls}}
  src -> out [style=invis]; flt -> scr [style=invis]; emb -> hum [style=invis];
  smp -> nam [style=invis]; tag -> clu [style=invis];
}}""", width="stretch")
    st.caption("Follow the arrows: the top row runs left to right, then the flow turns down and the bottom row runs right to left.")
    st.caption("Legend: grey = data · blue = rules / local ML (free, repeatable) · yellow = LLM · red octagon = automatic quality gate · "
               "green, thick border = human review. Labels in [brackets] say the same thing for anyone who cannot rely on colour.")

    st.subheader("Step by step")
    st.caption("Open any step to see exactly how it was done.")
    STEPS = [
        ("1 · Collect", "Scrapers / public APIs", f"{meta['raw_rows']:,} posts", "Source and date kept for every post",
         ["Pulled public posts from Google Play, App Store, Reddit, TradingQ&A / ValuePickr, X / LinkedIn and YouTube comments (Nov 2024 – Nov 2026).",
          "Mapped all six exports to one schema: source, group, thread title, text, author, date, rating, URL, is-reply.",
          "Kept the thread title next to every reply so a short comment still has context."]),
        ("2 · Rule filter", "Rules (regex)", f"{meta['filtered']:,} posts", "Counts logged per rule",
         ["Dropped deleted / empty text and platform-staff replies.",
          "Dropped trading-only posts (F&O, intraday, IPO) that never mention SIPs or mutual funds.",
          "Dropped spam: Telegram links, referral codes, phone numbers, fewer than 25 letters.",
          "Dropped posts under 60 characters, then removed exact duplicates.",
          "Flagged priority posts: describes an action, a feeling, a first-time investor or a journey stage."]),
        ("3 · Embed", "Local ML (multilingual MiniLM)", "1 vector per post", "Runs offline, free, repeatable",
         ["Encoded every post with paraphrase-multilingual-MiniLM-L12-v2 (handles Hindi and Hinglish).",
          "Stored a 384-dimension vector per post, so posts with similar meaning sit close together."]),
        ("4 · Stratified sample", "Rules", f"{meta['ai_tagged']:,} posts", "Fixed quota per source",
         ["Set a quota per source so Reddit volume does not drown out app reviews.",
          "Filled 70% of each quota with priority posts and 30% at random, so the AI also sees irrelevant posts.",
          "Split the sample into 10 batches for tagging."]),
        ("5 · AI relevance + codebook tagging", "LLM", f"{meta['ai_relevant']:,} relevant", "Validator: schema, allowed values, verbatim quote",
         ["The LLM first decides whether the post is about a real person's own SIP / MF experience.",
          "If relevant, it codes 24 fields from closed lists (stage, action, triggers, checking, outside-app help, segment, unmet need).",
          "It must copy a word-for-word evidence quote from the post.",
          "A rule-based validator checks every output; failures go back for re-coding until 0 errors remain."]),
        ("6 · Scale to full corpus", "Local ML (logistic regression)", f"{meta['relevant_full']:,} relevant (est.)",
         f"5-fold CV: accuracy {meta['cv']['rel_acc']:.0%}, precision {meta['cv']['rel_p']:.0%}",
         ["Trained a classifier on the AI's relevance labels using embeddings plus rule features.",
          "Checked it with 5-fold cross-validation before trusting it.",
          "Applied it to every filtered post; prevalence is re-weighted for the sampling design."]),
        ("7 · Cluster", "Local ML (k-means)", "30 clusters", "Representative posts kept per cluster",
         ["Grouped all relevant posts by meaning with k-means (k = 30) on the embeddings.",
          "Extracted each cluster's distinctive terms (c-TF-IDF) and its most central posts."]),
        ("8 · Name clusters", "LLM", "30 draft names", "Names checked against the posts",
         ["The LLM proposed a name for each cluster from its terms and representative posts.",
          "Names are drafts only; a human reviews them in the next step."]),
        ("9 · Review and merge", "Human", "10 themes, codebook v2", "Every change logged in the Evidence Log",
         ["Merged near-duplicate clusters into 10 themes and mapped each to the metric tree (M1–M6).",
          "Fixed codebook gaps (e.g. added 'irregular income vs fixed SIP' and 'platform cancelled SIP').",
          "Standardised rules where AI coders disagreed (most-severe action wins, contract work = variable income).",
          "Rated each theme's addressability (1–5) for Groww without fund advice or fear / FOMO."]),
        ("10 · Quantify and rank", "Rules", "Ranked opportunities", "Small-sample numbers flagged",
         ["Scored each theme on reach, severity, first-year share and addressability (weights below).",
          "Ran cross-tabs (expectation × outcome, stage × outcome, influence × outcome) to find the strongest signals.",
          "Ranked themes and wrote the findings, gaps and limitations."]),
    ]
    for title, by, out_, qc, sub in STEPS:
        with st.expander(f"**{title}**  —  {by}  ·  {out_}"):
            st.markdown("\n".join(f"{i}. {x}" for i, x in enumerate(sub, 1)))
            st.caption(f"Done by: {by}  ·  Output: {out_}  ·  Quality check: {qc}")

    a, b = st.columns(2)
    with a:
        st.subheader("Opportunity score")
        st.markdown("""
| Factor | Weight | Source |
|---|---|---|
| Reach (√ share of relevant posts) | 25% | full corpus |
| Severity (share at risk or dropped) | 30% | AI-tagged posts |
| First-year share | 15% | AI-tagged posts |
| Addressability (1–5) | 30% | human rating |

Each factor is min-max normalised, then weighted. Addressability asks how much a Groww product change could move the behaviour
without recommending funds or using fear / FOMO.
""")
    with b:
        st.subheader("The codebook (24 fields)")
        st.markdown("""
- **Journey:** relevance, content type, journey stage (0–3, 3–6, 6–12, 12+ months)
- **Why they started:** motivation, expectation
- **What happened:** SIP action (continued, paused, skipped, reduced, stopped, redeemed, switched…), triggers (19 values)
- **Checking:** how often, what they look at
- **Feelings and help:** emotion, where they went outside the app, what confused them in the app
- **Segment signals:** income stability, age band, money source, who influenced them, experience
- **Outcome:** continuation signal, unmet need, verbatim evidence quote, confidence
""")

    st.subheader("Where a human corrected the AI")
    st.markdown("""
1. **Inconsistent relevance rules between AI coders** (generic "don't stop your SIP" advice and industry statistics coded both ways) → one rule in codebook v2.
2. **Missing codebook values** (irregular income vs a fixed SIP; a SIP cancelled by the platform) → added to codebook v2.
3. **Over-split clusters**: "review my portfolio" posts split into 13 near-duplicates → merged into 2 themes.
4. **Unreliable classifier**: stop/pause classifier precision was only """ + f"{meta['cv']['stop_p']:.0%}" + """ → severity taken from AI-tagged posts instead.
""")

    st.subheader("How the live features work")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ask the data**")
        st.graphviz_chart("""
digraph A { rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, penwidth=0.8, fillcolor="#dbeafe"];
  q [label="Question"]; g [label="Advice guard [rules]\\nfund-pick questions declined", shape=octagon, fillcolor="#fecaca"];
  r [label="Retrieve 18 posts [BM25]"]; l [label="Answer only from them [LLM]", fillcolor="#fde68a"];
  v [label="Checks [rules]: ≥2 citations, no invented P#,\\nverbatim quotes, no advice", shape=octagon, fillcolor="#fecaca"];
  s [label="Show answer + checks", fillcolor="#bbf7d0"]; f [label="Withhold → show raw evidence", fillcolor="#e5e7eb"];
  q -> g -> r -> l -> v; v -> s [label=" pass", fontsize=9]; v -> f [label=" fail", fontsize=8]; }""", width="content")
    with c2:
        st.markdown("**Tag a review**")
        st.graphviz_chart("""
digraph B { rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, penwidth=0.8, fillcolor="#dbeafe"];
  p [label="Pasted post"]; l [label="Code against codebook v2 [LLM]", fillcolor="#fde68a"];
  v [label="Checks [rules]: all fields, allowed values,\\nverbatim quote, action ↔ signal agree, no advice", shape=octagon, fillcolor="#fecaca"];
  s [label="Show tags + theme it feeds", fillcolor="#bbf7d0"]; f [label="Withhold tags, show which check failed", fillcolor="#e5e7eb"];
  p -> l -> v; v -> s [label=" pass", fontsize=9]; v -> f [label=" fail", fontsize=8]; }""", width="content")

    st.subheader("Stack")
    st.markdown("Python · pandas · sentence-transformers (multilingual MiniLM) · scikit-learn (logistic regression, k-means) · "
                "LLM tagging with a JSON-schema codebook · Mistral (live app) · Streamlit · code on "
                "[GitHub](https://github.com/BhavyaGoyal0904/groww-sip-discovery-engine).")

# ---------------- Quality & limits ----------------
with tab_eval:
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
