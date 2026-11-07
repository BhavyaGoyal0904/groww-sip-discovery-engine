# Groww SIP continuation: AI discovery engine results

Data source: the 6 CSVs in `~/Downloads/Groww_SIP_MF_User_Feedback_Report`, covering Nov 2024 to Nov 2026.

## 1. Pipeline and volumes

| Stage | What happens | Output |
|---|---|---|
| 0. Collect | 6 sources (Play Store, App Store, Reddit, forums, social, YouTube) normalised to one schema | **430,169** rows |
| 1. Rule filter (free) | Keeps posts that mention SIPs or MFs. Drops posts under 60 characters, deleted posts, Zerodha staff and admin replies, trading-only posts (F&O, intraday, IPO) without MF terms, spam (Telegram, referral codes, phone numbers, text that isn't really words) and duplicates | **73,625** posts |
| 2. Local embeddings | Multilingual MiniLM model (handles Hindi and Hinglish), run on the laptop | 73,625 × 384 vectors |
| 3. AI relevance and tagging | Claude agents tagged a stratified sample of 1,500 posts (70% high rule priority, 30% random, across all 6 sources) against a 24-field codebook. An automatic validator checks the schema, the allowed values, and that each **quote appears word for word** in the post | 1,500 tagged posts, **993 relevant**, 0 validator errors |
| 3b. Scale to full corpus | A logistic-regression classifier trained on the AI labels (embeddings plus rule features) | Relevance at **78% accuracy** (5-fold CV) and **87% precision** on the random stratum. **38.6k relevant posts** across the full corpus |
| 4. Themes | K-means (k=30) on relevant posts, AI names each cluster, then a human review merges them | 30 clusters merged into **10 themes** |
| 5. Quantify and rank | For each theme: reach, at-risk-or-dropped share, first-year share, first-timer share, segment skew, addressability | Opportunity score (0–100) |

**Opportunity score** = 25% reach (square-root scaled) + 30% severity (at-risk or dropped) + 15% first-year share + 30% addressability. Each factor is min-max normalised.

## 2. Ranked opportunity areas

| # | Theme | Posts in corpus | AI-tagged n | At risk or dropped | Addressability (1–5) | Score | Metric node |
|---|---|---|---|---|---|---|---|
| 1 | SIP mechanics fail: mandate failures, debited but rejected, skip/modify/cancel flows | 1,152 | 56 | **57%** | 5 | **77.7** | M1 Instalment success |
| 2 | Early returns below expectation ("what am I doing wrong?") | 3,612 | 145 | 41% | 5 | **64.7** | M2 Voluntary stop: how returns are perceived |
| 3 | Money movement and account issues (redemption, bank, KYC) | 1,146 | 42 | 45% | 4 | 56.5 | M6 Platform trust |
| 4 | App can't show my portfolio clearly (bugs, updates) | 1,697 | 109 | 42% | 4 | 50.5 | M6 Platform trust |
| 5 | Portfolio tinkering: too many funds, overlap | 10,805 | 211 | 13% | 3 | 48.6 | M5 Portfolio churn |
| 6 | My specific fund is lagging: stay, stop or switch? | 3,231 | 68 | 34% | 3 | 46.4 | M2 Voluntary stop: how returns are perceived |
| 7 | SIP vs real-life cash flow (salary, loans, emergency fund) | 3,643 | 64 | 16% | 4 | 39.3 | M3 Affordability |
| 8 | Which funds should I start with? (pre-start) | 10,452 | 198 | 12% | 2 | 36.1 | M4 Start quality |
| 9 | Vernacular beginners' basic doubts | 1,266 | 67 | 24% | 4 | 35.8 | M4 Start quality |
| 10 | Chasing the hot asset (gold/silver) | 649 | 8 | 0% | 2 | 8.3 | M5 Portfolio churn |

Full table: `data/theme_scorecard.csv`. Unmet needs and quotes per theme: `data/theme_details.md`.

## 3. What the engine says about the discovery questions

Numbers are from the 993 AI-tagged relevant posts (`data/crosstabs.txt`). Small samples are flagged.

1. **The expectation people start with predicts whether they continue.** This is the strongest signal.
   - Posts showing the writer understood volatility: **74% likely to continue, 2% at risk.**
   - Posts expecting quick/high or FD-like returns: **30–32% at risk, about 10% likely to continue.**
   - Caveat: only about 8% of posts state an expectation at all (n≈76). The direction is strong, but the sample is small, so this is a hypothesis for interviews to test.
2. **Months 6–12 look like the danger zone.** Of the posts that state a journey stage:
   - Months 6–12: 10 of 19 are at risk or dropped (**53%**).
   - Months 0–3: 25%.
   - Year 1 and later: 39%.

   The typical voice: *"XIRR shows -2.2%… will it improve or is FD is the way to go?"* (month 7–8).
3. **What pushes people to stop.** Among at-risk or dropped posts, the triggers are:
   - App/UX friction: 42%
   - Distrust of the platform or MFs: 24%
   - Market fall or negative returns: 14%
   - Fees, exit load or tax: 13.5%
   - Returns below expectation: 13%

   The market-fall share is only 14%, so the brief's "investors stop when markets fall" explains a small part of the drop-off. Note that app-store reviews inflate the friction share (see §5).
4. **Emotion separates people who continue from people who leave.** "Confident/calm" posts: 86% continue. "Anxious": 50% at risk. "Disappointed": 62% at risk or dropped.
5. **Who influenced them may matter.** Posts influenced by finfluencers: **38%** at risk or dropped (n=21). Self-researched: **8%** (n=25). Treat this as directional only.
6. **What people do when unsure: they ask strangers.**
   - 55% of relevant posts are questions to a community.
   - Reddit or forums are named in 10.8% of posts, YouTube in 8.5%, ChatGPT in 0.6%.
   - The app itself is almost never cited as the place they went for an answer.
7. **What confuses them in the app.** Fund selection (17%), overlap (5.5%), tax and exit load (5%), the pause/skip/cancel flow (4%), and SIP debit date or status (3%).
8. **Gap: how often people check is almost never stated** (10 posts). Public posts can't answer this question, so the Part 3 interviews and the Google Form must cover it.
9. **Gap: segment data is thin.** Income stability is known for only 80 posts and money source for 91. Patterns by income stability and money source **can't be concluded from public data**, and this is a key reason to do primary research.

## 4. Recommended focus for Part 3

**Opportunity:** #2, early returns below expectation, combined with the months 6–12 risk window.
- It scores high (64.7), it's fully within Groww's control, and it can be solved without recommending funds or using fear or FOMO.
- It also matches the strongest signal in the data: the expectation people start with predicts whether they continue.
- #1 (SIP mechanics) scores higher, but it's an ops and reliability fix. Include it in the metric breakdown as a hygiene item.

**Segment to interview:** first-time investors who started a SIP **3–12 months ago**, aged about 21–32, salaried or early-career, especially those who started because of a finfluencer, friend or family member.

**Hypotheses for the interviews to test:**
- H1. People start with an unstated "FD-plus" expectation. The first negative or flat XIRR at months 4–9 breaks it, and nothing in the app resets it.
- H2. When unsure, they go to Reddit, YouTube or friends rather than the app, and they get conflicting advice.
- H3. Finfluencer-led starters drop off more than self-researched ones.
- H4. Some stops are really failed skips or mandates. The user meant to pause and ended up cancelling.

## 5. Material for the AI Evidence Log (AI output I rejected or corrected)

1. **The AI coders disagreed on relevance rules, and I standardised them.** One batch coded generic "don't stop your SIP" advice and industry SIP-stoppage statistics as relevant; another coded the same kind of post as irrelevant. In codebook v2: generic advice, stats and news with no specific investor are not relevant.
2. **The codebook was missing categories, and I added them.** Coders forced "no fixed income, so stopped the SIP and moved to lump sums" into *insufficient_balance* and *flat_returns*, and platform-initiated SIP cancellations into *app friction*. I added `irregular_income_vs_fixed_sip` and `platform_cancelled_sip`, and a rule that contract or gig work counts as `variable_income`.
3. **I rejected the AI cluster structure.** K-means split "review my portfolio" posts into 13 near-duplicate clusters, which inflated their apparent importance. I merged them into 2 themes ("which funds to start with" and "portfolio tinkering").
4. **I rejected the classifier's numbers for severity.** The stop/pause classifier had only 0.33 precision against the AI labels (5-fold CV), so severity now comes from the AI-tagged posts, not the classifier. The relevance classifier (0.87 precision) is used only to size themes.
5. **My own estimate was wrong.** I expected 8–15k relevant posts. The design-weighted estimate is about 49k and the classifier gives 38.6k. The codebook's definition of relevant is broader than "first-person journey story": 71% of relevant posts also meet the stricter journey-story test.

## 6. Known biases and limitations

- **Reddit dominates:** 73% of filtered posts and 87% of relevant ones. These posters are more engaged and English-speaking than the typical first-time Groww investor.
- **App-store reviews over-represent complaints**, which inflates the app-friction share among at-risk posts.
- **Survivor bias:** people who quietly stopped rarely post. Silent drop-off is under-represented.
- **Thread context is lost for replies:** the v1 codebook coded reply comments without their thread title.
- **Agents tagged the posts, not a scripted API run.** The tags come from Claude agents in Claude Code and are validated, but re-running them requires the agent step or an API key.

## Files

- `pipeline/01_filter.py`, `03_embed.py`, `04_make_tag_batches.py`, `codebook.py` (v2), `validate_tags.py`, `05_relevance_and_clusters.py`, `06_quantify.py`, `07_crosstabs.py`
- `data/tags_ai.parquet` (1,500 AI tags), `data/corpus_scored.parquet` (73.6k posts with relevance and cluster), `data/theme_scorecard.csv`, `data/theme_details.md`, `data/crosstabs.txt`, `data/clusters_for_naming.md`, `data/themes.json`
