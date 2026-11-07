# SIP Drop-off Discovery Engine (Groww case study)

An AI-assisted discovery engine for one question: why do first-time mutual-fund investors in India stop their SIPs within 12 months?

- `pipeline/`: rule filter, local multilingual embeddings, LLM codebook tagging (validated), relevance classifier, clustering, theme scoring
- `app/`: Streamlit app with a themes dashboard, "Ask the data" (answers cited to posts and checked before they are shown), and a live review tagger with visible quality checks
- `DISCOVERY_FINDINGS.md`: results, opportunity ranking, AI corrections, limitations

Research tool only. It does not recommend funds or stocks and is not investment advice.

Run locally: `pip install -r app/requirements.txt && MISTRAL_API_KEY=... streamlit run app/streamlit_app.py`
