# Price & Promo Copilot

Demo agent that answers price, promotion, elasticity and policy questions over the Kaggle
Online Retail II dataset (Postgres + Chroma + scikit-learn/statsmodels + LangGraph + Streamlit).

Promotions are inferred from price drops and unit cost is assumed at 60% of median price: demo assumptions, not real client data.

## Setup
1. Install PostgreSQL, create a database `promo`, and a Kaggle API token in `~/.kaggle/kaggle.json`.
2. `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
3. `cp .env.example .env` and fill in values. **Set `APP_PASS`**; the UI has no lockout or hashing, so keep it on 127.0.0.1.
4. `PYTHONPATH=. streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501`, sign in, click **Run data pipeline**.
5. The agent uses a local Ollama model (`ollama pull qwen2.5:7b-instruct`).
