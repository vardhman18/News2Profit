# 📈 NEWS2PROFIT

**AI-Powered NSE Stock Movement Prediction**

NEWS2PROFIT is a decision-support system that fuses market data, financial news, and social sentiment to generate daily stock movement predictions for NSE-listed companies. It delivers real-time insights, model-driven recommendations, and a streamlined dashboard for analysis and reporting.

---

## 🎯 Problem Statement

Investors and analysts face challenges including:

- Overwhelming, fragmented financial news streams and social chatter.
- Difficulty quantifying sentiment impact on short-term price movement.
- Manual technical analysis that is time-consuming and inconsistent.
- Lack of unified, reproducible workflows for data → features → models → insights.

NEWS2PROFIT addresses these by combining historical prices, engineered technical indicators, and multi-source sentiment into actionable, model-backed signals.

---

## 🚀 Key Features

### 1) 📰 Multi-Source Sentiment Fusion
- Ingests financial news (NewsAPI) and social signals (Reddit).
- Applies VADER, TextBlob, and BERT to score sentiment.
- Aligns sentiment with trading dates for robust labeling and features.

### 2) 📉 Technical Indicators & Features
- Computes SMA, EMA, RSI, returns, and custom features via `ta` library and in-house engineering.
- Merges technical signals with sentiment to enhance predictive power.

### 3) 🤖 Predictive Modeling
- Trains and compares Logistic Regression, XGBoost, and LSTM models.
- Tracks performance with Accuracy, Precision, Recall, F1, ROC-AUC.
- Saves model artifacts and comparison metrics for transparency.

### 4) 📊 Interactive Streamlit Dashboard
- Explore predictions, features, and recent sentiment in an intuitive UI.
- Filter by ticker and date range; view latest predictions and trends.

### 5) 🧾 Reporting & Artifacts
- Stores inputs/outputs in `data/` (raw, processed, models) for reproducibility.
- Exposes `latest_predictions.csv` and `model_comparison.csv` for quick reporting.

---

## 🏗️ System Architecture

1. **Frontend (Streamlit UI)**
   - Single-page dashboard in `src/dashboard.py` for exploration and monitoring.

2. **Data Pipeline (Python Modules)**
   - `src/data_collection.py`: yfinance OHLCV + NewsAPI/Reddit ingestion.
   - `src/preprocessing.py`: cleaning, alignment, technical indicators, feature engineering.
   - `src/sentiment.py`: VADER, TextBlob, BERT sentiment scoring.
   - `src/model.py`: model training, evaluation, prediction.

3. **Configuration & Storage**
   - `config/config.py`: symbols, hyperparameters, data paths, model settings.
   - `data/` directory for raw/processed data, models, and outputs.

---

## 📌 Example Use Cases

- **Open-Close Direction**: Daily signal on whether a stock is likely to move up/down.
- **Event Impact Check**: Measure sentiment around earnings/news and observe predictive effect.
- **Model Comparison**: Choose between LR, XGBoost, and LSTM using recent metrics.
- **Coverage Set Analysis**: Monitor a curated set of NSE tickers  ( RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS).
- **Backtesting Exploration**: Use notebooks to investigate strategy ideas and feature importance.

---

## 🧰 Tech Stack

- **Core**: Python, pandas, numpy, scikit-learn, xgboost, TensorFlow/Keras (LSTM)
- **NLP**: VADER, TextBlob, BERT (`nlptown/bert-base-multilingual-uncased-sentiment`)
- **Market Data**: yfinance
- **News & Social**: NewsAPI, Reddit API
- **Indicators**: `ta` library
- **UI**: Streamlit

---

## ✅ Current Configuration (Highlights)

Defined in `config/config.py`:
- **Symbols**: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`
- **Indicators**: SMA: [5, 10, 20, 50], EMA: [12, 26], RSI: 14
- **Sentiment Models**: `['vader', 'textblob', 'bert']`
- **Models & Params**: Logistic Regression, XGBoost, LSTM (sequence length 30, dropout 0.2)
- **Data Dirs**: `data/raw`, `data/processed`, `data/models`

---

### 🎯 **Target Stocks**
Currently, predictions are limited to five target NSE stocks:

- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- ICICIBANK.NS (ICICI Bank)

---

## 👥 Target Users

- **Analysts & Researchers**: Quantify sentiment + TA for short-term signals.
- **Active Traders**: Supplement decision-making with model-backed, reproducible insights.
- **Educators & Students**: Study end-to-end ML workflow on financial data.

> NEWS2PROFIT is a decision-support tool, not financial advice.

---

## 🌟 Advantages

- **Unified Pipeline**: From data collection to visualization in one repo.
- **Multi-Model View**: Compare classical and deep learning approaches.
- **Reproducible Artifacts**: Saved outputs enable auditability and iteration.
- **Modular Design**: Swap models, features, and sources with minimal friction.

---

## 🚀 Quickstart (Windows PowerShell)

1) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -r requirements.txt
```

3) Configure environment variables:
- Copy `.env.template` to `.env` and set:
   - `NEWS_API_KEY`, optional `SERPAPI_KEY`
   - `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
  
   Note: Twitter integration is not used in this project.

4) (Optional) Train/refresh models and datasets:
```powershell
python .\train_models.py
```

5) Launch the dashboard:
```powershell
python -m streamlit run src/dashboard.py
```

Or via VS Code task: "Launch NEWS2PROFIT Dashboard".

---

## 🧪 Tests

Run the test suite (if `pytest` is installed):
```powershell
pytest -q
```

---

## 📂 Key Outputs

- `data/processed/latest_predictions.csv`: Most recent predictions.
- `data/models/model_comparison.csv`: Cross-model performance snapshot.

---

## 🧭 Vision

Expand coverage, improve sentiment domain adaptation, and enrich feature sets (e.g., intra-day signals). Iterate on explainability and portfolio-aware modeling while keeping the system modular, auditable, and easy to extend.

---

## 📌 Positioning

NEWS2PROFIT is a research and decision-support platform for market participants who value transparent, data-driven workflows. It is not a brokerage tool, auto-trader, or a source of investment advice.
