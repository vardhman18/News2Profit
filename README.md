# NEWS2PROFIT - Predicting Stock Movement with ML & Financial News Sentiment

A data science project for predicting NSE stock price movements using machine learning and financial news sentiment analysis.

> Looking for a narrative, high-level overview? See `docs/NEWS2PROFIT_Overview.md`.

## Project Overview

This project predicts whether an NSE stock's price will go up, down, or stay neutral the next day by combining:
- Historical stock OHLCV data from yfinance
- Financial news sentiment analysis
- Technical indicators and engineered features
- Multiple ML models (Logistic Regression, XGBoost, LSTM)

## Dataset

- **Stock Data**: Historical OHLCV data for NSE stocks (`.NS` tickers like RELIANCE.NS, TCS.NS, INFY.NS, NIFTY50)
- **News Data**: Financial news headlines from NewsAPI, Twitter API, and Reddit
- **Sentiment Analysis**: Using VADER, TextBlob, and BERT models
- **Features**: SMA, EMA, RSI, daily returns, sentiment scores

## Workflow

1. **Data Collection**: Fetch stock and news data
2. **Preprocessing**: Clean and engineer features
3. **Sentiment Analysis**: Score news sentiment and align with trading dates
4. **Model Training**: Train Logistic Regression, XGBoost, and LSTM models
5. **Evaluation**: Assess using Accuracy, Precision, Recall, F1-score
6. **Visualization**: Interactive Streamlit dashboard

## Project Structure

```
NEWS2PROFIT/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── model.py
│   └── dashboard.py
├── notebooks/
├── config/
│   └── config.py
├── tests/
├── requirements.txt
├── .env.template
└── README.md
```

## Setup

1. Clone the repository and navigate to the project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.template` to `.env` and add your API keys:
   - NewsAPI key
   - Twitter API credentials
   - Reddit API credentials

## Usage

### Data Collection
```python
from src.data_collection import StockDataCollector, NewsDataCollector

# Collect stock data
stock_collector = StockDataCollector()
stock_data = stock_collector.fetch_stock_data(['RELIANCE.NS', 'TCS.NS'])

# Collect news data
news_collector = NewsDataCollector()
news_data = news_collector.fetch_news_data()
```

### Sentiment Analysis
```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_scores = analyzer.analyze_sentiment(news_data)
```

### Model Training
```python
from src.model import StockPredictor

predictor = StockPredictor()
predictor.train_models(processed_data)
predictions = predictor.predict(test_data)
```

### Dashboard
```bash
streamlit run src/dashboard.py
```

## Models

- **Logistic Regression**: Baseline linear model
- **XGBoost**: Gradient boosting for tabular data
- **LSTM**: Deep learning for sequential patterns

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## API Keys Required

- **NewsAPI**: Get from https://newsapi.org/
- **Twitter API**: Get from https://developer.twitter.com/
- **Reddit API**: Get from https://www.reddit.com/prefs/apps

## License

MIT License