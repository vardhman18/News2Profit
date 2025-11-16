"""
Configuration settings for NEWS2PROFIT project
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")

# API Keys (support Streamlit Cloud secrets if available)
_secrets = {}
try:
    import streamlit as st  # type: ignore
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}

def _get_secret(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or _secrets.get(name) or default

# Prefer SERPAPI_KEY; allow legacy NEWS_API_KEY as fallback
NEWS_API_KEY = _get_secret('NEWS_API_KEY')
SERPAPI_KEY = _get_secret('SERPAPI_KEY', NEWS_API_KEY)

TWITTER_BEARER_TOKEN = _get_secret('TWITTER_BEARER_TOKEN')
TWITTER_API_KEY = _get_secret('TWITTER_API_KEY')
TWITTER_API_SECRET = _get_secret('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = _get_secret('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = _get_secret('TWITTER_ACCESS_TOKEN_SECRET')

REDDIT_CLIENT_ID = _get_secret('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = _get_secret('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = _get_secret('REDDIT_USER_AGENT', 'NEWS2PROFIT/1.0')

# NSE Stock Symbols
NSE_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
    'ICICIBANK.NS'
]

# Nifty Index
NIFTY_INDEX = '^NSEI'

# Data paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Technical indicators parameters
SMA_PERIODS = [5, 10, 20, 50]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14

# Sentiment analysis models
SENTIMENT_MODELS = ['vader', 'textblob', 'bert']
BERT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'

# News sources and keywords
NEWS_KEYWORDS = [
    'stock market', 'NSE', 'BSE', 'Indian stocks', 'equity',
    'share price', 'market news', 'financial news', 'economy India'
]

TWITTER_KEYWORDS = [
    '#NSE', '#BSE', '#IndianStocks', '#StockMarket', '#Nifty50'
]

REDDIT_SUBREDDITS = [
    'IndiaInvestments', 'indianstreetbets', 'StockMarket', 'investing'
]

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

LSTM_PARAMS = {
    'units': 50,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'sequence_length': 30
}

# Streamlit dashboard settings
DASHBOARD_TITLE = "NEWS2PROFIT - Stock Movement Prediction Dashboard"
DASHBOARD_PORT = 8501

# Environment validation (optional - remove for production)
# Uncomment the lines below to debug environment variable loading
# print("Debugging Environment Variables:")
# print(f"NEWS_API_KEY: {NEWS_API_KEY}")
# print(f"TWITTER_API_KEY: {TWITTER_API_KEY}")
# print(f"REDDIT_CLIENT_ID: {REDDIT_CLIENT_ID}")

# Add missing library warning
try:
    import ta
except ImportError:
    print("Warning: 'ta' library is not installed. Please install it using 'pip install ta'.")