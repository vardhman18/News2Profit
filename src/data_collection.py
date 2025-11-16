"""
Data Collection Module for NEWS2PROFIT

This module handles the collection of stock data from yfinance
and news data from various sources including NewsAPI, Twitter, and Reddit.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import praw
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *
from dateutil import parser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collects historical stock data from yfinance"""
    
    def __init__(self):
        self.data = {}

    def _normalize_stock_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()
        if not df.index.name and 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        # Reset index to get date as column
        df.reset_index(inplace=True)
        # Clean column names
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
        # Ensure date column exists
        if 'date' not in df.columns:
            # Try common alternatives
            for alt in ['datetime', 'index']:
                if alt in df.columns:
                    df.rename(columns={alt: 'date'}, inplace=True)
                    break
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].dt.date
        df['symbol'] = symbol
        # Keep standard columns if present
        keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'symbol'] if c in df.columns]
        return df[keep] if keep else df

    def _load_existing_raw(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            fname = os.path.join(base_dir, RAW_DATA_DIR, f"{symbol.replace('.', '_')}_historical.csv")
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                if not df.empty:
                    logger.warning(f"Using existing raw CSV as fallback for {symbol}: {fname}")
                    return self._normalize_stock_df(df, symbol)
        except Exception as e:
            logger.error(f"Fallback read failed for {symbol}: {e}")
        return None
    
    def fetch_stock_data(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for given symbols
        
        Args:
            symbols: List of stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            Dictionary mapping symbols to their historical data
        """
        logger.info(f"Fetching stock data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Attempt 1: Ticker.history with explicit params
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=period,
                    interval='1d',
                    auto_adjust=True,
                    actions=False,
                    repair=True,
                    raise_errors=False
                )

                if data is None or data.empty:
                    logger.warning(f"No data via Ticker.history for {symbol}; trying yf.download ...")
                    # Attempt 2: yf.download fallback
                    try:
                        dl = yf.download(
                            tickers=symbol,
                            period=period,
                            interval='1d',
                            auto_adjust=True,
                            actions=False,
                            threads=False
                        )
                    except TypeError:
                        # Older yfinance versions may not support some args
                        dl = yf.download(tickers=symbol, period=period, interval='1d', threads=False)
                    data = dl

                if data is None or data.empty:
                    logger.warning(f"No Yahoo data for {symbol}; attempting local CSV fallback ...")
                    fallback_df = self._load_existing_raw(symbol)
                    if fallback_df is not None and not fallback_df.empty:
                        self.data[symbol] = fallback_df
                        logger.info(f"Loaded {len(fallback_df)} fallback records for {symbol}")
                        continue
                    else:
                        logger.warning(f"No data found for {symbol}")
                        continue

                norm = self._normalize_stock_df(data, symbol)
                if norm is not None and not norm.empty:
                    self.data[symbol] = norm
                    logger.info(f"Successfully fetched {len(norm)} records for {symbol}")
                else:
                    logger.warning(f"Data normalization produced empty frame for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return self.data
    
    def fetch_nifty_data(self, period: str = "2y") -> pd.DataFrame:
        """Fetch Nifty 50 index data"""
        return self.fetch_stock_data([NIFTY_INDEX], period).get(NIFTY_INDEX)
    
    def save_stock_data(self, filepath: str = None) -> None:
        """Save collected stock data to CSV files"""
        if filepath is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            filepath = os.path.join(base_dir, RAW_DATA_DIR)
        
        os.makedirs(filepath, exist_ok=True)
        
        for symbol, data in self.data.items():
            filename = os.path.join(filepath, f"{symbol.replace('.', '_')}_historical.csv")
            data.to_csv(filename, index=False)
            logger.info(f"Saved {symbol} data to {filename}")


class NewsDataCollector:
    """Collects news data from various sources"""
    
    def __init__(self):
        self.news_data = []

        # Initialize API clients
        self._setup_news_api()
        self._setup_reddit_api()
    
    def _setup_news_api(self):
        """Setup SerpApi client for Google News Light"""
        # Prefer configured key; try Streamlit secrets if available when running on Streamlit
        api_key = None
        try:
            import streamlit as st  # type: ignore
            api_key = st.secrets.get('SERPAPI_KEY', None)
        except Exception:
            api_key = None
        # Fallback to config-provided env var mappings
        try:
            # SERPAPI_KEY may be provided via config import
            api_key = api_key or SERPAPI_KEY or NEWS_API_KEY
        except Exception:
            pass
        self.news_api_key = api_key
        self.news_base_url = "https://serpapi.com/search"

    
    # Twitter fetching removed by request — Twitter client and related functions have been removed.
    
    def _setup_reddit_api(self):
        """Setup Reddit API client"""
        if all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
            try:
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Reddit API: {str(e)}")
                self.reddit = None
        else:
            self.reddit = None
            logger.warning("Reddit API credentials not found")
    
    def fetch_news_api_data(self, days_back: int = 30) -> List[Dict]:
        """
        Fetch stock news from SerpApi's Google News Light engine for 5 specific companies.
        """
        if not self.news_api_key:
            logger.warning("SerpApi key not found")
            return []

        articles = []
        
        # Define the 5 companies with their search queries
        companies = {
            "RELIANCE": "Reliance Industries stock news India",
            "TCS": "TCS stock news India", 
            "INFY": "Infosys stock news India",
            "HDFCBANK": "HDFC Bank stock news India",
            "ICICIBANK": "ICICI Bank stock news India"
        }

        try:
            logger.info("Fetching stock news from SerpApi Google News Light...")
            
            for company_symbol, query in companies.items():
                logger.info(f"Fetching news for {company_symbol}...")
                
                # SerpApi Google News Light parameters
                params = {
                    "engine": "google_news_light",
                    "q": query,
                    "gl": "IN",  # Country: India
                    "hl": "en",  # Language: English
                    "num": 20,   # Number of results per company
                    "api_key": self.news_api_key
                }

                response = requests.get("https://serpapi.com/search", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    # Get results from different possible keys
                    news_results = data.get("top_stories", []) or data.get("news_results", [])
                    
                    for article in news_results:
                        # Parse date if available
                        published_date = article.get('date', '')
                        try:
                            if published_date:
                                parsed_date = parser.parse(published_date)
                            else:
                                parsed_date = datetime.now()
                        except:
                            parsed_date = datetime.now()
                        
                        articles.append({
                            'source': 'serpapi_google_news',
                            'title': article.get('title', ''),
                            'description': article.get('snippet', ''),
                            'content': article.get('snippet', ''),
                            'url': article.get('link', ''),
                            'published_at': parsed_date,
                            'keyword': company_symbol
                        })
                    
                    logger.info(f"Fetched {len(news_results)} articles for {company_symbol}")
                    
                else:
                    logger.error(f"SerpApi error for {company_symbol}: {response.status_code} - {response.text}")
                
                # Rate limiting - pause between requests
                time.sleep(1)

            logger.info(f"Total articles fetched: {len(articles)}")

        except Exception as e:
            logger.error(f"Error fetching stock news: {str(e)}")

        return articles
    
    def fetch_twitter_data(self, keywords: List[str] = None, count: int = 2) -> List[Dict]:
        """
        Fetch tweets using Twitter API v2 for Indian stock companies.
        Uses company names instead of stock symbols for better results.
        """
        # Twitter fetching removed. This method retained as a stub for backward compatibility but returns empty.
        logger.info("Twitter fetching has been removed from the project.")
        return []
    
    def fetch_reddit_data(self, subreddits: List[str] = None, days_back: int = 7, limit: int = 100) -> List[Dict]:
        """
        Fetch posts from Reddit
        
        Args:
            subreddits: List of subreddit names
            days_back: Number of days to look back
            limit: Number of posts to fetch per subreddit
        
        Returns:
            List of Reddit posts
        """
        if not self.reddit:
            logger.warning("Reddit API not initialized")
            return []
        
        if subreddits is None:
            subreddits = REDDIT_SUBREDDITS
        
        posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for post in subreddit.new(limit=limit):
                    post_date = datetime.fromtimestamp(post.created_utc)
                    
                    if post_date >= datetime.now() - timedelta(days=days_back):
                        posts.append({
                            'source': 'reddit',
                            'title': post.title,
                            'description': post.selftext[:500] if post.selftext else post.title,
                            'content': post.selftext,
                            'url': post.url,
                            'published_at': post_date.isoformat(),
                            'keyword': subreddit_name,
                            'score': post.score,
                            'num_comments': post.num_comments
                        })
                
                logger.info(f"Fetched posts from r/{subreddit_name}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching Reddit data from r/{subreddit_name}: {str(e)}")
        
        return posts
    
    def fetch_all_news_data(self, days_back: int = 30, save_to_csv: bool = True) -> pd.DataFrame:
        """
        Fetch news from all sources and combine into DataFrame
        Focus on 5 specific stocks: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK
        
        Args:
            days_back: Number of days to look back
            save_to_csv: Whether to save individual source data to CSV files
        
        Returns:
            Combined DataFrame with all news data
        """
        logger.info("Fetching news from all sources for 5 specific stocks...")
        
        # Fetch from NewsAPI
        logger.info("Fetching NewsAPI data...")
        news_articles = self.fetch_news_api_data(days_back=days_back)
        if news_articles and save_to_csv:
            self._save_source_data_to_csv(news_articles, 'newsapi')
        
        # Twitter fetching removed — skip Twitter source
        tweets = []
        
        # Fetch from Reddit
        logger.info("Fetching Reddit data...")
        reddit_posts = self.fetch_reddit_data(days_back=days_back)
        if reddit_posts and save_to_csv:
            self._save_source_data_to_csv(reddit_posts, 'reddit')
        
        # Combine all data
        all_news = news_articles + reddit_posts
        
        # Convert to DataFrame
        if all_news:
            df = pd.DataFrame(all_news)
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df['date'] = df['published_at'].dt.date
            
            # Remove duplicates based on title and content
            df = df.drop_duplicates(subset=['title', 'content'])
            
            logger.info(f"Collected {len(df)} unique news items from all sources")
            
            # Save combined data
            if save_to_csv:
                self._save_source_data_to_csv(df.to_dict('records'), 'combined')
            
            return df
        else:
            logger.warning("No news data collected")
            return pd.DataFrame()
    
    def _save_source_data_to_csv(self, data, source_name: str) -> None:
        """
        Save data from a specific source to CSV file
        
        Args:
            data: List of dictionaries or DataFrame records
            source_name: Name of the data source (newsapi, twitter, reddit, combined)
        """
        try:
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
            elif hasattr(data, 'to_dict'):
                df = data
            else:
                logger.warning(f"No data to save for {source_name}")
                return
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(RAW_DATA_DIR, f"{source_name}_data_{timestamp}.csv")
            
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df)} records to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving {source_name} data to CSV: {str(e)}")
    
    def save_news_data(self, df: pd.DataFrame, filepath: str = None) -> None:
        """Save news data to CSV file"""
        if filepath is None:
            filepath = os.path.join(RAW_DATA_DIR, "news_data.csv")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved news data to {filepath}")


def main():
    """Main function to test data collection"""
    # Test stock data collection
    stock_collector = StockDataCollector()
    
    # Fetch data for a few stocks
    # Use the 5 specific NSE stocks used across the project
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    stock_data = stock_collector.fetch_stock_data(test_symbols, period="6mo")
    
    # Save stock data
    stock_collector.save_stock_data()
    
    # Test news data collection
    news_collector = NewsDataCollector()
    news_df = news_collector.fetch_all_news_data(days_back=7)
    
    if not news_df.empty:
        news_collector.save_news_data(news_df)
        print(f"Collected {len(news_df)} news items")
        print("\nNews sources distribution:")
        print(news_df['source'].value_counts())
    else:
        print("No news data collected. Please check API credentials.")


if __name__ == "__main__":
    main()