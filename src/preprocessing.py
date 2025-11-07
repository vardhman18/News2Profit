"""
Data Preprocessing Module for NEWS2PROFIT

This module handles data cleaning, feature engineering, and preprocessing
for both stock data and news data.
"""

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """Preprocesses stock market data and creates technical indicators"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values and outliers
        
        Args:
            df: Raw stock data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning stock data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Ensure date column is datetime
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Sort by date
        df_clean = df_clean.sort_values('date')
        
        # Handle missing values
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Forward fill missing values
                df_clean[col] = df_clean[col].ffill()
                
                # Backward fill remaining missing values
                df_clean[col] = df_clean[col].bfill()
        
        # Remove rows where all OHLC values are NaN
        df_clean = df_clean.dropna(subset=['open', 'high', 'low', 'close'], how='all')
        
        # Handle volume = 0 (replace with previous day's volume)
        if 'volume' in df_clean.columns:
            df_clean['volume'] = df_clean['volume'].replace(0, np.nan)
            df_clean['volume'] = df_clean['volume'].ffill()
            df_clean['volume'] = df_clean['volume'].bfill()
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators for stock data
        
        Args:
            df: Cleaned stock data DataFrame
        
        Returns:
            DataFrame with technical indicators
        """
        logger.info("Creating technical indicators...")
        
        df_tech = df.copy()
        
        # Price-based indicators
        df_tech['daily_return'] = df_tech['close'].pct_change()
        df_tech['price_change'] = df_tech['close'] - df_tech['open']
        df_tech['price_change_pct'] = (df_tech['close'] - df_tech['open']) / df_tech['open'] * 100
        
        # High-Low indicators
        df_tech['high_low_pct'] = (df_tech['high'] - df_tech['low']) / df_tech['low'] * 100
        df_tech['close_vs_high'] = (df_tech['close'] - df_tech['high']) / df_tech['high'] * 100
        df_tech['close_vs_low'] = (df_tech['close'] - df_tech['low']) / df_tech['low'] * 100
        
        # Moving Averages
        for period in SMA_PERIODS:
            df_tech[f'sma_{period}'] = ta.trend.sma_indicator(df_tech['close'], window=period)
            df_tech[f'close_sma_{period}_ratio'] = df_tech['close'] / df_tech[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in EMA_PERIODS:
            df_tech[f'ema_{period}'] = ta.trend.ema_indicator(df_tech['close'], window=period)
            df_tech[f'close_ema_{period}_ratio'] = df_tech['close'] / df_tech[f'ema_{period}']
        
        # MACD
        df_tech['macd'] = ta.trend.macd(df_tech['close'])
        df_tech['macd_signal'] = ta.trend.macd_signal(df_tech['close'])
        df_tech['macd_diff'] = ta.trend.macd_diff(df_tech['close'])
        
        # RSI
        df_tech['rsi'] = ta.momentum.rsi(df_tech['close'], window=RSI_PERIOD)
        
        # Bollinger Bands
        df_tech['bb_high'] = ta.volatility.bollinger_hband(df_tech['close'])
        df_tech['bb_low'] = ta.volatility.bollinger_lband(df_tech['close'])
        df_tech['bb_mid'] = ta.volatility.bollinger_mavg(df_tech['close'])
        df_tech['bb_width'] = (df_tech['bb_high'] - df_tech['bb_low']) / df_tech['bb_mid']
        df_tech['bb_position'] = (df_tech['close'] - df_tech['bb_low']) / (df_tech['bb_high'] - df_tech['bb_low'])
        
        # Volume indicators
        if 'volume' in df_tech.columns:
            # Simple volume moving average
            df_tech['volume_sma'] = df_tech['volume'].rolling(window=20).mean()
            df_tech['volume_ratio'] = df_tech['volume'] / df_tech['volume'].rolling(window=20).mean()
            df_tech['price_volume'] = df_tech['close'] * df_tech['volume']
        
        # Volatility
        df_tech['volatility'] = df_tech['daily_return'].rolling(window=20).std()
        
        # Support and Resistance levels
        df_tech['support'] = df_tech['low'].rolling(window=20).min()
        df_tech['resistance'] = df_tech['high'].rolling(window=20).max()
        df_tech['support_distance'] = (df_tech['close'] - df_tech['support']) / df_tech['support'] * 100
        df_tech['resistance_distance'] = (df_tech['resistance'] - df_tech['close']) / df_tech['close'] * 100
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df_tech[f'close_lag_{lag}'] = df_tech['close'].shift(lag)
            df_tech[f'return_lag_{lag}'] = df_tech['daily_return'].shift(lag)
            df_tech[f'volume_lag_{lag}'] = df_tech['volume'].shift(lag) if 'volume' in df_tech.columns else 0
        
        logger.info(f"Created technical indicators. New shape: {df_tech.shape}")
        return df_tech
    
    def create_target_variable(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Create target variable for prediction (next day price movement)
        
        Args:
            df: DataFrame with stock data
            threshold: Threshold for classifying as neutral (±threshold%)
        
        Returns:
            DataFrame with target variable
        """
        logger.info("Creating target variable...")
        
        df_target = df.copy()
        
        # Calculate next day return
        df_target['next_day_return'] = df_target['daily_return'].shift(-1)
        df_target['next_day_close'] = df_target['close'].shift(-1)
        df_target['next_day_price_change'] = df_target['next_day_close'] - df_target['close']
        df_target['next_day_price_change_pct'] = df_target['next_day_price_change'] / df_target['close'] * 100
        
        # Create categorical target with better balance
        def classify_movement(return_pct):
            if pd.isna(return_pct):
                return 'NEUTRAL'
            elif return_pct > threshold:
                return 'UP'
            elif return_pct < -threshold:
                return 'DOWN'
            else:
                return 'NEUTRAL'
        
        df_target['target'] = df_target['next_day_price_change_pct'].apply(classify_movement)
        
        # For ML models, convert NEUTRAL to binary classification if too few samples
        target_counts = df_target['target'].value_counts()
        logger.info(f"Target distribution:\n{target_counts}")
        
        # If NEUTRAL class has < 50 samples, convert to binary classification
        if target_counts.get('NEUTRAL', 0) < 50:
            logger.info("Converting to binary classification (UP/DOWN) due to insufficient NEUTRAL samples")
            df_target['target_ml'] = df_target['target'].apply(lambda x: 'UP' if x in ['UP', 'NEUTRAL'] else 'DOWN')
        else:
            df_target['target_ml'] = df_target['target']
        
        # Create binary target (UP vs DOWN, excluding NEUTRAL)
        df_target['target_binary'] = df_target['target'].map({'UP': 1, 'DOWN': 0, 'NEUTRAL': np.nan})
        
        # Create numeric target
        df_target['target_numeric'] = df_target['target'].map({'UP': 1, 'NEUTRAL': 0, 'DOWN': -1})
        
        # Remove the last row as it doesn't have next day data
        df_target = df_target[:-1]
        
        logger.info(f"Target distribution:\n{df_target['target'].value_counts()}")
        return df_target
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning
        
        Args:
            df: DataFrame with all features
        
        Returns:
            DataFrame ready for ML
        """
        logger.info("Preparing features for ML...")
        
        df_ml = df.copy()
        
        # Select feature columns (exclude target and metadata columns)
        exclude_cols = [
            'date', 'symbol', 'target', 'target_ml', 'target_binary', 'target_numeric',
            'next_day_return', 'next_day_close', 'next_day_price_change',
            'next_day_price_change_pct'
        ]
        
        feature_cols = [col for col in df_ml.columns if col not in exclude_cols]
        
        # Separate numeric and categorical features
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if df_ml[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        
        # Handle infinite values in numeric columns only
        if numeric_cols:
            df_ml[numeric_cols] = df_ml[numeric_cols].replace([np.inf, -np.inf], np.nan)
            # Fill remaining NaN values with column median for numeric columns
            df_ml[numeric_cols] = df_ml[numeric_cols].fillna(df_ml[numeric_cols].median())
        
        # Handle categorical columns
        for col in categorical_cols:
            # Fill NaN values with mode (most frequent value)
            if df_ml[col].isna().any():
                mode_value = df_ml[col].mode().iloc[0] if not df_ml[col].mode().empty else 'UNKNOWN'
                df_ml[col] = df_ml[col].fillna(mode_value)
        
        logger.info(f"Prepared {len(feature_cols)} features for ML")
        return df_ml


class NewsDataPreprocessor:
    """Preprocesses news data for sentiment analysis"""
    
    def __init__(self):
        pass
    
    def clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean news data
        
        Args:
            df: Raw news DataFrame
        
        Returns:
            Cleaned news DataFrame
        """
        logger.info("Cleaning news data...")
        
        df_clean = df.copy()
        
        # Ensure date columns are datetime
        if 'published_at' in df_clean.columns:
            df_clean['published_at'] = pd.to_datetime(df_clean['published_at'])
            df_clean['date'] = df_clean['published_at'].dt.date
        
        # Clean text columns
        text_columns = ['title', 'description', 'content']
        
        for col in text_columns:
            if col in df_clean.columns:
                # Fill NaN values
                df_clean[col] = df_clean[col].fillna('')
                
                # Convert to string and strip whitespace
                df_clean[col] = df_clean[col].astype(str).str.strip()
                
                # Remove very short texts (less than 10 characters)
                df_clean = df_clean[df_clean[col].str.len() >= 10]
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['title', 'description'])
        
        # Filter by date (keep only recent news)
        if 'date' in df_clean.columns:
            cutoff_date = datetime.now().date() - timedelta(days=365)  # Keep last year
            df_clean = df_clean[df_clean['date'] >= cutoff_date]
        
        logger.info(f"Cleaned news data shape: {df_clean.shape}")
        return df_clean
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text-based features from news data
        
        Args:
            df: News DataFrame
        
        Returns:
            DataFrame with text features
        """
        logger.info("Extracting text features...")
        
        df_features = df.copy()
        
        # Text length features
        df_features['title_length'] = df_features['title'].str.len()
        df_features['description_length'] = df_features['description'].str.len()
        df_features['content_length'] = df_features['content'].str.len()
        
        # Word count features
        df_features['title_word_count'] = df_features['title'].str.split().str.len()
        df_features['description_word_count'] = df_features['description'].str.split().str.len()
        df_features['content_word_count'] = df_features['content'].str.split().str.len()
        
        # Time-based features
        if 'published_at' in df_features.columns:
            df_features['hour'] = df_features['published_at'].dt.hour
            df_features['day_of_week'] = df_features['published_at'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            df_features['is_market_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 15)).astype(int)
        
        # Source-based features
        if 'source' in df_features.columns:
            df_features['source_encoded'] = LabelEncoder().fit_transform(df_features['source'])
        
        # Financial keywords
        financial_keywords = [
            'profit', 'loss', 'revenue', 'earnings', 'growth', 'decline',
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'upgrade', 'downgrade',
            'merger', 'acquisition', 'ipo', 'dividend', 'split', 'bonus'
        ]
        
        # Count financial keywords
        combined_text = (df_features['title'] + ' ' + df_features['description']).str.lower()
        
        for keyword in financial_keywords:
            df_features[f'keyword_{keyword}'] = combined_text.str.count(keyword)
        
        df_features['total_financial_keywords'] = df_features[[f'keyword_{kw}' for kw in financial_keywords]].sum(axis=1)
        
        logger.info(f"Extracted text features. New shape: {df_features.shape}")
        return df_features


class DataIntegrator:
    """Integrates stock data with news data and sentiment scores"""
    
    def __init__(self):
        pass
    
    def align_data_by_date(self, stock_df: pd.DataFrame, news_df: pd.DataFrame, 
                          sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Align stock data with news data by date
        
        Args:
            stock_df: Stock data DataFrame
            news_df: News data DataFrame
            sentiment_df: Sentiment scores DataFrame (optional)
        
        Returns:
            Integrated DataFrame
        """
        logger.info("Aligning stock and news data by date...")
        
        # Ensure date columns are the same type
        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Aggregate news data by date
        news_agg = self._aggregate_news_by_date(news_df, sentiment_df)
        
        # Merge stock data with aggregated news data
        integrated_df = stock_df.merge(news_agg, on='date', how='left')
        
        # Fill missing news features with defaults
        news_columns = [col for col in integrated_df.columns if col.startswith(('news_', 'sentiment_'))]
        integrated_df[news_columns] = integrated_df[news_columns].fillna(0)
        
        logger.info(f"Integrated data shape: {integrated_df.shape}")
        return integrated_df
    
    def _aggregate_news_by_date(self, news_df: pd.DataFrame, 
                               sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Aggregate news data and sentiment scores by date
        
        Args:
            news_df: News data DataFrame
            sentiment_df: Sentiment scores DataFrame
        
        Returns:
            Aggregated news DataFrame by date
        """
        # Merge sentiment scores if provided
        if sentiment_df is not None:
            news_with_sentiment = news_df.merge(sentiment_df, left_index=True, right_index=True, how='left')
        else:
            news_with_sentiment = news_df
        
        # Aggregate by date
        agg_dict = {
            'title_length': ['mean', 'sum', 'count'],
            'description_length': ['mean', 'sum'],
            'content_length': ['mean', 'sum'],
            'title_word_count': ['mean', 'sum'],
            'description_word_count': ['mean', 'sum'],
            'content_word_count': ['mean', 'sum'],
            'total_financial_keywords': ['sum', 'mean']
        }
        
        # Add sentiment aggregations if available (only numeric columns)
        sentiment_cols = [col for col in news_with_sentiment.columns if col.startswith('sentiment_') and news_with_sentiment[col].dtype in ['float64', 'int64']]
        for col in sentiment_cols:
            agg_dict[col] = ['mean', 'sum', 'std']
            
        # Add VADER/TextBlob numeric sentiment scores
        numeric_sentiment_cols = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                                 'textblob_polarity', 'textblob_subjectivity',
                                 'bert_positive', 'bert_negative', 'bert_neutral', 'bert_confidence']
        
        for col in numeric_sentiment_cols:
            if col in news_with_sentiment.columns:
                agg_dict[col] = ['mean', 'sum', 'std']
        
        # Handle categorical sentiment labels separately
        categorical_sentiment_cols = ['vader_label', 'textblob_label', 'bert_label', 'composite_label']
        
        # First aggregate numeric columns
        news_agg = news_with_sentiment.groupby('date').agg(agg_dict)
        
        # Flatten column names for the main aggregation
        news_agg.columns = ['news_' + '_'.join(col).strip() for col in news_agg.columns]
        
        # Then handle categorical columns by counting occurrences
        for cat_col in categorical_sentiment_cols:
            if cat_col in news_with_sentiment.columns:
                cat_counts = news_with_sentiment.groupby(['date', cat_col]).size().unstack(fill_value=0)
                cat_counts.columns = [f'news_{cat_col}_{col}' for col in cat_counts.columns]
                # Reset index for both DataFrames to ensure proper joining
                if not news_agg.index.equals(cat_counts.index):
                    news_agg = news_agg.reset_index()
                    cat_counts = cat_counts.reset_index()
                    news_agg = news_agg.merge(cat_counts, on='date', how='left')
                    news_agg = news_agg.set_index('date')
                else:
                    news_agg = news_agg.join(cat_counts, how='left')
        
        # Reset index
        news_agg = news_agg.reset_index()
        
        return news_agg
    
    def create_lagged_features(self, df: pd.DataFrame, lag_days: list = None) -> pd.DataFrame:
        """
        Create lagged features for time series prediction
        
        Args:
            df: Integrated DataFrame
            lag_days: List of lag periods
        
        Returns:
            DataFrame with lagged features
        """
        if lag_days is None:
            lag_days = [1, 2, 3, 5, 7]
            
        logger.info(f"Creating lagged features for {lag_days} days...")
        
        df_lagged = df.copy()
        df_lagged = df_lagged.sort_values(['symbol', 'date'])
        
        # Features to lag
        lag_features = [
            'daily_return', 'close', 'volume', 'rsi', 'macd', 'volatility'
        ]
        
        # Add sentiment features if available
        sentiment_features = [col for col in df_lagged.columns if col.startswith('sentiment_')]
        lag_features.extend(sentiment_features)
        
        # Create lagged features
        for symbol in df_lagged['symbol'].unique():
            symbol_mask = df_lagged['symbol'] == symbol
            
            for feature in lag_features:
                if feature in df_lagged.columns:
                    for lag in lag_days:
                        df_lagged.loc[symbol_mask, f'{feature}_lag_{lag}'] = \
                            df_lagged.loc[symbol_mask, feature].shift(lag)
        
        logger.info(f"Created lagged features. New shape: {df_lagged.shape}")
        return df_lagged


def main():
    """Main function to test preprocessing"""
    # Test with sample data
    logger.info("Testing preprocessing modules...")
    
    # Create sample stock data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_stock_data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates)),
        'symbol': 'RELIANCE.NS'
    })
    
    # Test stock preprocessing
    stock_preprocessor = StockDataPreprocessor()
    
    # Clean data
    cleaned_stock = stock_preprocessor.clean_stock_data(sample_stock_data)
    
    # Create technical indicators
    stock_with_indicators = stock_preprocessor.create_technical_indicators(cleaned_stock)
    
    # Create target variable
    stock_with_target = stock_preprocessor.create_target_variable(stock_with_indicators)
    
    # Prepare features
    ml_ready_data = stock_preprocessor.prepare_features(stock_with_target)
    
    print(f"Final processed stock data shape: {ml_ready_data.shape}")
    print(f"Target distribution:\n{stock_with_target['target'].value_counts()}")


if __name__ == "__main__":
    main()