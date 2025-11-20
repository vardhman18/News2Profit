"""
Example notebook for data exploration and analysis

This notebook demonstrates how to use the NEWS2PROFIT modules
for stock market analysis and prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import NEWS2PROFIT modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import *
from src.data_collection import StockDataCollector, NewsDataCollector
from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor, DataIntegrator
from src.sentiment import SentimentAnalyzer
from src.model import StockPredictor, LogisticRegressionModel, XGBoostModel

def example_data_collection():
    """Example: How to collect stock and news data"""
    print("=== DATA COLLECTION EXAMPLE ===")
    
    # Initialize collectors
    stock_collector = StockDataCollector()
    news_collector = NewsDataCollector()
    
    # Fetch stock data for configured NSE symbols
    from config.config import NSE_STOCKS
    symbols = NSE_STOCKS
    print(f"Fetching stock data for: {symbols}")
    
    stock_data = stock_collector.fetch_stock_data(symbols, period="6mo")
    
    # Combine all stock data
    all_stock_data = []
    for symbol, data in stock_data.items():
        all_stock_data.append(data)
    
    combined_stock = pd.concat(all_stock_data, ignore_index=True) if all_stock_data else pd.DataFrame()
    
    print(f"Stock data shape: {combined_stock.shape}")
    print(f"Stock data columns: {list(combined_stock.columns)}")
    print(f"Date range: {combined_stock['date'].min()} to {combined_stock['date'].max()}")
    
    # Fetch news data
    print("\\nFetching news data...")
    news_data = news_collector.fetch_all_news_data(days_back=30)
    
    if not news_data.empty:
        print(f"News data shape: {news_data.shape}")
        print(f"News sources: {news_data['source'].value_counts().to_dict()}")
        print(f"Date range: {news_data['date'].min()} to {news_data['date'].max()}")
    else:
        print("No news data collected (check API keys)")
    
    return combined_stock, news_data

def example_preprocessing(stock_data):
    """Example: How to preprocess stock data"""
    print("\\n=== PREPROCESSING EXAMPLE ===")
    
    if stock_data.empty:
        print("No stock data to preprocess")
        return pd.DataFrame()
    
    # Initialize preprocessor
    stock_preprocessor = StockDataPreprocessor()
    
    # Clean data
    print("Cleaning stock data...")
    cleaned_data = stock_preprocessor.clean_stock_data(stock_data)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Create technical indicators
    print("Creating technical indicators...")
    indicators_data = stock_preprocessor.create_technical_indicators(cleaned_data)
    
    # Show some technical indicators
    tech_indicators = [col for col in indicators_data.columns if col in [
        'sma_5', 'sma_20', 'rsi', 'macd', 'daily_return', 'volatility'
    ]]
    
    print(f"Technical indicators added: {tech_indicators}")
    
    # Create target variable
    print("Creating target variable...")
    target_data = stock_preprocessor.create_target_variable(indicators_data)
    
    print(f"Target distribution:")
    print(target_data['target'].value_counts())
    
    return target_data

def example_sentiment_analysis(news_data):
    """Example: How to perform sentiment analysis"""
    print("\\n=== SENTIMENT ANALYSIS EXAMPLE ===")
    
    if news_data.empty:
        print("No news data for sentiment analysis")
        return pd.DataFrame()
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(['vader', 'textblob'])
    
    # Analyze sentiment
    print("Analyzing news sentiment...")
    news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_data)
    
    # Show sentiment results
    sentiment_cols = [col for col in news_with_sentiment.columns if 'sentiment' in col or 'vader' in col or 'textblob' in col]
    print(f"Sentiment columns added: {len(sentiment_cols)}")
    
    # Show sentiment distribution
    if 'vader_label' in news_with_sentiment.columns:
        print("\\nVADER sentiment distribution:")
        print(news_with_sentiment['vader_label'].value_counts())
    
    if 'textblob_label' in news_with_sentiment.columns:
        print("\\nTextBlob sentiment distribution:")
        print(news_with_sentiment['textblob_label'].value_counts())
    
    return news_with_sentiment

def example_model_training(processed_data):
    """Example: How to train and evaluate models"""
    print("\\n=== MODEL TRAINING EXAMPLE ===")
    
    if processed_data.empty or len(processed_data) < 50:
        print("Not enough data for model training")
        return None
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Train models
    print("Training models...")
    try:
        predictor.train_all_models(processed_data, target_col='target')
        
        # Show model comparison
        comparison = predictor.get_model_comparison()
        print("\\nModel Performance Comparison:")
        print(comparison.to_string(index=False))
        
        print(f"\\nBest model: {predictor.best_model}")
        
        return predictor
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return None

def example_visualization(stock_data, news_data=None):
    """Example: How to create visualizations"""
    print("\\n=== VISUALIZATION EXAMPLE ===")
    
    if stock_data.empty:
        print("No data to visualize")
        return
    
    # Set up plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Stock price over time
    for symbol in stock_data['symbol'].unique()[:3]:  # Limit to first 3 symbols
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        axes[0, 0].plot(symbol_data['date'], symbol_data['close'], label=symbol.replace('.NS', ''))
    
    axes[0, 0].set_title('Stock Prices Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price (₹)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Volume distribution
    axes[0, 1].hist(stock_data['volume'], bins=30, alpha=0.7)
    axes[0, 1].set_title('Trading Volume Distribution')
    axes[0, 1].set_xlabel('Volume')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Daily returns distribution
    if 'daily_return' in stock_data.columns:
        daily_returns = stock_data['daily_return'].dropna()
        axes[1, 0].hist(daily_returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
    else:
        # Calculate daily returns
        stock_data_sorted = stock_data.sort_values(['symbol', 'date'])
        stock_data_sorted['daily_return'] = stock_data_sorted.groupby('symbol')['close'].pct_change()
        daily_returns = stock_data_sorted['daily_return'].dropna()
        axes[1, 0].hist(daily_returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
    
    # Plot 4: News sentiment over time (if available)
    if news_data is not None and not news_data.empty and 'vader_compound' in news_data.columns:
        daily_sentiment = news_data.groupby('date')['vader_compound'].mean()
        axes[1, 1].plot(daily_sentiment.index, daily_sentiment.values)
        axes[1, 1].set_title('Average Daily News Sentiment')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No sentiment data\\navailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('News Sentiment Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join('notebooks', 'analysis_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'analysis_plots.png'")

def main():
    """Main function to run all examples"""
    print("NEWS2PROFIT - Data Analysis Example")
    print("=" * 50)
    
    # Step 1: Data Collection
    stock_data, news_data = example_data_collection()
    
    # Step 2: Data Preprocessing
    processed_stock = example_preprocessing(stock_data)
    
    # Step 3: Sentiment Analysis
    if not news_data.empty:
        news_with_sentiment = example_sentiment_analysis(news_data)
    else:
        news_with_sentiment = pd.DataFrame()
    
    # Step 4: Data Integration (if both stock and news data available)
    if not processed_stock.empty and not news_with_sentiment.empty:
        print("\\n=== DATA INTEGRATION ===")
        integrator = DataIntegrator()
        integrated_data = integrator.align_data_by_date(processed_stock, news_with_sentiment)
        print(f"Integrated data shape: {integrated_data.shape}")
        final_data = integrated_data
    else:
        final_data = processed_stock
    
    # Step 5: Model Training
    predictor = example_model_training(final_data)
    
    # Step 6: Visualization
    example_visualization(stock_data, news_with_sentiment if not news_with_sentiment.empty else None)
    
    # Step 7: Generate sample predictions (if models trained)
    if predictor is not None:
        print("\\n=== SAMPLE PREDICTIONS ===")
        try:
            # Get latest data for prediction
            for symbol in final_data['symbol'].unique()[:3]:
                symbol_data = final_data[final_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) > 0:
                    # Get features for latest date
                    feature_cols = [col for col in symbol_data.columns if col not in [
                        'date', 'symbol', 'target', 'target_binary', 'target_numeric',
                        'next_day_return', 'next_day_close', 'next_day_price_change',
                        'next_day_price_change_pct'
                    ]]
                    
                    latest_features = symbol_data[feature_cols].iloc[-1:].fillna(0)
                    latest_features = latest_features.replace([np.inf, -np.inf], 0)
                    
                    # Make prediction
                    prediction = predictor.predict(latest_features)
                    proba = predictor.models[predictor.best_model].predict_proba(latest_features)
                    
                    # Convert prediction
                    if hasattr(predictor.models[predictor.best_model].label_encoder, 'inverse_transform'):
                        pred_label = predictor.models[predictor.best_model].label_encoder.inverse_transform(prediction)[0]
                    else:
                        pred_label = ['DOWN', 'NEUTRAL', 'UP'][prediction[0]]
                    
                    confidence = np.max(proba)
                    print(f"{symbol}: {pred_label} (confidence: {confidence:.2%})")
        
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
    
    print("\\n" + "=" * 50)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
