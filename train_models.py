"""
Main training script for NEWS2PROFIT models

This script orchestrates the entire pipeline:
1. Data collection
2. Preprocessing 
3. Sentiment analysis
4. Model training
5. Model evaluation and saving
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *
from src.data_collection import StockDataCollector, NewsDataCollector
from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor, DataIntegrator
from src.sentiment import SentimentAnalyzer
from src.model import StockPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collect_data():
    """Step 1: Collect stock and news data"""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 50)
    
    # Collect stock data
    logger.info("Collecting stock data...")
    stock_collector = StockDataCollector()
    
    # Fetch data for selected NSE stocks
    selected_stocks = NSE_STOCKS  # Use all 5 target stocks
    stock_data = stock_collector.fetch_stock_data(selected_stocks, period="2y")
    
    # Save stock data
    stock_collector.save_stock_data()
    
    # Combine all stock data
    all_stock_data = []
    for symbol, data in stock_data.items():
        all_stock_data.append(data)
    
    combined_stock_data = pd.concat(all_stock_data, ignore_index=True) if all_stock_data else pd.DataFrame()
    
    # Collect news data
    logger.info("Collecting news data...")
    news_collector = NewsDataCollector()
    news_data = news_collector.fetch_all_news_data(days_back=90)
    
    # Save news data
    if not news_data.empty:
        news_collector.save_news_data(news_data)
    
    logger.info(f"Collected data for {len(stock_data)} stocks and {len(news_data)} news items")
    
    return combined_stock_data, news_data


def preprocess_data(stock_data, news_data):
    """Step 2: Preprocess stock and news data"""
    logger.info("=" * 50)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 50)
    
    # Preprocess stock data
    logger.info("Preprocessing stock data...")
    stock_preprocessor = StockDataPreprocessor()
    
    # Clean stock data
    cleaned_stock = stock_preprocessor.clean_stock_data(stock_data)
    
    # Create technical indicators
    stock_with_indicators = stock_preprocessor.create_technical_indicators(cleaned_stock)
    
    # Create target variable
    stock_with_target = stock_preprocessor.create_target_variable(stock_with_indicators)
    
    # Preprocess news data if available
    processed_news = pd.DataFrame()
    if not news_data.empty:
        logger.info("Preprocessing news data...")
        news_preprocessor = NewsDataPreprocessor()
        
        # Clean news data
        cleaned_news = news_preprocessor.clean_news_data(news_data)
        
        # Extract text features
        processed_news = news_preprocessor.extract_text_features(cleaned_news)
    
    logger.info(f"Preprocessed stock data shape: {stock_with_target.shape}")
    logger.info(f"Preprocessed news data shape: {processed_news.shape}")
    
    return stock_with_target, processed_news


def analyze_sentiment(news_data):
    """Step 3: Analyze sentiment of news data"""
    logger.info("=" * 50)
    logger.info("STEP 3: SENTIMENT ANALYSIS")
    logger.info("=" * 50)
    
    if news_data.empty:
        logger.warning("No news data available for sentiment analysis")
        return pd.DataFrame()
    
    logger.info("Analyzing news sentiment...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Perform sentiment analysis
    news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_data)
    
    # Get sentiment summary
    sentiment_summary = sentiment_analyzer.get_sentiment_summary(news_with_sentiment)
    
    logger.info("Sentiment Analysis Summary:")
    for key, value in sentiment_summary.items():
        logger.info(f"  {key}: {value}")
    
    return news_with_sentiment


def integrate_data(stock_data, news_data):
    """Step 4: Integrate stock and news data"""
    logger.info("=" * 50)
    logger.info("STEP 4: DATA INTEGRATION")
    logger.info("=" * 50)
    
    if news_data.empty:
        logger.info("No news data to integrate, using stock data only")
        return stock_data
    
    logger.info("Integrating stock and news data...")
    integrator = DataIntegrator()
    
    # Align data by date
    integrated_data = integrator.align_data_by_date(stock_data, news_data)
    
    # Create lagged features
    final_data = integrator.create_lagged_features(integrated_data)
    
    logger.info(f"Integrated data shape: {final_data.shape}")
    
    return final_data


def prepare_for_ml(integrated_data):
    """Step 5: Prepare data for machine learning"""
    logger.info("=" * 50)
    logger.info("STEP 5: ML DATA PREPARATION")
    logger.info("=" * 50)
    
    logger.info("Preparing data for machine learning...")
    stock_preprocessor = StockDataPreprocessor()
    
    # Prepare features
    ml_ready_data = stock_preprocessor.prepare_features(integrated_data)

    # Determine which target column to use for training
    preferred_targets = ['target_ml', 'target', 'target_binary', 'target_numeric']
    chosen_target = None
    for t in preferred_targets:
        if t in ml_ready_data.columns:
            chosen_target = t
            break

    if chosen_target is None:
        logger.warning("No target column found in data. Attempting to create target from stock data...")

        # Try to create technical indicators and target variable if possible
        try:
            if 'daily_return' not in integrated_data.columns:
                logger.info("Adding technical indicators to integrated data")
                integrated_data = stock_preprocessor.create_technical_indicators(integrated_data)

            logger.info("Creating target variable from integrated data")
            integrated_with_target = stock_preprocessor.create_target_variable(integrated_data)

            # Re-run feature preparation
            ml_ready_data = stock_preprocessor.prepare_features(integrated_with_target)

            # Re-evaluate chosen target
            for t in preferred_targets:
                if t in ml_ready_data.columns:
                    chosen_target = t
                    break

            if chosen_target is None:
                logger.error("Unable to create target column automatically. Available columns: %s", list(ml_ready_data.columns))
                raise KeyError(f"Missing target column after attempting auto-generation. Available columns: {list(ml_ready_data.columns)}")

        except Exception as e:
            logger.error("Failed to auto-generate target: %s", str(e))
            raise

    # Remove rows with NaN targets (only drop for the chosen target)
    ml_ready_data = ml_ready_data.dropna(subset=[chosen_target])

    logger.info(f"ML-ready data shape: {ml_ready_data.shape}")
    logger.info(f"Using target column: {chosen_target}")
    logger.info(f"Original target distribution:")
    logger.info(ml_ready_data[chosen_target].value_counts())

    # If we used 'target_ml' or others, we keep it available for training
    ml_ready_data['training_target'] = ml_ready_data[chosen_target]

    return ml_ready_data, chosen_target


def train_models(ml_data, target_col: str = 'training_target'):
    """Step 6: Train machine learning models"""
    logger.info("=" * 50)
    logger.info("STEP 6: MODEL TRAINING")
    logger.info("=" * 50)
    
    logger.info("Training machine learning models...")
    
    # Initialize stock predictor
    predictor = StockPredictor()
    
    # Train all models using the chosen training target
    predictor.train_all_models(ml_data, target_col=target_col)
    
    # Get model comparison
    comparison_df = predictor.get_model_comparison()
    
    logger.info("Model Performance Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    # Save all models
    predictor.save_all_models()
    
    # Save model comparison
    os.makedirs(MODELS_DIR, exist_ok=True)
    comparison_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
    
    logger.info(f"Best performing model: {predictor.best_model}")
    
    return predictor


def generate_predictions(predictor, ml_data):
    """Step 7: Generate sample predictions"""
    logger.info("=" * 50)
    logger.info("STEP 7: PREDICTION GENERATION")
    logger.info("=" * 50)
    
    logger.info("Generating sample predictions...")
    
    # Get latest data for each stock
    predictions_summary = []
    
    for symbol in ml_data['symbol'].unique():
        symbol_data = ml_data[ml_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        if len(symbol_data) >= 10:  # Need enough data for prediction
            # Use last 5 days for prediction
            test_data = symbol_data.tail(5)
            
            # Get features
            # Use feature names from the trained model if available, otherwise infer
            model = predictor.models.get(predictor.best_model)
            if model and getattr(model, 'feature_names', None):
                feature_cols = model.feature_names
            else:
                feature_cols = [col for col in test_data.columns if col not in [
                    'date', 'symbol', 'target', 'target_ml', 'target_binary', 'target_numeric',
                    'training_target', 'next_day_return', 'next_day_close', 'next_day_price_change',
                    'next_day_price_change_pct'
                ]]

            # Ensure training_target and other non-feature columns are not included
            feature_cols = [c for c in feature_cols if c in test_data.columns]

            X_test = test_data[feature_cols].copy()
            # Force numeric where possible and fill missing/infinite
            X_test = X_test.apply(pd.to_numeric, errors='coerce')
            X_test = X_test.fillna(0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Make predictions
            try:
                predictions = predictor.predict(X_test)
                probabilities = predictor.models[predictor.best_model].predict_proba(X_test)
                
                # Get latest prediction
                latest_pred = predictions[-1]
                latest_proba = probabilities[-1] if len(probabilities.shape) > 1 else probabilities
                
                # Convert prediction back to label using label encoder if available
                le = predictor.models[predictor.best_model].label_encoder if predictor.best_model in predictor.models else None
                try:
                    if le is not None:
                        pred_label = le.inverse_transform([latest_pred])[0]
                    else:
                        raise ValueError("No label encoder available")
                except Exception:
                    # Fallback mapping for common integer classes
                    try:
                        pred_label = ['DOWN', 'NEUTRAL', 'UP'][int(latest_pred)]
                    except Exception:
                        pred_label = str(latest_pred)
                
                predictions_summary.append({
                    'symbol': symbol,
                    'prediction': pred_label,
                    'confidence': np.max(latest_proba) if hasattr(latest_proba, '__iter__') else latest_proba,
                    'date': test_data['date'].iloc[-1]
                })
                
            except Exception as e:
                logger.warning(f"Could not generate prediction for {symbol}: {str(e)}")
    
    # Save predictions
    if predictions_summary:
        predictions_df = pd.DataFrame(predictions_summary)
        predictions_path = os.path.join(PROCESSED_DATA_DIR, 'latest_predictions.csv')
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        
        logger.info("Sample Predictions:")
        logger.info(predictions_df.to_string(index=False))
    
    return predictions_summary


def main():
    """Main training pipeline"""
    start_time = datetime.now()
    
    logger.info("Starting NEWS2PROFIT training pipeline...")
    logger.info(f"Start time: {start_time}")
    
    try:
        # Step 1: Data Collection
        stock_data, news_data = collect_data()
        
        # Step 2: Data Preprocessing
        processed_stock, processed_news = preprocess_data(stock_data, news_data)
        
        # Step 3: Sentiment Analysis
        news_with_sentiment = analyze_sentiment(processed_news)
        
        # Step 4: Data Integration
        integrated_data = integrate_data(processed_stock, news_with_sentiment)
        
        # Step 5: ML Data Preparation
        ml_ready_data, chosen_target = prepare_for_ml(integrated_data)

        # Check if we have enough data
        if len(ml_ready_data) < 100:
            logger.error("Not enough data for model training. Need at least 100 samples.")
            return

        # Step 6: Model Training
        predictor = train_models(ml_ready_data, target_col='training_target')

        # Step 7: Generate Sample Predictions
        predictions = generate_predictions(predictor, ml_ready_data)
        
        # Training completed successfully
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Total training time: {duration}")
        logger.info(f"Data samples processed: {len(ml_ready_data)}")
        logger.info(f"Models trained: {len(predictor.models)}")
        logger.info(f"Best model: {predictor.best_model}")
        logger.info(f"Predictions generated: {len(predictions)}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
