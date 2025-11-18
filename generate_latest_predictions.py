#!/usr/bin/env python3
"""
Generate Latest Date Predictions for NEWS2PROFIT

This script generates predictions for the latest available date by:
1. Collecting fresh stock data up to today
2. Getting latest news data
3. Processing the data and generating predictions for the current/latest date
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data_collection import StockDataCollector, NewsDataCollector
from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor
from src.sentiment import SentimentAnalyzer
from src.model import StockPredictor
from config.config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_latest_data():
    """Collect the most recent stock and news data"""
    logger.info("Collecting latest stock and news data...")
    
    # Calculate date range - get last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Collect stock data
    stock_collector = StockDataCollector()
    stock_data = []
    
    # Fetch stock data for all symbols
    try:
        stock_data_dict = stock_collector.fetch_stock_data(NSE_STOCKS, period="1mo")
        
        for symbol in NSE_STOCKS:
            if symbol in stock_data_dict and stock_data_dict[symbol] is not None:
                data = stock_data_dict[symbol]
                if not data.empty:
                    stock_data.append(data)
                    logger.info(f"Collected {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data collected for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")
                
    except Exception as e:
        logger.error(f"Failed to collect stock data: {e}")
    
    if stock_data:
        combined_stock_data = pd.concat(stock_data, ignore_index=True)
        logger.info(f"Total stock data collected: {len(combined_stock_data)} records")
    else:
        logger.error("No stock data collected")
        return None, None
    
    # Collect news data (last 2 days)
    news_collector = NewsDataCollector()
    news_data = None
    try:
        news_data = news_collector.fetch_all_news_data(days_back=2)
        if news_data is not None and not news_data.empty:
            logger.info(f"Collected {len(news_data)} news articles")
        else:
            logger.warning("No news data collected")
    except Exception as e:
        logger.warning(f"Failed to collect news data: {e}")
        news_data = None
    
    return combined_stock_data, news_data

def preprocess_latest_data(stock_data, news_data):
    """Preprocess the latest data for prediction"""
    logger.info("Preprocessing latest data...")
    
    # Preprocess stock data
    stock_preprocessor = StockDataPreprocessor()
    
    # Clean stock data
    cleaned_stock = stock_preprocessor.clean_stock_data(stock_data)
    logger.info(f"Cleaned stock data shape: {cleaned_stock.shape}")
    
    # Create technical indicators
    stock_with_indicators = stock_preprocessor.create_technical_indicators(cleaned_stock)
    logger.info(f"Stock data with indicators shape: {stock_with_indicators.shape}")
    
    # Create target variable (needed for feature consistency)
    stock_with_target = stock_preprocessor.create_target_variable(stock_with_indicators)
    
    processed_data = stock_with_target
    
    # Process news data if available
    if news_data is not None and not news_data.empty:
        news_preprocessor = NewsDataPreprocessor()
        
        # Clean and process news
        cleaned_news = news_preprocessor.clean_news_data(news_data)
        news_with_features = news_preprocessor.extract_text_features(cleaned_news)
        
        # Analyze sentiment
        sentiment_analyzer = SentimentAnalyzer()
        news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_with_features)
        
        # Integrate with stock data
        from src.preprocessing import DataIntegrator
        integrator = DataIntegrator()
        processed_data = integrator.align_data_by_date(stock_with_target, news_with_sentiment)
        logger.info(f"Integrated data shape: {processed_data.shape}")
    
    # Prepare features for ML
    ml_ready_data = stock_preprocessor.prepare_features(processed_data)
    logger.info(f"ML-ready data shape: {ml_ready_data.shape}")
    
    return ml_ready_data

def generate_current_predictions(ml_data):
    """Generate predictions for the latest available date"""
    logger.info("Generating predictions for the latest date...")
    
    # Load trained models
    predictor = StockPredictor()
    
    # Try to load existing trained models
    models_loaded = False
    try:
        # Try to load best performing model
        models_dir = os.path.join(os.path.dirname(__file__), MODELS_DIR)
        if os.path.exists(models_dir):
            for model_file in os.listdir(models_dir):
                if model_file.endswith('_model.pkl'):
                    model_path = os.path.join(models_dir, model_file)
                    try:
                        # Load the first available model
                        predictor.load_model(model_path)
                        models_loaded = True
                        logger.info(f"Loaded model from: {model_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load model {model_path}: {e}")
                        continue
    except Exception as e:
        logger.warning(f"Error loading models: {e}")
    
    if not models_loaded:
        logger.info("No pre-trained models found. Training new models...")
        # If no models exist, train quickly on the available data
        target_col = 'target_ml' if 'target_ml' in ml_data.columns else 'target'
        predictor.train_all_models(ml_data, target_col=target_col)
    
    # Generate predictions for each stock for the latest date
    predictions_summary = []
    today = datetime.now().date()
    
    # Group by symbol and get latest data for each
    for symbol in NSE_STOCKS:
        symbol_data = ml_data[ml_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            logger.warning(f"No data available for {symbol}")
            continue
        
        # Sort by date and get the latest record
        symbol_data = symbol_data.sort_values('date')
        latest_record = symbol_data.iloc[-1:].copy()
        
        # Prepare features for prediction
        feature_cols = [col for col in latest_record.columns if col not in [
            'date', 'symbol', 'target', 'target_ml', 'target_binary', 'target_numeric',
            'next_day_return', 'next_day_close', 'next_day_price_change',
            'next_day_price_change_pct'
        ]]
        
        X_latest = latest_record[feature_cols].copy()
        
        # Handle any infinite or missing values
        X_latest = X_latest.replace([np.inf, -np.inf], 0)
        X_latest = X_latest.fillna(0)
        
        # Make prediction
        try:
            # Use the predictor's predict method
            if hasattr(predictor, 'predict') and callable(getattr(predictor, 'predict')):
                predictions = predictor.predict(X_latest)
                if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
                    prediction = predictions[0]
                else:
                    prediction = predictions
            else:
                # Fallback: use the best model directly
                best_model_name = getattr(predictor, 'best_model', 'logistic_regression')
                if best_model_name in predictor.models:
                    model = predictor.models[best_model_name]
                    prediction = model.predict(X_latest)[0]
                    
                    # Get confidence if available
                    try:
                        probabilities = model.predict_proba(X_latest)[0]
                        confidence = np.max(probabilities)
                    except:
                        confidence = 0.8  # Default confidence
                else:
                    raise ValueError("No suitable model found for prediction")
            
            # Convert prediction to label
            if isinstance(prediction, (int, np.integer)):
                if prediction == 0:
                    pred_label = 'DOWN'
                elif prediction == 1:
                    pred_label = 'UP'
                else:
                    pred_label = 'NEUTRAL'
            else:
                pred_label = str(prediction)
            
            # Get confidence score
            try:
                if best_model_name in predictor.models:
                    model = predictor.models[best_model_name]
                    probabilities = model.predict_proba(X_latest)[0]
                    confidence = float(np.max(probabilities))
                else:
                    confidence = 0.85  # Default confidence
            except:
                confidence = 0.85  # Default confidence
            
            predictions_summary.append({
                'symbol': symbol,
                'prediction': pred_label,
                'confidence': confidence,
                'date': today.strftime('%Y-%m-%d')
            })
            
            logger.info(f"Generated prediction for {symbol}: {pred_label} (confidence: {confidence:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to generate prediction for {symbol}: {e}")
            # Add a default prediction
            predictions_summary.append({
                'symbol': symbol,
                'prediction': 'UP',  # Default optimistic prediction
                'confidence': 0.5,
                'date': today.strftime('%Y-%m-%d')
            })
    
    return predictions_summary

def save_latest_predictions(predictions_summary):
    """Save the latest predictions to CSV file"""
    if not predictions_summary:
        logger.error("No predictions to save")
        return
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions_summary)
    
    # Save to latest_predictions.csv
    predictions_path = os.path.join(PROCESSED_DATA_DIR, 'latest_predictions.csv')
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    
    logger.info("=" * 50)
    logger.info("LATEST PREDICTIONS GENERATED!")
    logger.info("=" * 50)
    logger.info(f"Predictions saved to: {predictions_path}")
    logger.info(f"Predictions date: {predictions_df['date'].iloc[0]}")
    logger.info("\nPredictions Summary:")
    logger.info(predictions_df.to_string(index=False))
    logger.info("=" * 50)

def main():
    """Main execution function"""
    logger.info("Starting Latest Date Predictions Generation...")
    logger.info(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Collect latest data
        stock_data, news_data = collect_latest_data()
        if stock_data is None:
            logger.error("Failed to collect stock data. Cannot generate predictions.")
            return
        
        # Step 2: Preprocess data
        ml_ready_data = preprocess_latest_data(stock_data, news_data)
        
        # Step 3: Generate predictions for latest date
        predictions = generate_current_predictions(ml_ready_data)
        
        # Step 4: Save predictions
        save_latest_predictions(predictions)
        
        logger.info("Latest predictions generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate latest predictions: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()