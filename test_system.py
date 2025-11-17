#!/usr/bin/env python3
"""
Quick test script to validate NEWS2PROFIT setup
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        # Core data science libraries
        import pandas as pd
        import numpy as np
        print(" Core libraries (pandas, numpy)")
        
        # Machine learning libraries
        import sklearn
        import xgboost as xgb
        import tensorflow as tf
        print(" ML libraries (sklearn, xgboost, tensorflow)")
        
        # Visualization libraries
        import matplotlib.pyplot as plt
        import plotly.express as px
        import streamlit as st
        print(" Visualization libraries (matplotlib, plotly, streamlit)")
        
        # Financial data library
        import yfinance as yf
        print(" Financial data library (yfinance)")
        
        # Sentiment analysis libraries
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob
        print(" Sentiment analysis libraries (VADER, TextBlob)")
        
        # NEWS2PROFIT modules
        from src.data_collection import StockDataCollector, NewsDataCollector
        from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor
        from src.sentiment import SentimentAnalyzer
        from src.model import StockPredictor, LogisticRegressionModel
        from src.dashboard import News2ProfitDashboard
        print(" NEWS2PROFIT modules")
        
        return True
        
    except Exception as e:
        print(f" Import error: {str(e)}")
        return False

def test_data_collection():
    """Test basic data collection"""
    print("\nTesting data collection...")
    
    try:
        from src.data_collection import StockDataCollector
        
        collector = StockDataCollector()
        data = collector.fetch_stock_data(['RELIANCE.NS'], period='5d')
        
        if data and 'RELIANCE.NS' in data:
            stock_data = data['RELIANCE.NS']
            print(f" Collected {len(stock_data)} days of stock data")
            print(f"   Columns: {list(stock_data.columns)}")
            return True
        else:
            print(" No stock data collected")
            return False
            
    except Exception as e:
        print(f" Data collection error: {str(e)}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("\nTesting preprocessing...")
    
    try:
        from src.data_collection import StockDataCollector
        from src.preprocessing import StockDataPreprocessor
        
        # Get some data
        collector = StockDataCollector()
        data = collector.fetch_stock_data(['RELIANCE.NS'], period='1mo')
        
        if not data:
            print(" No data for preprocessing test")
            return False
            
        stock_data = list(data.values())[0]
        
        # Test preprocessing
        preprocessor = StockDataPreprocessor()
        cleaned_data = preprocessor.clean_stock_data(stock_data)
        
        if len(cleaned_data) > 0:
            print(f" Cleaned data shape: {cleaned_data.shape}")
            
            # Test technical indicators
            indicators_data = preprocessor.create_technical_indicators(cleaned_data)
            print(f" Added technical indicators. Shape: {indicators_data.shape}")
            
            return True
        else:
            print(" Preprocessing failed - no data")
            return False
            
    except Exception as e:
        print(f" Preprocessing error: {str(e)}")
        return False

def test_sentiment():
    """Test sentiment analysis"""
    print("\nTesting sentiment analysis...")
    
    try:
        from src.sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(['vader', 'textblob'])
        
        test_text = "The stock market is performing very well today with great gains!"
        sentiment = analyzer.analyze_text(test_text)
        
        if sentiment and 'vader_compound' in sentiment:
            print(f"   Sentiment analysis working")
            print(f"   Sample: '{test_text[:50]}...'")
            print(f"   VADER: {sentiment.get('vader_label')} ({sentiment.get('vader_compound', 0):.3f})")
            return True
        else:
            print(" Sentiment analysis failed")
            return False
            
    except Exception as e:
        print(f" Sentiment analysis error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("NEWS2PROFIT - System Validation Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Collection Test", test_data_collection),
        ("Preprocessing Test", test_preprocessing),
        ("Sentiment Analysis Test", test_sentiment),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f" {test_name} failed with error: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for i, (test_name, _) in enumerate(tests):
        status = " PASSED" if results[i] else " FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All systems operational! NEWS2PROFIT is ready to use.")
        print("\n  Next steps:")
        print("   1. Add API keys to .env file")
        print("   2. Run: streamlit run src/dashboard.py")
        print("   3. Open: http://localhost:8501")
    else:
        print("  Some issues detected. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)