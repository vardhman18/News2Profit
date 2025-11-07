#!/usr/bin/env python3
"""
Test script for updated data collection focusing on 5 specific stocks
with CSV file saving functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.data_collection import StockDataCollector, NewsDataCollector
from config.config import RAW_DATA_DIR

def test_stock_collection():
    """Test stock data collection for 5 specific stocks"""
    print("=" * 60)
    print("TESTING STOCK DATA COLLECTION")
    print("=" * 60)
    
    collector = StockDataCollector()
    
    # Test with 5 specific stocks
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    stock_data = collector.fetch_stock_data(test_stocks, period='5d')
    
    for symbol in test_stocks:
        if symbol in stock_data and not stock_data[symbol].empty:
            print(f"✅ {symbol}: {len(stock_data[symbol])} records")
        else:
            print(f"❌ {symbol}: No data collected")
    
    # Save to CSV
    collector.save_stock_data()
    print(f"💾 Stock data saved to {RAW_DATA_DIR}")

def test_news_collection():
    """Test news data collection with CSV saving"""
    print("\n" + "=" * 60)
    print("TESTING NEWS DATA COLLECTION WITH CSV SAVING")
    print("=" * 60)
    
    collector = NewsDataCollector()
    
    # Test news collection with CSV saving enabled
    try:
        # Test with smaller date range to avoid rate limits
        news_df = collector.fetch_all_news_data(days_back=7, save_to_csv=True)
        
        if not news_df.empty:
            print(f"✅ Total news collected: {len(news_df)} articles")
            print(f"📊 Sources: {news_df['source'].value_counts().to_dict()}")
            print(f"🎯 Keywords: {list(news_df['keyword'].unique())}")
        else:
            print("⚠️ No news data collected")
            
    except Exception as e:
        print(f"❌ News collection error: {e}")
    
    # Check what CSV files were created
    print(f"\n📁 CSV Files in {RAW_DATA_DIR}:")
    if os.path.exists(RAW_DATA_DIR):
        csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        for file in csv_files:
            file_path = os.path.join(RAW_DATA_DIR, file)
            size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"   📄 {file} ({size:.1f} KB)")
    else:
        print("   No CSV files found")

def main():
    """Run all tests"""
    print("TESTING UPDATED DATA COLLECTION FOR 5 SPECIFIC STOCKS")
    print("Stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS")
    print("Features: Focused data collection + CSV file saving")
    
    # Test stock collection
    test_stock_collection()
    
    # Test news collection
    test_news_collection()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("💡 Check the data/raw/ directory for generated CSV files")
    print("💡 Run the updated notebook cells to see the changes in action")

if __name__ == "__main__":
    main()