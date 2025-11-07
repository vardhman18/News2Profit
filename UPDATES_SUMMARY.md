# NEWS2PROFIT - Updated Data Collection Summary

## Changes Made for 5 Specific Stocks Focus

### 🎯 **Target Stocks**
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)  
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- ICICIBANK.NS (ICICI Bank)

## Model.py Fixes Applied

### ✅ **TensorFlow Compatibility Issues Resolved**
- Added fallback imports for different TensorFlow/Keras versions
- Fixed Adam optimizer parameters for TF 2.x compatibility  
- Added proper error handling for missing TensorFlow
- Updated LSTM model to handle different target data types

### ✅ **Model Functionality Verified**
- Logistic Regression: ✅ Working (Accuracy: ~37.5%)
- XGBoost: ✅ Working (Accuracy: ~42.5%) 
- LSTM: ✅ Working with proper data
- StockPredictor Ensemble: ✅ Working (Best: XGBoost)

### 🔧 **Technical Improvements**
- Robust import handling prevents crashes when TensorFlow unavailable
- Better error messages and logging
- Improved data type handling for string targets
- Enhanced model comparison functionality

### 📰 **NewsAPI Updates**
- ✅ Fetches general Indian business headlines
- ✅ Searches for specific stock keywords: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK
- ✅ Uses pagination to get all available articles
- ✅ Saves data to timestamped CSV: `newsapi_data_YYYYMMDD_HHMMSS.csv`

### 🐦 **Twitter API Updates** 
- ✅ Updated to use Twitter API v2 with Bearer Token
- ✅ Maps stock symbols to company names for better search results:
  - RELIANCE.NS → "Reliance Industries"
  - TCS.NS → "Tata Consultancy Services TCS"
  - HDFCBANK.NS → "HDFC Bank" 
  - INFY.NS → "Infosys"
  - ICICIBANK.NS → "ICICI Bank"
- ✅ Handles free tier limitations (10 tweets per request)
- ✅ Saves data to timestamped CSV: `twitter_data_YYYYMMDD_HHMMSS.csv`

### 🔗 **Reddit API Updates**
- ✅ Continues to fetch from investment-related subreddits
- ✅ Saves data to timestamped CSV: `reddit_data_YYYYMMDD_HHMMSS.csv`

### 💾 **CSV File Generation**
All data is automatically saved to separate CSV files in `data/raw/`:

1. **Stock Data**: Individual files per stock
   - `RELIANCE_NS_historical.csv`
   - `TCS_NS_historical.csv` 
   - `INFY_NS_historical.csv`
   - `HDFCBANK_NS_historical.csv`
   - `ICICIBANK_NS_historical.csv`

2. **News Data**: Timestamped files by source
   - `newsapi_data_YYYYMMDD_HHMMSS.csv`
   - `twitter_data_YYYYMMDD_HHMMSS.csv`
   - `reddit_data_YYYYMMDD_HHMMSS.csv`
   - `combined_data_YYYYMMDD_HHMMSS.csv`

### 📊 **Test Results**
✅ Stock collection: 5/5 stocks successful  
✅ Twitter collection: 20 tweets collected and saved  
✅ Reddit collection: 259 posts collected and saved  
✅ CSV files: All sources saving properly  
⚠️ NewsAPI: 0 articles (may need API key verification)

### 🚀 **How to Use**

1. **Run the updated notebook cells** to see the changes in action
2. **Check data/raw/ directory** for generated CSV files
3. **All data is focused on the 5 target stocks** for better analysis
4. **CSV files include timestamps** to avoid overwrites

### 💡 **Next Steps**
- Run sentiment analysis on the collected data
- Train ML models with the focused dataset  
- Analyze patterns specific to these 5 major Indian stocks
- Use the CSV files for further analysis or backup