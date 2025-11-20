"""
Streamlit Dashboard for NEWS2PROFIT

This module creates an interactive dashboard for visualizing stock predictions,
sentiment analysis results, and model performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Graceful imports for Streamlit Cloud
try:
    from config.config import *
    from src.data_collection import StockDataCollector, NewsDataCollector
    from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor, DataIntegrator
    from src.sentiment import SentimentAnalyzer
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"Some ML components not available: {e}")
    ML_AVAILABLE = False

# Optional ML model imports
try:
    from src.model import StockPredictor, LogisticRegressionModel, XGBoostModel, LSTMModel
    MODELS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced ML models not available: {e}")
    MODELS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def generate_and_save_predictions():
    """
    Generate predictions for the latest date using the latest data and trained models.
    """
    try:
        import subprocess
        import sys
        
        # Run the latest predictions generation script
        script_path = os.path.join(os.path.dirname(__file__), '..', 'generate_latest_predictions.py')
        
        with st.spinner('Generating predictions for the latest date...'):
            # Run the script and capture output
            result = subprocess.run(
                [sys.executable, script_path], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(script_path)
            )
            
            if result.returncode == 0:
                # Load the generated predictions
                predictions_path = os.path.join(PROCESSED_DATA_DIR, 'latest_predictions.csv')
                if os.path.exists(predictions_path):
                    predictions_df = pd.read_csv(predictions_path)
                    st.success(f"✅ Latest predictions generated for {predictions_df['date'].iloc[0]}!")
                    return predictions_df
                else:
                    st.error("❌ Predictions file not found after generation")
                    return None
            else:
                st.error(f"❌ Error generating predictions: {result.stderr}")
                st.error(f"Script output: {result.stdout}")
                return None
                
    except Exception as e:
        st.error(f"❌ Error running prediction generation: {e}")
        return None


class News2ProfitDashboard:
    """Main dashboard class for NEWS2PROFIT application"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="NEWS2PROFIT - Stock Movement Prediction",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.25rem solid #1f77b4;
        }
        
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            font-weight: bold;
            font-size: 1.2rem;
        }
        
        .bullish { background-color: #d4edda; color: #155724; }
        .bearish { background-color: #f8d7da; color: #721c24; }
        .neutral { background-color: #fff3cd; color: #856404; }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        
        if 'stock_data' not in st.session_state:
            st.session_state.stock_data = None
        
        if 'news_data' not in st.session_state:
            st.session_state.news_data = None
        
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
            # Try to load existing predictions
            self.load_existing_predictions()
        
        # Control rendering: show predictions only after explicit button click
        if 'show_predictions' not in st.session_state:
            st.session_state.show_predictions = False
    
    def load_existing_predictions(self):
        """Load existing predictions from CSV file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                os.path.join('data', 'processed', 'latest_predictions.csv'),
                os.path.join('..', 'data', 'processed', 'latest_predictions.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'latest_predictions.csv')
            ]
            
            for predictions_path in possible_paths:
                if os.path.exists(predictions_path):
                    predictions_df = pd.read_csv(predictions_path)
                    # Convert to the format expected by render_predictions
                    predictions_dict = {}
                    for _, row in predictions_df.iterrows():
                        predictions_dict[row['symbol']] = {
                            'prediction': row['prediction'],
                            'confidence': row['confidence']
                        }
                    st.session_state.predictions = predictions_dict
                    break
        except Exception as e:
            print(f"Error loading predictions: {e}")
    
    def show_latest_predictions_from_file(self):
        """Show predictions directly from file if session state is empty"""
        try:
            # Try to find and display the predictions file
            possible_paths = [
                os.path.join('data', 'processed', 'latest_predictions.csv'),
                os.path.join('..', 'data', 'processed', 'latest_predictions.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'latest_predictions.csv')
            ]
            
            predictions_df = None
            for path in possible_paths:
                if os.path.exists(path):
                    predictions_df = pd.read_csv(path)
                    break
            
            if predictions_df is not None:
                st.subheader("🔮 Latest Stock Movement Predictions")
                st.success("📊 **Pre-trained ML Models Results** (Logistic Regression: 76.46% accuracy)")
                
                # Filter for our 5 target stocks only
                target_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
                predictions_df = predictions_df[predictions_df['symbol'].isin(target_stocks)]
                
                # Add sentiment data
                try:
                    # Try to load sentiment summary
                    sentiment_paths = [
                        'data/processed/sentiment_summary.csv',
                        '../data/processed/sentiment_summary.csv',
                        os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sentiment_summary.csv')
                    ]
                    
                    sentiment_df = None
                    for path in sentiment_paths:
                        if os.path.exists(path):
                            sentiment_df = pd.read_csv(path, index_col=0)
                            break
                    
                    if sentiment_df is not None:
                        # Add sentiment to predictions
                        predictions_df['sentiment_score'] = predictions_df['symbol'].apply(
                            lambda x: sentiment_df.loc[x.replace('.NS', ''), 'avg_sentiment'] 
                            if x.replace('.NS', '') in sentiment_df.index else 0.0
                        )
                        predictions_df['sentiment_emoji'] = predictions_df['sentiment_score'].apply(
                            lambda x: "😊 Positive" if x > 0.1 else "😐 Neutral" if x > -0.1 else "😟 Negative"
                        )
                except:
                    predictions_df['sentiment_score'] = 0.0
                    predictions_df['sentiment_emoji'] = "📰 Analyzing..."
                
                # Format display
                predictions_df['confidence'] = predictions_df['confidence'].apply(lambda x: f"{x:.1%}")
                predictions_df['symbol'] = predictions_df['symbol'].str.replace('.NS', '')
                
                # Add emojis for predictions
                # Normalize prediction values and add a human-readable trend column
                predictions_df['prediction'] = predictions_df['prediction'].astype(str).str.upper()
                predictions_df['trend'] = predictions_df['prediction'].apply(lambda x: 'UP' if x == 'UP' else ('DOWN' if x == 'DOWN' else 'NEUTRAL'))
                
                # Reorder columns
                if 'sentiment_emoji' in predictions_df.columns:
                    display_df = predictions_df[['symbol', 'trend', 'confidence', 'sentiment_emoji', 'date']].copy()
                    display_df.columns = ['Stock', 'ML Prediction', 'Confidence', 'News Sentiment', 'Date']
                else:
                    display_df = predictions_df[['symbol', 'trend', 'confidence', 'date']].copy()
                    display_df.columns = ['Stock', 'ML Prediction', 'Confidence', 'Date']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Show summary stats
                up_count = len(predictions_df[predictions_df['prediction'] == 'UP'])
                down_count = len(predictions_df[predictions_df['prediction'] == 'DOWN'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Bullish Predictions", up_count)
                with col2:
                    st.metric("📉 Bearish Predictions", down_count)
                with col3:
                    st.metric("🎯 Total Stocks", len(predictions_df))
                
                # Show individual stock cards
                self.show_stock_cards(predictions_df)
                
                # Add sentiment analysis section
                self.show_sentiment_analysis()
                    
                st.markdown("---")
            else:
                st.info("📊 No predictions available yet. Train the models first by running `python train_models.py` or click 'Generate Predictions' in the sidebar.")
                
        except Exception as e:
            st.error(f"Error displaying predictions: {e}")
    
    def show_stock_cards(self, predictions_df):
        """Show individual cards for each stock with prediction and sentiment"""
        try:
            st.subheader("🎯 Individual Stock Analysis")
            
            # Ensure we only have the 5 target stocks
            target_stocks_order = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
            
            # Create 5 columns for the 5 stocks  
            cols = st.columns(5)
            
            # Sort predictions to match our target order
            predictions_list = []
            for stock in target_stocks_order:
                stock_with_ns = stock + '.NS'
                matching_row = predictions_df[predictions_df['symbol'] == stock_with_ns]
                if not matching_row.empty:
                    predictions_list.append(matching_row.iloc[0])
            
            for i, row in enumerate(predictions_list[:5]):
                with cols[i]:
                    stock_symbol = row['symbol']
                    prediction = row['prediction']
                    confidence = row['confidence']
                    
                    # Determine colors and styling
                    if prediction == 'UP':
                        bg_color = "#d4edda"
                        border_color = "#28a745"
                        emoji = "📈"
                    else:
                        bg_color = "#f8d7da" 
                        border_color = "#dc3545"
                        emoji = "📉"
                    
                    # Get sentiment if available
                    sentiment_text = row.get('sentiment_emoji', '📰 Analyzing...')
                    
                    # Create card HTML
                    card_html = f"""
                    <div style="
                        background-color: {bg_color};
                        border: 2px solid {border_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin: 5px;
                        text-align: center;
                        height: 160px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    ">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #333;">
                            {stock_symbol}
                        </div>
                        <div style="font-size: 2rem;">
                            {emoji}
                        </div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: {border_color};">
                            {prediction}
                        </div>
                        <div style="font-size: 0.9rem; color: #666;">
                            {confidence}
                        </div>
                        <div style="font-size: 0.8rem; color: #555;">
                            {sentiment_text}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error creating stock cards: {e}")
    
    def show_sentiment_analysis(self):
        """Display sentiment analysis for the 5 target stocks"""
        try:
            st.subheader("📰 News Sentiment Analysis")
            
            # Try to generate sentiment on the fly from recent news
            stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
            
            # Load recent news
            news_paths = [
                'data/raw/news_data.csv',
                '../data/raw/news_data.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'news_data.csv')
            ]
            
            news_df = None
            for path in news_paths:
                if os.path.exists(path):
                    news_df = pd.read_csv(path)
                    break
            
            if news_df is not None:
                # Quick sentiment calculation for display
                sentiment_data = []
                for stock in stocks:
                    stock_news = news_df[news_df['keyword'] == stock]
                    if len(stock_news) > 0:
                        # Simple sentiment based on keywords in titles
                        positive_keywords = ['profit', 'gain', 'up', 'rise', 'bullish', 'buy', 'growth', 'strong']
                        negative_keywords = ['loss', 'fall', 'down', 'drop', 'bearish', 'sell', 'weak', 'decline']
                        
                        total_news = len(stock_news)
                        positive_count = 0
                        negative_count = 0
                        
                        for title in stock_news['title']:
                            title_lower = str(title).lower()
                            if any(keyword in title_lower for keyword in positive_keywords):
                                positive_count += 1
                            elif any(keyword in title_lower for keyword in negative_keywords):
                                negative_count += 1
                        
                        neutral_count = total_news - positive_count - negative_count
                        
                        # Calculate overall sentiment
                        if positive_count > negative_count:
                            sentiment_label = "😊 Positive"
                            sentiment_color = "green"
                        elif negative_count > positive_count:
                            sentiment_label = "😟 Negative"  
                            sentiment_color = "red"
                        else:
                            sentiment_label = "😐 Neutral"
                            sentiment_color = "gray"
                            
                        sentiment_data.append({
                            'Stock': stock,
                            'Total News': total_news,
                            'Positive': positive_count,
                            'Negative': negative_count,
                            'Neutral': neutral_count,
                            'Overall': sentiment_label
                        })
                
                if sentiment_data:
                    sentiment_df = pd.DataFrame(sentiment_data)
                    st.dataframe(sentiment_df, use_container_width=True)
                    
                    # Show sentiment distribution chart
                    if len(sentiment_df) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart of overall sentiment distribution
                            sentiment_counts = sentiment_df['Overall'].value_counts()
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="📊 Overall News Sentiment Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Bar chart by stock
                            fig_bar = px.bar(
                                sentiment_df,
                                x='Stock',
                                y=['Positive', 'Negative', 'Neutral'],
                                title="📈 Sentiment by Stock",
                                barmode='stack'
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No recent news data available for sentiment analysis")
            else:
                st.info("📰 News sentiment data not available. Run data collection to get latest news.")
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">📈 NEWS2PROFIT</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predicting Movement for Top 5 NSE Stocks with ML & News Sentiment</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1rem; color: #888;">🎯 RELIANCE • TCS • HDFCBANK • INFY • ICICIBANK</p>', unsafe_allow_html=True)
        
        # Show latest predictions summary only after user generates them in this session
        if st.session_state.get('show_predictions') and st.session_state.get('predictions'):
            st.success("🎯 **Latest ML Predictions Generated!**")
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.header("⚙️ Controls")
        
        # Stock selection
        # Limit stock selection to five predefined stocks
        st.sidebar.subheader("📊 Stock Selection")
        selected_stocks = st.sidebar.multiselect(
            "Select NSE Stocks:",
            ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
            default=["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        )
        
        # Date range selection
        st.sidebar.subheader("📅 Date Range")
        end_date = st.sidebar.date_input("End Date", datetime.now().date())
        start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))
        
        # Model selection
        st.sidebar.subheader("🤖 Model Settings")
        model_type = st.sidebar.selectbox(
            "Select Model:",
            ["Auto (Best)", "Logistic Regression", "XGBoost", "LSTM"]
        )
        
        # Action buttons
        st.sidebar.subheader("🎯 Actions")
        
        # Quick Prediction Generation (uses latest data and trained models)
        if st.sidebar.button("⚡ Generate Latest Predictions", type="primary"):
            predictions_df = generate_and_save_predictions()
            if predictions_df is not None:
                st.session_state.show_predictions = True
                # Convert to session dict format
                preds = {}
                for _, row in predictions_df.iterrows():
                    preds[row['symbol']] = {
                        'prediction': row['prediction'], 
                        'confidence': float(row['confidence']), 
                        'date': row['date']
                    }
                st.session_state.predictions = preds
                # Streamlit compatibility: try new rerun() first, fallback to experimental_rerun()
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Advanced Options:**")
        
        if st.sidebar.button("📥 Load Data"):
            self.load_data(selected_stocks, start_date, end_date)
        
        if st.sidebar.button("🧠 Train Models"):
            if st.session_state.data_loaded:
                self.train_models()
            else:
                st.sidebar.error("Please load data first!")
        
        return selected_stocks, start_date, end_date, model_type
    
    def load_data(self, stocks, start_date, end_date):
        """Load stock and news data"""
        with st.spinner("Loading stock and news data..."):
            try:
                # Load stock data
                stock_collector = StockDataCollector()
                period = f"{(end_date - start_date).days // 30}mo" if (end_date - start_date).days > 30 else "1mo"
                stock_data = stock_collector.fetch_stock_data(stocks, period=period)
                
                # Combine all stock data
                all_stock_data = []
                for symbol, data in stock_data.items():
                    all_stock_data.append(data)
                
                if all_stock_data:
                    combined_stock_data = pd.concat(all_stock_data, ignore_index=True)
                    st.session_state.stock_data = combined_stock_data
                
                # Load news data
                news_collector = NewsDataCollector()
                news_data = news_collector.fetch_all_news_data(days_back=30)
                st.session_state.news_data = news_data
                
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded data for {len(stocks)} stocks and {len(news_data)} news items")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    def train_models(self):
        """Train machine learning models"""
        with st.spinner("Training machine learning models..."):
            try:
                if st.session_state.stock_data is None:
                    st.error("No stock data available")
                    return
                
                # Preprocess stock data
                stock_preprocessor = StockDataPreprocessor()
                cleaned_stock = stock_preprocessor.clean_stock_data(st.session_state.stock_data)
                stock_with_indicators = stock_preprocessor.create_technical_indicators(cleaned_stock)
                stock_with_target = stock_preprocessor.create_target_variable(stock_with_indicators)
                
                # Process news data if available
                if st.session_state.news_data is not None and len(st.session_state.news_data) > 0:
                    news_preprocessor = NewsDataPreprocessor()
                    sentiment_analyzer = SentimentAnalyzer()
                    
                    cleaned_news = news_preprocessor.clean_news_data(st.session_state.news_data)
                    news_with_features = news_preprocessor.extract_text_features(cleaned_news)
                    news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_with_features)
                    
                    # Integrate data
                    integrator = DataIntegrator()
                    final_data = integrator.align_data_by_date(stock_with_target, news_with_sentiment)
                else:
                    final_data = stock_with_target
                
                # Prepare for ML
                ml_data = stock_preprocessor.prepare_features(final_data)
                
                # Train models
                predictor = StockPredictor()
                predictor.train_all_models(ml_data, target_col='target')
                
                st.session_state.predictor = predictor
                st.session_state.processed_data = ml_data
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Show model performance
                comparison_df = predictor.get_model_comparison()
                st.subheader("📊 Model Performance Comparison")
                st.dataframe(comparison_df)
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
    
    def generate_predictions(self, model_type):
        """Generate predictions using the selected model"""
        with st.spinner("Generating predictions..."):
            try:
                if st.session_state.stock_data is None:
                    st.error("No stock data available")
                    return
                # Prepare ML-ready data
                if st.session_state.get('processed_data') is not None:
                    ml_data = st.session_state.processed_data
                else:
                    # Recreate processed data from loaded stock and news
                    stock_preprocessor = StockDataPreprocessor()
                    cleaned_stock = stock_preprocessor.clean_stock_data(st.session_state.stock_data)
                    stock_with_indicators = stock_preprocessor.create_technical_indicators(cleaned_stock)
                    stock_with_target = stock_preprocessor.create_target_variable(stock_with_indicators)

                    # Integrate news if available
                    if st.session_state.get('news_data') is not None and len(st.session_state.get('news_data')) > 0:
                        news_preprocessor = NewsDataPreprocessor()
                        cleaned_news = news_preprocessor.clean_news_data(st.session_state.news_data)
                        news_with_features = news_preprocessor.extract_text_features(cleaned_news)
                        sentiment_analyzer = SentimentAnalyzer()
                        news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_with_features)
                        integrator = DataIntegrator()
                        final_data = integrator.align_data_by_date(stock_with_target, news_with_sentiment)
                    else:
                        final_data = stock_with_target

                    ml_data = stock_preprocessor.prepare_features(final_data)

                # Determine training target column
                target_col = 'training_target' if 'training_target' in ml_data.columns else ('target_ml' if 'target_ml' in ml_data.columns else 'target')

                # Initialize predictor and train (or use existing trained predictor)
                if st.session_state.get('predictor') and st.session_state.get('models_trained'):
                    predictor = st.session_state.predictor
                else:
                    predictor = StockPredictor()
                    predictor.train_all_models(ml_data, target_col=target_col)
                    st.session_state.predictor = predictor
                    st.session_state.processed_data = ml_data
                    st.session_state.models_trained = True

                # Generate predictions for the latest date
                try:
                    predictions_summary = []
                    today = datetime.now().date()
                    
                    # Generate predictions for each stock for the latest date
                    for symbol in NSE_STOCKS:
                        symbol_data = ml_data[ml_data['symbol'] == symbol].copy()
                        
                        if symbol_data.empty:
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
                        X_latest = X_latest.replace([np.inf, -np.inf], 0)
                        X_latest = X_latest.fillna(0)
                        
                        # Make prediction
                        try:
                            prediction = predictor.predict(X_latest)
                            if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                                pred_value = prediction[0]
                            else:
                                pred_value = prediction
                            
                            # Convert prediction to label
                            if isinstance(pred_value, (int, np.integer)):
                                pred_label = ['DOWN', 'NEUTRAL', 'UP'][min(int(pred_value), 2)]
                            else:
                                pred_label = str(pred_value)
                            
                            # Get confidence
                            try:
                                best_model_name = getattr(predictor, 'best_model', 'logistic_regression')
                                if best_model_name in predictor.models:
                                    model = predictor.models[best_model_name]
                                    probabilities = model.predict_proba(X_latest)[0]
                                    confidence = float(np.max(probabilities))
                                else:
                                    confidence = 0.85
                            except:
                                confidence = 0.85
                            
                            predictions_summary.append({
                                'symbol': symbol,
                                'prediction': pred_label,
                                'confidence': confidence,
                                'date': today.strftime('%Y-%m-%d')
                            })
                            
                        except Exception as e:
                            st.warning(f"Could not generate prediction for {symbol}: {str(e)}")
                    
                    # Save predictions
                    if predictions_summary:
                        predictions_df = pd.DataFrame(predictions_summary)
                        predictions_path = os.path.join(PROCESSED_DATA_DIR, 'latest_predictions.csv')
                        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
                        predictions_df.to_csv(predictions_path, index=False)
                        
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {e}")
                    return

                # Load saved predictions into session state
                predictions_path = os.path.join(PROCESSED_DATA_DIR, 'latest_predictions.csv')
                if os.path.exists(predictions_path):
                    try:
                        predictions_df = pd.read_csv(predictions_path)
                        # Convert to session dict format
                        preds = {}
                        for _, row in predictions_df.iterrows():
                            preds[row['symbol']] = {'prediction': row['prediction'], 'confidence': float(row.get('confidence', 0)), 'date': row.get('date')}
                        st.session_state.predictions = preds
                        # Gate to allow predictions to render in main content
                        st.session_state.show_predictions = True
                    except Exception:
                        pass

                # Confirm generation; rendering happens in main content to avoid duplicates
                st.success("✅ Predictions generated successfully. Scroll to see them below.")

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
    
    def render_main_content(self, stocks, start_date, end_date):
        """Render the main content area"""
        # Show predictions only when user has explicitly generated them
        if st.session_state.get('show_predictions'):
            try:
                self.show_predictions_for_selected_stocks(stocks)
            except Exception:
                pass
        
        # Data Overview
        if st.session_state.data_loaded:
            self.render_data_overview()
        
        # Charts and Analysis
        if st.session_state.stock_data is not None:
            self.render_stock_charts(stocks)
        
        if st.session_state.news_data is not None:
            self.render_sentiment_analysis()

    def show_predictions_for_selected_stocks(self, stocks):
        """Load latest predictions and display only for the selected stocks"""
        try:
            # Try session state first
            if st.session_state.get('predictions'):
                # Convert session dict to DataFrame
                preds = st.session_state.predictions
                rows = []
                for sym, p in preds.items():
                    row = {'symbol': sym, 'prediction': p.get('prediction'), 'confidence': p.get('confidence'), 'date': datetime.now().date()}
                    rows.append(row)
                predictions_df = pd.DataFrame(rows)
            else:
                # Try loading from file
                possible_paths = [
                    os.path.join('data', 'processed', 'latest_predictions.csv'),
                    os.path.join('..', 'data', 'processed', 'latest_predictions.csv'),
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'latest_predictions.csv')
                ]
                predictions_df = None
                for path in possible_paths:
                    if os.path.exists(path):
                        predictions_df = pd.read_csv(path)
                        break

            if predictions_df is None or predictions_df.empty:
                return

            # Filter by selected stocks
            if stocks:
                predictions_df = predictions_df[predictions_df['symbol'].isin(stocks)]
            else:
                # If none selected, default to our 5 target stocks
                target_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
                predictions_df = predictions_df[predictions_df['symbol'].isin(target_stocks)]

            if predictions_df.empty:
                return

            # Format and display similar to existing logic
            st.subheader("🔮 Stock Movement Predictions")
            # Normalize prediction values
            predictions_df['prediction'] = predictions_df['prediction'].astype(str).str.upper()
            predictions_df['confidence'] = predictions_df['confidence'].astype(float).apply(lambda x: f"{x:.1%}")
            predictions_df['symbol'] = predictions_df['symbol'].str.replace('.NS', '')
            # Provide a clear UP/DOWN/NEUTRAL label for ML Prediction column
            predictions_df['trend'] = predictions_df['prediction'].apply(lambda x: 'UP' if x == 'UP' else ('DOWN' if x == 'DOWN' else 'NEUTRAL'))

            display_df = predictions_df[['symbol', 'trend', 'confidence', 'date']].copy()
            display_df.columns = ['Stock', 'ML Prediction', 'Confidence', 'Date']
            st.dataframe(display_df, use_container_width=True)

            # Show individual cards for selected stocks
            self.show_stock_cards(predictions_df)

        except Exception as e:
            st.error(f"Error showing selected predictions: {e}")
    
    def render_data_overview(self):
        """Render data overview section"""
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Loaded", len(st.session_state.stock_data['symbol'].unique()) if st.session_state.stock_data is not None else 0)
        
        with col2:
            st.metric("Data Points", len(st.session_state.stock_data) if st.session_state.stock_data is not None else 0)
        
        with col3:
            st.metric("News Articles", len(st.session_state.news_data) if st.session_state.news_data is not None else 0)
        
        with col4:
            st.metric("Models Trained", "Yes" if st.session_state.models_trained else "No")
        
        st.markdown("---")
    
    def render_predictions(self):
        """Render predictions section"""
        st.subheader("🔮 Stock Movement Predictions")
        
        predictions = st.session_state.predictions
        
        # Create columns for predictions
        cols = st.columns(min(len(predictions), 4))
        
        for i, (symbol, pred_data) in enumerate(predictions.items()):
            with cols[i % 4]:
                prediction = str(pred_data.get('prediction', '')).upper()
                confidence = pred_data.get('confidence', 0)

                # Determine styling and display text
                if prediction == 'UP':
                    css_class = 'bullish'
                    emoji = '📈'
                    display_text = 'UP'
                elif prediction == 'DOWN':
                    css_class = 'bearish'
                    emoji = '📉'
                    display_text = 'DOWN'
                else:
                    css_class = 'neutral'
                    emoji = '➡️'
                    display_text = prediction or 'N/A'

                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    <div>{symbol.replace('.NS', '')}</div>
                    <div>{emoji} {display_text}</div>
                    <div style="font-size: 0.9rem;">Confidence: {float(confidence):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
    
    def render_stock_charts(self, selected_stocks):
        """Render stock price charts"""
        st.subheader("📈 Individual Stock Analysis")

        if st.session_state.stock_data is None:
            st.info("No stock data available. Please load data first.")
            return

        # Use provided selected_stocks if any, otherwise show the default top-5
        from config.config import NSE_STOCKS
        symbols_to_show = selected_stocks if selected_stocks else NSE_STOCKS

        # Ensure we only show the five project symbols and keep order
        symbols_to_show = [s for s in NSE_STOCKS if s in symbols_to_show] or NSE_STOCKS

        tabs = st.tabs([s.replace('.NS', '') for s in symbols_to_show])

        for tab, symbol in zip(tabs, symbols_to_show):
            with tab:
                stock_df = st.session_state.stock_data[st.session_state.stock_data['symbol'] == symbol].copy()
                if stock_df.empty:
                    st.info(f"No data for {symbol}. Try loading data for a wider date range.")
                    continue

                # Ensure date is datetime
                try:
                    stock_df['date'] = pd.to_datetime(stock_df['date'])
                except Exception:
                    pass

                stock_df = stock_df.sort_values('date')

                # Compute simple moving averages for visual cues
                if 'close' in stock_df.columns:
                    stock_df['sma_20'] = stock_df['close'].rolling(window=20, min_periods=1).mean()
                    stock_df['sma_50'] = stock_df['close'].rolling(window=50, min_periods=1).mean()

                # Create subplots: candlestick (row 1) + volume (row 2)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7, 0.3], vertical_spacing=0.03)

                fig.add_trace(go.Candlestick(
                    x=stock_df['date'],
                    open=stock_df['open'],
                    high=stock_df['high'],
                    low=stock_df['low'],
                    close=stock_df['close'],
                    name=f"{symbol}"
                ), row=1, col=1)

                # Add SMA overlays if computed
                if 'sma_20' in stock_df.columns:
                    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['sma_20'],
                                             mode='lines', name='SMA 20', line=dict(width=1.5, color='blue')),
                                  row=1, col=1)
                if 'sma_50' in stock_df.columns:
                    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['sma_50'],
                                             mode='lines', name='SMA 50', line=dict(width=1.5, color='orange')),
                                  row=1, col=1)

                # Volume bars
                fig.add_trace(go.Bar(x=stock_df['date'], y=stock_df['volume'], name='Volume', marker_color='rgba(0, 123, 255, 0.6)'), row=2, col=1)

                # Layout tweaks
                fig.update(layout_xaxis_rangeslider_visible=False)
                fig.update_layout(height=650, showlegend=True,
                                  title_text=f"{symbol} Price Chart and Volume")

                # Add some improved axis labels and formatting
                fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Quick metrics for the symbol
                latest = stock_df.iloc[-1]
                prev = stock_df.iloc[-2] if len(stock_df) > 1 else latest
                price = latest['close']
                change = price - prev['close']
                pct_change = (change / prev['close']) * 100 if prev['close'] != 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label=f"{symbol.replace('.NS', '')} Price", value=f"₹{price:,.2f}", delta=f"{change:,.2f} ({pct_change:.2f}%)")
                with col2:
                    high_52 = stock_df['high'].rolling(window=252, min_periods=1).max().iloc[-1]
                    st.metric(label="52-week High", value=f"₹{high_52:,.2f}")
                with col3:
                    low_52 = stock_df['low'].rolling(window=252, min_periods=1).min().iloc[-1]
                    st.metric(label="52-week Low", value=f"₹{low_52:,.2f}")

                st.markdown("---")
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis section"""
        st.subheader("😊 News Sentiment Analysis")
        
        if st.session_state.news_data is None or len(st.session_state.news_data) == 0:
            st.info("No news data available.")
            return
        
        news_data = st.session_state.news_data
        
        # Check if sentiment analysis has been performed
        sentiment_cols = [col for col in news_data.columns if 'sentiment' in col or 'vader' in col]
        
        if sentiment_cols:
            # Sentiment distribution
            if 'vader_label' in news_data.columns:
                sentiment_counts = news_data['vader_label'].value_counts()
                
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="News Sentiment Distribution"
                )
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Sentiment over time
            if 'date' in news_data.columns and 'vader_compound' in news_data.columns:
                daily_sentiment = news_data.groupby('date')['vader_compound'].mean().reset_index()
                
                fig_time = px.line(
                    daily_sentiment,
                    x='date',
                    y='vader_compound',
                    title="Average Daily Sentiment Score"
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent news headlines
        st.subheader("📰 Recent News Headlines")
        
        if 'title' in news_data.columns:
            recent_news = news_data.sort_values('published_at', ascending=False).head(10)
            
            for _, article in recent_news.iterrows():
                with st.expander(f"📄 {article['title'][:100]}..."):
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    st.write(f"**Published:** {article.get('published_at', 'Unknown')}")
                    if 'description' in article:
                        st.write(f"**Description:** {article['description']}")
                    if 'vader_label' in article:
                        sentiment = article['vader_label']
                        if sentiment == 'POSITIVE':
                            st.success(f"😊 Sentiment: {sentiment}")
                        elif sentiment == 'NEGATIVE':
                            st.error(f"😞 Sentiment: {sentiment}")
                        else:
                            st.info(f"😐 Sentiment: {sentiment}")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Sidebar
        selected_stocks, start_date, end_date, model_type = self.render_sidebar()
        
        # Main content
        self.render_main_content(selected_stocks, start_date, end_date)
        
        # Footer
        st.markdown("---")
        st.markdown("**NEWS2PROFIT** - Predicting Stock Movement with ML & Financial News Sentiment")


def main():
    """Main function to run the dashboard"""
    dashboard = News2ProfitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
