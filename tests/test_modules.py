"""
Test module for NEWS2PROFIT

This module contains unit tests for all major components of the project.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection import StockDataCollector, NewsDataCollector
from src.preprocessing import StockDataPreprocessor, NewsDataPreprocessor, DataIntegrator
from src.sentiment import SentimentAnalyzer, VaderSentimentAnalyzer, TextBlobSentimentAnalyzer
from src.model import LogisticRegressionModel, XGBoostModel, StockPredictor


class TestDataCollection(unittest.TestCase):
    """Test data collection modules"""
    
    def setUp(self):
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsDataCollector()
    
    def test_stock_data_collector_initialization(self):
        """Test stock data collector initialization"""
        self.assertIsInstance(self.stock_collector.data, dict)
    
    def test_fetch_stock_data_structure(self):
        """Test stock data fetching returns correct structure"""
        # This would require actual API calls, so we'll mock it
        sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': np.random.uniform(100, 200, 10),
            'high': np.random.uniform(150, 250, 10),
            'low': np.random.uniform(90, 180, 10),
            'close': np.random.uniform(100, 200, 10),
            'volume': np.random.uniform(1000000, 5000000, 10),
            'symbol': 'TEST.NS'
        })
        
        # Test data structure
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        for col in required_columns:
            self.assertIn(col, sample_data.columns)
    
    def test_news_collector_initialization(self):
        """Test news data collector initialization"""
        self.assertIsNotNone(self.news_collector)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing modules"""
    
    def setUp(self):
        self.stock_preprocessor = StockDataPreprocessor()
        self.news_preprocessor = NewsDataPreprocessor()
        
        # Create sample stock data
        self.sample_stock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(90, 180, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000000, 5000000, 100),
            'symbol': 'TEST.NS'
        })
        
        # Create sample news data
        self.sample_news_data = pd.DataFrame({
            'title': ['Stock market rises today', 'Economic growth expected', 'Market volatility increases'],
            'description': ['Positive market movement', 'Economy showing growth', 'Increased uncertainty'],
            'content': ['Full article about market rise', 'Full article about growth', 'Full article about volatility'],
            'published_at': pd.date_range('2023-01-01', periods=3),
            'source': ['newsapi', 'twitter', 'reddit'],
            'date': pd.date_range('2023-01-01', periods=3).date
        })
    
    def test_clean_stock_data(self):
        """Test stock data cleaning"""
        cleaned_data = self.stock_preprocessor.clean_stock_data(self.sample_stock_data)
        
        # Check that data is cleaned
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)
        
        # Check required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, cleaned_data.columns)
    
    def test_create_technical_indicators(self):
        """Test technical indicators creation"""
        cleaned_data = self.stock_preprocessor.clean_stock_data(self.sample_stock_data)
        indicators_data = self.stock_preprocessor.create_technical_indicators(cleaned_data)
        
        # Check that technical indicators are added
        expected_indicators = ['daily_return', 'sma_5', 'sma_10', 'rsi', 'macd']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators_data.columns)
    
    def test_create_target_variable(self):
        """Test target variable creation"""
        cleaned_data = self.stock_preprocessor.clean_stock_data(self.sample_stock_data)
        indicators_data = self.stock_preprocessor.create_technical_indicators(cleaned_data)
        target_data = self.stock_preprocessor.create_target_variable(indicators_data)
        
        # Check target columns exist
        self.assertIn('target', target_data.columns)
        self.assertIn('target_binary', target_data.columns)
        self.assertIn('target_numeric', target_data.columns)
        
        # Check target values are valid
        valid_targets = {'UP', 'DOWN', 'NEUTRAL'}
        unique_targets = set(target_data['target'].dropna().unique())
        self.assertTrue(unique_targets.issubset(valid_targets))
    
    def test_clean_news_data(self):
        """Test news data cleaning"""
        cleaned_news = self.news_preprocessor.clean_news_data(self.sample_news_data)
        
        # Check that data is cleaned
        self.assertIsInstance(cleaned_news, pd.DataFrame)
        self.assertGreater(len(cleaned_news), 0)


class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis modules"""
    
    def setUp(self):
        self.vader_analyzer = VaderSentimentAnalyzer()
        self.textblob_analyzer = TextBlobSentimentAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(['vader', 'textblob'])
        
        self.test_texts = [
            "The stock market is performing very well today with great gains!",
            "Market crash expected due to economic uncertainty and bad news.",
            "The stock is trading sideways in a normal pattern.",
            "",  # Empty text
            "HODL! This stock is going to the moon! 🚀"
        ]
    
    def test_vader_sentiment_analysis(self):
        """Test VADER sentiment analysis"""
        for text in self.test_texts:
            result = self.vader_analyzer.analyze_sentiment(text)
            
            # Check required keys exist
            required_keys = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 'vader_label']
            for key in required_keys:
                self.assertIn(key, result)
            
            # Check score ranges
            self.assertGreaterEqual(result['vader_compound'], -1)
            self.assertLessEqual(result['vader_compound'], 1)
            self.assertIn(result['vader_label'], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    
    def test_textblob_sentiment_analysis(self):
        """Test TextBlob sentiment analysis"""
        for text in self.test_texts:
            result = self.textblob_analyzer.analyze_sentiment(text)
            
            # Check required keys exist
            required_keys = ['textblob_polarity', 'textblob_subjectivity', 'textblob_label']
            for key in required_keys:
                self.assertIn(key, result)
            
            # Check score ranges
            self.assertGreaterEqual(result['textblob_polarity'], -1)
            self.assertLessEqual(result['textblob_polarity'], 1)
            self.assertIn(result['textblob_label'], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    
    def test_sentiment_analyzer_dataframe(self):
        """Test sentiment analysis on DataFrame"""
        test_df = pd.DataFrame({
            'title': self.test_texts[:3],
            'description': ['Description ' + str(i) for i in range(3)]
        })
        
        result_df = self.sentiment_analyzer.analyze_dataframe(test_df)
        
        # Check that sentiment columns are added
        sentiment_columns = [col for col in result_df.columns if 'sentiment' in col or 'vader' in col or 'textblob' in col]
        self.assertGreater(len(sentiment_columns), 0)


class TestModels(unittest.TestCase):
    """Test machine learning models"""
    
    def setUp(self):
        # Create sample ML data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Generate sample features
        X_sample = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate sample target
        target_probs = np.random.rand(n_samples)
        y_sample = np.where(target_probs < 0.33, 'DOWN', 
                           np.where(target_probs < 0.66, 'NEUTRAL', 'UP'))
        
        self.sample_ml_data = X_sample.copy()
        self.sample_ml_data['target'] = y_sample
        self.sample_ml_data['date'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        self.sample_ml_data['symbol'] = 'TEST.NS'
    
    def test_logistic_regression_model(self):
        """Test logistic regression model"""
        model = LogisticRegressionModel()
        
        # Test training
        metrics = model.train(self.sample_ml_data)
        
        # Check that model is fitted
        self.assertTrue(model.is_fitted)
        
        # Check metrics exist
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertGreater(metrics[metric], 0)
    
    def test_stock_predictor(self):
        """Test stock predictor ensemble"""
        predictor = StockPredictor()
        
        # Test training
        predictor.train_all_models(self.sample_ml_data)
        
        # Check that models are trained
        self.assertGreater(len(predictor.models), 0)
        self.assertIsNotNone(predictor.best_model)
        
        # Test prediction
        feature_cols = [col for col in self.sample_ml_data.columns if col.startswith('feature_')]
        test_features = self.sample_ml_data[feature_cols].iloc[:5]
        
        predictions = predictor.predict(test_features)
        self.assertEqual(len(predictions), 5)


class TestDataIntegration(unittest.TestCase):
    """Test data integration"""
    
    def setUp(self):
        # Create sample stock data
        self.stock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10).date,
            'close': np.random.uniform(100, 200, 10),
            'volume': np.random.uniform(1000000, 5000000, 10),
            'symbol': 'TEST.NS',
            'target': np.random.choice(['UP', 'DOWN', 'NEUTRAL'], 10)
        })
        
        # Create sample news data
        self.news_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10).date,
            'sentiment_vader_compound': np.random.uniform(-1, 1, 10),
            'sentiment_bullish_words': np.random.randint(0, 5, 10),
            'sentiment_bearish_words': np.random.randint(0, 5, 10)
        })
        
        self.integrator = DataIntegrator()
    
    def test_align_data_by_date(self):
        """Test data alignment by date"""
        aligned_data = self.integrator.align_data_by_date(self.stock_data, self.news_data)
        
        # Check that data is aligned
        self.assertIsInstance(aligned_data, pd.DataFrame)
        self.assertGreater(len(aligned_data), 0)
        
        # Check that both stock and news columns exist
        self.assertIn('close', aligned_data.columns)
        self.assertTrue(any(col.startswith('news_') for col in aligned_data.columns))


def run_all_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestDataCollection,
        TestPreprocessing,
        TestSentimentAnalysis,
        TestModels,
        TestDataIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
