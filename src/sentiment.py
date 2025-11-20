"""
Sentiment Analysis Module for NEWS2PROFIT

VADER, TextBlob, and BERT-based models.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple
import logging
import os
import sys

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# BERT-based sentiment
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("Transformers library not available. BERT sentiment analysis will be disabled.")


    def ensure_bert_ready(model_name: str = None) -> bool:
        """Check if BERT dependencies and model are available.

        Returns True if the transformers library and the configured model can be loaded.
        If not available, prints actionable install hints and returns False.
        """
        if model_name is None:
            try:
                from config.config import BERT_MODEL as model_name
            except Exception:
                model_name = None

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except Exception:
            logger.error("Transformers not installed. To enable BERT run: pip install transformers")
            logger.info("If using TensorFlow/Keras backend with newer Keras 3, install the TF-Keras bridge: pip install tf-keras")
            return False

        if model_name is None:
            logger.warning("No BERT model configured in config.BERT_MODEL")
            return False

        try:
            logger.info(f"Checking availability of BERT model: {model_name} ...")
            AutoTokenizer.from_pretrained(model_name, use_fast=True)
            AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("BERT model and tokenizer loaded successfully (cached locally or downloaded)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BERT model '{model_name}': {e}")
            logger.info("If you see errors related to Keras, try: pip install tf-keras")
            logger.info("To install transformers: pip install transformers")
            return False


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and preprocesses text data for sentiment analysis"""
    
    def __init__(self):
        self.financial_slang = {
            'hodl': 'hold',
            'mooning': 'rising rapidly',
            'diamond hands': 'strong holder',
            'paper hands': 'weak seller',
            'to the moon': 'very bullish',
            'bear trap': 'false bearish signal',
            'bull trap': 'false bullish signal',
            'fud': 'fear uncertainty doubt',
            'fomo': 'fear of missing out',
            'btfd': 'buy the dip',
            'yolo': 'high risk investment'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace financial slang
        for slang, meaning in self.financial_slang.items():
            text = text.replace(slang, meaning)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#](\w+)', r'\\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text) < 10:
            return ""
        
        return text
    
    def extract_financial_entities(self, text: str) -> Dict[str, int]:
        """
        Extract financial entities and keywords from text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of financial entities and their counts
        """
        financial_terms = {
            'bullish_words': ['bull', 'bullish', 'buy', 'long', 'moon', 'pump', 'rally', 'surge', 'rise', 'gain', 'profit'],
            'bearish_words': ['bear', 'bearish', 'sell', 'short', 'dump', 'crash', 'fall', 'drop', 'loss', 'decline'],
            'neutral_words': ['hold', 'sideways', 'consolidate', 'stable', 'flat', 'range'],
            'uncertainty_words': ['volatile', 'uncertain', 'risk', 'caution', 'warning', 'concern']
        }
        
        text_lower = text.lower()
        entity_counts = {}
        
        for category, words in financial_terms.items():
            count = sum(len(re.findall(r'\\b' + word + r'\\b', text_lower)) for word in words)
            entity_counts[category] = count
        
        return entity_counts


class VaderSentimentAnalyzer:
    """VADER sentiment analysis implementation"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'vader_compound': 0.0,
                'vader_pos': 0.0,
                'vader_neu': 0.0,
                'vader_neg': 0.0,
                'vader_label': 'NEUTRAL'
            }
        
        scores = self.analyzer.polarity_scores(text)
        
        # Determine label based on compound score
        if scores['compound'] >= 0.05:
            label = 'POSITIVE'
        elif scores['compound'] <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'vader_compound': scores['compound'],
            'vader_pos': scores['pos'],
            'vader_neu': scores['neu'],
            'vader_neg': scores['neg'],
            'vader_label': label
        }


class TextBlobSentimentAnalyzer:
    """TextBlob sentiment analysis implementation"""
    
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_label': 'NEUTRAL'
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine label based on polarity
            if polarity > 0.1:
                label = 'POSITIVE'
            elif polarity < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity,
                'textblob_label': label
            }
        
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {str(e)}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_label': 'NEUTRAL'
            }


class BertSentimentAnalyzer:
    """BERT-based sentiment analysis implementation"""
    
    def __init__(self, model_name: str = None):
        self.pipeline = None
        
        if not BERT_AVAILABLE:
            logger.warning("BERT sentiment analysis not available - transformers library not installed")
            return
        
        if model_name is None:
            model_name = BERT_MODEL
        
        try:
            # Check for Keras 3 compatibility issue
            try:
                import keras
                if hasattr(keras, '__version__') and keras.__version__.startswith('3.'):
                    logger.warning("Keras 3 detected - BERT sentiment analysis disabled due to compatibility issues")
                    logger.info("To enable BERT: pip install tf-keras")
                    return
            except ImportError:
                pass  # Keras not installed, proceed
            
            # Try to initialize BERT pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            logger.info(f"Initialized BERT sentiment analyzer with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {str(e)}")
            if "tf-keras" in str(e).lower() or "keras 3" in str(e).lower():
                logger.info("💡 Fix: pip install tf-keras")
            self.pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using BERT
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not self.pipeline or not text or not isinstance(text, str):
            return {
                'bert_positive': 0.0,
                'bert_negative': 0.0,
                'bert_neutral': 0.0,
                'bert_label': 'NEUTRAL',
                'bert_confidence': 0.0
            }
        
        try:
            # Truncate text if too long
            if len(text) > 500:
                text = text[:500]
            
            results = self.pipeline(text)[0]
            
            # Initialize scores
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            # Extract scores based on model output
            for result in results:
                label = result['label'].lower()
                if 'pos' in label or label == '5' or label == '4':
                    scores['positive'] = result['score']
                elif 'neg' in label or label == '1' or label == '2':
                    scores['negative'] = result['score']
                else:
                    scores['neutral'] = result['score']
            
            # Determine best label and confidence
            best_label = max(scores, key=scores.get)
            confidence = scores[best_label]
            
            return {
                'bert_positive': scores['positive'],
                'bert_negative': scores['negative'],
                'bert_neutral': scores['neutral'],
                'bert_label': best_label.upper(),
                'bert_confidence': confidence
            }
        
        except Exception as e:
            logger.warning(f"BERT analysis failed: {str(e)}")
            return {
                'bert_positive': 0.0,
                'bert_negative': 0.0,
                'bert_neutral': 0.0,
                'bert_label': 'NEUTRAL',
                'bert_confidence': 0.0
            }


class SentimentAnalyzer:
    """Main sentiment analysis class that combines multiple approaches"""
    
    def __init__(self, models: List[str] = None):
        if models is None:
            models = SENTIMENT_MODELS
        
        self.text_cleaner = TextCleaner()
        self.analyzers = {}
        
        # Initialize sentiment analyzers
        if 'vader' in models:
            self.analyzers['vader'] = VaderSentimentAnalyzer()
        
        if 'textblob' in models:
            self.analyzers['textblob'] = TextBlobSentimentAnalyzer()
        
        if 'bert' in models and BERT_AVAILABLE:
            bert_analyzer = BertSentimentAnalyzer()
            # Only add BERT if it was successfully initialized
            if bert_analyzer.pipeline is not None:
                self.analyzers['bert'] = bert_analyzer
            else:
                logger.warning("BERT analyzer failed to initialize - excluding from available analyzers")
        
        # Log which analyzers are actually available
        available_analyzers = list(self.analyzers.keys())
        logger.info(f"Initialized sentiment analyzers: {available_analyzers}")
        
        if not available_analyzers:
            logger.error("No sentiment analyzers available!")
        elif 'bert' not in available_analyzers and 'bert' in models:
            logger.info("💡 BERT unavailable - using VADER and TextBlob only")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using all available methods
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with all sentiment scores
        """
        # Clean text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        if not cleaned_text:
            return self._get_empty_sentiment_dict()
        
        # Extract financial entities
        financial_entities = self.text_cleaner.extract_financial_entities(cleaned_text)
        
        # Analyze with each method
        sentiment_scores = {}
        
        for name, analyzer in self.analyzers.items():
            scores = analyzer.analyze_sentiment(cleaned_text)
            sentiment_scores.update(scores)
        
        # Add financial entity counts
        sentiment_scores.update({f'sentiment_{k}': v for k, v in financial_entities.items()})
        
        # Create composite score
        sentiment_scores.update(self._calculate_composite_scores(sentiment_scores))
        
        return sentiment_scores
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame
        
        Args:
            df: DataFrame with text data
            text_column: Column name containing text to analyze
        
        Returns:
            DataFrame with sentiment scores added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        df_sentiment = df.copy()
        
        # Combine title and description if both exist
        if 'title' in df_sentiment.columns and 'description' in df_sentiment.columns:
            combined_text = (df_sentiment['title'].fillna('') + ' ' + 
                           df_sentiment['description'].fillna('')).str.strip()
        else:
            combined_text = df_sentiment[text_column].fillna('')
        
        # Analyze sentiment for each text
        sentiment_results = []
        
        for i, text in enumerate(combined_text):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(combined_text)} texts")
            
            sentiment_scores = self.analyze_text(text)
            sentiment_results.append(sentiment_scores)
        
        # Convert to DataFrame and merge
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([df_sentiment.reset_index(drop=True), sentiment_df], axis=1)
        
        logger.info(f"Completed sentiment analysis. Added {len(sentiment_df.columns)} sentiment features.")
        return result_df
    
    def add_text_features(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """
        Add text-based features like length, word count, and financial keyword count.
        
        Args:
            df: DataFrame containing text data
            text_column: Column name containing text to analyze
        
        Returns:
            DataFrame with added text features
        """
        df = df.copy()
        
        # Add text length and word count
        df[f'{text_column}_length'] = df[text_column].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df[f'{text_column}_word_count'] = df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Add financial keyword count (example logic)
        financial_keywords = ['profit', 'loss', 'revenue', 'growth', 'decline']
        df['total_financial_keywords'] = df[text_column].apply(
            lambda x: sum(1 for word in x.split() if word.lower() in financial_keywords) if isinstance(x, str) else 0
        )
        
        return df
    
    def _get_empty_sentiment_dict(self) -> Dict[str, float]:
        """Return empty sentiment scores"""
        empty_scores = {}
        
        # VADER scores
        empty_scores.update({
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neu': 0.0,
            'vader_neg': 0.0,
            'vader_label': 'NEUTRAL'
        })
        
        # TextBlob scores
        empty_scores.update({
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'textblob_label': 'NEUTRAL'
        })
        
        # BERT scores
        empty_scores.update({
            'bert_positive': 0.0,
            'bert_negative': 0.0,
            'bert_neutral': 0.0,
            'bert_label': 'NEUTRAL',
            'bert_confidence': 0.0
        })
        
        # Financial entities
        empty_scores.update({
            'sentiment_bullish_words': 0,
            'sentiment_bearish_words': 0,
            'sentiment_neutral_words': 0,
            'sentiment_uncertainty_words': 0
        })
        
        # Composite scores
        empty_scores.update({
            'sentiment_composite_score': 0.0,
            'sentiment_composite_label': 'NEUTRAL',
            'sentiment_confidence': 0.0
        })
        
        return empty_scores
    
    def _calculate_composite_scores(self, sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate composite sentiment scores from individual analyzer results
        
        Args:
            sentiment_scores: Dictionary with individual sentiment scores
        
        Returns:
            Dictionary with composite scores
        """
        # Extract numeric scores
        vader_score = sentiment_scores.get('vader_compound', 0.0)
        textblob_score = sentiment_scores.get('textblob_polarity', 0.0)
        
        # BERT score (convert to -1 to 1 scale)
        bert_pos = sentiment_scores.get('bert_positive', 0.0)
        bert_neg = sentiment_scores.get('bert_negative', 0.0)
        bert_score = bert_pos - bert_neg
        
        # Financial sentiment based on keyword counts
        bullish_count = sentiment_scores.get('sentiment_bullish_words', 0)
        bearish_count = sentiment_scores.get('sentiment_bearish_words', 0)
        
        if bullish_count + bearish_count > 0:
            financial_score = (bullish_count - bearish_count) / (bullish_count + bearish_count)
        else:
            financial_score = 0.0
        
        # Weighted composite score
        weights = {'vader': 0.3, 'textblob': 0.2, 'bert': 0.3, 'financial': 0.2}
        
        scores = [vader_score, textblob_score, bert_score, financial_score]
        valid_scores = [score for score in scores if score != 0.0]
        
        if valid_scores:
            composite_score = np.average(scores, weights=list(weights.values()))
        else:
            composite_score = 0.0
        
        # Determine composite label
        if composite_score > 0.1:
            composite_label = 'POSITIVE'
        elif composite_score < -0.1:
            composite_label = 'NEGATIVE'
        else:
            composite_label = 'NEUTRAL'
        
        # Calculate confidence as standard deviation of individual scores
        if len(valid_scores) > 1:
            confidence = 1.0 - np.std(valid_scores)
        else:
            confidence = 0.5
        
        return {
            'sentiment_composite_score': composite_score,
            'sentiment_composite_label': composite_label,
            'sentiment_confidence': max(0.0, min(1.0, confidence))
        }
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics of sentiment analysis results
        
        Args:
            df: DataFrame with sentiment scores
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Label distributions
        for analyzer in ['vader', 'textblob', 'bert', 'composite']:
            label_col = f'sentiment_{analyzer}_label' if analyzer == 'composite' else f'{analyzer}_label'
            if label_col in df.columns:
                summary[f'{analyzer}_label_distribution'] = df[label_col].value_counts().to_dict()
        
        # Score statistics
        score_columns = [col for col in df.columns if 'sentiment_' in col and col.endswith('_score')]
        for col in score_columns:
            summary[f'{col}_stats'] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return summary


def main():
    """Main function to test sentiment analysis"""
    # Check BERT availability and give user actionable hints
    bert_ready = ensure_bert_ready()

    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "The stock market is showing strong bullish momentum with great earnings.",
        "Market crash expected due to economic uncertainty and bearish sentiment.",
        "The stock is holding steady in a sideways trading pattern.",
        "HODL! This stock is going to the moon! ",
        "FUD is spreading, but diamond hands will prevail in this bull market."
    ]
    
    print("Testing individual text analysis:")
    for i, text in enumerate(test_texts, 1):
        scores = analyzer.analyze_text(text)
        print(f"\\nText {i}: {text}")
        print(f"VADER: {scores.get('vader_label')} ({scores.get('vader_compound', 0):.3f})")
        print(f"TextBlob: {scores.get('textblob_label')} ({scores.get('textblob_polarity', 0):.3f})")
        print(f"Composite: {scores.get('sentiment_composite_label')} ({scores.get('sentiment_composite_score', 0):.3f})")
    
    # Test DataFrame analysis
    test_df = pd.DataFrame({
        'title': test_texts,
        'description': ['Description for text ' + str(i) for i in range(1, 6)]
    })
    
    print("\\n\\nTesting DataFrame analysis:")
    result_df = analyzer.analyze_dataframe(test_df)
    
    print(f"\\nDataFrame shape after sentiment analysis: {result_df.shape}")
    print("\\nSentiment columns added:")
    sentiment_cols = [col for col in result_df.columns if 'sentiment' in col or 'vader' in col or 'textblob' in col or 'bert' in col]
    print(sentiment_cols)
    
    # Get summary
    summary = analyzer.get_sentiment_summary(result_df)
    print("\\nSentiment Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()