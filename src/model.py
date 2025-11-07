"""
Machine Learning Models Module for NEWS2PROFIT

This module implements logger.info(f"Data shape after filtering: {df.shape}")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [
            'date', 'symbol', 'target', 'target_ml', 'target_binary', 'target_numeric',
            'next_day_return', 'next_day_close', 'next_day_price_change',
            'next_day_price_change_pct'
        ]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()models for predicting stock price movements:
- Logistic Regression
- XGBoost
- LSTM Neural Network
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    # Fallback for older TensorFlow versions
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
    except ImportError as e:
        print(f"Warning: Could not import Keras/TensorFlow: {e}")
        Sequential = None
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all prediction models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = []
        self.metrics = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target', 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data for {self.model_name}...")
        
        # Validate target column exists; try common alternatives if missing
        if target_col not in df.columns:
            alt_cols = ['target_ml', 'target', 'target_binary', 'target_numeric']
            found = None
            for alt in alt_cols:
                if alt in df.columns:
                    found = alt
                    break
            if found is not None:
                logger.info(f"Target column '{target_col}' not found. Using '{found}' instead.")
                target_col = found
            else:
                raise KeyError(f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}")

        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [
            'date', 'symbol', 'target', 'target_ml', 'target_binary', 'target_numeric',
            'next_day_return', 'next_day_close', 'next_day_price_change',
            'next_day_price_change_pct'
        ]]

        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical and numeric columns separately
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle categorical features - encode them
        if categorical_cols:
            logger.info(f"Encoding {len(categorical_cols)} categorical features")
            for col in categorical_cols:
                # Simple label encoding for categorical features
                unique_values = X[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                X[col] = X[col].map(value_map)
                
        # Handle numeric columns
        if numeric_cols:
            # Handle missing values in numeric columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # Remove infinite values in numeric columns
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(X[numeric_cols].median())
        
        # Now handle any remaining NaN values in categorical columns
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels if needed
        if y_train.dtype == 'object':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
        
        logger.info(f"Data prepared: Train shape {X_train_scaled.shape}, Test shape {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def evaluate_model(self, X_test, y_test, y_pred) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.metrics = metrics
        logger.info(f"{self.model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"{self.model_name.lower()}_model.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a model from the specified filepath.
        """
        try:
            self.model = joblib.load(filepath)
            self.is_fitted = True
            logger.info(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {filepath}")
            raise FileNotFoundError("Model file not found.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def load_auto_model(self):
        """
        Load the best-performing model automatically.
        """
        auto_model_path = os.path.join(os.path.dirname(__file__), '../data/models/model_comparison.csv')
        try:
            # Read model comparison file to determine the best model
            comparison_df = pd.read_csv(auto_model_path)
            best_model = comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model_name']
            best_model_path = os.path.join(os.path.dirname(__file__), f'../data/models/{best_model}.pkl')
            self.load_model(best_model_path)
            logger.info(f"Auto model loaded: {best_model}")
        except FileNotFoundError:
            logger.error(f"Model comparison file not found at {auto_model_path}")
            raise FileNotFoundError("Model comparison file not found.")
        except Exception as e:
            logger.error(f"Error loading auto model: {e}")
            raise e


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for stock movement prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("Logistic_Regression")
        self.params = {**LOGISTIC_REGRESSION_PARAMS, **kwargs}
        self.model = LogisticRegression(**self.params)
    
    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        """Train logistic regression model"""
        logger.info(f"Training {self.model_name} model...")
        
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        self.evaluate_model(X_test, y_test, y_pred)
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # For multiclass, take the mean of absolute coefficients
        if len(self.model.coef_.shape) > 1:
            importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            importance = np.abs(self.model.coef_[0])
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class XGBoostModel(BaseModel):
    """XGBoost model for stock movement prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost")
        self.params = {**XGBOOST_PARAMS, **kwargs}
        self.model = xgb.XGBClassifier(**self.params)
    
    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        """Train XGBoost model"""
        logger.info(f"Training {self.model_name} model...")
        
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        self.evaluate_model(X_test, y_test, y_pred)
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class LSTMModel(BaseModel):
    """LSTM Neural Network model for stock movement prediction"""
    
    def __init__(self, **kwargs):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras is required for LSTM model but not available")
        super().__init__("LSTM")
        self.params = {**LSTM_PARAMS, **kwargs}
        self.sequence_length = self.params.get('sequence_length', 30)
        self.model = None
    
    def create_sequences(self, data, target, sequence_length):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, df: pd.DataFrame, target_col: str = 'target'):
        """Prepare data for LSTM training"""
        logger.info(f"Preparing LSTM data with sequence length {self.sequence_length}...")
        
        # Sort by symbol and date
        df_sorted = df.sort_values(['symbol', 'date'])
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [
            'date', 'symbol', 'target', 'target_binary', 'target_numeric',
            'next_day_return', 'next_day_close', 'next_day_price_change',
            'next_day_price_change_pct'
        ]]
        
        X_data = []
        y_data = []
        
        # Create sequences for each symbol
        for symbol in df_sorted['symbol'].unique():
            symbol_data = df_sorted[df_sorted['symbol'] == symbol]
            
            if len(symbol_data) >= self.sequence_length + 1:
                features = symbol_data[feature_cols].values
                targets = symbol_data[target_col].values
                
                # Handle missing values
                features = np.nan_to_num(features)
                
                # Create sequences
                X_seq, y_seq = self.create_sequences(features, targets, self.sequence_length)
                
                if len(X_seq) > 0:
                    X_data.append(X_seq)
                    y_data.append(y_seq)
        
        if X_data:
            X = np.vstack(X_data)
            y = np.hstack(y_data)
            
            # Encode labels - handle both string and numeric targets
            if y.dtype == 'object' or y.dtype.kind in ['U', 'S']:  # Unicode or byte strings
                y = self.label_encoder.fit_transform(y)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_reshaped = X_train.reshape(-1, n_features)
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
            
            n_samples_test, _, _ = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
            
            self.feature_names = feature_cols
            logger.info(f"LSTM data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            raise ValueError("Not enough data to create sequences")
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.params['units'], return_sequences=True, input_shape=input_shape),
            Dropout(self.params['dropout']),
            LSTM(self.params['units'] // 2, return_sequences=False),
            Dropout(self.params['dropout']),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        # Handle different TensorFlow versions
        try:
            optimizer = Adam(learning_rate=0.001)
        except TypeError:
            # Fallback for older versions
            optimizer = Adam(lr=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        """Train LSTM model"""
        logger.info(f"Training {self.model_name} model...")
        
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(df, target_col)
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        
        # Build model
        self.model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_fitted = True
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1) if num_classes > 2 else (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        self.evaluate_model(X_test, y_test, y_pred)
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_proba = self.model.predict(X_scaled)
        return np.argmax(y_pred_proba, axis=1) if y_pred_proba.shape[1] > 1 else (y_pred_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        return self.model.predict(X_scaled)


class StockPredictor:
    """Main class that manages multiple models for stock prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_performances = {}
    
    def add_model(self, model_name: str, model_instance: BaseModel):
        """Add a model to the predictor"""
        self.models[model_name] = model_instance
        logger.info(f"Added {model_name} model")
    
    def train_all_models(self, df: pd.DataFrame, target_col: str = 'target'):
        """Train all models and compare performance"""
        logger.info("Training all models...")
        
        # Initialize models
        self.models['logistic_regression'] = LogisticRegressionModel()
        self.models['xgboost'] = XGBoostModel()
        
        # Only add LSTM if we have enough data and TensorFlow is available
        # Add LSTM if any symbol has enough sequential data (sequence_length + 1)
        try:
            seq_len = LSTM_PARAMS.get('sequence_length', 30)
        except Exception:
            seq_len = 30

        # Re-check TensorFlow/Keras availability at runtime (helps when TF was installed after module import)
        try:
            import tensorflow as _tf
            from tensorflow.keras.models import Sequential as _TFSequential  # noqa: F401
            tf_available = True
        except Exception:
            tf_available = False

        if tf_available:
            # Check per-symbol counts
            try:
                symbols = df['symbol'].unique() if 'symbol' in df.columns else []
                has_enough = False
                for sym in symbols:
                    sym_count = len(df[df['symbol'] == sym])
                    if sym_count >= seq_len + 1:
                        has_enough = True
                        break

                if has_enough:
                    try:
                        self.models['lstm'] = LSTMModel()
                        logger.info("LSTM model added (sufficient per-symbol data found)")
                    except ImportError as e:
                        logger.warning(f"LSTM model not available: {e}")
                else:
                    logger.info("Not adding LSTM: insufficient per-symbol sequential data (need >= %d), consider increasing historical data.", seq_len + 1)
            except Exception as e:
                logger.warning(f"Error checking data for LSTM eligibility: {e}")
        else:
            logger.info("TensorFlow/Keras not available - LSTM model will not be added. Install tensorflow to enable LSTM training.")
        
        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                metrics = model.train(df, target_col)
                self.model_performances[name] = metrics
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                self.model_performances[name] = {'accuracy': 0.0}
        
        # Determine best model
        best_accuracy = 0
        for name, metrics in self.model_performances.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                self.best_model = name
        
        logger.info(f"Best model: {self.best_model} (accuracy: {best_accuracy:.4f})")
    
    def predict(self, X, model_name: str = None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def load_and_predict(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Load trained models and make predictions on new data"""
        if model_name is None:
            model_name = self.best_model

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        if not model.is_fitted:
            model.load_model()

        # Prepare data
        X = df[model.feature_names].copy()
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Make predictions
        predictions = model.predict(X)
        df['predictions'] = predictions

        return df
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all model performances"""
        comparison_data = []
        
        for name, metrics in self.model_performances.items():
            row = {'model': name}
            row.update(metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('accuracy', ascending=False)
    
    def save_all_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            if model.is_fitted:
                model.save_model()
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        comparison_df = self.get_model_comparison()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                sns.barplot(data=comparison_df, x='model', y=metric, ax=axes[i])
                axes[i].set_title(f'Model Comparison - {metric.capitalize()}')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to test the models"""
    # Create sample data for testing
    logger.info("Testing models with sample data...")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate sample features
    X_sample = np.random.randn(n_samples, n_features)
    
    # Generate sample target (simulate stock movements)
    target_probs = np.random.rand(n_samples)
    y_sample = np.where(target_probs < 0.33, 'DOWN', 
                       np.where(target_probs < 0.66, 'NEUTRAL', 'UP'))
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    sample_df = pd.DataFrame(X_sample, columns=feature_names)
    sample_df['target'] = y_sample
    sample_df['date'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    sample_df['symbol'] = 'SAMPLE.NS'
    
    # Test individual models
    print("Testing Logistic Regression...")
    lr_model = LogisticRegressionModel()
    lr_metrics = lr_model.train(sample_df)
    
    print("\\nTesting XGBoost...")
    xgb_model = XGBoostModel()
    xgb_metrics = xgb_model.train(sample_df)
    
    # Test ensemble predictor
    print("\\nTesting Stock Predictor...")
    predictor = StockPredictor()
    predictor.train_all_models(sample_df)
    
    # Show comparison
    comparison = predictor.get_model_comparison()
    print("\\nModel Performance Comparison:")
    print(comparison)


if __name__ == "__main__":
    main()
