"""
Forecasting models for power grid prediction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 2)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def train(self, train_data: pd.Series) -> None:
        """
        Train the ARIMA model.
        
        Args:
            train_data: Training time series data
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            print(f"Training ARIMA{self.order} model...")
            self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            print("ARIMA model trained successfully")
            
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            raise
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)


class LSTMModel:
    """LSTM neural network model for time series forecasting."""
    
    def __init__(self, sequence_length: int = 24, units: int = 50, epochs: int = 50):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units
            epochs: Number of training epochs
        """
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.model = None
        self.scaler = None
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, train_data: pd.Series) -> None:
        """
        Train the LSTM model.
        
        Args:
            train_data: Training time series data
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            from sklearn.preprocessing import MinMaxScaler
            
            print(f"Training LSTM model with {self.units} units...")
            
            # Scale the data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            self.model = keras.Sequential([
                layers.LSTM(self.units, activation='relu', input_shape=(self.sequence_length, 1)),
                layers.Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train the model
            self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
            print("LSTM model trained successfully")
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            last_sequence: Last sequence of values to start prediction
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale the last sequence
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = last_sequence_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()


class SimpleMovingAverage:
    """Simple moving average baseline model."""
    
    def __init__(self, window: int = 24):
        """
        Initialize moving average model.
        
        Args:
            window: Window size for moving average
        """
        self.window = window
        self.train_data = None
        
    def train(self, train_data: pd.Series) -> None:
        """
        Store training data.
        
        Args:
            train_data: Training time series data
        """
        self.train_data = train_data
        print(f"Simple Moving Average (window={self.window}) ready")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions using moving average.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.train_data is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use last window values
        last_values = self.train_data.values[-self.window:]
        prediction = np.mean(last_values)
        
        # Repeat prediction for all steps (naive forecast)
        return np.full(steps, prediction)
