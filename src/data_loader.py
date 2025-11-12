"""
Data loading and preprocessing module for power grid forecasting.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Class for loading and preprocessing power grid data."""
    
    def __init__(self, filepath: str):
        """
        Initialize the DataLoader.
        
        Args:
            filepath: Path to the CSV file containing power data
        """
        self.filepath = filepath
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the power consumption data
        """
        try:
            self.data = pd.read_csv(self.filepath, parse_dates=['timestamp'])
            self.data.set_index('timestamp', inplace=True)
            print(f"Data loaded successfully: {len(self.data)} records")
            return self.data
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")
            print("Generating synthetic data for demonstration...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_samples: int = 8760) -> pd.DataFrame:
        """
        Generate synthetic power consumption data.
        
        Args:
            n_samples: Number of samples to generate (default: 8760 for 1 year hourly)
            
        Returns:
            DataFrame with synthetic power consumption data
        """
        # Generate timestamps (hourly for one year)
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
        
        # Base load with daily and seasonal patterns
        hours = np.arange(n_samples)
        daily_pattern = 1000 * np.sin(2 * np.pi * hours / 24) + 2000
        seasonal_pattern = 500 * np.sin(2 * np.pi * hours / (24 * 365))
        
        # Add random noise
        noise = np.random.normal(0, 200, n_samples)
        
        # Combine patterns
        power = daily_pattern + seasonal_pattern + noise
        power = np.maximum(power, 500)  # Ensure non-negative with minimum load
        
        self.data = pd.DataFrame({
            'power_consumption': power
        }, index=timestamps)
        
        self.data.index.name = 'timestamp'
        print(f"Synthetic data generated: {len(self.data)} records")
        return self.data
    
    def preprocess(self, fill_missing: bool = True) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and outliers.
        
        Args:
            fill_missing: Whether to fill missing values
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Handle missing values
        if fill_missing and self.data.isnull().any().any():
            print(f"Missing values found: {self.data.isnull().sum().sum()}")
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            print("Missing values filled using forward/backward fill")
        
        return self.data
    
    def create_features(self, lag_hours: list = [1, 24, 168]) -> pd.DataFrame:
        """
        Create lag features for the model.
        
        Args:
            lag_hours: List of lag periods in hours
            
        Returns:
            DataFrame with added lag features
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data.copy()
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Add lag features
        for lag in lag_hours:
            df[f'lag_{lag}h'] = df['power_consumption'].shift(lag)
        
        # Add rolling mean features
        df['rolling_mean_24h'] = df['power_consumption'].rolling(window=24, min_periods=1).mean()
        df['rolling_std_24h'] = df['power_consumption'].rolling(window=24, min_periods=1).std()
        
        # Drop rows with NaN values created by lag features
        df = df.dropna()
        
        print(f"Features created. Shape: {df.shape}")
        return df
    
    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        split_idx = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        print(f"Data split: Train={len(train_data)}, Test={len(test_data)}")
        return train_data, test_data
