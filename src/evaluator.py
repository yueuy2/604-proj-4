"""
Model evaluation utilities.
"""
import numpy as np
from typing import Dict
import pandas as pd


class Evaluator:
    """Class for evaluating forecasting models."""
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value as percentage
        """
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R2 score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    @classmethod
    def evaluate_model(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                       model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model using multiple metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for display
            
        Returns:
            Dictionary of metric names and values
        """
        rmse = cls.calculate_rmse(y_true, y_pred)
        mae = cls.calculate_mae(y_true, y_pred)
        mape = cls.calculate_mape(y_true, y_pred)
        r2 = cls.calculate_r2(y_true, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        
        print(f"\n{model_name} Evaluation Metrics:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        return metrics
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            results: Dictionary of model names to their metrics
            
        Returns:
            DataFrame comparing all models
        """
        df = pd.DataFrame(results).T
        df = df.sort_values('RMSE')
        
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        print(df.to_string())
        print("="*60)
        
        return df
