"""
Visualization utilities for power grid forecasting.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import seaborn as sns


class Visualizer:
    """Class for creating visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    @staticmethod
    def plot_time_series(data: pd.DataFrame, column: str = 'power_consumption',
                        title: str = 'Power Consumption Over Time',
                        figsize: tuple = (15, 6)) -> None:
        """
        Plot time series data.
        
        Args:
            data: DataFrame with time series data
            column: Column name to plot
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(data.index, data[column], linewidth=1, alpha=0.8)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power Consumption (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('power_consumption_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'power_consumption_timeseries.png'")
    
    @staticmethod
    def plot_predictions(actual: pd.Series, predictions: Dict[str, np.ndarray],
                        title: str = 'Model Predictions vs Actual',
                        figsize: tuple = (15, 8)) -> None:
        """
        Plot predictions from multiple models against actual values.
        
        Args:
            actual: Actual values
            predictions: Dictionary of model names to prediction arrays
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot actual values
        plt.plot(actual.index, actual.values, label='Actual', 
                linewidth=2, alpha=0.8, color='black')
        
        # Plot predictions from each model
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(actual.index[:len(pred)], pred, 
                    label=model_name, linewidth=2, 
                    alpha=0.7, linestyle='--',
                    color=colors[i % len(colors)])
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power Consumption (MW)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'predictions_comparison.png'")
    
    @staticmethod
    def plot_residuals(actual: np.ndarray, predicted: np.ndarray,
                      model_name: str = 'Model',
                      figsize: tuple = (15, 5)) -> None:
        """
        Plot residuals analysis.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
            figsize: Figure size
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals over time
        axes[0].plot(residuals, linewidth=1)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{model_name} - Residuals Over Time')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Residual')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_title(f'{model_name} - Residual Distribution')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title(f'{model_name} - Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'residuals_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved as 'residuals_{model_name.lower().replace(' ', '_')}.png'")
    
    @staticmethod
    def plot_metrics_comparison(results: pd.DataFrame,
                               title: str = 'Model Performance Comparison',
                               figsize: tuple = (12, 6)) -> None:
        """
        Plot comparison of model metrics.
        
        Args:
            results: DataFrame with model metrics
            title: Plot title
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot RMSE and MAE
        results[['RMSE', 'MAE']].plot(kind='bar', ax=axes[0], rot=45)
        axes[0].set_title('Error Metrics')
        axes[0].set_ylabel('Error Value')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot R2 score
        results['R2'].plot(kind='bar', ax=axes[1], rot=45, color='green')
        axes[1].set_title('R² Score (Higher is Better)')
        axes[1].set_ylabel('R² Score')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'metrics_comparison.png'")
    
    @staticmethod
    def plot_seasonal_patterns(data: pd.DataFrame, column: str = 'power_consumption',
                             figsize: tuple = (15, 8)) -> None:
        """
        Plot seasonal patterns in the data.
        
        Args:
            data: DataFrame with time series data
            column: Column name to analyze
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Hourly pattern
        hourly_avg = data.groupby(data.index.hour)[column].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[0, 0].set_title('Average Power by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Power (MW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Day of week pattern
        daily_avg = data.groupby(data.index.dayofweek)[column].mean()
        axes[0, 1].bar(daily_avg.index, daily_avg.values)
        axes[0, 1].set_title('Average Power by Day of Week')
        axes[0, 1].set_xlabel('Day (0=Monday)')
        axes[0, 1].set_ylabel('Power (MW)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly pattern
        monthly_avg = data.groupby(data.index.month)[column].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', color='green')
        axes[1, 0].set_title('Average Power by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Power (MW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot by month
        data_copy = data.copy()
        data_copy['month'] = data_copy.index.month
        data_copy.boxplot(column=column, by='month', ax=axes[1, 1])
        axes[1, 1].set_title('Power Distribution by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Power (MW)')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=0)
        
        plt.suptitle('Seasonal Patterns Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'seasonal_patterns.png'")
