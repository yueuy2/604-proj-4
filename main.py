"""
Main execution script for Power Grid Forecasting.
"""
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_loader import DataLoader
from models import ARIMAModel, LSTMModel, SimpleMovingAverage
from evaluator import Evaluator
from visualizer import Visualizer


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        return {
            'data': {'filepath': 'data/sample_data.csv', 'test_size': 0.2},
            'forecast': {'horizon': 168},
            'models': {
                'arima': {'enabled': True, 'order': [5, 1, 2]},
                'lstm': {'enabled': True, 'sequence_length': 24, 'units': 50, 'epochs': 50},
                'moving_average': {'enabled': True, 'window': 24}
            },
            'visualization': {'generate_plots': True, 'plot_seasonal_patterns': True, 'plot_residuals': True}
        }


def main():
    """Main execution function."""
    print("="*70)
    print("Power Grid Forecasting - CS604 Project 4".center(70))
    print("="*70)
    print()
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = DataLoader(config['data']['filepath'])
    visualizer = Visualizer()
    evaluator = Evaluator()
    
    # Load and preprocess data
    print("Step 1: Loading data...")
    print("-" * 70)
    data = data_loader.load_data()
    data = data_loader.preprocess()
    print()
    
    # Split data
    print("Step 2: Splitting data...")
    print("-" * 70)
    train_data, test_data = data_loader.split_data(test_size=config['data']['test_size'])
    print()
    
    # Visualize seasonal patterns if enabled
    if config['visualization'].get('plot_seasonal_patterns', True):
        print("Step 3: Analyzing seasonal patterns...")
        print("-" * 70)
        visualizer.plot_seasonal_patterns(train_data)
        print()
    
    # Visualize time series
    if config['visualization'].get('generate_plots', True):
        print("Step 4: Visualizing time series...")
        print("-" * 70)
        visualizer.plot_time_series(train_data)
        print()
    
    # Train models and make predictions
    print("Step 5: Training models and making predictions...")
    print("-" * 70)
    
    results = {}
    predictions = {}
    forecast_horizon = min(config['forecast']['horizon'], len(test_data))
    
    # ARIMA Model
    if config['models']['arima']['enabled']:
        try:
            print("\n[1] ARIMA Model")
            arima = ARIMAModel(order=tuple(config['models']['arima']['order']))
            arima.train(train_data['power_consumption'])
            arima_pred = arima.predict(forecast_horizon)
            predictions['ARIMA'] = arima_pred
            
            # Evaluate
            y_true = test_data['power_consumption'].values[:forecast_horizon]
            results['ARIMA'] = evaluator.evaluate_model(y_true, arima_pred, "ARIMA")
            
            # Plot residuals if enabled
            if config['visualization'].get('plot_residuals', True):
                visualizer.plot_residuals(y_true, arima_pred, "ARIMA")
        except Exception as e:
            print(f"ARIMA model failed: {e}")
    
    # LSTM Model
    if config['models']['lstm']['enabled']:
        try:
            print("\n[2] LSTM Model")
            lstm = LSTMModel(
                sequence_length=config['models']['lstm']['sequence_length'],
                units=config['models']['lstm']['units'],
                epochs=config['models']['lstm']['epochs']
            )
            lstm.train(train_data['power_consumption'])
            
            # Get last sequence for prediction
            last_sequence = train_data['power_consumption'].values[-lstm.sequence_length:]
            lstm_pred = lstm.predict(last_sequence, forecast_horizon)
            predictions['LSTM'] = lstm_pred
            
            # Evaluate
            y_true = test_data['power_consumption'].values[:forecast_horizon]
            results['LSTM'] = evaluator.evaluate_model(y_true, lstm_pred, "LSTM")
            
            # Plot residuals if enabled
            if config['visualization'].get('plot_residuals', True):
                visualizer.plot_residuals(y_true, lstm_pred, "LSTM")
        except Exception as e:
            print(f"LSTM model failed: {e}")
    
    # Simple Moving Average (Baseline)
    if config['models']['moving_average']['enabled']:
        try:
            print("\n[3] Moving Average (Baseline)")
            ma = SimpleMovingAverage(window=config['models']['moving_average']['window'])
            ma.train(train_data['power_consumption'])
            ma_pred = ma.predict(forecast_horizon)
            predictions['Moving Average'] = ma_pred
            
            # Evaluate
            y_true = test_data['power_consumption'].values[:forecast_horizon]
            results['Moving Average'] = evaluator.evaluate_model(y_true, ma_pred, "Moving Average")
        except Exception as e:
            print(f"Moving Average model failed: {e}")
    
    print()
    
    # Compare models
    if len(results) > 1:
        print("Step 6: Comparing models...")
        print("-" * 70)
        import pandas as pd
        comparison_df = evaluator.compare_models(results)
        print()
        
        if config['visualization'].get('generate_plots', True):
            visualizer.plot_metrics_comparison(comparison_df)
    
    # Plot predictions
    if config['visualization'].get('generate_plots', True) and predictions:
        print("Step 7: Visualizing predictions...")
        print("-" * 70)
        test_subset = test_data['power_consumption'].iloc[:forecast_horizon]
        visualizer.plot_predictions(test_subset, predictions)
        print()
    
    print("="*70)
    print("Forecasting complete!".center(70))
    print("="*70)


if __name__ == "__main__":
    main()
