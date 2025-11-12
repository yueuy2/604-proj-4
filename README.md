# Power Grid Forecasting - CS604 Project 4

A machine learning project for forecasting power grid demand and generation using time series analysis.

## Project Overview

This project implements multiple forecasting models to predict power consumption in electrical grids. The implementation includes:

- Data preprocessing and feature engineering
- Multiple forecasting models (ARIMA, LSTM)
- Model evaluation and comparison
- Visualization of predictions vs actual values
- Configurable parameters for experimentation

## Features

- **Data Processing**: Load, clean, and prepare time series data
- **Multiple Models**: Support for ARIMA and LSTM neural networks
- **Evaluation Metrics**: RMSE, MAE, MAPE for model comparison
- **Visualization**: Interactive plots for data exploration and results
- **Configuration**: Easy-to-modify parameters via config file

## Installation

```bash
# Clone the repository
git clone https://github.com/yueuy2/604-proj-4.git
cd 604-proj-4

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
604-proj-4/
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── models.py            # Forecasting model implementations
│   ├── evaluator.py         # Model evaluation metrics
│   └── visualizer.py        # Visualization utilities
├── data/
│   └── sample_data.csv      # Sample power consumption data
├── notebooks/
│   └── demo.ipynb           # Demonstration notebook
├── config.yaml              # Configuration parameters
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### Quick Start

Run the complete forecasting pipeline:

```bash
python main.py
```

### Using Jupyter Notebook

Explore the project interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

### Configuration

Modify `config.yaml` to adjust:
- Model parameters
- Training/test split ratio
- Forecast horizon
- Data source

## Models

### ARIMA (AutoRegressive Integrated Moving Average)
- Traditional statistical model for time series
- Good for linear trends and seasonality
- Fast training and inference

### LSTM (Long Short-Term Memory)
- Deep learning approach
- Captures complex non-linear patterns
- Requires more data and computational resources

## Evaluation Metrics

- **RMSE** (Root Mean Square Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average magnitude of errors
- **MAPE** (Mean Absolute Percentage Error): Percentage-based accuracy

## Sample Results

The models are evaluated on historical power consumption data. Example output:

```
Model: ARIMA
RMSE: 125.4
MAE: 98.7
MAPE: 5.2%

Model: LSTM
RMSE: 112.3
MAE: 87.5
MAPE: 4.8%
```

## License

MIT License - See LICENSE file for details

## Author

CS604 Fall 2025 - Project 4