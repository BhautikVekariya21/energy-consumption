# ⚡ Multi-Horizon Electricity Demand Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-yellow.svg)](https://colab.research.google.com/)

A deep learning system for forecasting electricity demand across multiple time horizons using Transformer-based attention mechanisms. Trained on real PJM Interconnection data covering 145,000+ hourly records.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Results](#-performance-results)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Configuration](#-configuration)
- [Training Your Own Model](#-training-your-own-model)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a **multi-horizon electricity demand forecasting system** that predicts power consumption for:

| Horizon | Forecast Window | Use Case |
|---------|-----------------|----------|
| **Daily** | 24 hours ahead | Day-ahead market, unit commitment |
| **Weekly** | 168 hours (7 days) | Maintenance scheduling, fuel procurement |
| **Monthly** | 720 hours (30 days) | Capacity planning, contract negotiations |

The system uses lightweight Transformer models optimized for time-series forecasting, achieving **industry-competitive accuracy** while remaining deployable on consumer hardware.

---

## ✨ Key Features

- 🔮 **Multi-Horizon Forecasting** - Single codebase for daily, weekly, and monthly predictions
- 🧠 **Transformer Architecture** - Self-attention mechanisms capture long-range temporal dependencies
- ⚡ **Memory Optimized** - Runs on Google Colab T4 GPU (16GB) without crashes
- 📊 **Comprehensive Features** - Temporal encodings, lag features, rolling statistics
- 💾 **Production Ready** - Saved models, scalers, and inference pipeline included
- 📈 **Visualization Suite** - Automated plotting of results and diagnostics
- 🔄 **Reproducible** - Fixed random seeds and documented configurations

---

## 📊 Performance Results

### Model Metrics

| Horizon | MAE (MW) | RMSE (MW) | MAPE (%) | R² Score | Training Time |
|---------|----------|-----------|----------|----------|---------------|
| **Daily** | 1,944 | 2,650 | 4.20% | 0.9075 | 3.8 min |
| **Weekly** | 3,979 | 5,433 | 8.23% | 0.5593 | 3.5 min |
| **Monthly** | 3,784 | 5,083 | 7.99% | 0.6585 | 2.0 min |

### Industry Comparison

| Metric | Our Model (Daily) | Industry Benchmark | Status |
|--------|-------------------|-------------------|--------|
| MAPE | 4.20% | 3-5% | ✅ Competitive |
| R² | 0.9075 | 0.85-0.95 | ✅ Excellent |

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT (Lookback × Features)                   │
│                    Daily: 72×16, Weekly: 168×16                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Dense Projection (d_model)                   │
│                         + LayerNorm                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Conv1D Downsampling (for long sequences)            │
│                  Stride=2, GELU activation                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer Encoder Blocks                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Multi-Head Attention (4 heads, key_dim=16)               │  │
│  │  + Residual Connection + LayerNorm                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                         × N_LAYERS                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Global Average Pooling 1D                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                Dense → GELU → Dropout → Dense                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT (Horizon values)                       │
│                Daily: 24, Weekly: 168, Monthly: 720             │
└─────────────────────────────────────────────────────────────────┘
```

### Model Parameters

| Horizon | Lookback | d_model | Layers | Heads | Parameters |
|---------|----------|---------|--------|-------|------------|
| Daily | 72h | 64 | 2 | 4 | 40,472 |
| Weekly | 168h | 64 | 2 | 4 | 115,304 |
| Monthly | 336h | 64 | 2 | 4 | 273,744 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/electricity-forecasting/blob/main/notebooks/train_models.ipynb)

Simply open the notebook in Colab - all dependencies are pre-installed.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/electricity-forecasting.git
cd electricity-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker

```bash
# Build image
docker build -t electricity-forecast .

# Run container
docker run --gpus all -v $(pwd)/outputs:/app/outputs electricity-forecast
```

---

## ⚡ Quick Start

### 1. Using Pre-trained Models

```python
from src.inference import ElectricityForecaster

# Initialize forecaster
forecaster = ElectricityForecaster(model_dir='models/')

# Load your data (last 72 hours for daily forecast)
import pandas as pd
data = pd.read_csv('your_data.csv', parse_dates=['datetime'], index_col='datetime')

# Generate forecast
forecast = forecaster.predict(data, horizon='daily')

print(f"Next 24 hours forecast: {forecast}")
```

### 2. Training New Models

```python
from src.train import train_all_horizons

# Train all three horizons
results = train_all_horizons(
    data_path='data/pjm_data.csv',
    save_dir='models/',
    epochs=30
)

print(f"Daily MAE: {results['daily']['mae']:.0f} MW")
```

---

## 📖 Usage

### Loading Models

```python
import pickle
import json
from tensorflow import keras
from pathlib import Path

MODEL_DIR = Path('models/')

def load_forecaster(horizon='daily'):
    """Load trained model and scalers."""
    
    # Load Keras model
    model = keras.models.load_model(MODEL_DIR / f'{horizon}_model.keras')
    
    # Load scalers
    with open(MODEL_DIR / 'scalers' / f'{horizon}_feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(MODEL_DIR / 'scalers' / f'{horizon}_target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load feature columns
    with open(MODEL_DIR / 'data' / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    
    return {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols
    }
```

### Making Predictions

```python
import numpy as np
import pandas as pd

def prepare_features(df):
    """Create features matching training data."""
    
    df = df.copy()
    
    # Temporal features
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binary features
    df['is_weekend'] = (df['dow'] >= 5).astype(float)
    
    # Lag features
    for lag in [1, 24, 48, 168]:
        df[f'lag_{lag}'] = df['Load_MW'].shift(lag)
    
    # Rolling statistics
    shifted = df['Load_MW'].shift(1)
    for w in [24, 168]:
        df[f'roll_{w}_mean'] = shifted.rolling(w, min_periods=1).mean()
    
    return df.ffill().bfill()


def forecast(df, horizon='daily'):
    """Generate electricity demand forecast."""
    
    # Configuration
    lookback = {'daily': 72, 'weekly': 168, 'monthly': 336}[horizon]
    
    # Load model
    forecaster = load_forecaster(horizon)
    model = forecaster['model']
    feat_scaler = forecaster['feature_scaler']
    tgt_scaler = forecaster['target_scaler']
    feat_cols = forecaster['feature_cols']
    
    # Prepare features
    df = prepare_features(df)
    
    # Get last lookback hours
    X = df[feat_cols].tail(lookback).values
    
    # Scale
    X_scaled = feat_scaler.transform(X)
    X_scaled = X_scaled.reshape(1, lookback, -1)
    
    # Predict
    pred_scaled = model.predict(X_scaled, verbose=0)
    pred = tgt_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    return pred


# Example usage
df = pd.read_csv('data/electricity.csv', parse_dates=['datetime'], index_col='datetime')

# Daily forecast (24 hours)
daily_forecast = forecast(df, horizon='daily')
print(f"Hour 1:  {daily_forecast[0]:,.0f} MW")
print(f"Hour 12: {daily_forecast[11]:,.0f} MW")
print(f"Hour 24: {daily_forecast[23]:,.0f} MW")

# Weekly forecast (168 hours)
weekly_forecast = forecast(df, horizon='weekly')
print(f"\nDay 1 avg: {weekly_forecast[:24].mean():,.0f} MW")
print(f"Day 7 avg: {weekly_forecast[-24:].mean():,.0f} MW")
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_forecast(actual, predicted, horizon='daily', save_path=None):
    """Plot forecast vs actual values."""
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    hours = range(len(predicted))
    
    ax.plot(hours, actual, 'gray', alpha=0.8, linewidth=1.5, label='Actual')
    ax.plot(hours, predicted, 'blue', alpha=0.9, linewidth=1.5, label='Forecast')
    
    # Error band
    ax.fill_between(hours, actual, predicted, alpha=0.3, color='blue')
    
    # Metrics
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    ax.set_xlabel('Hours')
    ax.set_ylabel('Load (MW)')
    ax.set_title(f'⚡ {horizon.title()} Forecast | MAE: {mae:,.0f} MW | MAPE: {mape:.2f}%')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {'mae': mae, 'mape': mape}
```

---

## 📁 Project Structure

```
electricity-forecasting/
│
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore               # Git ignore rules
├── 📄 Dockerfile               # Docker configuration
│
├── 📂 notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_inference_demo.ipynb
│
├── 📂 src/                     # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py        # Data download utilities
│   │   └── preprocess.py      # Data preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py     # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py     # Model architecture
│   │   └── train.py           # Training loop
│   ├── inference.py           # Prediction utilities
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Plotting functions
│
├── 📂 configs/                 # Configuration files
│   ├── default.yaml
│   ├── daily.yaml
│   ├── weekly.yaml
│   └── monthly.yaml
│
├── 📂 tests/                   # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── 📂 data/                    # Data directory (gitignored)
│   ├── raw/
│   │   └── pjm_hourly.csv
│   └── processed/
│       └── features.pkl
│
├── 📂 models/                  # Saved models (gitignored)
│   ├── daily_model.keras
│   ├── weekly_model.keras
│   ├── monthly_model.keras
│   └── scalers/
│       ├── daily_feature_scaler.pkl
│       └── daily_target_scaler.pkl
│
└── 📂 outputs/                 # Output files (gitignored)
    ├── plots/
    ├── predictions/
    └── logs/
```

---

## 📊 Data Sources

### Primary Dataset

**PJM Interconnection Hourly Energy Consumption**

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| Time Range | 2002 - 2018 |
| Frequency | Hourly |
| Records | 145,000+ |
| Regions | PJME, AEP |

### Data Format

```csv
Datetime,Load_MW
2002-01-01 01:00:00,28521.0
2002-01-01 02:00:00,27394.0
2002-01-01 03:00:00,26570.0
...
```

### Features Generated

| Category | Features |
|----------|----------|
| **Temporal** | hour, day_of_week, month |
| **Cyclical** | hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos |
| **Binary** | is_weekend |
| **Lag** | lag_1, lag_24, lag_48, lag_168 |
| **Rolling** | roll_24_mean, roll_168_mean |

---

## ⚙️ Configuration

### Default Configuration

```python
# configs/default.py

HORIZONS = {
    'daily': {
        'hours': 24,
        'lookback': 72,
        'd_model': 64,
        'n_layers': 2,
        'n_heads': 4,
        'batch_size': 128,
    },
    'weekly': {
        'hours': 168,
        'lookback': 168,
        'd_model': 64,
        'n_layers': 2,
        'n_heads': 4,
        'batch_size': 32,
    },
    'monthly': {
        'hours': 720,
        'lookback': 336,
        'd_model': 64,
        'n_layers': 2,
        'n_heads': 4,
        'batch_size': 16,
    }
}

TRAINING = {
    'epochs': 30,
    'patience': 10,
    'learning_rate': 1e-3,
    'min_lr': 1e-6,
    'dropout': 0.1,
}

DATA = {
    'train_split': 0.70,
    'val_split': 0.85,
    'random_seed': 42,
}
```

### YAML Configuration

```yaml
# configs/daily.yaml

model:
  name: daily
  horizon_hours: 24
  lookback_hours: 72
  d_model: 64
  n_layers: 2
  n_heads: 4
  dropout: 0.1

training:
  batch_size: 128
  epochs: 30
  patience: 10
  learning_rate: 0.001
  min_learning_rate: 0.000001
  
data:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
```

---

## 🎓 Training Your Own Model

### Step 1: Prepare Your Data

```python
import pandas as pd

# Your data should have datetime index and Load_MW column
df = pd.read_csv('your_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Ensure hourly frequency
df = df.resample('H').mean().interpolate()

# Save processed data
df.to_csv('data/processed/my_data.csv')
```

### Step 2: Train Models

```python
from src.train import MultiHorizonTrainer

trainer = MultiHorizonTrainer(
    data_path='data/processed/my_data.csv',
    save_dir='models/',
    config_path='configs/default.yaml'
)

# Train all horizons
results = trainer.train_all()

# Or train specific horizon
daily_results = trainer.train('daily', epochs=50)
```

### Step 3: Evaluate

```python
from src.evaluate import evaluate_model

metrics = evaluate_model(
    model_path='models/daily_model.keras',
    test_data='data/processed/test.csv',
    horizon='daily'
)

print(f"MAE: {metrics['mae']:,.0f} MW")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"R²: {metrics['r2']:.4f}")
```

---

## 📚 API Reference

### ElectricityForecaster

```python
class ElectricityForecaster:
    """Main forecasting interface."""
    
    def __init__(self, model_dir: str, horizon: str = 'daily'):
        """
        Initialize forecaster.
        
        Args:
            model_dir: Path to saved models directory
            horizon: 'daily', 'weekly', or 'monthly'
        """
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate forecast.
        
        Args:
            data: DataFrame with Load_MW and datetime index.
                  Must have at least `lookback` hours of data.
        
        Returns:
            Array of forecasted MW values for each hour
        """
        
    def predict_with_intervals(
        self, 
        data: pd.DataFrame, 
        confidence: float = 0.95
    ) -> dict:
        """
        Generate forecast with prediction intervals.
        
        Returns:
            {
                'forecast': np.ndarray,
                'lower': np.ndarray,
                'upper': np.ndarray
            }
        """
```

### Training Functions

```python
def train_model(
    data: pd.DataFrame,
    horizon: str,
    config: dict = None,
    save_path: str = None
) -> dict:
    """
    Train a single horizon model.
    
    Args:
        data: Training data with Load_MW column
        horizon: 'daily', 'weekly', or 'monthly'
        config: Optional configuration overrides
        save_path: Path to save trained model
    
    Returns:
        Dictionary with metrics and training history
    """
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

### 1. Fork the Repository

```bash
git clone https://github.com/yourusername/electricity-forecasting.git
cd electricity-forecasting
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Submit Pull Request

- Describe your changes
- Reference any related issues
- Include test results

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/ --check

# Run tests with coverage
pytest --cov=src tests/
```

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{electricity_forecasting_2024,
  author = {Your Name},
  title = {Multi-Horizon Electricity Demand Forecasting},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/electricity-forecasting}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **PJM Interconnection** - For providing the hourly energy consumption data
- **TensorFlow Team** - For the excellent deep learning framework
- **Kaggle** - For hosting the dataset
- **Google Colab** - For free GPU resources

---

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

## 🗺️ Roadmap

- [x] Multi-horizon forecasting (daily, weekly, monthly)
- [x] Transformer-based architecture
- [x] Memory-optimized training
- [ ] Add weather data integration
- [ ] Probabilistic forecasting
- [ ] Real-time streaming predictions
- [ ] REST API deployment
- [ ] Web dashboard

---

## 📄 Requirements Files

### requirements.txt

```txt
# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Deep Learning
tensorflow>=2.12.0

# Machine Learning
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Data Handling
requests>=2.26.0
tqdm>=4.62.0

# Utilities
pyyaml>=5.4.0
python-dateutil>=2.8.0
```

### requirements-dev.txt

```txt
# Include base requirements
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-xdist>=2.5.0

# Code Quality
flake8>=4.0.0
black>=22.0.0
isort>=5.10.0
mypy>=0.950

# Pre-commit
pre-commit>=2.17.0

# Documentation
sphinx>=4.4.0
sphinx-rtd-theme>=1.0.0

# Jupyter
jupyterlab>=3.3.0
ipywidgets>=7.7.0
```

---
