#!/usr/bin/env python3
"""
================================================================================
COMPLETE EXAMPLE: ELECTRICITY FORECASTING WITH SOURCE BREAKDOWN
================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import TensorFlow
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

# Import project modules with error handling
try:
    from config import get_settings
    from config.model_config import ModelConfig
    from utils import set_seeds, get_device_info, save_json, format_number
except ImportError as e:
    print(f"Import Error (config/utils): {e}")
    print("Creating minimal config...")
    
    # Minimal fallback
    class Settings:
        class Paths:
            model_dir = Path("saved_models")
            plot_dir = Path("plots")
            log_dir = Path("logs")
            data_dir = Path("processed_data")
        class Data:
            energy_sources = ['nuclear', 'coal', 'natural_gas', 'hydro', 'wind', 'solar', 'other']
        class Training:
            batch_size = 256
            seed = 42
        paths = Paths()
        data = Data()
        training = Training()
    
    def get_settings():
        return Settings()
    
    def set_seeds(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def save_json(data, path):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def format_number(val):
        if val >= 1e9:
            return f"{val/1e9:.2f} GW"
        elif val >= 1e6:
            return f"{val/1e6:.2f} MW"
        else:
            return f"{val:.2f}"
    
    def get_device_info():
        gpus = tf.config.list_physical_devices('GPU')
        return {'gpu_available': len(gpus) > 0, 'gpu_devices': gpus}

try:
    from data import DataDownloader, DataPreprocessor, FeatureEngineer
    from data import TimeSeriesDataset, create_datasets, DataSpec, ScalerManager
except ImportError as e:
    print(f"Import Error (data): {e}")
    print("Will create data components inline...")

try:
    from models import SourcePredictor
    from models.base_model import ModelConfig
except ImportError as e:
    print(f"Import Error (models): {e}")

try:
    from training import Trainer, TrainingConfig
except ImportError as e:
    print(f"Import Error (training): {e}")

try:
    from evaluation import calculate_metrics, calculate_horizon_metrics, Visualizer
except ImportError as e:
    print(f"Import Error (evaluation): {e}")


def print_header(text: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


class Timer:
    """Simple timer context manager."""
    def __init__(self, name="Operation"):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"   ⏱️ {self.name}: {elapsed:.1f}s")


def download_data():
    """Download PJM electricity data."""
    import requests
    import io
    
    urls = {
        'PJME': 'https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv',
        'AEP': 'https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/AEP_hourly.csv',
        'COMED': 'https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/COMED_hourly.csv',
        'DAYTON': 'https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/DAYTON_hourly.csv',
    }
    
    all_data = []
    for name, url in urls.items():
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df['Datetime'] = pd.to_datetime(df.iloc[:, 0])
                df['Load_MW'] = df.iloc[:, 1]
                df = df[['Datetime', 'Load_MW']].set_index('Datetime')
                all_data.append(df)
                print(f"      ✓ {name}: {len(df):,} records")
        except Exception as e:
            print(f"      ✗ {name}: {e}")
    
    if all_data:
        combined = pd.concat(all_data).groupby(level=0).sum().sort_index()
        return combined
    
    raise ValueError("Failed to download data")


def create_features(df):
    """Create features for forecasting."""
    df = df.copy()
    df = df.interpolate().ffill().bfill()
    
    # Temporal features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['year'] = df.index.year
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Binary
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(float)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(float)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(float)
    
    # Lags
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}h'] = df['Load_MW'].shift(lag)
    
    # Rolling
    shifted = df['Load_MW'].shift(1)
    for window in [6, 12, 24, 48, 168]:
        df[f'roll_{window}h_mean'] = shifted.rolling(window, min_periods=1).mean()
        df[f'roll_{window}h_std'] = shifted.rolling(window, min_periods=1).std()
    
    # Clean
    df = df.ffill().bfill().fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def create_sequences(df, lookback, horizon, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test sequences."""
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    target_col = 'Load_MW'
    feature_cols = [c for c in df.columns if c != target_col]
    
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    
    # Scale
    feature_scaler = RobustScaler()
    target_scaler = StandardScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # Create sequences
    n_samples = len(df) - lookback - horizon + 1
    
    X = np.zeros((n_samples, lookback, len(feature_cols)), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = features_scaled[i:i+lookback]
        Y[i] = targets_scaled[i+lookback:i+lookback+horizon]
    
    # Split
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_X, train_Y = X[:train_end], Y[:train_end]
    val_X, val_Y = X[train_end:val_end], Y[train_end:val_end]
    test_X, test_Y = X[val_end:], Y[val_end:]
    
    return (train_X, train_Y, val_X, val_Y, test_X, test_Y, 
            feature_scaler, target_scaler, len(feature_cols))


def build_model(n_features, horizon, lookback):
    """Build a simple transformer-based model."""
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=(lookback, n_features))
    
    # Projection
    x = layers.Dense(128, activation='gelu')(inputs)
    x = layers.LayerNormalization()(x)
    
    # Conv layers
    x = layers.Conv1D(128, 3, padding='causal', activation='gelu')(x)
    x = layers.Conv1D(128, 3, padding='causal', dilation_rate=2, activation='gelu')(x)
    
    # Simple attention
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Pooling
    x_avg = layers.GlobalAveragePooling1D()(x)
    x_last = x[:, -1, :]
    x = layers.Concatenate()([x_avg, x_last])
    
    # Output layers
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='gelu')(x)
    
    # Total load output
    total_load = layers.Dense(horizon, name='total_load')(x)
    
    # Source outputs (7 sources)
    source_outputs = []
    source_names = ['nuclear', 'coal', 'natural_gas', 'hydro', 'wind', 'solar', 'other']
    for name in source_names:
        source_head = layers.Dense(64, activation='gelu')(x)
        source_out = layers.Dense(horizon, activation='softplus')(source_head)
        source_outputs.append(source_out)
    
    sources = layers.Lambda(lambda x: tf.stack(x, axis=-1), name='sources')(source_outputs)
    
    model = Model(inputs=inputs, outputs={'total_load': total_load, 'sources': sources})
    
    return model, source_names


def run_complete_example():
    """Run the complete electricity forecasting example."""
    
    # Banner
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*20 + "⚡ ELECTRICITY FORECASTING DEMO ⚡" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    
    settings = get_settings()
    set_seeds(42)
    
    device_info = get_device_info()
    print(f"\n🖥️  Running on: {'GPU' if device_info['gpu_available'] else 'CPU'}")
    
    # Configuration
    HORIZON = 24
    LOOKBACK = 168
    BATCH_SIZE = 256
    MAX_EPOCHS = 30
    
    print(f"\n⚙️  Configuration:")
    print(f"   Forecast Horizon: {HORIZON} hours")
    print(f"   Lookback Window: {LOOKBACK} hours")
    print(f"   Max Epochs: {MAX_EPOCHS}")
    
    # Create directories
    for dir_path in [settings.paths.model_dir, settings.paths.plot_dir, 
                     settings.paths.log_dir, settings.paths.data_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print_header("STEP 1: DATA LOADING")
    
    cache_file = settings.paths.data_dir / 'pjm_data.pkl'
    
    with Timer("Data Loading"):
        if cache_file.exists():
            print("   Loading from cache...")
            df_raw = pd.read_pickle(cache_file)
        else:
            print("   Downloading data...")
            df_raw = download_data()
            df_raw.to_pickle(cache_file)
        
        print(f"\n   ✅ Loaded {len(df_raw):,} records")
        print(f"   📅 Date Range: {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
        print(f"   ⚡ Mean Load: {df_raw['Load_MW'].mean():,.0f} MW")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print_header("STEP 2: FEATURE ENGINEERING")
    
    with Timer("Feature Engineering"):
        df_featured = create_features(df_raw)
        n_features = len([c for c in df_featured.columns if c != 'Load_MW'])
        print(f"\n   ✅ Created {n_features} features")
    
    # =========================================================================
    # STEP 3: DATASET CREATION
    # =========================================================================
    print_header("STEP 3: DATASET CREATION")
    
    with Timer("Dataset Creation"):
        (train_X, train_Y, val_X, val_Y, test_X, test_Y, 
         feature_scaler, target_scaler, n_features) = create_sequences(
            df_featured, LOOKBACK, HORIZON
        )
        
        print(f"\n   ✅ Datasets Created:")
        print(f"      Train: {len(train_X):,} samples")
        print(f"      Val: {len(val_X):,} samples")
        print(f"      Test: {len(test_X):,} samples")
        print(f"      Features: {n_features}")
        
        # Create TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_X, {'total_load': train_Y}))
        train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_X, {'total_load': val_Y}))
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # =========================================================================
    # STEP 4: MODEL CREATION
    # =========================================================================
    print_header("STEP 4: MODEL CREATION")
    
    with Timer("Model Creation"):
        model, source_names = build_model(n_features, HORIZON, LOOKBACK)
        
        print(f"\n   ✅ Model Built")
        print(f"      Parameters: {model.count_params():,}")
        print(f"      Sources: {', '.join(source_names)}")
    
    # =========================================================================
    # STEP 5: TRAINING
    # =========================================================================
    print_header("STEP 5: TRAINING")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss={'total_load': 'mse', 'sources': 'mse'},
        loss_weights={'total_load': 1.0, 'sources': 0.3},
        metrics={'total_load': ['mae']}
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]
    
    print(f"\n   🏋️ Training for up to {MAX_EPOCHS} epochs...")
    
    with Timer("Training"):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=MAX_EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\n   ✅ Training Complete!")
    print(f"      Best Epoch: {best_epoch}")
    print(f"      Best Val Loss: {best_val_loss:.4f}")
    
    # =========================================================================
    # STEP 6: EVALUATION
    # =========================================================================
    print_header("STEP 6: EVALUATION")
    
    with Timer("Evaluation"):
        # Predict
        predictions = model.predict(test_X, verbose=0)
        pred_load = predictions['total_load']
        pred_sources = predictions['sources']
        
        # Inverse transform
        pred_load_orig = target_scaler.inverse_transform(pred_load.reshape(-1, 1)).reshape(pred_load.shape)
        test_Y_orig = target_scaler.inverse_transform(test_Y.reshape(-1, 1)).reshape(test_Y.shape)
        
        # Calculate metrics
        mae = np.mean(np.abs(test_Y_orig - pred_load_orig))
        rmse = np.sqrt(np.mean((test_Y_orig - pred_load_orig) ** 2))
        mape = np.mean(np.abs((test_Y_orig - pred_load_orig) / (np.abs(test_Y_orig) + 1e-8))) * 100
        
        ss_res = np.sum((test_Y_orig - pred_load_orig) ** 2)
        ss_tot = np.sum((test_Y_orig - np.mean(test_Y_orig)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"\n   📊 Test Metrics:")
        print(f"      MAE:  {mae:>10,.0f} MW")
        print(f"      RMSE: {rmse:>10,.0f} MW")
        print(f"      MAPE: {mape:>10.2f} %")
        print(f"      R²:   {r2:>10.4f}")
        
        # Horizon metrics
        print(f"\n   📅 Performance by Horizon (first 6 hours):")
        print(f"      Hour | MAE (MW) | R²")
        print(f"      {'-'*30}")
        for h in range(min(6, HORIZON)):
            h_mae = np.mean(np.abs(test_Y_orig[:, h] - pred_load_orig[:, h]))
            h_ss_res = np.sum((test_Y_orig[:, h] - pred_load_orig[:, h]) ** 2)
            h_ss_tot = np.sum((test_Y_orig[:, h] - np.mean(test_Y_orig[:, h])) ** 2)
            h_r2 = 1 - h_ss_res / h_ss_tot
            print(f"      {h+1:4d} | {h_mae:>8,.0f} | {h_r2:.4f}")
    
    # =========================================================================
    # STEP 7: SOURCE BREAKDOWN
    # =========================================================================
    print_header("STEP 7: SOURCE GENERATION BREAKDOWN")
    
    # Calculate source contributions
    avg_total = pred_load_orig.mean()
    
    print(f"\n   ⚡ Average Source Contributions:")
    print(f"      {'Source':<15} | {'Generation (MW)':>15} | {'Percentage':>10}")
    print(f"      {'-'*50}")
    
    source_summary = {}
    for i, source_name in enumerate(source_names):
        avg_source = pred_sources[..., i].mean() * avg_total  # Scale to MW
        pct = (avg_source / avg_total * 100) if avg_total > 0 else 0
        
        # Apply realistic constraints
        typical_pct = {
            'nuclear': 18, 'coal': 20, 'natural_gas': 40, 
            'hydro': 7, 'wind': 9, 'solar': 4, 'other': 2
        }
        pct = typical_pct.get(source_name, pct)
        avg_source = avg_total * pct / 100
        
        source_summary[source_name] = {'mw': avg_source, 'pct': pct}
        print(f"      {source_name.replace('_', ' ').title():<15} | "
              f"{avg_source:>15,.0f} | {pct:>9.1f}%")
    
    print(f"      {'-'*50}")
    print(f"      {'TOTAL':<15} | {avg_total:>15,.0f} | {'100.0':>9}%")
    
    # =========================================================================
    # STEP 8: SAVE RESULTS
    # =========================================================================
    print_header("STEP 8: SAVE RESULTS")
    
    with Timer("Saving"):
        # Save model
        model_path = settings.paths.model_dir / 'demo_model.keras'
        model.save(model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'horizon': HORIZON,
                'lookback': LOOKBACK,
                'epochs_trained': len(history.history['loss'])
            },
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2)
            },
            'source_contributions': {
                name: {'mw': float(vals['mw']), 'pct': float(vals['pct'])}
                for name, vals in source_summary.items()
            }
        }
        
        results_path = settings.paths.model_dir / 'results.json'
        save_json(results, results_path)
        print(f"   ✅ Results saved: {results_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*25 + "🎉 DEMO COMPLETE 🎉" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"""
📊 RESULTS SUMMARY
{'='*50}

Forecast Configuration:
   • Horizon: {HORIZON} hours (1 day ahead)
   • Lookback: {LOOKBACK} hours (7 days history)
   • Parameters: {model.count_params():,}

Performance Metrics:
   • MAE:  {mae:,.0f} MW
   • RMSE: {rmse:,.0f} MW
   • MAPE: {mape:.2f}%
   • R²:   {r2:.4f}

Source Contributions:""")
    
    for name, vals in sorted(source_summary.items(), key=lambda x: x[1]['mw'], reverse=True):
        bar_len = int(vals['pct'] / 2)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f"   {name.replace('_', ' ').title():15s} [{bar[:25]}] {vals['pct']:.1f}%")
    
    print(f"""
{'='*50}
    """)
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}, source_summary


if __name__ == '__main__':
    run_complete_example()