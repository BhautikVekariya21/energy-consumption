#!/usr/bin/env python3
"""
================================================================================
INDUSTRY-GRADE ELECTRICITY FORECASTING SYSTEM
================================================================================

Main entry point for the electricity load and source-wise generation
forecasting system.

Features:
- Multi-horizon forecasting (1 day to 1 year)
- Source-wise generation prediction (Solar, Wind, Nuclear, Coal, Gas, Hydro)
- TensorFlow/Keras implementation with GPU optimization
- Comprehensive evaluation and visualization

Usage:
    python main.py --mode train --horizon 24
    python main.py --mode predict --model saved_models/best_model
    python main.py --mode evaluate --model saved_models/best_model
    python main.py --mode full_pipeline

Author: Electricity Forecast System
Version: 1.0.0
================================================================================
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf

# Import project modules
from config import get_settings, Settings, get_model_config
from config.model_config import ModelConfig
from utils import (
    get_logger, setup_logging, set_seeds, 
    get_device_info, Timer, save_json, format_time
)
from data import (
    DataDownloader, download_all_data,
    DataPreprocessor, preprocess_data,
    FeatureEngineer, create_features,
    TimeSeriesDataset, create_datasets, DataPipeline, DataSpec
)
from models import (
    TransformerModel, TemporalFusionTransformer,
    SourcePredictor, create_model, load_model,
    ModelOutput
)
from training import (
    Trainer, TrainingConfig, TrainingResult,
    train_model, get_default_callbacks
)
from evaluation import (
    calculate_metrics, calculate_horizon_metrics,
    MetricsCalculator, ForecastMetrics,
    Visualizer, create_forecast_report
)

# Initialize logger
logger = get_logger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ⚡ ELECTRICITY LOAD FORECASTING SYSTEM ⚡                                  ║
║                                                                              ║
║   Industry-Grade Multi-Horizon Forecasting with Source-wise Generation      ║
║                                                                              ║
║   Features:                                                                  ║
║   • Transformer & Temporal Fusion Transformer Models                        ║
║   • Source-wise Generation: Nuclear, Coal, Gas, Hydro, Wind, Solar          ║
║   • Horizons: 1 Day to 1 Year                                               ║
║   • Uncertainty Estimation                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_environment():
    """Setup environment and print system info."""
    settings = get_settings()
    
    # Set seeds for reproducibility
    set_seeds(settings.training.seed)
    
    # Get device info
    device_info = get_device_info()
    
    print("\n" + "="*70)
    print("🖥️  SYSTEM CONFIGURATION")
    print("="*70)
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   TensorFlow Version: {tf.__version__}")
    print(f"   NumPy Version: {np.__version__}")
    print(f"   ")
    print(f"   CPU Cores: {device_info['cpu_count']}")
    print(f"   GPU Available: {device_info['gpu_available']}")
    
    if device_info['gpu_available']:
        print(f"   GPU Count: {device_info['gpu_count']}")
        for i, gpu in enumerate(device_info['gpu_devices']):
            print(f"   GPU {i}: {gpu.get('name', 'Unknown')}")
        print(f"   Mixed Precision: {device_info['mixed_precision_available']}")
    
    print("="*70)
    
    return settings, device_info


def run_data_pipeline(
    settings: Settings,
    horizon_hours: int = 24,
    lookback_hours: int = 168,
    is_daily: bool = False,
    force_download: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DataSpec, any]:
    """
    Run the complete data pipeline.
    
    Args:
        settings: Settings object
        horizon_hours: Forecast horizon in hours
        lookback_hours: Lookback window in hours
        is_daily: Whether to use daily aggregation
        force_download: Force re-download data
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds, data_spec, scaler)
    """
    print("\n" + "="*70)
    print("📥 DATA PIPELINE")
    print("="*70)
    
    with Timer("Data Download", logger):
        downloader = DataDownloader()
        df_raw = downloader.download_all(force_download=force_download)
        
        data_info = downloader.get_data_info(df_raw)
        print(f"\n   📊 Data Summary:")
        print(f"      Records: {data_info['n_records']:,}")
        print(f"      Date Range: {data_info['start_date'][:10]} to {data_info['end_date'][:10]}")
        print(f"      Mean Load: {data_info['mean_load_mw']:,.0f} MW")
        print(f"      Max Load: {data_info['max_load_mw']:,.0f} MW")
        print(f"      Data Quality: {data_info['data_quality']}")
    
    with Timer("Preprocessing", logger):
        preprocessor = DataPreprocessor()
        resample_freq = 'D' if is_daily else None
        df_clean = preprocessor.preprocess(df_raw, resample_freq=resample_freq)
        
        stats = preprocessor.get_stats()
        print(f"\n   🧹 Preprocessing Stats:")
        print(f"      Original rows: {stats['original_rows']:,}")
        print(f"      Final rows: {stats['final_rows']:,}")
        print(f"      Outliers removed: {stats['outliers_replaced']}")
    
    with Timer("Feature Engineering", logger):
        engineer = FeatureEngineer()
        df_featured = engineer.create_features(df_clean, is_daily=is_daily)
        
        feature_groups = engineer.get_feature_importance_groups()
        print(f"\n   🔧 Features Created:")
        for group, features in feature_groups.items():
            if features:
                print(f"      {group}: {len(features)} features")
        print(f"      Total: {len(engineer.get_feature_names())} features")
    
    with Timer("Dataset Creation", logger):
        # Adjust for daily data
        if is_daily:
            lookback = lookback_hours // 24
            horizon = horizon_hours // 24
        else:
            lookback = lookback_hours
            horizon = horizon_hours
        
        train_ds, val_ds, test_ds, spec, scaler = create_datasets(
            df_featured,
            lookback=lookback,
            horizon=horizon,
            batch_size=settings.training.batch_size
        )
        
        print(f"\n   📦 Datasets Created:")
        print(f"      Lookback: {lookback} steps")
        print(f"      Horizon: {horizon} steps")
        print(f"      Features: {spec.n_features}")
        print(f"      Batch Size: {settings.training.batch_size}")
    
    return train_ds, val_ds, test_ds, spec, scaler


def train_forecast_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    spec: DataSpec,
    model_type: str = 'transformer',
    output_dir: Optional[Path] = None,
    training_config: Optional[TrainingConfig] = None
) -> Tuple[any, TrainingResult]:
    """
    Train a forecasting model.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        spec: Data specification
        model_type: Type of model to train
        output_dir: Output directory
        training_config: Training configuration
    
    Returns:
        Tuple of (trained model, training result)
    """
    print("\n" + "="*70)
    print(f"🚀 TRAINING: {model_type.upper()}")
    print("="*70)
    
    settings = get_settings()
    output_dir = output_dir or settings.paths.model_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model configuration
    model_config = ModelConfig(
        n_features=spec.n_features,
        horizon=spec.horizon,
        lookback=spec.lookback,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        predict_sources=True,
        predict_uncertainty=True
    )
    
    # Create model
    print(f"\n   📐 Model Configuration:")
    print(f"      Type: {model_type}")
    print(f"      d_model: {model_config.d_model}")
    print(f"      Layers: {model_config.n_layers}")
    print(f"      Heads: {model_config.n_heads}")
    
    model = create_model(model_type, model_config)
    model.build()
    
    print(f"      Parameters: {model.get_total_params():,}")
    
    # Create trainer
    training_config = training_config or TrainingConfig(
        max_epochs=150,
        early_stopping_patience=25,
        learning_rate=1e-3,
        batch_size=settings.training.batch_size,
        use_mixed_precision=True
    )
    
    trainer = Trainer(model, training_config, output_dir)
    trainer.compile()
    
    print(f"\n   ⚙️ Training Configuration:")
    print(f"      Max Epochs: {training_config.max_epochs}")
    print(f"      Early Stopping Patience: {training_config.early_stopping_patience}")
    print(f"      Learning Rate: {training_config.learning_rate}")
    print(f"      Mixed Precision: {training_config.use_mixed_precision}")
    
    # Train
    print(f"\n   🏋️ Training...")
    result = trainer.train(train_ds, val_ds)
    
    # Save
    trainer.save()
    
    print(f"\n   ✅ Training Complete!")
    print(f"      Best Epoch: {result.best_epoch + 1}")
    print(f"      Best Val Loss: {result.best_val_loss:.4f}")
    print(f"      Training Time: {format_time(result.total_time_seconds)}")
    
    return model, result


def evaluate_model(
    model: any,
    test_ds: tf.data.Dataset,
    scaler: any,
    spec: DataSpec,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        test_ds: Test dataset
        scaler: Data scaler
        spec: Data specification
        output_dir: Output directory for plots
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*70)
    print("📊 EVALUATION")
    print("="*70)
    
    settings = get_settings()
    output_dir = output_dir or settings.paths.plot_dir
    
    # Get predictions
    print("\n   🔮 Generating predictions...")
    
    all_preds = []
    all_targets = []
    all_sources = []
    
    for batch_x, batch_y in test_ds:
        preds = model.model.predict(batch_x, verbose=0)
        
        if isinstance(preds, dict):
            total_preds = preds.get('total_load', preds.get('output'))
            source_preds = preds.get('sources')
        else:
            total_preds = preds
            source_preds = None
        
        all_preds.append(total_preds)
        
        if isinstance(batch_y, dict):
            all_targets.append(batch_y.get('total_load', batch_y).numpy())
        else:
            all_targets.append(batch_y.numpy())
        
        if source_preds is not None:
            all_sources.append(source_preds)
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Inverse transform
    if scaler:
        predictions = scaler.inverse_transform_targets(predictions)
        targets = scaler.inverse_transform_targets(targets)
    
    # Calculate metrics
    print("\n   📈 Calculating metrics...")
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate(targets, predictions, 'test')
    
    print(f"\n   {metrics.summary()}")
    
    # Horizon-specific metrics
    if len(predictions.shape) == 2:
        h_metrics = calculate_horizon_metrics(targets, predictions)
        
        print(f"\n   📅 Horizon Performance (first 5 steps):")
        for i in range(min(5, spec.horizon)):
            print(f"      Step {i+1}: MAE={h_metrics['horizon_mae'][i]:,.0f} MW, "
                  f"R²={h_metrics['horizon_r2'][i]:.4f}")
    
    # Source breakdown
    if all_sources:
        sources = np.concatenate(all_sources)
        source_dict = {}
        
        for i, source_name in enumerate(settings.data.energy_sources):
            if i < sources.shape[-1]:
                source_dict[source_name] = sources[..., i].mean(axis=0)
        
        print(f"\n   ⚡ Average Source Contribution:")
        total = predictions.mean()
        for source_name, source_vals in source_dict.items():
            avg_mw = source_vals.mean() if len(source_vals.shape) > 0 else source_vals
            pct = avg_mw / total * 100 if total > 0 else 0
            print(f"      {source_name.replace('_', ' ').title():15s}: "
                  f"{avg_mw:>10,.0f} MW ({pct:>5.1f}%)")
    
    # Generate plots
    print(f"\n   📊 Generating visualizations...")
    
    visualizer = Visualizer(save_dir=output_dir)
    
    # Predictions plot
    visualizer.plot_predictions(
        targets.flatten()[:500],
        predictions.flatten()[:500],
        title="Load Forecast vs Actual (First 500 Samples)",
        show=False,
        save_name='test_predictions'
    )
    
    # Horizon performance
    if len(predictions.shape) == 2:
        visualizer.plot_horizon_performance(
            h_metrics['horizon_mae'],
            h_metrics['horizon_r2'],
            title="Forecast Performance by Horizon",
            show=False,
            save_name='horizon_performance'
        )
    
    # Source breakdown
    if all_sources:
        visualizer.plot_source_breakdown(
            predictions.mean(axis=0) if len(predictions.shape) > 1 else predictions[:24],
            {k: v[:24] if len(v) > 24 else v for k, v in source_dict.items()},
            title="Generation by Source",
            show=False,
            save_name='source_breakdown'
        )
    
    print(f"\n   ✅ Evaluation complete! Plots saved to {output_dir}")
    
    return metrics.to_dict()


def predict_future(
    model: any,
    last_known_data: np.ndarray,
    scaler: any,
    n_steps: int = 24
) -> ModelOutput:
    """
    Generate future predictions.
    
    Args:
        model: Trained model
        last_known_data: Last known historical data
        scaler: Data scaler
        n_steps: Number of steps to predict
    
    Returns:
        ModelOutput with predictions
    """
    print("\n" + "="*70)
    print("🔮 GENERATING FORECAST")
    print("="*70)
    
    # Prepare input
    if len(last_known_data.shape) == 2:
        last_known_data = last_known_data[np.newaxis, ...]
    
    # Generate prediction
    output = model.predict(last_known_data, return_sources=True, return_uncertainty=True)
    
    # Inverse transform
    if scaler:
        output.total_load = scaler.inverse_transform_targets(output.total_load)
    
    # Print summary
    print(f"\n   📊 Forecast Summary:")
    print(f"      Horizon: {output.total_load.shape[-1]} steps")
    print(f"      Mean Load: {output.total_load.mean():,.0f} MW")
    print(f"      Max Load: {output.total_load.max():,.0f} MW")
    print(f"      Min Load: {output.total_load.min():,.0f} MW")
    
    if output.source_generation:
        print(f"\n   ⚡ Source Breakdown (First Step):")
        summary = output.get_source_summary(timestep=0)
        for source, values in summary.items():
            print(f"      {source.replace('_', ' ').title():15s}: "
                  f"{values['mw']:>10,.0f} MW ({values['percentage']:>5.1f}%)")
    
    if output.lower_bound is not None:
        print(f"\n   📉 Uncertainty (95% CI):")
        print(f"      Lower Bound: {output.lower_bound.mean():,.0f} MW")
        print(f"      Upper Bound: {output.upper_bound.mean():,.0f} MW")
    
    return output


def run_full_pipeline(
    horizon_hours: int = 24,
    lookback_hours: int = 168,
    model_type: str = 'transformer',
    is_daily: bool = False
) -> Dict:
    """
    Run the complete pipeline: data → training → evaluation.
    
    Args:
        horizon_hours: Forecast horizon in hours
        lookback_hours: Lookback window in hours
        model_type: Type of model
        is_daily: Whether to use daily data
    
    Returns:
        Dictionary with all results
    """
    print_banner()
    settings, device_info = setup_environment()
    
    results = {
        'horizon_hours': horizon_hours,
        'lookback_hours': lookback_hours,
        'model_type': model_type,
        'is_daily': is_daily,
        'timestamp': datetime.now().isoformat()
    }
    
    with Timer("Full Pipeline", logger):
        # Data pipeline
        train_ds, val_ds, test_ds, spec, scaler = run_data_pipeline(
            settings,
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
            is_daily=is_daily
        )
        
        results['data_spec'] = {
            'n_features': spec.n_features,
            'horizon': spec.horizon,
            'lookback': spec.lookback
        }
        
        # Training
        model, train_result = train_forecast_model(
            train_ds, val_ds, spec, model_type
        )
        
        results['training'] = train_result.to_dict()
        
        # Evaluation
        eval_metrics = evaluate_model(model, test_ds, scaler, spec)
        results['evaluation'] = eval_metrics
    
    # Save results
    results_path = settings.paths.model_dir / 'pipeline_results.json'
    save_json(results, results_path)
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE")
    print("="*70)
    print(f"\n   📁 Results saved to: {results_path}")
    print(f"\n   📊 Final Metrics:")
    print(f"      MAE: {eval_metrics['mae']:,.0f} MW")
    print(f"      RMSE: {eval_metrics['rmse']:,.0f} MW")
    print(f"      MAPE: {eval_metrics['mape']:.2f}%")
    print(f"      R²: {eval_metrics['r2']:.4f}")
    
    return results


def run_multi_horizon_experiment():
    """
    Run experiment with multiple forecast horizons.
    """
    print_banner()
    settings, device_info = setup_environment()
    
    horizons = [
        {'name': '1 Day', 'hours': 24, 'lookback': 168, 'daily': False},
        {'name': '1 Week', 'hours': 168, 'lookback': 336, 'daily': False},
        {'name': '1 Month', 'hours': 720, 'lookback': 720, 'daily': True},
        {'name': '1 Quarter', 'hours': 2160, 'lookback': 1440, 'daily': True},
    ]
    
    all_results = {}
    
    print("\n" + "="*70)
    print("🔬 MULTI-HORIZON EXPERIMENT")
    print("="*70)
    
    for config in horizons:
        print(f"\n\n{'='*70}")
        print(f"📊 HORIZON: {config['name']}")
        print(f"{'='*70}")
        
        try:
            with Timer(f"{config['name']} Pipeline", logger):
                results = run_full_pipeline(
                    horizon_hours=config['hours'],
                    lookback_hours=config['lookback'],
                    model_type='source_predictor',
                    is_daily=config['daily']
                )
                
                all_results[config['name']] = results['evaluation']
                
        except Exception as e:
            logger.error(f"Error in {config['name']}: {e}")
            all_results[config['name']] = {'error': str(e)}
    
    # Summary table
    print("\n\n" + "="*80)
    print("📊 MULTI-HORIZON RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Horizon':<15} | {'MAE (MW)':>12} | {'RMSE (MW)':>12} | {'MAPE (%)':>10} | {'R²':>10}")
    print("-"*80)
    
    for name, metrics in all_results.items():
        if 'error' in metrics:
            print(f"{name:<15} | {'ERROR':>12} | {'-':>12} | {'-':>10} | {'-':>10}")
        else:
            print(f"{name:<15} | {metrics['mae']:>12,.0f} | {metrics['rmse']:>12,.0f} | "
                  f"{metrics['mape']:>9.2f}% | {metrics['r2']:>10.4f}")
    
    print("="*80)
    
    # Save summary
    save_json(all_results, settings.paths.model_dir / 'multi_horizon_results.json')
    
    return all_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Industry-Grade Electricity Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode train --horizon 24
  %(prog)s --mode train --horizon 168 --model-type source_predictor
  %(prog)s --mode evaluate --model-dir saved_models/transformer
  %(prog)s --mode full_pipeline --horizon 24
  %(prog)s --mode multi_horizon
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full_pipeline',
        choices=['train', 'evaluate', 'predict', 'full_pipeline', 'multi_horizon'],
        help='Mode of operation'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=168,
        help='Lookback window in hours (default: 168)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='source_predictor',
        choices=['transformer', 'tft', 'source_predictor'],
        help='Type of model to use (default: source_predictor)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory containing saved model (for evaluate/predict modes)'
    )
    
    parser.add_argument(
        '--daily',
        action='store_true',
        help='Use daily data aggregation'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Maximum training epochs (default: 150)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    settings = get_settings()
    if not args.quiet:
        setup_logging(settings.paths.log_dir)
    
    # Update settings
    settings.training.batch_size = args.batch_size
    settings.training.max_epochs = args.epochs
    
    # Run appropriate mode
    if args.mode == 'full_pipeline':
        run_full_pipeline(
            horizon_hours=args.horizon,
            lookback_hours=args.lookback,
            model_type=args.model_type,
            is_daily=args.daily
        )
    
    elif args.mode == 'multi_horizon':
        run_multi_horizon_experiment()
    
    elif args.mode == 'train':
        print_banner()
        settings, _ = setup_environment()
        
        train_ds, val_ds, test_ds, spec, scaler = run_data_pipeline(
            settings,
            horizon_hours=args.horizon,
            lookback_hours=args.lookback,
            is_daily=args.daily
        )
        
        model, result = train_forecast_model(
            train_ds, val_ds, spec, args.model_type
        )
    
    elif args.mode == 'evaluate':
        if args.model_dir is None:
            print("Error: --model-dir required for evaluate mode")
            sys.exit(1)
        
        print_banner()
        settings, _ = setup_environment()
        
        # Load model
        model = load_model(Path(args.model_dir))
        
        # Load data
        train_ds, val_ds, test_ds, spec, scaler = run_data_pipeline(
            settings,
            horizon_hours=args.horizon,
            lookback_hours=args.lookback,
            is_daily=args.daily
        )
        
        evaluate_model(model, test_ds, scaler, spec)
    
    elif args.mode == 'predict':
        if args.model_dir is None:
            print("Error: --model-dir required for predict mode")
            sys.exit(1)
        
        print_banner()
        
        # Load model
        model = load_model(Path(args.model_dir))
        
        # For prediction, we need some input data
        print("\n⚠️ Predict mode requires input data.")
        print("   Please use the Python API for custom predictions.")


if __name__ == '__main__':
    main()