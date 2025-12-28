"""
================================================================================
TRAINING ORCHESTRATOR
================================================================================
Main training class that orchestrates the entire training process.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from datetime import datetime

from config import get_settings, get_model_config
from models import BaseModel, create_model
from models.base_model import ModelConfig
from data import DataSpec, ScalerManager
from utils import get_logger, save_json, load_json, Timer, format_time

from .callbacks import get_default_callbacks, MetricsLogger
from .losses import get_loss_function, CombinedForecastLoss

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Epochs
    max_epochs: int = 200
    early_stopping_patience: int = 25
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Batch size
    batch_size: int = 256
    
    # Loss
    loss_type: str = 'combined'
    
    # Learning rate schedule
    lr_schedule: str = 'cosine'
    warmup_epochs: int = 5
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Validation
    validation_freq: int = 1
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Results from training."""
    
    # Metrics
    best_val_loss: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    
    # History
    history: Dict[str, List[float]]
    
    # Timing
    total_time_seconds: float
    epochs_trained: int
    
    # Model info
    model_path: Optional[str] = None
    
    # Additional metrics
    best_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch),
            'final_train_loss': float(self.final_train_loss),
            'final_val_loss': float(self.final_val_loss),
            'total_time_seconds': float(self.total_time_seconds),
            'epochs_trained': int(self.epochs_trained),
            'model_path': self.model_path,
            'best_metrics': self.best_metrics,
            'history': {k: [float(v) for v in vals] for k, vals in self.history.items()}
        }
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "Training Results",
            "=" * 50,
            f"Best Validation Loss: {self.best_val_loss:.4f}",
            f"Best Epoch: {self.best_epoch + 1}",
            f"Epochs Trained: {self.epochs_trained}",
            f"Total Time: {format_time(self.total_time_seconds)}",
        ]
        
        if self.best_metrics:
            lines.append("\nBest Metrics:")
            for name, value in self.best_metrics.items():
                lines.append(f"  {name}: {value:.4f}")
        
        return "\n".join(lines)


class Trainer:
    """
    Main trainer class for electricity forecasting models.
    
    Handles:
    - Model compilation
    - Training loop with callbacks
    - Validation and evaluation
    - Model saving and loading
    - Mixed precision training
    - Multi-GPU training
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: Optional[TrainingConfig] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            output_dir: Directory for outputs
        """
        self.model = model
        self.config = config or TrainingConfig()
        
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else settings.paths.model_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.result: Optional[TrainingResult] = None
        self.scaler: Optional[ScalerManager] = None
        
        # Setup mixed precision if enabled
        if self.config.use_mixed_precision:
            self._setup_mixed_precision()
        
        # Setup multi-GPU if available
        self.strategy = self._setup_strategy()
        
        logger.info(f"Trainer initialized with config: {self.config.to_dict()}")
    
    def _setup_mixed_precision(self):
        """Enable mixed precision training."""
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled (float16)")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
    
    def _setup_strategy(self) -> tf.distribute.Strategy:
        """Setup distribution strategy for multi-GPU."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
        elif len(gpus) == 1:
            strategy = tf.distribute.get_strategy()
            logger.info("Using single GPU")
        else:
            strategy = tf.distribute.get_strategy()
            logger.info("Using CPU")
        
        return strategy
    
    def compile(
        self,
        loss: Optional[Union[str, keras.losses.Loss]] = None,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        metrics: Optional[List] = None
    ):
        """
        Compile the model.
        
        Args:
            loss: Loss function or name
            optimizer: Optimizer instance
            metrics: List of metrics
        """
        # Get loss function
        if loss is None:
            loss = get_loss_function(self.config.loss_type)
        elif isinstance(loss, str):
            loss = get_loss_function(loss)
        
        # Get optimizer
        if optimizer is None:
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                clipnorm=self.config.gradient_clip_norm
            )
        
        # Default metrics
        if metrics is None:
            metrics = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        
        # Build and compile model
        if not self.model.is_built:
            self.model.build()
        
        self.model.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer.__class__.__name__} and {loss}")
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            callbacks: List of callbacks
        
        Returns:
            TrainingResult with training history and metrics
        """
        # Compile if not already done
        if self.model.model.optimizer is None:
            self.compile()
        
        # Get callbacks
        if callbacks is None:
            callbacks = get_default_callbacks(
                model_dir=self.output_dir,
                patience=self.config.early_stopping_patience,
                total_epochs=self.config.max_epochs
            )
        
        logger.info(f"Starting training for up to {self.config.max_epochs} epochs...")
        
        start_time = time.time()
        
        # Train
        history = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.max_epochs,
            callbacks=callbacks,
            verbose=0  # We use custom progress callback
        )
        
        total_time = time.time() - start_time
        
        # Extract results
        hist_dict = history.history
        epochs_trained = len(hist_dict['loss'])
        
        # Find best epoch
        val_losses = hist_dict.get('val_loss', hist_dict['loss'])
        best_epoch = int(np.argmin(val_losses))
        best_val_loss = float(val_losses[best_epoch])
        
        # Get best metrics
        best_metrics = {}
        for key in hist_dict:
            if key.startswith('val_') and key != 'val_loss':
                metric_name = key[4:]  # Remove 'val_' prefix
                best_metrics[metric_name] = float(hist_dict[key][best_epoch])
        
        # Create result
        self.result = TrainingResult(
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            final_train_loss=float(hist_dict['loss'][-1]),
            final_val_loss=float(val_losses[-1]),
            history=hist_dict,
            total_time_seconds=total_time,
            epochs_trained=epochs_trained,
            model_path=str(self.output_dir / 'best_model.keras'),
            best_metrics=best_metrics
        )
        
        # Save training info
        self._save_training_info()
        
        logger.info(self.result.summary())
        
        return self.result
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        scaler: Optional[ScalerManager] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            scaler: Scaler for inverse transforming predictions
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        self.scaler = scaler
        
        # Get predictions
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in test_dataset:
            preds = self.model.model.predict(batch_x, verbose=0)
            
            # Handle multi-output
            if isinstance(preds, dict):
                preds = preds.get('total_load', preds.get('output'))
            
            all_preds.append(preds)
            
            if isinstance(batch_y, dict):
                batch_y = batch_y.get('total_load', batch_y.get('output'))
            
            all_targets.append(batch_y.numpy())
        
        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # Inverse transform if scaler provided
        if scaler:
            predictions = scaler.inverse_transform_targets(predictions)
            targets = scaler.inverse_transform_targets(targets)
        
        # Calculate metrics
        from evaluation import calculate_metrics
        metrics = calculate_metrics(targets, predictions)
        
        logger.info(f"Test Metrics: MAE={metrics['mae']:.2f}, "
                   f"RMSE={metrics['rmse']:.2f}, "
                   f"MAPE={metrics['mape']:.2f}%, "
                   f"R²={metrics['r2']:.4f}")
        
        return metrics
    
    def _save_training_info(self):
        """Save training configuration and results."""
        info = {
            'config': self.config.to_dict(),
            'model_config': self.model.config.to_dict(),
            'result': self.result.to_dict() if self.result else None,
            'timestamp': datetime.now().isoformat()
        }
        
        save_json(info, self.output_dir / 'training_info.json')
    
    def save(self, path: Optional[Path] = None):
        """
        Save trainer state and model.
        
        Args:
            path: Directory to save to
        """
        path = path or self.output_dir
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save training config
        save_json(self.config.to_dict(), path / 'training_config.json')
        
        # Save result
        if self.result:
            save_json(self.result.to_dict(), path / 'training_result.json')
        
        # Save scaler
        if self.scaler:
            self.scaler.save(path / 'scaler.pkl')
        
        logger.info(f"Trainer saved to {path}")
    
    @classmethod
    def load(cls, path: Path, model_class: type = None) -> 'Trainer':
        """
        Load trainer from directory.
        
        Args:
            path: Directory to load from
            model_class: Model class to use for loading
        
        Returns:
            Loaded Trainer instance
        """
        path = Path(path)
        
        # Load training config
        config_path = path / 'training_config.json'
        if config_path.exists():
            config = TrainingConfig.from_dict(load_json(config_path))
        else:
            config = TrainingConfig()
        
        # Load model
        from models import load_model
        model = load_model(path)
        
        # Create trainer
        trainer = cls(model, config, output_dir=path)
        
        # Load result
        result_path = path / 'training_result.json'
        if result_path.exists():
            result_dict = load_json(result_path)
            trainer.result = TrainingResult(**result_dict)
        
        # Load scaler
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            trainer.scaler = ScalerManager.load(scaler_path)
        
        logger.info(f"Trainer loaded from {path}")
        
        return trainer


def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    model_config: ModelConfig,
    training_config: Optional[TrainingConfig] = None,
    model_type: str = 'transformer',
    output_dir: Optional[Path] = None
) -> Tuple[BaseModel, TrainingResult]:
    """
    Convenience function to train a model.
    
    Args:
        train_dataset: Training data
        val_dataset: Validation data
        model_config: Model configuration
        training_config: Training configuration
        model_type: Type of model to create
        output_dir: Output directory
    
    Returns:
        Tuple of (trained model, training result)
    """
    # Create model
    model = create_model(model_type, model_config)
    
    # Create trainer
    trainer = Trainer(model, training_config, output_dir)
    
    # Compile and train
    trainer.compile()
    result = trainer.train(train_dataset, val_dataset)
    
    # Save
    trainer.save()
    
    return model, result


def cross_validate(
    create_dataset_fn: callable,
    model_config: ModelConfig,
    training_config: Optional[TrainingConfig] = None,
    n_folds: int = 5,
    model_type: str = 'transformer'
) -> Dict[str, List[float]]:
    """
    Perform cross-validation.
    
    Args:
        create_dataset_fn: Function that takes fold index and returns (train, val) datasets
        model_config: Model configuration
        training_config: Training configuration
        n_folds: Number of folds
        model_type: Type of model
    
    Returns:
        Dictionary of metrics across folds
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    
    all_metrics = {
        'val_loss': [],
        'mae': [],
        'rmse': [],
        'mape': [],
        'r2': []
    }
    
    for fold in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")
        
        # Create datasets for this fold
        train_ds, val_ds = create_dataset_fn(fold)
        
        # Create and train model
        model = create_model(model_type, model_config)
        trainer = Trainer(model, training_config)
        trainer.compile()
        
        result = trainer.train(train_ds, val_ds)
        
        # Store metrics
        all_metrics['val_loss'].append(result.best_val_loss)
        for name, value in result.best_metrics.items():
            if name in all_metrics:
                all_metrics[name].append(value)
    
    # Calculate summary statistics
    summary = {}
    for name, values in all_metrics.items():
        if values:
            summary[f'{name}_mean'] = float(np.mean(values))
            summary[f'{name}_std'] = float(np.std(values))
    
    logger.info("\nCross-Validation Results:")
    logger.info("=" * 50)
    for name in ['val_loss', 'mae', 'rmse', 'mape', 'r2']:
        if f'{name}_mean' in summary:
            logger.info(f"{name}: {summary[f'{name}_mean']:.4f} ± {summary[f'{name}_std']:.4f}")
    
    return summary