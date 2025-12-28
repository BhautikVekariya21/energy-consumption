"""
================================================================================
CUSTOM CALLBACKS
================================================================================
Training callbacks for monitoring, logging, and controlling the training process.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
from datetime import datetime

from config import get_settings
from utils import get_logger, format_time, save_json

logger = get_logger(__name__)


class ProgressCallback(callbacks.Callback):
    """
    Enhanced progress callback with detailed metrics and ETA.
    """
    
    def __init__(
        self,
        total_epochs: int,
        print_freq: int = 1,
        metrics_to_show: Optional[List[str]] = None
    ):
        """
        Initialize progress callback.
        
        Args:
            total_epochs: Total number of epochs
            print_freq: How often to print (in epochs)
            metrics_to_show: List of metrics to display
        """
        super().__init__()
        
        self.total_epochs = total_epochs
        self.print_freq = print_freq
        self.metrics_to_show = metrics_to_show or ['loss', 'val_loss', 'mae', 'val_mae']
        
        self.epoch_times = []
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info(f"Training started for {self.total_epochs} epochs")
        print("\n" + "="*80)
        print(f"{'Epoch':^8} | {'Loss':^12} | {'Val Loss':^12} | {'MAE':^10} | {'Time':^8} | {'ETA':^10}")
        print("="*80)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - (self.epoch_start if hasattr(self, 'epoch_start') else self.start_time)
        self.epoch_times.append(epoch_time)
        
        if (epoch + 1) % self.print_freq == 0:
            # Calculate ETA
            avg_epoch_time = np.mean(self.epoch_times[-10:])  # Last 10 epochs
            remaining_epochs = self.total_epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs
            
            # Format metrics
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            mae = logs.get('mae', logs.get('mean_absolute_error', 0))
            
            print(f"{epoch+1:^8} | {loss:^12.4f} | {val_loss:^12.4f} | "
                  f"{mae:^10.4f} | {format_time(epoch_time):^8} | {format_time(eta):^10}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("="*80)
        logger.info(f"Training completed in {format_time(total_time)}")


class MetricsLogger(callbacks.Callback):
    """
    Logs detailed metrics to file and tracks best values.
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_name: str = 'training_metrics'
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            log_name: Name of log file
        """
        super().__init__()
        
        settings = get_settings()
        self.log_dir = Path(log_dir) if log_dir else settings.paths.log_dir
        self.log_name = log_name
        
        self.metrics_history = {
            'epochs': [],
            'timestamps': [],
            'train': {},
            'val': {},
            'best': {}
        }
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['timestamps'].append(datetime.now().isoformat())
        
        # Separate train and val metrics
        for key, value in logs.items():
            if key.startswith('val_'):
                metric_name = key[4:]  # Remove 'val_' prefix
                if metric_name not in self.metrics_history['val']:
                    self.metrics_history['val'][metric_name] = []
                self.metrics_history['val'][metric_name].append(float(value))
            else:
                if key not in self.metrics_history['train']:
                    self.metrics_history['train'][key] = []
                self.metrics_history['train'][key].append(float(value))
        
        # Track best values
        if 'loss' in self.metrics_history['val']:
            val_losses = self.metrics_history['val']['loss']
            best_idx = np.argmin(val_losses)
            self.metrics_history['best'] = {
                'epoch': int(best_idx),
                'val_loss': float(val_losses[best_idx]),
                'train_loss': float(self.metrics_history['train']['loss'][best_idx])
            }
    
    def on_train_end(self, logs=None):
        # Save metrics to file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f'{self.log_name}.json'
        save_json(self.metrics_history, log_path)
        logger.info(f"Metrics saved to {log_path}")
    
    def get_best_epoch(self) -> int:
        """Get the epoch with best validation loss."""
        return self.metrics_history['best'].get('epoch', 0)


class GradientMonitor(callbacks.Callback):
    """
    Monitors gradient statistics to detect training issues.
    """
    
    def __init__(
        self,
        check_frequency: int = 10,
        gradient_threshold: float = 100.0,
        log_histograms: bool = False
    ):
        """
        Initialize gradient monitor.
        
        Args:
            check_frequency: How often to check gradients (in batches)
            gradient_threshold: Warn if gradient norm exceeds this
            log_histograms: Whether to log gradient histograms
        """
        super().__init__()
        
        self.check_frequency = check_frequency
        self.gradient_threshold = gradient_threshold
        self.log_histograms = log_histograms
        
        self.batch_count = 0
        self.gradient_norms = []
    
    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        
        if self.batch_count % self.check_frequency == 0:
            # Get gradient norms
            gradients = []
            for var in self.model.trainable_variables:
                if var.name.endswith('kernel:0'):
                    # Approximate gradient from variable change
                    gradients.append(tf.norm(var).numpy())
            
            if gradients:
                avg_norm = np.mean(gradients)
                self.gradient_norms.append(avg_norm)
                
                if avg_norm > self.gradient_threshold:
                    logger.warning(
                        f"High gradient norm detected: {avg_norm:.2f} "
                        f"(threshold: {self.gradient_threshold})"
                    )
    
    def on_train_end(self, logs=None):
        if self.gradient_norms:
            logger.info(f"Gradient norm stats: "
                       f"mean={np.mean(self.gradient_norms):.4f}, "
                       f"max={np.max(self.gradient_norms):.4f}")


class LearningRateScheduler(callbacks.Callback):
    """
    Custom learning rate scheduler with multiple strategies.
    
    Supports:
    - Cosine annealing with warm restarts
    - One-cycle policy
    - Exponential decay
    - Step decay
    """
    
    def __init__(
        self,
        schedule_type: str = 'cosine',
        initial_lr: float = 1e-3,
        min_lr: float = 1e-7,
        warmup_epochs: int = 5,
        decay_epochs: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LR scheduler.
        
        Args:
            schedule_type: Type of schedule ('cosine', 'exponential', 'step', 'one_cycle')
            initial_lr: Starting learning rate
            min_lr: Minimum learning rate
            warmup_epochs: Epochs for warmup phase
            decay_epochs: Total epochs for decay (for cosine)
            **kwargs: Additional schedule-specific arguments
        """
        super().__init__()
        
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.kwargs = kwargs
        
        self.lr_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        # Get current learning rate
        current_lr = self._calculate_lr(epoch)
        
        # Set learning rate
        keras.backend.set_value(
            self.model.optimizer.learning_rate,
            current_lr
        )
        
        self.lr_history.append(current_lr)
    
    def _calculate_lr(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        
        adjusted_epoch = epoch - self.warmup_epochs
        total_epochs = (self.decay_epochs or 100) - self.warmup_epochs
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            progress = adjusted_epoch / total_epochs
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
            
        elif self.schedule_type == 'exponential':
            # Exponential decay
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            lr = self.initial_lr * (decay_rate ** adjusted_epoch)
            
        elif self.schedule_type == 'step':
            # Step decay
            step_size = self.kwargs.get('step_size', 10)
            gamma = self.kwargs.get('gamma', 0.5)
            lr = self.initial_lr * (gamma ** (adjusted_epoch // step_size))
            
        elif self.schedule_type == 'one_cycle':
            # One-cycle policy
            if adjusted_epoch < total_epochs * 0.3:
                # Increase phase
                progress = adjusted_epoch / (total_epochs * 0.3)
                lr = self.initial_lr + (self.initial_lr * 9) * progress
            else:
                # Decrease phase
                progress = (adjusted_epoch - total_epochs * 0.3) / (total_epochs * 0.7)
                max_lr = self.initial_lr * 10
                lr = max_lr - (max_lr - self.min_lr) * progress
        else:
            lr = self.initial_lr
        
        return max(lr, self.min_lr)


class ModelCheckpoint(callbacks.Callback):
    """
    Enhanced model checkpoint with multiple save options.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        save_freq: str = 'epoch',
        verbose: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path template for saved models
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when monitored metric improves
            save_weights_only: Save weights only (not full model)
            save_freq: 'epoch' or integer for batch frequency
            verbose: Verbosity level
        """
        super().__init__()
        
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch
            
            # Format filepath
            filepath = self.filepath.format(
                epoch=epoch + 1,
                **logs
            )
            
            # Save model
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)
            
            if self.verbose > 0:
                logger.info(f"Epoch {epoch+1}: {self.monitor} improved to {current:.4f}, "
                           f"saving model to {filepath}")
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best
        else:
            return current > self.best


class EarlyStoppingWithRestore(callbacks.Callback):
    """
    Early stopping that restores best weights and provides detailed info.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 20,
        min_delta: float = 0.0001,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: int = 1
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best weights
            verbose: Verbosity level
        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best = None
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, logs=None):
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        logger.info(f"Restoring model weights from epoch {self.best_epoch + 1}")
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(f"Early stopping triggered at epoch {self.stopped_epoch + 1}")
            logger.info(f"Best {self.monitor}: {self.best:.4f} at epoch {self.best_epoch + 1}")
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta


class WarmupCallback(callbacks.Callback):
    """
    Learning rate warmup callback.
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        initial_lr: float = 1e-3,
        warmup_lr: float = 1e-6
    ):
        super().__init__()
        
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * (epoch / self.warmup_epochs)
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)


def get_default_callbacks(
    model_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    patience: int = 25,
    total_epochs: int = 100
) -> List[callbacks.Callback]:
    """
    Get default set of callbacks for training.
    
    Args:
        model_dir: Directory to save model checkpoints
        log_dir: Directory for logs
        patience: Early stopping patience
        total_epochs: Total training epochs
    
    Returns:
        List of Keras callbacks
    """
    settings = get_settings()
    
    model_dir = model_dir or settings.paths.model_dir
    log_dir = log_dir or settings.paths.log_dir
    
    callback_list = [
        # Progress display
        ProgressCallback(total_epochs=total_epochs),
        
        # Metrics logging
        MetricsLogger(log_dir=log_dir),
        
        # Early stopping
        EarlyStoppingWithRestore(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Terminate on NaN
        callbacks.TerminateOnNaN()
    ]
    
    return callback_list