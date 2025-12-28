"""
================================================================================
TRAINING MODULE
================================================================================
Training utilities, callbacks, and loss functions for electricity forecasting.
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    TrainingResult,
    train_model,
    cross_validate
)
from .callbacks import (
    ProgressCallback,
    MetricsLogger,
    GradientMonitor,
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStoppingWithRestore,
    get_default_callbacks
)
from .losses import (
    WeightedMSELoss,
    QuantileLoss,
    HuberLoss,
    MAPELoss,
    CombinedForecastLoss,
    SourceConstraintLoss,
    get_loss_function
)

__all__ = [
    # Trainer
    'Trainer', 'TrainingConfig', 'TrainingResult',
    'train_model', 'cross_validate',
    
    # Callbacks
    'ProgressCallback', 'MetricsLogger', 'GradientMonitor',
    'LearningRateScheduler', 'ModelCheckpoint',
    'EarlyStoppingWithRestore', 'get_default_callbacks',
    
    # Losses
    'WeightedMSELoss', 'QuantileLoss', 'HuberLoss',
    'MAPELoss', 'CombinedForecastLoss', 'SourceConstraintLoss',
    'get_loss_function'
]