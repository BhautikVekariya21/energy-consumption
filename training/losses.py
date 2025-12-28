"""
================================================================================
CUSTOM LOSS FUNCTIONS
================================================================================
Specialized loss functions for electricity load forecasting.
Includes weighted losses, quantile losses, and source-specific losses.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Optional, List, Dict, Union, Callable

from utils import get_logger

logger = get_logger(__name__)


class WeightedMSELoss(keras.losses.Loss):
    """
    Weighted Mean Squared Error Loss.
    
    Applies different weights to different forecast horizons,
    typically giving more weight to near-term predictions.
    """
    
    def __init__(
        self,
        horizon: int,
        decay_rate: float = 0.95,
        name: str = 'weighted_mse'
    ):
        """
        Initialize weighted MSE loss.
        
        Args:
            horizon: Forecast horizon length
            decay_rate: Exponential decay rate for weights
            name: Loss name
        """
        super().__init__(name=name)
        
        self.horizon = horizon
        self.decay_rate = decay_rate
        
        # Create exponentially decaying weights
        weights = np.array([decay_rate ** i for i in range(horizon)])
        weights = weights / weights.sum() * horizon  # Normalize
        self.weights = tf.constant(weights, dtype=tf.float32)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate weighted MSE."""
        squared_error = tf.square(y_true - y_pred)
        weighted_error = squared_error * self.weights
        return tf.reduce_mean(weighted_error)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'horizon': self.horizon,
            'decay_rate': self.decay_rate
        })
        return config


class QuantileLoss(keras.losses.Loss):
    """
    Quantile Loss (Pinball Loss) for probabilistic forecasting.
    
    Used to predict specific quantiles of the distribution.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        name: str = 'quantile_loss'
    ):
        """
        Initialize quantile loss.
        
        Args:
            quantiles: List of quantiles to predict
            name: Loss name
        """
        super().__init__(name=name)
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate quantile loss.
        
        Args:
            y_true: True values of shape (batch, horizon)
            y_pred: Predicted quantiles of shape (batch, horizon, n_quantiles)
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            if len(y_pred.shape) == 3:
                pred_q = y_pred[..., i]
            else:
                pred_q = y_pred
            
            error = y_true - pred_q
            loss = tf.maximum(q * error, (q - 1) * error)
            losses.append(loss)
        
        return tf.reduce_mean(tf.stack(losses))
    
    def get_config(self):
        config = super().get_config()
        config.update({'quantiles': self.quantiles})
        return config


class HuberLoss(keras.losses.Loss):
    """
    Huber Loss (Smooth L1 Loss).
    
    Less sensitive to outliers than MSE while maintaining
    good gradients for small errors.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        name: str = 'huber_loss'
    ):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between L1 and L2
            name: Loss name
        """
        super().__init__(name=name)
        self.delta = delta
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate Huber loss."""
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * tf.square(quadratic) + self.delta * linear
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


class MAPELoss(keras.losses.Loss):
    """
    Mean Absolute Percentage Error Loss.
    
    Useful when relative errors are more important than absolute errors.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        name: str = 'mape_loss'
    ):
        """
        Initialize MAPE loss.
        
        Args:
            epsilon: Small constant to avoid division by zero
            name: Loss name
        """
        super().__init__(name=name)
        self.epsilon = epsilon
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate MAPE."""
        diff = tf.abs(y_true - y_pred)
        pct = diff / (tf.abs(y_true) + self.epsilon)
        return tf.reduce_mean(pct) * 100
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class SMAPELoss(keras.losses.Loss):
    """
    Symmetric Mean Absolute Percentage Error Loss.
    
    More stable than MAPE when true values are close to zero.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        name: str = 'smape_loss'
    ):
        super().__init__(name=name)
        self.epsilon = epsilon
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate SMAPE."""
        diff = tf.abs(y_true - y_pred)
        denominator = tf.abs(y_true) + tf.abs(y_pred) + self.epsilon
        return tf.reduce_mean(2 * diff / denominator) * 100


class CombinedForecastLoss(keras.losses.Loss):
    """
    Combined loss function for multi-objective optimization.
    
    Combines MSE, MAE, and MAPE with configurable weights.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,
        mape_weight: float = 0.1,
        horizon_decay: float = 0.98,
        name: str = 'combined_loss'
    ):
        """
        Initialize combined loss.
        
        Args:
            mse_weight: Weight for MSE component
            mae_weight: Weight for MAE component
            mape_weight: Weight for MAPE component
            horizon_decay: Decay rate for horizon weighting
            name: Loss name
        """
        super().__init__(name=name)
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mape_weight = mape_weight
        self.horizon_decay = horizon_decay
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate combined loss."""
        # MSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # MAE
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # MAPE
        mape = tf.reduce_mean(
            tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-8)
        )
        
        # Horizon weighting
        if len(y_true.shape) > 1:
            horizon = tf.cast(tf.shape(y_true)[1], tf.float32)
            weights = tf.pow(self.horizon_decay, tf.range(horizon))
            weights = weights / tf.reduce_sum(weights) * horizon
            
            weighted_mse = tf.reduce_mean(
                tf.square(y_true - y_pred) * weights
            )
            mse = 0.5 * mse + 0.5 * weighted_mse
        
        # Combine
        total_loss = (
            self.mse_weight * mse +
            self.mae_weight * mae +
            self.mape_weight * mape
        )
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mse_weight': self.mse_weight,
            'mae_weight': self.mae_weight,
            'mape_weight': self.mape_weight,
            'horizon_decay': self.horizon_decay
        })
        return config


class SourceConstraintLoss(keras.losses.Loss):
    """
    Loss function with source generation constraints.
    
    Ensures that:
    1. Individual source predictions match targets
    2. Sources sum to total load
    3. Source fractions stay within physical bounds
    """
    
    def __init__(
        self,
        n_sources: int = 7,
        sum_constraint_weight: float = 0.5,
        bound_constraint_weight: float = 0.1,
        source_bounds: Optional[Dict[int, tuple]] = None,
        name: str = 'source_constraint_loss'
    ):
        """
        Initialize source constraint loss.
        
        Args:
            n_sources: Number of energy sources
            sum_constraint_weight: Weight for sum-to-total constraint
            bound_constraint_weight: Weight for bound constraints
            source_bounds: Dict of {source_idx: (min_frac, max_frac)}
            name: Loss name
        """
        super().__init__(name=name)
        
        self.n_sources = n_sources
        self.sum_constraint_weight = sum_constraint_weight
        self.bound_constraint_weight = bound_constraint_weight
        
        # Default bounds if not provided
        if source_bounds is None:
            source_bounds = {i: (0.0, 1.0) for i in range(n_sources)}
        self.source_bounds = source_bounds
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        total_load: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Calculate source constraint loss.
        
        Args:
            y_true: True source values (batch, horizon, n_sources)
            y_pred: Predicted source values (batch, horizon, n_sources)
            total_load: Total load for constraint (batch, horizon)
        """
        # Base MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Sum constraint: sources should sum to total
        if total_load is not None:
            source_sum = tf.reduce_sum(y_pred, axis=-1)
            sum_error = tf.square(source_sum - total_load)
            sum_loss = tf.reduce_mean(sum_error)
        else:
            sum_loss = 0.0
        
        # Bound constraints: soft penalty for exceeding bounds
        bound_loss = 0.0
        for i, (min_frac, max_frac) in self.source_bounds.items():
            if i < self.n_sources:
                source_pred = y_pred[..., i]
                
                # Penalty for going below minimum
                below_min = tf.nn.relu(min_frac - source_pred)
                
                # Penalty for going above maximum  
                above_max = tf.nn.relu(source_pred - max_frac)
                
                bound_loss += tf.reduce_mean(tf.square(below_min) + tf.square(above_max))
        
        # Combine losses
        total_loss = (
            mse_loss +
            self.sum_constraint_weight * sum_loss +
            self.bound_constraint_weight * bound_loss
        )
        
        return total_loss


class UncertaintyLoss(keras.losses.Loss):
    """
    Negative Log-Likelihood loss for uncertainty estimation.
    
    Assumes Gaussian distribution and learns both mean and variance.
    """
    
    def __init__(
        self,
        name: str = 'uncertainty_loss'
    ):
        super().__init__(name=name)
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        log_variance: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate NLL loss.
        
        Args:
            y_true: True values
            y_pred: Predicted mean
            log_variance: Predicted log variance
        """
        precision = tf.exp(-log_variance)
        diff = y_true - y_pred
        
        nll = 0.5 * (log_variance + precision * tf.square(diff))
        
        return tf.reduce_mean(nll)


def get_loss_function(
    loss_type: str,
    **kwargs
) -> keras.losses.Loss:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('mse', 'mae', 'huber', 'mape', 'quantile', 'combined')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Keras loss function
    """
    loss_map = {
        'mse': keras.losses.MeanSquaredError,
        'mae': keras.losses.MeanAbsoluteError,
        'huber': HuberLoss,
        'mape': MAPELoss,
        'smape': SMAPELoss,
        'quantile': QuantileLoss,
        'combined': CombinedForecastLoss,
        'weighted_mse': WeightedMSELoss,
        'source_constraint': SourceConstraintLoss
    }
    
    loss_class = loss_map.get(loss_type.lower())
    
    if loss_class is None:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available: {list(loss_map.keys())}")
    
    return loss_class(**kwargs)