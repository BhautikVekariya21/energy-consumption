"""
================================================================================
BASE MODEL CLASS
================================================================================
Abstract base class for all forecasting models with common functionality.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

from config import get_settings, get_model_config
from utils import get_logger, save_json, load_json

logger = get_logger(__name__)


@dataclass
class ModelOutput:
    """Container for model predictions."""
    
    # Main predictions
    total_load: np.ndarray  # Shape: (batch, horizon)
    
    # Source-wise predictions (optional)
    source_generation: Optional[Dict[str, np.ndarray]] = None  # {source_name: (batch, horizon)}
    source_percentages: Optional[Dict[str, np.ndarray]] = None  # {source_name: (batch, horizon)}
    
    # Uncertainty estimates (optional)
    lower_bound: Optional[np.ndarray] = None  # Shape: (batch, horizon)
    upper_bound: Optional[np.ndarray] = None  # Shape: (batch, horizon)
    std: Optional[np.ndarray] = None  # Shape: (batch, horizon)
    
    # Attention weights for interpretability (optional)
    attention_weights: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {'total_load': self.total_load.tolist()}
        
        if self.source_generation:
            result['source_generation'] = {
                k: v.tolist() for k, v in self.source_generation.items()
            }
        
        if self.source_percentages:
            result['source_percentages'] = {
                k: v.tolist() for k, v in self.source_percentages.items()
            }
        
        if self.lower_bound is not None:
            result['lower_bound'] = self.lower_bound.tolist()
            result['upper_bound'] = self.upper_bound.tolist()
        
        return result
    
    def get_source_summary(self, timestep: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Get summary of source contributions at a specific timestep.
        
        Args:
            timestep: Which forecast timestep to summarize
        
        Returns:
            Dictionary with source names and their MW/percentage values
        """
        if self.source_generation is None:
            return {}
        
        summary = {}
        total = self.total_load[0, timestep] if len(self.total_load.shape) > 1 else self.total_load[timestep]
        
        for source_name in self.source_generation.keys():
            gen = self.source_generation[source_name]
            mw = gen[0, timestep] if len(gen.shape) > 1 else gen[timestep]
            pct = (mw / total * 100) if total > 0 else 0
            
            summary[source_name] = {
                'mw': float(mw),
                'gw': float(mw / 1000),
                'percentage': float(pct)
            }
        
        return summary


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Input/Output dimensions
    n_features: int
    horizon: int
    lookback: int
    
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Output configuration
    n_sources: int = 7
    predict_sources: bool = True
    predict_uncertainty: bool = True
    
    # Training
    learning_rate: float = 1e-3
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        return cls(**d)


class BaseModel(ABC):
    """
    Abstract base class for forecasting models.
    
    Provides common functionality:
    - Model building and compilation
    - Training with callbacks
    - Prediction with uncertainty
    - Model saving/loading
    - Source-wise generation prediction
    """
    
    def __init__(
        self,
        config: ModelConfig,
        name: str = "BaseModel"
    ):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            name: Model name for logging and saving
        """
        self.config = config
        self.name = name
        self.model: Optional[Model] = None
        self.history: Optional[Dict] = None
        self.is_built = False
        self.is_trained = False
        
        # Source information
        settings = get_settings()
        self.source_names = settings.data.energy_sources
        
        logger.info(f"Initialized {name} with config: d_model={config.d_model}, "
                   f"n_layers={config.n_layers}, horizon={config.horizon}")
    
    @abstractmethod
    def build(self) -> Model:
        """
        Build the Keras model.
        
        Returns:
            Compiled Keras Model
        """
        pass
    
    def compile(
        self,
        optimizer: Optional[optimizers.Optimizer] = None,
        loss: Optional[Union[str, keras.losses.Loss]] = None,
        metrics: Optional[List] = None
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Keras optimizer
            loss: Loss function
            metrics: List of metrics
        """
        if self.model is None:
            self.model = self.build()
        
        if optimizer is None:
            optimizer = optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=1e-4
            )
        
        if loss is None:
            loss = keras.losses.MeanSquaredError()
        
        if metrics is None:
            metrics = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.is_built = True
        logger.info(f"Model compiled with {optimizer.__class__.__name__}")
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        callbacks_list: Optional[List[callbacks.Callback]] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum epochs
            callbacks_list: List of Keras callbacks
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        if not self.is_built:
            self.compile()
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        logger.info(f"Starting training for up to {epochs} epochs...")
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.history = history.history
        self.is_trained = True
        
        # Log final metrics
        final_loss = self.history['loss'][-1]
        final_val_loss = self.history.get('val_loss', [None])[-1]
        
        logger.info(f"Training complete: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
        
        return self.history
    
    def predict(
        self,
        inputs: Union[np.ndarray, tf.data.Dataset],
        return_sources: bool = True,
        return_uncertainty: bool = True
    ) -> ModelOutput:
        """
        Generate predictions.
        
        Args:
            inputs: Input data (numpy array or TF dataset)
            return_sources: Whether to compute source-wise generation
            return_uncertainty: Whether to compute uncertainty estimates
        
        Returns:
            ModelOutput with predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() or load() first.")
        
        # Get raw predictions
        raw_preds = self.model.predict(inputs, verbose=0)
        
        # Handle multi-output models
        if isinstance(raw_preds, dict):
            total_load = raw_preds.get('total_load', raw_preds.get('output', None))
            source_preds = raw_preds.get('sources', None)
        elif isinstance(raw_preds, (list, tuple)):
            total_load = raw_preds[0]
            source_preds = raw_preds[1] if len(raw_preds) > 1 else None
        else:
            total_load = raw_preds
            source_preds = None
        
        # Ensure correct shape
        if len(total_load.shape) == 1:
            total_load = total_load.reshape(1, -1)
        
        # Build output
        output = ModelOutput(total_load=total_load)
        
        # Process source predictions
        if return_sources and source_preds is not None:
            output.source_generation = {}
            output.source_percentages = {}
            
            for i, source_name in enumerate(self.source_names):
                if i < source_preds.shape[-1]:
                    source_mw = source_preds[..., i]
                    output.source_generation[source_name] = source_mw
                    output.source_percentages[source_name] = source_mw / (total_load + 1e-10) * 100
        
        # Estimate uncertainty using MC dropout if available
        if return_uncertainty and hasattr(self, '_estimate_uncertainty'):
            lower, upper, std = self._estimate_uncertainty(inputs)
            output.lower_bound = lower
            output.upper_bound = upper
            output.std = std
        
        return output
    
    def _estimate_uncertainty(
        self,
        inputs: np.ndarray,
        n_samples: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using MC Dropout.
        
        Args:
            inputs: Input data
            n_samples: Number of MC samples
        
        Returns:
            Tuple of (lower_bound, upper_bound, std)
        """
        # Enable dropout during inference
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(inputs, training=True)
            if isinstance(pred, dict):
                pred = pred.get('total_load', pred.get('output'))
            predictions.append(pred.numpy())
        
        predictions = np.stack(predictions, axis=0)
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        # 95% confidence interval
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        
        return lower, upper, std
    
    def _get_default_callbacks(self) -> List[callbacks.Callback]:
        """Get default training callbacks."""
        settings = get_settings()
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=settings.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=settings.training.reduce_lr_patience,
                min_lr=settings.training.min_lr,
                verbose=1
            ),
            callbacks.TerminateOnNaN()
        ]
        
        return callback_list
    
    def save(self, directory: Union[str, Path]):
        """
        Save model and configuration.
        
        Args:
            directory: Directory to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = directory / 'model.keras'
        self.model.save(model_path)
        
        # Save config
        config_path = directory / 'config.json'
        save_json(self.config.to_dict(), config_path)
        
        # Save history
        if self.history:
            history_path = directory / 'history.json'
            save_json(self.history, history_path)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'is_trained': self.is_trained,
            'source_names': self.source_names
        }
        save_json(metadata, directory / 'metadata.json')
        
        logger.info(f"Model saved to {directory}")
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'BaseModel':
        """
        Load model from directory.
        
        Args:
            directory: Directory to load from
        
        Returns:
            Loaded model instance
        """
        directory = Path(directory)
        
        # Load config
        config_path = directory / 'config.json'
        config_dict = load_json(config_path)
        config = ModelConfig.from_dict(config_dict)
        
        # Load metadata
        metadata_path = directory / 'metadata.json'
        metadata = load_json(metadata_path)
        
        # Create instance
        instance = cls(config, name=metadata.get('name', 'LoadedModel'))
        
        # Load Keras model
        model_path = directory / 'model.keras'
        instance.model = keras.models.load_model(model_path)
        instance.is_built = True
        instance.is_trained = metadata.get('is_trained', True)
        
        # Load history if available
        history_path = directory / 'history.json'
        if history_path.exists():
            instance.history = load_json(history_path)
        
        logger.info(f"Model loaded from {directory}")
        
        return instance
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        if self.model is None:
            self.build()
        return self.model.count_params()


def create_model(
    model_type: str,
    config: ModelConfig,
    **kwargs
) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('transformer', 'tft', 'source_predictor')
        config: Model configuration
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    from .transformer import TransformerModel
    from .temporal_fusion import TemporalFusionTransformer
    from .source_predictor import SourcePredictor
    
    models = {
        'transformer': TransformerModel,
        'tft': TemporalFusionTransformer,
        'temporal_fusion': TemporalFusionTransformer,
        'source_predictor': SourcePredictor,
        'source': SourcePredictor
    }
    
    model_class = models.get(model_type.lower())
    
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return model_class(config, **kwargs)


def load_model(directory: Union[str, Path]) -> BaseModel:
    """
    Load model from directory (auto-detects type).
    
    Args:
        directory: Model directory
    
    Returns:
        Loaded model instance
    """
    directory = Path(directory)
    
    # Load metadata to determine model type
    metadata_path = directory / 'metadata.json'
    
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        model_name = metadata.get('name', 'TransformerModel')
    else:
        model_name = 'TransformerModel'
    
    # Import and load appropriate class
    if 'SourcePredictor' in model_name:
        from .source_predictor import SourcePredictor
        return SourcePredictor.load(directory)
    elif 'TemporalFusion' in model_name or 'TFT' in model_name:
        from .temporal_fusion import TemporalFusionTransformer
        return TemporalFusionTransformer.load(directory)
    else:
        from .transformer import TransformerModel
        return TransformerModel.load(directory)