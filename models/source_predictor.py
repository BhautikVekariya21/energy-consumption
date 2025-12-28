"""
================================================================================
SOURCE-WISE GENERATION PREDICTOR
================================================================================
Multi-output model that predicts both total load and source-wise generation.

Predicts contribution from:
- Nuclear
- Coal
- Natural Gas
- Hydro
- Wind
- Solar
- Other

Outputs in MW/GW and percentage contribution.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from .base_model import BaseModel, ModelConfig, ModelOutput
from .transformer import TransformerBlock, PositionalEncoding, TemporalConvBlock
from config import get_settings
from utils import get_logger, format_number

logger = get_logger(__name__)


@dataclass
class SourceOutput:
    """Container for source-wise predictions."""
    
    # Total load
    total_load_mw: np.ndarray
    
    # Per-source generation in MW
    source_mw: Dict[str, np.ndarray]
    
    # Per-source percentage
    source_pct: Dict[str, np.ndarray]
    
    # Horizon information
    horizon: int
    
    def get_summary(self, timestep: int = 0) -> str:
        """
        Get formatted summary string for a timestep.
        
        Args:
            timestep: Which forecast timestep
        
        Returns:
            Formatted string summary
        """
        lines = []
        
        # Get total for this timestep
        if len(self.total_load_mw.shape) > 1:
            total = self.total_load_mw[0, timestep]
        else:
            total = self.total_load_mw[timestep]
        
        lines.append(f"Forecast Timestep {timestep + 1}/{self.horizon}")
        lines.append(f"{'='*50}")
        lines.append(f"Total Load: {format_number(total)}")
        lines.append(f"\nSource Contributions:")
        lines.append(f"{'-'*50}")
        lines.append(f"{'Source':<15} {'Generation':>15} {'Percentage':>12}")
        lines.append(f"{'-'*50}")
        
        # Sort sources by contribution
        source_vals = []
        for source_name, mw_array in self.source_mw.items():
            if len(mw_array.shape) > 1:
                mw = mw_array[0, timestep]
            else:
                mw = mw_array[timestep]
            pct = mw / total * 100 if total > 0 else 0
            source_vals.append((source_name, mw, pct))
        
        source_vals.sort(key=lambda x: x[1], reverse=True)
        
        for source_name, mw, pct in source_vals:
            lines.append(f"{source_name:<15} {format_number(mw):>15} {pct:>10.1f}%")
        
        lines.append(f"{'-'*50}")
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        import pandas as pd
        
        data = {'total_load_mw': self.total_load_mw.flatten()}
        
        for source_name, mw_array in self.source_mw.items():
            data[f'{source_name}_mw'] = mw_array.flatten()
            data[f'{source_name}_pct'] = self.source_pct[source_name].flatten()
        
        return pd.DataFrame(data)


class SourceHead(layers.Layer):
    """
    Prediction head for a single energy source.
    
    Incorporates source-specific constraints and patterns.
    """
    
    def __init__(
        self,
        source_name: str,
        hidden_units: int,
        horizon: int,
        min_fraction: float = 0.0,
        max_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initialize source head.
        
        Args:
            source_name: Name of the energy source
            hidden_units: Hidden layer size
            horizon: Forecast horizon
            min_fraction: Minimum fraction of total load
            max_fraction: Maximum fraction of total load
        """
        super().__init__(**kwargs)
        
        self.source_name = source_name
        self.horizon = horizon
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        
        # Source-specific layers
        self.dense1 = layers.Dense(hidden_units, activation='gelu')
        self.dropout = layers.Dropout(0.1)
        self.dense2 = layers.Dense(hidden_units // 2, activation='gelu')
        self.output_layer = layers.Dense(horizon)
    
    def call(
        self,
        x: tf.Tensor,
        total_load: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Predict source generation.
        
        Args:
            x: Hidden features
            total_load: Predicted total load (for constraints)
            training: Whether in training mode
        
        Returns:
            Source generation prediction
        """
        hidden = self.dense1(x)
        hidden = self.dropout(hidden, training=training)
        hidden = self.dense2(hidden)
        
        # Raw prediction
        raw_pred = self.output_layer(hidden)
        
        # Apply softplus for positive values
        positive_pred = tf.nn.softplus(raw_pred)
        
        # Constrain to fraction of total load
        # This ensures physical feasibility
        min_gen = self.min_fraction * tf.abs(total_load)
        max_gen = self.max_fraction * tf.abs(total_load)
        
        # Clip to constraints
        constrained = tf.clip_by_value(positive_pred, min_gen, max_gen)
        
        return constrained


class SourcePredictor(BaseModel):
    """
    Multi-output model for source-wise generation prediction.
    
    Predicts:
    1. Total electricity load (MW)
    2. Generation from each source (MW):
       - Nuclear
       - Coal
       - Natural Gas
       - Hydro
       - Wind
       - Solar
       - Other
    
    Features:
    - Shared encoder for common patterns
    - Source-specific heads with constraints
    - Automatic normalization to ensure sources sum to total
    - Uncertainty estimation for each source
    """
    
    def __init__(
        self,
        config: ModelConfig,
        name: str = "SourcePredictor"
    ):
        super().__init__(config, name)
        
        # Source constraints (capacity factors)
        self.source_constraints = {
            'nuclear': (0.10, 0.30),      # Baseload
            'coal': (0.05, 0.35),          # Baseload + intermediate
            'natural_gas': (0.20, 0.50),   # Flexible
            'hydro': (0.03, 0.15),         # Seasonal
            'wind': (0.05, 0.20),          # Variable
            'solar': (0.0, 0.15),          # Variable, zero at night
            'other': (0.0, 0.10)           # Miscellaneous
        }
    
    def build(self) -> Model:
        """Build the source predictor model."""
        
        cfg = self.config
        
        # === INPUT ===
        inputs = layers.Input(
            shape=(cfg.lookback, cfg.n_features),
            name='input'
        )
        
        # === SHARED ENCODER ===
        # Initial projection
        x = layers.Conv1D(cfg.d_model, 1, activation='gelu')(inputs)
        x = layers.LayerNormalization()(x)
        
        # Temporal convolutions
        x = TemporalConvBlock(cfg.d_model, kernel_size=3, dilation_rate=1)(x)
        x = TemporalConvBlock(cfg.d_model, kernel_size=3, dilation_rate=2)(x)
        x = TemporalConvBlock(cfg.d_model, kernel_size=3, dilation_rate=4)(x)
        
        # Positional encoding
        x = PositionalEncoding(cfg.d_model, max_length=cfg.lookback)(x)
        
        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                name=f'encoder_block_{i}'
            )(x)
        
        x = layers.LayerNormalization()(x)
        
        # === POOLING ===
        x_avg = layers.GlobalAveragePooling1D()(x)
        x_last = x[:, -1, :]
        x_max = layers.GlobalMaxPooling1D()(x)
        
        shared_features = layers.Concatenate()([x_avg, x_last, x_max])
        
        # Shared hidden layers
        shared = layers.Dense(cfg.d_ff, activation='gelu')(shared_features)
        shared = layers.Dropout(cfg.dropout)(shared)
        shared = layers.Dense(cfg.d_ff // 2, activation='gelu')(shared)
        
        # === TOTAL LOAD PREDICTION ===
        total_hidden = layers.Dense(cfg.d_ff // 4, activation='gelu')(shared)
        total_load = layers.Dense(cfg.horizon, name='total_load')(total_hidden)
        
        # === SOURCE-SPECIFIC HEADS ===
        source_predictions = {}
        source_heads = []
        
        for source_name in self.source_names:
            constraints = self.source_constraints.get(source_name, (0.0, 1.0))
            
            head = SourceHead(
                source_name=source_name,
                hidden_units=cfg.d_ff // 4,
                horizon=cfg.horizon,
                min_fraction=constraints[0],
                max_fraction=constraints[1],
                name=f'head_{source_name}'
            )
            
            source_pred = head(shared, total_load)
            source_predictions[source_name] = source_pred
            source_heads.append(source_pred)
        
        # Stack sources: (batch, horizon, n_sources)
        sources_stacked = layers.Lambda(
            lambda x: tf.stack(x, axis=-1),
            name='sources_raw'
        )(source_heads)
        
        # === NORMALIZATION LAYER ===
        # Ensure sources sum to total load
        sources_sum = tf.reduce_sum(sources_stacked, axis=-1, keepdims=True)
        scale_factor = tf.abs(total_load[..., tf.newaxis]) / (sources_sum + 1e-10)
        sources_normalized = sources_stacked * scale_factor
        sources_normalized = tf.identity(sources_normalized, name='sources')
        
        # === UNCERTAINTY ESTIMATION ===
        if cfg.predict_uncertainty:
            uncertainty_hidden = layers.Dense(cfg.d_ff // 4, activation='gelu')(shared)
            
            # Total load uncertainty
            total_uncertainty = layers.Dense(
                cfg.horizon,
                activation='softplus',
                name='total_uncertainty'
            )(uncertainty_hidden)
            
            # Per-source uncertainty
            source_uncertainties = []
            for source_name in self.source_names:
                source_unc = layers.Dense(
                    cfg.horizon,
                    activation='softplus',
                    name=f'uncertainty_{source_name}'
                )(uncertainty_hidden)
                source_uncertainties.append(source_unc)
            
            source_unc_stacked = layers.Lambda(
                lambda x: tf.stack(x, axis=-1),
                name='source_uncertainties'
            )(source_uncertainties)
        
        # === BUILD MODEL ===
        outputs = {
            'total_load': total_load,
            'sources': sources_normalized
        }
        
        if cfg.predict_uncertainty:
            outputs['total_uncertainty'] = total_uncertainty
            outputs['source_uncertainties'] = source_unc_stacked
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        logger.info(f"Built {self.name}: {self.model.count_params():,} parameters")
        logger.info(f"  Sources: {', '.join(self.source_names)}")
        
        return self.model
    
    def predict_sources(
        self,
        inputs: np.ndarray,
        inverse_transform_fn: Optional[callable] = None
    ) -> SourceOutput:
        """
        Predict total load and source-wise generation.
        
        Args:
            inputs: Input data of shape (batch, lookback, features)
            inverse_transform_fn: Function to inverse transform predictions
        
        Returns:
            SourceOutput with all predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Get predictions
        preds = self.model.predict(inputs, verbose=0)
        
        total_load = preds['total_load']
        sources = preds['sources']
        
        # Inverse transform if provided
        if inverse_transform_fn:
            total_load = inverse_transform_fn(total_load)
            # Transform each source
            for i in range(sources.shape[-1]):
                sources[..., i] = inverse_transform_fn(sources[..., i])
        
        # Build source dictionaries
        source_mw = {}
        source_pct = {}
        
        for i, source_name in enumerate(self.source_names):
            source_mw[source_name] = sources[..., i]
            source_pct[source_name] = sources[..., i] / (total_load + 1e-10) * 100
        
        return SourceOutput(
            total_load_mw=total_load,
            source_mw=source_mw,
            source_pct=source_pct,
            horizon=self.config.horizon
        )
    
    def get_source_summary_table(
        self,
        inputs: np.ndarray,
        timesteps: Optional[List[int]] = None
    ) -> str:
        """
        Get formatted table of source predictions.
        
        Args:
            inputs: Input data
            timesteps: Which timesteps to include (default: first, middle, last)
        
        Returns:
            Formatted string table
        """
        output = self.predict_sources(inputs)
        
        if timesteps is None:
            h = output.horizon
            timesteps = [0, h // 2, h - 1]
        
        tables = []
        for t in timesteps:
            tables.append(output.get_summary(t))
            tables.append("")
        
        return "\n".join(tables)
    
    def compile_with_source_loss(
        self,
        total_weight: float = 1.0,
        source_weight: float = 0.5,
        uncertainty_weight: float = 0.1
    ):
        """
        Compile model with custom loss for sources.
        
        Args:
            total_weight: Weight for total load loss
            source_weight: Weight for source prediction loss
            uncertainty_weight: Weight for uncertainty loss
        """
        if self.model is None:
            self.build()
        
        losses = {
            'total_load': keras.losses.MeanSquaredError(),
            'sources': keras.losses.MeanSquaredError()
        }
        
        loss_weights = {
            'total_load': total_weight,
            'sources': source_weight
        }
        
        if 'total_uncertainty' in [o.name for o in self.model.outputs]:
            losses['total_uncertainty'] = self._uncertainty_loss
            loss_weights['total_uncertainty'] = uncertainty_weight
        
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics={
                'total_load': [
                    keras.metrics.MeanAbsoluteError(name='mae'),
                    keras.metrics.RootMeanSquaredError(name='rmse')
                ]
            }
        )
        
        self.is_built = True
        logger.info("Model compiled with source-specific loss")
    
    @staticmethod
    def _uncertainty_loss(y_true, y_pred):
        """Negative log-likelihood loss for uncertainty."""
        # Assume y_pred is log variance
        return tf.reduce_mean(y_pred + tf.square(y_true) * tf.exp(-y_pred))


def create_source_predictor(
    n_features: int,
    horizon: int,
    lookback: int = 168,
    d_model: int = 256,
    n_layers: int = 4,
    compile_model: bool = True
) -> SourcePredictor:
    """
    Factory function to create a source predictor model.
    
    Args:
        n_features: Number of input features
        horizon: Forecast horizon
        lookback: Historical window size
        d_model: Model dimension
        n_layers: Number of transformer layers
        compile_model: Whether to compile the model
    
    Returns:
        SourcePredictor instance
    """
    config = ModelConfig(
        n_features=n_features,
        horizon=horizon,
        lookback=lookback,
        d_model=d_model,
        n_layers=n_layers,
        predict_sources=True,
        predict_uncertainty=True
    )
    
    model = SourcePredictor(config)
    model.build()
    
    if compile_model:
        model.compile_with_source_loss()
    
    return model