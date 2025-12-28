"""
================================================================================
TEMPORAL FUSION TRANSFORMER (TFT)
================================================================================
Implementation of the Temporal Fusion Transformer for interpretable
multi-horizon time series forecasting.

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon 
           Time Series Forecasting" (Lim et al., 2021)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Optional, Dict, List, Tuple

from .base_model import BaseModel, ModelConfig
from utils import get_logger

logger = get_logger(__name__)


class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit (GLU) activation."""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.linear = layers.Dense(units * 2)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.linear(x)
        x, gate = tf.split(x, 2, axis=-1)
        return x * tf.nn.sigmoid(gate)


class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN).
    
    Applies non-linear processing with gating and residual connections.
    Used throughout TFT for flexible feature transformation.
    """
    
    def __init__(
        self,
        units: int,
        dropout: float = 0.1,
        use_time_distributed: bool = False,
        return_gate: bool = False,
        **kwargs
    ):
        """
        Initialize GRN.
        
        Args:
            units: Number of output units
            dropout: Dropout rate
            use_time_distributed: Whether to apply across time dimension
            return_gate: Whether to return gating weights
        """
        super().__init__(**kwargs)
        
        self.units = units
        self.return_gate = return_gate
        
        # Core layers
        self.dense1 = layers.Dense(units, activation='elu')
        self.dense2 = layers.Dense(units)
        self.glu = GatedLinearUnit(units)
        
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
        
        # Residual connection
        self.residual_dense = layers.Dense(units)
    
    def call(
        self,
        x: tf.Tensor,
        context: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Apply GRN transformation.
        
        Args:
            x: Input tensor
            context: Optional context vector
            training: Whether in training mode
        
        Returns:
            Transformed tensor
        """
        # Initial transformation
        hidden = self.dense1(x)
        
        # Add context if provided
        if context is not None:
            context_proj = layers.Dense(self.units)(context)
            if len(context_proj.shape) < len(hidden.shape):
                context_proj = tf.expand_dims(context_proj, 1)
            hidden = hidden + context_proj
        
        hidden = self.dropout(hidden, training=training)
        hidden = self.dense2(hidden)
        
        # Gating
        gated = self.glu(hidden)
        
        # Residual
        residual = self.residual_dense(x)
        
        # Combine and normalize
        output = self.layer_norm(residual + gated)
        
        return output


class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network (VSN).
    
    Learns which input features are most important at each timestep.
    Provides interpretability through learned feature weights.
    """
    
    def __init__(
        self,
        n_features: int,
        units: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize VSN.
        
        Args:
            n_features: Number of input features
            units: Hidden units
            dropout: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.n_features = n_features
        self.units = units
        
        # Feature-level GRNs
        self.feature_grns = [
            GatedResidualNetwork(units, dropout)
            for _ in range(n_features)
        ]
        
        # Variable selection GRN
        self.variable_grn = GatedResidualNetwork(n_features, dropout)
        
        # Softmax for weights
        self.softmax = layers.Softmax(axis=-1)
    
    def call(
        self,
        x: tf.Tensor,
        context: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply variable selection.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            context: Optional context vector
            training: Whether in training mode
        
        Returns:
            Tuple of (selected_features, variable_weights)
        """
        # Process each feature through its GRN
        feature_outputs = []
        for i, grn in enumerate(self.feature_grns):
            # Extract single feature: (batch, seq_len, 1) -> (batch, seq_len, units)
            feature = x[..., i:i+1]
            processed = grn(feature, context, training)
            feature_outputs.append(processed)
        
        # Stack: (batch, seq_len, n_features, units)
        stacked = tf.stack(feature_outputs, axis=-2)
        
        # Flatten features for selection: (batch, seq_len, n_features)
        flat_x = tf.reduce_mean(stacked, axis=-1)
        
        # Variable selection weights
        weights = self.variable_grn(flat_x, context, training)
        weights = self.softmax(weights)  # (batch, seq_len, n_features)
        
        # Weight and combine features
        weights_expanded = tf.expand_dims(weights, -1)  # (batch, seq_len, n_features, 1)
        selected = tf.reduce_sum(stacked * weights_expanded, axis=-2)  # (batch, seq_len, units)
        
        return selected, weights


class InterpretableMultiHeadAttention(layers.Layer):
    """
    Interpretable Multi-Head Attention.
    
    Modified attention mechanism that provides interpretable attention weights.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)
        self.output_dense = layers.Dense(d_model)
        
        self.dropout = layers.Dropout(dropout)
        
        # Store attention for interpretability
        self.attention_weights = None
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply interpretable attention.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = tf.shape(query)[0]
        
        # Project
        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)
        
        # Reshape for multi-head
        q = tf.reshape(q, (batch_size, -1, self.n_heads, self.d_head))
        k = tf.reshape(k, (batch_size, -1, self.n_heads, self.d_head))
        v = tf.reshape(v, (batch_size, -1, self.n_heads, self.d_head))
        
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Attention scores
        scale = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        
        if mask is not None:
            scores += mask * -1e9
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention
        context = tf.matmul(attention_weights, v)
        
        # Reshape back
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        
        output = self.output_dense(context)
        
        # Average attention across heads for interpretability
        avg_attention = tf.reduce_mean(attention_weights, axis=1)
        
        return output, avg_attention


class TemporalFusionTransformer(BaseModel):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.
    
    Features:
    - Variable Selection Networks for feature importance
    - Gated Residual Networks for flexible transformations
    - Interpretable Multi-Head Attention
    - Multi-horizon output with uncertainty
    
    Provides high interpretability through:
    - Feature importance scores
    - Temporal attention patterns
    - Contribution analysis
    """
    
    def __init__(
        self,
        config: ModelConfig,
        name: str = "TemporalFusionTransformer"
    ):
        super().__init__(config, name)
        
        self.variable_weights = None
        self.temporal_attention = None
    
    def build(self) -> Model:
        """Build the TFT model."""
        
        cfg = self.config
        
        # Input
        inputs = layers.Input(
            shape=(cfg.lookback, cfg.n_features),
            name='input'
        )
        
        # === VARIABLE SELECTION ===
        # Select important features at each timestep
        vsn = VariableSelectionNetwork(
            n_features=cfg.n_features,
            units=cfg.d_model,
            dropout=cfg.dropout,
            name='variable_selection'
        )
        selected_features, variable_weights = vsn(inputs)
        
        # === LSTM ENCODER ===
        # Capture local temporal patterns
        lstm_output = layers.LSTM(
            cfg.d_model,
            return_sequences=True,
            dropout=cfg.dropout,
            name='lstm_encoder'
        )(selected_features)
        
        # Gate and add residual
        encoder_output = GatedResidualNetwork(
            cfg.d_model,
            cfg.dropout,
            name='encoder_grn'
        )(lstm_output)
        
        # === STATIC ENRICHMENT ===
        # Enrich temporal features with static context
        static_context = layers.GlobalAveragePooling1D()(encoder_output)
        static_context = GatedResidualNetwork(
            cfg.d_model,
            cfg.dropout,
            name='static_grn'
        )(static_context)
        
        # === TEMPORAL SELF-ATTENTION ===
        # Capture long-range dependencies
        attention = InterpretableMultiHeadAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            name='temporal_attention'
        )
        
        attended, attention_weights = attention(
            encoder_output, encoder_output, encoder_output
        )
        
        # Gate and add residual
        attended = GatedResidualNetwork(
            cfg.d_model,
            cfg.dropout,
            name='attention_grn'
        )(attended + encoder_output)
        
        # === POSITION-WISE FEED-FORWARD ===
        ff_output = GatedResidualNetwork(
            cfg.d_model,
            cfg.dropout,
            name='ff_grn'
        )(attended)
        
        # === OUTPUT LAYER ===
        # Combine temporal and static information
        temporal_features = layers.GlobalAveragePooling1D()(ff_output)
        combined = layers.Concatenate()([temporal_features, static_context])
        
        # Final processing
        output_hidden = layers.Dense(cfg.d_ff, activation='gelu')(combined)
        output_hidden = layers.Dropout(cfg.dropout)(output_hidden)
        output_hidden = layers.Dense(cfg.d_ff // 2, activation='gelu')(output_hidden)
        output_hidden = layers.Dropout(cfg.dropout)(output_hidden)
        
        # Total load prediction
        total_load = layers.Dense(cfg.horizon, name='total_load')(output_hidden)
        
        outputs = {'total_load': total_load}
        
        # Source predictions
        if cfg.predict_sources:
            source_outputs = []
            
            for source_name in self.source_names:
                source_hidden = layers.Dense(cfg.d_ff // 4, activation='gelu')(output_hidden)
                source_pred = layers.Dense(cfg.horizon, activation='softplus')(source_hidden)
                source_outputs.append(source_pred)
            
            sources = layers.Lambda(
                lambda x: tf.stack(x, axis=-1),
                name='sources'
            )(source_outputs)
            outputs['sources'] = sources
        
        # Uncertainty (quantiles)
        if cfg.predict_uncertainty:
            # Predict lower, median, upper quantiles
            lower = layers.Dense(cfg.horizon, name='lower_quantile')(output_hidden)
            upper = layers.Dense(cfg.horizon, name='upper_quantile')(output_hidden)
            outputs['lower_quantile'] = lower
            outputs['upper_quantile'] = upper
        
        # Build model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        logger.info(f"Built {self.name}: {self.model.count_params():,} parameters")
        
        return self.model
    
    def get_feature_importance(
        self,
        inputs: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get feature importance scores from Variable Selection Network.
        
        Args:
            inputs: Sample inputs
            feature_names: List of feature names
        
        Returns:
            Dictionary of feature importance scores
        """
        # Get VSN layer
        vsn_layer = None
        for layer in self.model.layers:
            if 'variable_selection' in layer.name:
                vsn_layer = layer
                break
        
        if vsn_layer is None:
            return {}
        
        # Get intermediate output
        intermediate_model = Model(
            inputs=self.model.input,
            outputs=vsn_layer.output
        )
        
        _, weights = intermediate_model.predict(inputs, verbose=0)
        
        # Average weights across samples and time
        avg_weights = np.mean(weights, axis=(0, 1))
        
        # Create importance dict
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(avg_weights))]
        
        importance = {
            name: float(weight)
            for name, weight in zip(feature_names, avg_weights)
        }
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def get_temporal_attention(self, inputs: np.ndarray) -> np.ndarray:
        """
        Get temporal attention weights.
        
        Args:
            inputs: Sample inputs
        
        Returns:
            Attention weight matrix
        """
        # Find attention layer
        for layer in self.model.layers:
            if isinstance(layer, InterpretableMultiHeadAttention):
                # Run forward pass
                _ = self.model.predict(inputs, verbose=0)
                return layer.attention_weights.numpy()
        
        return None