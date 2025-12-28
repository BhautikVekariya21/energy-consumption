"""
================================================================================
TRANSFORMER MODEL
================================================================================
Transformer-based model for time series forecasting.
Includes positional encoding, multi-head attention, and feed-forward layers.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Optional, Tuple, List

from .base_model import BaseModel, ModelConfig
from utils import get_logger

logger = get_logger(__name__)


class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer for transformers.
    
    Supports both sinusoidal and learnable positional encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_length: int = 5000,
        learnable: bool = True,
        **kwargs
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            learnable: Whether to use learnable positional embeddings
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_length = max_length
        self.learnable = learnable
        
        if learnable:
            self.pos_embedding = layers.Embedding(
                input_dim=max_length,
                output_dim=d_model
            )
        else:
            # Pre-compute sinusoidal encodings
            self.pos_encoding = self._create_sinusoidal_encoding(max_length, d_model)
    
    def _create_sinusoidal_encoding(
        self,
        max_length: int,
        d_model: int
    ) -> tf.Tensor:
        """Create sinusoidal positional encoding."""
        positions = np.arange(max_length)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        
        # Apply sin to even indices, cos to odd
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = tf.shape(x)[1]
        
        if self.learnable:
            positions = tf.range(seq_len)
            pos_enc = self.pos_embedding(positions)
            return x + pos_enc
        else:
            return x + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'max_length': self.max_length,
            'learnable': self.learnable
        })
        return config


class MultiHeadAttention(layers.Layer):
    """
    Multi-head self-attention layer.
    
    Implements scaled dot-product attention with multiple heads.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(**kwargs)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)
        self.output_dense = layers.Dense(d_model)
        
        self.dropout = layers.Dropout(dropout)
        self.attention_weights = None  # Store for interpretability
    
    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split tensor into multiple heads."""
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.d_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def _merge_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Merge heads back together."""
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            training: Whether in training mode
        
        Returns:
            Attention output
        """
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        attention_scores = tf.matmul(query, key, transpose_b=True) / scale
        
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        self.attention_weights = attention_weights  # Store for analysis
        
        attention_weights = self.dropout(attention_weights, training=training)
        
        attention_output = tf.matmul(attention_weights, value)
        
        # Merge heads
        output = self._merge_heads(attention_output, batch_size)
        output = self.output_dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads
        })
        return config


class FeedForward(layers.Layer):
    """Position-wise feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        **kwargs
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__(**kwargs)
        
        self.dense1 = layers.Dense(d_ff, activation=activation)
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply feed-forward transformation."""
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


class TransformerBlock(layers.Layer):
    """
    Single Transformer encoder block.
    
    Contains multi-head attention, feed-forward network, and layer normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        **kwargs
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            norm_first: Whether to apply layer norm before (pre-norm) or after (post-norm)
        """
        super().__init__(**kwargs)
        
        self.norm_first = norm_first
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """Apply transformer block."""
        
        if self.norm_first:
            # Pre-norm (better for deep networks)
            attn_input = self.norm1(x)
            attn_output = self.attention(attn_input, attn_input, attn_input, mask, training)
            x = x + self.dropout1(attn_output, training=training)
            
            ff_input = self.norm2(x)
            ff_output = self.feed_forward(ff_input, training)
            x = x + self.dropout2(ff_output, training=training)
        else:
            # Post-norm (original transformer)
            attn_output = self.attention(x, x, x, mask, training)
            x = self.norm1(x + self.dropout1(attn_output, training=training))
            
            ff_output = self.feed_forward(x, training)
            x = self.norm2(x + self.dropout2(ff_output, training=training))
        
        return x


class TemporalConvBlock(layers.Layer):
    """Temporal convolutional block for local pattern extraction."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.conv1 = layers.Conv1D(
            filters, kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='gelu'
        )
        self.conv2 = layers.Conv1D(
            filters, kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        
        self.residual = layers.Conv1D(filters, 1) if True else None
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = self.residual(x) if self.residual else x
        
        out = self.conv1(x)
        out = self.dropout(out, training=training)
        out = self.conv2(out)
        out = self.norm(out + residual)
        
        return tf.nn.gelu(out)


class TransformerModel(BaseModel):
    """
    Transformer-based model for electricity load forecasting.
    
    Architecture:
    1. Input projection with temporal convolutions
    2. Positional encoding
    3. Stack of transformer encoder blocks
    4. Global pooling and output projection
    """
    
    def __init__(
        self,
        config: ModelConfig,
        name: str = "TransformerModel"
    ):
        """
        Initialize Transformer model.
        
        Args:
            config: Model configuration
            name: Model name
        """
        super().__init__(config, name)
    
    def build(self) -> Model:
        """Build the Transformer model."""
        
        cfg = self.config
        
        # Input layer
        inputs = layers.Input(
            shape=(cfg.lookback, cfg.n_features),
            name='input'
        )
        
        # Input projection with convolutions
        x = layers.Conv1D(cfg.d_model, 1, activation='gelu', name='input_proj')(inputs)
        x = layers.LayerNormalization()(x)
        
        # Temporal convolutions for local patterns
        x = TemporalConvBlock(cfg.d_model, kernel_size=3, dilation_rate=1)(x)
        x = TemporalConvBlock(cfg.d_model, kernel_size=3, dilation_rate=2)(x)
        
        # Positional encoding
        x = PositionalEncoding(
            cfg.d_model,
            max_length=cfg.lookback,
            learnable=True,
            name='pos_encoding'
        )(x)
        
        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                norm_first=True,
                name=f'transformer_block_{i}'
            )(x)
        
        # Final normalization
        x = layers.LayerNormalization()(x)
        
        # Pooling strategies
        # 1. Global average pooling
        x_avg = layers.GlobalAveragePooling1D()(x)
        
        # 2. Last timestep
        x_last = x[:, -1, :]
        
        # 3. Max pooling
        x_max = layers.GlobalMaxPooling1D()(x)
        
        # Combine pooling strategies
        pooled = layers.Concatenate()([x_avg, x_last, x_max])
        
        # Output projection
        x = layers.Dense(cfg.d_ff, activation='gelu')(pooled)
        x = layers.Dropout(cfg.dropout)(x)
        x = layers.Dense(cfg.d_ff // 2, activation='gelu')(x)
        x = layers.Dropout(cfg.dropout)(x)
        
        # Main output: total load
        total_load = layers.Dense(cfg.horizon, name='total_load')(x)
        
        outputs = {'total_load': total_load}
        
        # Optional: Source-wise predictions
        if cfg.predict_sources:
            source_outputs = []
            
            for i, source_name in enumerate(self.source_names):
                source_head = layers.Dense(cfg.d_ff // 4, activation='gelu')(x)
                source_pred = layers.Dense(cfg.horizon, activation='softplus')(source_head)
                source_outputs.append(source_pred)
            
            # Stack sources: (batch, horizon, n_sources)
            sources = layers.Lambda(
                lambda x: tf.stack(x, axis=-1),
                name='sources'
            )(source_outputs)
            
            outputs['sources'] = sources
        
        # Optional: Uncertainty estimation
        if cfg.predict_uncertainty:
            uncertainty = layers.Dense(cfg.horizon, activation='softplus', name='uncertainty')(x)
            outputs['uncertainty'] = uncertainty
        
        # Build model
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        logger.info(f"Built {self.name}: {self.model.count_params():,} parameters")
        
        return self.model
    
    def get_attention_weights(self) -> Optional[List[np.ndarray]]:
        """
        Get attention weights from all transformer blocks.
        
        Returns:
            List of attention weight matrices
        """
        weights = []
        
        for layer in self.model.layers:
            if isinstance(layer, TransformerBlock):
                if layer.attention.attention_weights is not None:
                    weights.append(layer.attention.attention_weights.numpy())
        
        return weights if weights else None