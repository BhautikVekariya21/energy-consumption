"""
================================================================================
MODEL CONFIGURATION
================================================================================
Neural network architecture configurations for different model types.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ModelType(Enum):
    """Available model types."""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    TCN = "tcn"
    TEMPORAL_FUSION = "temporal_fusion"
    ENSEMBLE = "ensemble"


class ActivationType(Enum):
    """Activation functions."""
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    MISH = "mish"
    LEAKY_RELU = "leaky_relu"


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    d_ff: int = 1024
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Positional encoding
    max_seq_len: int = 1024
    use_learnable_pos_encoding: bool = True
    
    # Other
    activation: str = "gelu"
    norm_first: bool = True  # Pre-norm vs Post-norm


@dataclass
class LSTMConfig:
    """LSTM model configuration."""
    
    # Architecture
    hidden_size: int = 256
    n_layers: int = 3
    bidirectional: bool = True
    
    # Regularization
    dropout: float = 0.2
    recurrent_dropout: float = 0.1
    
    # Attention
    use_attention: bool = True
    attention_heads: int = 4


@dataclass
class TCNConfig:
    """Temporal Convolutional Network configuration."""
    
    # Architecture
    n_filters: int = 128
    kernel_size: int = 3
    n_layers: int = 6
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    
    # Regularization
    dropout: float = 0.2
    use_weight_norm: bool = True
    use_layer_norm: bool = True


@dataclass
class TemporalFusionConfig:
    """Temporal Fusion Transformer configuration."""
    
    # Hidden dimensions
    hidden_size: int = 256
    attention_heads: int = 4
    
    # LSTM encoder
    lstm_layers: int = 2
    
    # Variable selection
    n_static_vars: int = 0
    n_time_varying_known: int = 20
    n_time_varying_unknown: int = 10
    
    # Dropout
    dropout: float = 0.1
    
    # Quantiles for probabilistic forecasting
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class SourcePredictorConfig:
    """Configuration for source-wise generation predictor."""
    
    # Number of energy sources
    n_sources: int = 7
    
    # Source names
    source_names: List[str] = field(default_factory=lambda: [
        'nuclear', 'coal', 'natural_gas', 'hydro', 'wind', 'solar', 'other'
    ])
    
    # Base model config
    hidden_size: int = 256
    n_heads: int = 8
    n_layers: int = 4
    
    # Source-specific heads
    source_head_hidden: int = 128
    
    # Constraints
    enforce_sum_constraint: bool = True  # Ensure sources sum to total
    min_source_fraction: float = 0.0
    max_source_fraction: float = 1.0
    
    # Typical capacity factors (for realistic constraints)
    capacity_factors: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'nuclear': (0.85, 0.95),      # High capacity factor
        'coal': (0.40, 0.85),          # Variable
        'natural_gas': (0.30, 0.90),   # Flexible
        'hydro': (0.30, 0.60),         # Seasonal
        'wind': (0.25, 0.45),          # Variable
        'solar': (0.15, 0.30),         # Low, depends on sun
        'other': (0.20, 0.50),         # Misc
    })


@dataclass
class EnsembleConfig:
    """Ensemble model configuration."""
    
    # Models to include
    models: List[ModelType] = field(default_factory=lambda: [
        ModelType.TRANSFORMER,
        ModelType.LSTM,
        ModelType.TCN
    ])
    
    # Ensemble method
    method: str = "weighted_average"  # 'weighted_average', 'stacking', 'voting'
    
    # Learning weights
    learnable_weights: bool = True
    
    # Individual model configs
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    tcn_config: TCNConfig = field(default_factory=TCNConfig)


@dataclass
class ModelConfig:
    """Master model configuration."""
    
    # Model type
    model_type: ModelType = ModelType.TRANSFORMER
    
    # Individual configs
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    temporal_fusion: TemporalFusionConfig = field(default_factory=TemporalFusionConfig)
    source_predictor: SourcePredictorConfig = field(default_factory=SourcePredictorConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Output settings
    predict_uncertainty: bool = True
    predict_sources: bool = True
    
    def get_config_for_type(self, model_type: ModelType):
        """Get configuration for specific model type."""
        config_map = {
            ModelType.TRANSFORMER: self.transformer,
            ModelType.LSTM: self.lstm,
            ModelType.TCN: self.tcn,
            ModelType.TEMPORAL_FUSION: self.temporal_fusion,
            ModelType.ENSEMBLE: self.ensemble,
        }
        return config_map.get(model_type)


# Global model config
_model_config: Optional[ModelConfig] = None


def get_model_config() -> ModelConfig:
    """Get global model configuration."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def set_model_config(config: ModelConfig):
    """Set global model configuration."""
    global _model_config
    _model_config = config