"""
================================================================================
MODELS MODULE
================================================================================
Neural network models for electricity load and source-wise generation forecasting.

Available Models:
- TransformerModel: Standard transformer encoder for time series
- TemporalFusionTransformer: TFT for interpretable forecasting
- SourcePredictor: Multi-output model for source-wise generation
- EnsembleModel: Ensemble of multiple architectures
================================================================================
"""

from .base_model import (
    BaseModel,
    ModelOutput,
    create_model,
    load_model
)
from .transformer import (
    TransformerModel,
    TransformerBlock,
    MultiHeadAttention,
    PositionalEncoding
)
from .temporal_fusion import (
    TemporalFusionTransformer,
    VariableSelectionNetwork,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention
)
from .source_predictor import (
    SourcePredictor,
    SourceOutput,
    create_source_predictor
)

__all__ = [
    # Base
    'BaseModel', 'ModelOutput', 'create_model', 'load_model',
    
    # Transformer
    'TransformerModel', 'TransformerBlock', 
    'MultiHeadAttention', 'PositionalEncoding',
    
    # TFT
    'TemporalFusionTransformer', 'VariableSelectionNetwork',
    'GatedResidualNetwork', 'InterpretableMultiHeadAttention',
    
    # Source Predictor
    'SourcePredictor', 'SourceOutput', 'create_source_predictor'
]