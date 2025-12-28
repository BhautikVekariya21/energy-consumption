"""
================================================================================
MODELS MODULE
================================================================================
Neural network models for electricity load and source-wise generation forecasting.
"""

from .base_model import (
    BaseModel,
    ModelOutput,
    ModelConfig,  # ADD THIS
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
    'BaseModel', 'ModelOutput', 'ModelConfig', 'create_model', 'load_model',
    
    # Transformer
    'TransformerModel', 'TransformerBlock', 
    'MultiHeadAttention', 'PositionalEncoding',
    
    # TFT
    'TemporalFusionTransformer', 'VariableSelectionNetwork',
    'GatedResidualNetwork', 'InterpretableMultiHeadAttention',
    
    # Source Predictor
    'SourcePredictor', 'SourceOutput', 'create_source_predictor'
]