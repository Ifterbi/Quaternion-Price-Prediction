"""
Factory for instantiating the correct model architecture based on configuration.
"""
from typing import Optional
import config
from lstm_model import QuaternionLSTMPredictor
from mtl_model import MultiTaskQuaternionPredictor

def build_primary_model(model_type: Optional[str] = None):
    """
    Polymorphic factory function to build and return the primary predictor.
    If model_type is None, it uses config.MODEL_TYPE.
    
    Returns:
        The instantiated and built model predictor.
    """
    m_type = model_type or config.MODEL_TYPE
    
    if m_type == "mtl":
        predictor = MultiTaskQuaternionPredictor(
            sequence_length=config.SEQUENCE_LENGTH,
            n_features=config.N_FEATURES,
            lstm_units=config.LSTM_UNITS,
            aux_dense_units=16,
            dropout_rate=config.DROPOUT_RATE,
            learning_rate=config.LEARNING_RATE,
        )
    else:
        predictor = QuaternionLSTMPredictor(
            sequence_length=config.SEQUENCE_LENGTH,
            n_features=config.N_FEATURES,
            lstm_units=config.LSTM_UNITS,
            dense_units=config.DENSE_UNITS,
            dropout_rate=config.DROPOUT_RATE,
            learning_rate=config.LEARNING_RATE,
            output_size=config.N_FEATURES,
            dual_stream=config.DUAL_STREAM,
            n_context_features=config.N_CONTEXT_FEATURES if config.DUAL_STREAM else 5,
            context_lstm_units=config.CONTEXT_LSTM_UNITS if config.DUAL_STREAM else 32,
            fusion_strategy=config.FUSION_STRATEGY if config.DUAL_STREAM else "film",
            fusion_dense_units=config.FUSION_DENSE_UNITS if config.DUAL_STREAM else 48,
            context_dropout_rate=config.CONTEXT_DROPOUT_RATE if config.DUAL_STREAM else 0.15,
        )
        
    predictor.build_model()
    return predictor
