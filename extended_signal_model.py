"""
Feedback Oscillator Model and Extended Pipeline.

This module provides the FeedbackOscillator, which predicts a buy/sell signal
based on the predicted rotation and historical residuals. It also provides the
ExtendedPipeline class to seamlessly chain the predictor and the oscillator.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import logging
from typing import Dict, Optional
import signal_model

logger = logging.getLogger(__name__)

class FeedbackOscillator(signal_model.ResidualOscillator):
    """
    Inherits from the base ResidualOscillator but is adapted to work
    specifically with the new extended MTL rotation model's outputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self) -> Model:
        logger.info("Building Feedback Oscillator model...")
        return super().build_model()
        

class ExtendedPipeline:
    """
    A wrapper pipeline that chains the Extended MTL Predictor and the Feedback Oscillator.
    Provides a feedback loop during inference by computing expected residuals on the fly.
    """
    
    def __init__(self, predictor, oscillator, scaler):
        self.predictor = predictor
        self.oscillator = oscillator
        self.scaler = scaler
        logger.info("ExtendedPipeline initialized.")
        
    def predict_with_feedback(self, X_seq: np.ndarray, X_ctx: np.ndarray, q_current: np.ndarray, recent_actuals: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the full inference pipeline with the feedback loop.
        
        1. Predict Q_{next} using the Extended MTL model.
        2. Estimate the current residual using recent actuals and recent predictions.
           (For a true real-time loop, you provide the trailing N residuals).
        3. Pass Q_{next} and the trailing residuals to the oscillator to get the signal.
        """
        if self.predictor.model is None or self.oscillator.model is None:
            raise RuntimeError("Both predictor and oscillator must be built and trained.")
            
        # 1. Base Prediction
        q_next = self.predictor.predict_next_q(X_seq, X_ctx, q_current)
        
        # 2. Extract expected price
        # In a real scenario, you'd calculate the residual between the *last* prediction
        # and the *current* actual price. For this demo pipeline, we assume `recent_actuals`
        # is a pre-calculated sequence of residuals of shape (batch_size, seq_len, 1).
        residuals = recent_actuals
        
        # 3. Oscillator Signal
        signal = self.oscillator.predict(residuals, q_next)
        
        # 4. Decode to actual price (w component = next open)
        if q_next.ndim == 1:
            q_next_reshaped = q_next.reshape(1, -1)
        else:
            q_next_reshaped = q_next
            
        inverse = self.scaler.inverse_transform(q_next_reshaped)
        predicted_close = inverse[:, 0]
        
        return {
            "predicted_quaternion": q_next,
            "predicted_close_price": predicted_close,
            "trading_signal": signal
        }
