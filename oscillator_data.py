"""
Data preparation for the Self-Learning Oscillator.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
import quaternion_encoder

logger = logging.getLogger(__name__)

def prepare_oscillator_training_data(
    df: pd.DataFrame,
    sequence_length: int = 14,
    train_split: float = 0.8,
) -> Dict:
    """
    Prepare data for the Self-Learning Oscillator.
    
    The inputs to the oscillator will be:
      - X_price: The sequence of historical price changes (returns).
      - X_q: The simulated next predicted quaternion (or price).
             For simplicity, we use the actual next quaternion as a proxy for a perfect prediction, 
             but in a real setup, this would be the output of the MTL model.
             
    The target y is the actual future price change.
    """
    logger.info("Preparing data for Self-Learning Oscillator.")
    
    # Scale raw prices
    scaled_data, scaler = quaternion_encoder.encode_dataframe(df)
    
    # Calculate price changes (momentum)
    # We use scaled close prices (w-component, index 0)
    close_prices = scaled_data[:, 0]
    returns = np.zeros_like(close_prices)
    returns[1:] = close_prices[1:] - close_prices[:-1]
    
    X_price = []
    X_q = []
    y_target = []
    
    n_samples = len(scaled_data)
    for i in range(n_samples - sequence_length - 1):
        # Sequence up to t (using returns)
        seq_returns = returns[i : i + sequence_length]
        
        # Next predicted quaternion (we use actual for this demo)
        # In a real pipeline, we would plug in predictor.predict_next_q(...)
        next_q = scaled_data[i + sequence_length]
        
        # Target is the actual return at t+1
        target_return = returns[i + sequence_length + 1]
        
        X_price.append(seq_returns.reshape(-1, 1))
        X_q.append(next_q)
        y_target.append([target_return])
        
    X_price = np.array(X_price, dtype=np.float32)
    X_q = np.array(X_q, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    
    split_idx = int(len(X_price) * train_split)
    
    logger.info(f"Generated {len(X_price)} samples for Self-Learning Oscillator.")
    
    return {
        "X_price_train": X_price[:split_idx],
        "X_q_train": X_q[:split_idx],
        "y_train": y_target[:split_idx],
        "X_price_test": X_price[split_idx:],
        "X_q_test": X_q[split_idx:],
        "y_test": y_target[split_idx:],
        "scaler": scaler,
    }
