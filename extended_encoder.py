"""
Extended Encoder for OHLV Momentum and Quaternion Rotation.

This module provides data preparation functions for the Extended MTL model.
It calculates OHLV momentum as targets and provides the current state quaternion
alongside historical context to allow rotation-based prediction.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import quaternion_encoder

logger = logging.getLogger(__name__)

def prepare_extended_training_data(
    df: pd.DataFrame,
    sequence_length: int = 60,
    train_split: float = 0.8,
    volume_ma_window: int = 20,
) -> Dict:
    """
    Prepare data for the Extended MTL model.
    
    The targets for the 4 auxiliary heads will be the momentums of O, H, L, V.
    The main model input will be the historical quaternion sequences and context,
    plus the *current* state quaternion Q_t so that the model can apply the predicted
    momentum rotation to it to find Q_{t+1}.
    """
    logger.info("Preparing extended training data for momentum-based rotation model.")
    
    # 1. Base encodings (Scale raw prices to [0, 1])
    scaled_data, scaler = quaternion_encoder.encode_dataframe(df)
    
    # Calculate momentums (rate of change of scaled values)
    # momentum[t] = scaled_data[t+1] - scaled_data[t]
    # We pad the first element with 0 to maintain length, but we will shift later.
    momentums = np.zeros_like(scaled_data)
    momentums[1:] = scaled_data[1:] - scaled_data[:-1]
    
    # Calculate context features
    ctx_raw = quaternion_encoder.compute_context_features(df, volume_ma_window=volume_ma_window)
    
    # Sliding window generation
    X_seq = []      # Historical quaternions
    X_ctx = []      # Historical context
    X_current = []  # Q_t (the quaternion at the very end of the sequence window)
    y_mom = []      # Momentum targets for t+1
    y_next = []     # Actual Q_{t+1} (for the main head MSE loss)
    
    n_samples = len(scaled_data)
    for i in range(n_samples - sequence_length - 1):
        # Sequence up to t
        seq = scaled_data[i : i + sequence_length]
        ctx = ctx_raw[i : i + sequence_length]
        
        # Current state Q_t
        q_t = scaled_data[i + sequence_length - 1]
        
        # Targets for t+1
        mom_t_plus_1 = momentums[i + sequence_length]
        q_t_plus_1 = scaled_data[i + sequence_length]
        
        X_seq.append(seq)
        X_ctx.append(ctx)
        X_current.append(q_t)
        y_mom.append(mom_t_plus_1)
        y_next.append(q_t_plus_1)
        
    X_seq = np.array(X_seq, dtype=np.float32)
    X_ctx = np.array(X_ctx, dtype=np.float32)
    X_current = np.array(X_current, dtype=np.float32)
    y_mom = np.array(y_mom, dtype=np.float32)
    y_next = np.array(y_next, dtype=np.float32)
    
    # Train/Test Split
    split_idx = int(len(X_seq) * train_split)
    
    # Context scaling on train set
    context_scaler = MinMaxScaler(feature_range=(0, 1))
    n_tr, sl, nf = X_ctx[:split_idx].shape
    context_scaler.fit(X_ctx[:split_idx].reshape(-1, nf))
    
    X_ctx_scaled = context_scaler.transform(X_ctx.reshape(-1, nf)).reshape(len(X_ctx), sl, nf)
    
    logger.info(f"Generated {len(X_seq)} samples. X_seq shape: {X_seq.shape}")
    
    return {
        "X_train": [X_seq[:split_idx], X_ctx_scaled[:split_idx], X_current[:split_idx]],
        "y_train": {
            "mom_w": y_mom[:split_idx, 0:1],
            "mom_x": y_mom[:split_idx, 1:2],
            "mom_y": y_mom[:split_idx, 2:3],
            "mom_z": y_mom[:split_idx, 3:4],
            "out_main": y_next[:split_idx]
        },
        "X_test": [X_seq[split_idx:], X_ctx_scaled[split_idx:], X_current[split_idx:]],
        "y_test": {
            "mom_w": y_mom[split_idx:, 0:1],
            "mom_x": y_mom[split_idx:, 1:2],
            "mom_y": y_mom[split_idx:, 2:3],
            "mom_z": y_mom[split_idx:, 3:4],
            "out_main": y_next[split_idx:]
        },
        "scaler": scaler,
        "context_scaler": context_scaler,
        "encoding_info": {
            "train_samples": split_idx,
            "test_samples": len(df) - split_idx
        }
    }
