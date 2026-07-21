"""
Centralised configuration for the Quaternion LSTM Price Predictor.

All tuneable hyper-parameters, file paths, and project-wide constants
are defined here so that every other module can simply
``from config import *`` (or import individual names).
"""

import os

# ──────────────────────────────────────────────
# Ticker / data settings
# ──────────────────────────────────────────────
TICKER_PRIMARY: str = "F"
"""Primary ticker symbol to fetch from Yahoo Finance. (Ford Motor Co)"""

TICKER_FALLBACK: str = "CAT"
"""Fallback ticker if the primary is unavailable. (Caterpillar Inc)"""

DEFAULT_START_DATE: str = "2020-01-01"
"""Earliest date for historical data retrieval (YYYY-MM-DD)."""

DEFAULT_END_DATE: str | None = None
"""End date for data retrieval.  ``None`` means 'today'."""

DEFAULT_INTERVAL: str = "1d"
"""Candlestick interval (e.g. '1d', '5m')."""

# ──────────────────────────────────────────────
# Sequence / feature dimensions
# ──────────────────────────────────────────────
SEQUENCE_LENGTH: int = 30
"""Number of time-steps in each input sequence. (30 is roughly 1.5 months of daily candles)"""

N_FEATURES: int = 4
"""Number of quaternion components per time-step (w, x, y, z)."""

# ──────────────────────────────────────────────
# Dual-stream / context settings
# ──────────────────────────────────────────────
DUAL_STREAM: bool = True
"""Enable the dual-stream architecture (price + context LSTMs)."""

N_CONTEXT_FEATURES: int = 5
"""Number of context features per time-step (norm_volume, range_ratio,
vol_momentum, norm_sq, range_asymmetry)."""

CONTEXT_LSTM_UNITS: int = 32
"""Hidden units in the context stream LSTM."""

VOLUME_MA_WINDOW: int = 20
"""Window size for the volume moving average used in norm_volume."""

FUSION_STRATEGY: str = "film"
"""Fusion strategy for combining price and context streams.
Options: 'concat', 'gate', 'film'."""

FUSION_DENSE_UNITS: int = 48
"""Units in the Dense layer after fusion."""

CONTEXT_DROPOUT_RATE: float = 0.15
"""Probability of replacing a context timestep with stale data during
training.  Teaches the model to cope with imperfect context at inference."""

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
MODEL_TYPE: str = "lstm"
"""Architecture type to use ('lstm' or 'mtl')."""

LSTM_UNITS: int = 64
"""Hidden units in the LSTM layer."""

DENSE_UNITS: int = 32
"""Units in the fully-connected layer after the LSTM."""

DROPOUT_RATE: float = 0.2
"""Dropout probability applied after recurrent layers."""

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
LEARNING_RATE: float = 0.001
"""Adam optimiser learning rate."""

BATCH_SIZE: int = 32
"""Mini-batch size for training."""

EPOCHS: int = 2
"""Maximum number of training epochs."""

VALIDATION_SPLIT: float = 0.1
"""Fraction of training data reserved for validation."""

TRAIN_TEST_SPLIT: float = 0.8
"""Fraction of the full dataset used for training (remainder is test)."""

# ──────────────────────────────────────────────
# Oscillator Model Settings
# ──────────────────────────────────────────────

OSCILLATOR_SEQ_LEN: int = 14
"""Sequence length for the residual history passed to the oscillator."""

OSCILLATOR_LSTM_UNITS: int = 16
"""LSTM units for processing the residual sequence."""

OSCILLATOR_DENSE_UNITS: int = 8
"""Dense units before the final tanh activation."""

OSCILLATOR_EPOCHS: int = 20
"""Training epochs for the oscillator model."""

OSCILLATOR_LEARNING_RATE: float = 0.005
"""Adam learning rate for the oscillator model."""

MEAN_CENTER_OSCILLATOR: bool = True
"""If True, mean-centers the prediction residuals before passing them to tanh. 
This forces the oscillator to center at 0.0, capturing relative deviations 
rather than absolute model biases."""

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42
"""Global random seed for NumPy, TensorFlow, and scikit-learn."""

# ──────────────────────────────────────────────
# Persistence directories
# ──────────────────────────────────────────────
MODEL_SAVE_DIR: str = "saved_models"
"""Directory where trained model weights / checkpoints are stored."""

DATA_CACHE_DIR: str = "cached_data"
"""Directory for caching downloaded market data."""

VISUALIZATION_DIR: str = "visualizations"
"""Directory for saving plots and charts."""

# Create directories on import so downstream code can write immediately.
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
