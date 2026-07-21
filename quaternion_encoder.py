"""
Quaternion Encoder for Financial OHLCV Data.

This module handles the novel/custom encoding of financial OHLCV
(Open, High, Low, Close, Volume) data into quaternion representation
for use with the Quaternion LSTM Price Predictor.

The encoding maps price features into 4 quaternion components:
    w = open
    x = high
    y = low
    z = volume
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
import quaternion_ops
import config

logger = logging.getLogger(__name__)


def encode_ohlcv_to_quaternion(
    open_price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Map OHLCV price features into 4 quaternion components.

    Accepts single scalar values or 1-D arrays of equal length.

    Quaternion mapping:
        w = open
        x = high
        y = low
        z = volume

    Args:
        open_price: Opening price(s).
        high: High price(s).
        low: Low price(s).
        volume: Volume(s).

    Returns:
        np.ndarray of shape ``(4,)`` for scalar inputs or ``(N, 4)`` for
        array inputs, where the columns correspond to ``[w, x, y, z]``.
    """
    open_price = np.asarray(open_price, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    w = open_price
    x = high
    y = low
    z = volume

    if w.ndim == 0:
        # Scalar inputs → shape (4,)
        result = np.array([w.item(), x.item(), y.item(), z.item()])
        logger.debug("Encoded single OHLCV sample to quaternion: %s", result)
        return result

    # Array inputs → shape (N, 4)
    result = np.column_stack([w, x, y, z])
    logger.info(
        "Encoded %d OHLCV samples to quaternion representation (shape %s)",
        len(result),
        result.shape,
    )
    return result


def encode_dataframe(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Encode an OHLCV DataFrame into scaled quaternion components.

    Steps:
        1. Extract Open, High, Low, Volume columns.
        2. Compute the 4 quaternion components via
           :func:`encode_ohlcv_to_quaternion`.
        3. Scale the quaternion components to ``[0, 1]`` using a
           ``MinMaxScaler``.

    Args:
        df: DataFrame containing at least ``Open``, ``High``, ``Low``, and
            ``Volume`` columns.
        scaler: An existing ``MinMaxScaler`` instance.  If *None* and
            ``fit_scaler`` is *True*, a new scaler is created and fitted.
        fit_scaler: When *True* (and no *scaler* is supplied), create and fit
            a new ``MinMaxScaler``.  When *False*, the provided *scaler* is
            used for transform only (useful for test / inference data).

    Returns:
        A tuple ``(scaled_quaternions, scaler)`` where
        ``scaled_quaternions`` has shape ``(N, 4)`` and ``scaler`` is the
        fitted ``MinMaxScaler``.

    Raises:
        ValueError: If ``fit_scaler`` is *False* and no *scaler* is provided.
    """
    logger.info("Encoding DataFrame with %d rows to quaternion representation", len(df))

    # 1. Extract price columns
    open_price = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values

    # 2. Compute quaternion components
    q_array = encode_ohlcv_to_quaternion(open_price, high, low, volume)

    # 3. Scale using MinMaxScaler
    if fit_scaler:
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            logger.info("Created new MinMaxScaler and fitting on %d samples", len(q_array))
        else:
            logger.info("Fitting provided MinMaxScaler on %d samples", len(q_array))
        scaled = scaler.fit_transform(q_array)
    else:
        if scaler is None:
            raise ValueError(
                "A fitted scaler must be provided when fit_scaler=False."
            )
        logger.info("Transforming %d samples with existing scaler (no refit)", len(q_array))
        scaled = scaler.transform(q_array)

    logger.info("Scaled quaternion data shape: %s", scaled.shape)
    return scaled, scaler


def decode_quaternion_to_price(
    q_array: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Inverse-transform quaternion components back to a close-price estimate.

    Because the target is the *next* quaternion on the path, its `w` component 
    will be the *next* Open price. For 24/7 markets like Bitcoin, the *next* 
    Open price effectively is the *current* Close price.

    Args:
        q_array: Scaled quaternion component array of shape ``(N, 4)`` or
            ``(4,)``.
        scaler: The ``MinMaxScaler`` that was used to scale the data.

    Returns:
        1-D array of estimated close prices (the ``w`` component after
        inverse scaling).
    """
    if q_array.ndim == 1:
        q_array = q_array.reshape(1, -1)

    inverse = scaler.inverse_transform(q_array)
    close_prices = inverse[:, 0]  # w component = next open price = predicted close price

    logger.debug(
        "Decoded %d quaternion samples → predicted close prices (min=%.4f, max=%.4f)",
        len(close_prices),
        close_prices.min(),
        close_prices.max(),
    )
    return close_prices


def compute_quaternion_path(q_array: np.ndarray) -> np.ndarray:
    """Compute sequential path deltas between consecutive quaternion samples.

    For each consecutive pair ``(q[i], q[i+1])``, the relative rotation is
    computed as ``q[i]^{-1} * q[i+1]`` using :mod:`quaternion_ops`.

    Args:
        q_array: Array of quaternion components with shape ``(N, 4)``.

    Returns:
        Array of path deltas with shape ``(N-1, 4)``.
    """
    if q_array.ndim != 2 or q_array.shape[1] != 4:
        raise ValueError(
            f"Expected array of shape (N, 4), got {q_array.shape}"
        )

    n = len(q_array)
    path_deltas = np.zeros((n - 1, 4), dtype=np.float64)

    for i in range(n - 1):
        q_current = quaternion_ops.from_array(q_array[i])
        q_next = quaternion_ops.from_array(q_array[i + 1])
        delta = quaternion_ops.relative_rotation(q_current, q_next)
        path_deltas[i] = np.array(
            [delta.w, delta.x, delta.y, delta.z], dtype=np.float64
        )

    logger.info(
        "Computed %d quaternion path deltas from %d samples",
        len(path_deltas),
        n,
    )
    return path_deltas


def compute_context_features(
    df: pd.DataFrame,
    volume_ma_window: int = 20,
) -> np.ndarray:
    """Compute context features from raw OHLCV data.

    These features capture volume and range dynamics that constrain
    the space of valid quaternion rotations through normalisation.

    Features:
        f1 — norm_volume:     volume / MA(volume, window)
        f2 — range_ratio:     (high - low) / open
        f3 — vol_momentum:    (volume_t - volume_{t-1}) / volume_{t-1}
        f4 — norm_sq:         open² + high² + low² + volume²
        f5 — range_asymmetry: (close - low) / (high - low)

    Args:
        df: OHLCV DataFrame with Open, High, Low, Close, Volume columns.
        volume_ma_window: Rolling window size for the volume moving average.

    Returns:
        np.ndarray of shape ``(N, 5)`` containing the context features.
    """
    open_p = df["Open"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    close = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    # f1: norm_volume — volume relative to its moving average
    vol_ma = (
        pd.Series(volume)
        .rolling(window=volume_ma_window, min_periods=1)
        .mean()
        .values
    )
    # Avoid division by zero in case MA is exactly 0
    vol_ma = np.where(vol_ma < 1e-12, 1.0, vol_ma)
    norm_volume = volume / vol_ma

    # f2: range_ratio — normalised candlestick range
    open_safe = np.where(np.abs(open_p) < 1e-12, 1.0, open_p)
    range_ratio = (high - low) / np.abs(open_safe)

    # f3: vol_momentum — rate of change of volume
    vol_prev = np.roll(volume, 1)
    vol_prev[0] = volume[0]  # No previous value for first row
    vol_prev_safe = np.where(np.abs(vol_prev) < 1e-12, 1.0, vol_prev)
    vol_momentum = (volume - vol_prev) / np.abs(vol_prev_safe)
    vol_momentum[0] = 0.0  # First row has no momentum

    # f4: norm_sq — squared quaternion norm (from RAW values, not scaled)
    # This captures the actual magnitude relationship where volume dominates
    norm_sq = open_p**2 + high**2 + low**2 + volume**2

    # f5: range_asymmetry — where in the range the close settled
    hl_range = high - low
    hl_range_safe = np.where(np.abs(hl_range) < 1e-12, 1.0, hl_range)
    range_asymmetry = (close - low) / hl_range_safe
    # If high == low (zero range), default to 0.5 (neutral)
    range_asymmetry = np.where(np.abs(hl_range) < 1e-12, 0.5, range_asymmetry)

    features = np.column_stack([
        norm_volume,
        range_ratio,
        vol_momentum,
        norm_sq,
        range_asymmetry,
    ])

    logger.info(
        "Computed %d context feature vectors (5 features) — "
        "norm_volume range: [%.4f, %.4f], norm_sq range: [%.2f, %.2f]",
        len(features),
        norm_volume.min(), norm_volume.max(),
        norm_sq.min(), norm_sq.max(),
    )
    return features


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 60,
    train_split: float = 0.8,
    use_path_deltas: bool = False,
    dual_stream: bool = False,
    volume_ma_window: int = 20,
) -> Dict:
    """End-to-end data preparation pipeline for the Quaternion LSTM.

    Steps:
        1. Encode the DataFrame to quaternion representation and scale.
        2. Optionally compute path deltas.
        3. If ``dual_stream``, compute and align context features.
        4. Create sliding-window sequences (X, y pairs).
        5. Split into train / test sets (time-respecting, *not* random).

    Args:
        df: OHLCV DataFrame.
        sequence_length: Number of time-steps per input window.
        train_split: Fraction of data used for training (remainder is test).
        use_path_deltas: If *True*, use quaternion path deltas instead of
            raw quaternion values.  Forced to *True* when ``dual_stream``
            is enabled.
        dual_stream: If *True*, also compute context features and return
            them as ``ctx_X_train`` / ``ctx_X_test`` alongside the price
            sequences.
        volume_ma_window: Rolling window for the volume moving average
            used in the ``norm_volume`` context feature.

    Returns:
        Dictionary with keys:
            ``X_train``, ``y_train``, ``X_test``, ``y_test``,
            ``scaler``, ``encoding_info``.

        When ``dual_stream=True``, additionally includes:
            ``ctx_X_train``, ``ctx_X_test``, ``context_scaler``.
    """
    # Dual-stream always uses path deltas
    if dual_stream:
        use_path_deltas = True

    logger.info(
        "Preparing training data — sequence_length=%d, train_split=%.2f, "
        "use_path_deltas=%s, dual_stream=%s",
        sequence_length,
        train_split,
        use_path_deltas,
        dual_stream,
    )

    # 1. Encode to quaternion
    scaled_data, scaler = encode_dataframe(df)

    # 2. Optionally compute path deltas
    if use_path_deltas:
        data = compute_quaternion_path(scaled_data)
        logger.info("Using path deltas — data shape: %s", data.shape)
    else:
        data = scaled_data
        logger.info("Using raw quaternion values — data shape: %s", data.shape)

    # 3. Context features (dual-stream only)
    context_scaler = None
    ctx_aligned = None
    if dual_stream:
        ctx_raw = compute_context_features(df, volume_ma_window=volume_ma_window)
        # --- Alignment ---
        # Path deltas have length N-1 (computed from pairs of consecutive rows).
        # Context features have length N (one per row).
        # ctx[t] pairs with delta[t] (the starting day's context informs the
        # rotation FROM that day).  Drop the last context row (no delta pair).
        ctx_aligned = ctx_raw[:-1]  # shape (N-1, 5)
        assert len(ctx_aligned) == len(data), (
            f"Alignment error: context ({len(ctx_aligned)}) != "
            f"deltas ({len(data)}). Expected equal lengths after alignment."
        )
        logger.info(
            "Aligned context features to path deltas — "
            "both have %d rows",
            len(ctx_aligned),
        )
        # Scale context features
        context_scaler = MinMaxScaler(feature_range=(0, 1))
        # We scale AFTER the train/test split to avoid data leakage,
        # but we need to window first.  So we defer scaling to after
        # windowing + splitting below.

    # 4. Create sliding-window sequences
    X, y = [], []
    ctx_X = [] if dual_stream else None
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
        if dual_stream:
            ctx_X.append(ctx_aligned[i : i + sequence_length])

    X = np.array(X)
    y = np.array(y)
    logger.info("Created %d sequences — X shape: %s, y shape: %s", len(X), X.shape, y.shape)

    # 5. Time-respecting train/test split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(
        "Train set: %d samples | Test set: %d samples",
        len(X_train),
        len(X_test),
    )

    encoding_info = {
        "sequence_length": sequence_length,
        "n_features": 4,
        "use_path_deltas": use_path_deltas,
        "dual_stream": dual_stream,
        "train_split": train_split,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "encoding_info": encoding_info,
    }

    # 6. Scale and attach context data
    if dual_stream:
        ctx_X = np.array(ctx_X)
        ctx_X_train, ctx_X_test = ctx_X[:split_idx], ctx_X[split_idx:]

        # Fit scaler on training context only (avoid data leakage).
        # Reshape to 2-D for fitting, then back to 3-D.
        n_tr, sl, nf = ctx_X_train.shape
        context_scaler.fit(ctx_X_train.reshape(-1, nf))
        ctx_X_train = context_scaler.transform(
            ctx_X_train.reshape(-1, nf)
        ).reshape(n_tr, sl, nf)

        n_te = ctx_X_test.shape[0]
        ctx_X_test = context_scaler.transform(
            ctx_X_test.reshape(-1, nf)
        ).reshape(n_te, sl, nf)

        result["ctx_X_train"] = ctx_X_train
        result["ctx_X_test"] = ctx_X_test
        result["context_scaler"] = context_scaler
        encoding_info["n_context_features"] = nf

        logger.info(
            "Dual-stream context — ctx_X_train: %s, ctx_X_test: %s",
            ctx_X_train.shape,
            ctx_X_test.shape,
        )

    return result


def compute_residuals(
    predictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    ctx_X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute residuals and next predictions using the primary model.

    Args:
        predictor: The trained QuaternionLSTMPredictor (frozen).
        X_test: Price sequence inputs.
        y_test: Actual targets (scaled quaternions).
        scaler: The fitted MinMaxScaler.
        ctx_X_test: Optional context sequences.

    Returns:
        Tuple of (residuals, predicted_quaternions).
        residuals shape: (N,)
        predicted_quaternions shape: (N, 4)
    """
    model_input = [X_test, ctx_X_test] if ctx_X_test is not None else X_test
    
    # Get primary model's prediction for the next quaternion
    pred_q = predictor.predict(model_input)
    
    # Decode both to actual prices to compute error in meaningful units
    actual_price = decode_quaternion_to_price(y_test, scaler)
    pred_price = decode_quaternion_to_price(pred_q, scaler)
    
    residuals = actual_price - pred_price
    
    logger.info(
        "Computed %d residuals (mean=%.4f, std=%.4f)",
        len(residuals),
        np.mean(residuals),
        np.std(residuals),
    )
    return residuals, pred_q


def prepare_oscillator_data(
    residuals: np.ndarray,
    pred_q: np.ndarray,
    sequence_length: int = 14,
    train_split: float = 0.8,
) -> Dict[str, np.ndarray]:
    """Prepare sliding window sequences for the oscillator model.

    The target is the *future* normalized residual, squashed by tanh.
    This creates an implicit [-1, 1] target where 1 is highly overvalued
    and -1 is highly undervalued.

    Args:
        residuals: 1D array of prediction errors (shape: N,).
        pred_q: 2D array of predicted quaternions (shape: N, 4).
        sequence_length: Window size for past residuals.
        train_split: Fraction of data for training.

    Returns:
        Dictionary with X_res_train, X_q_train, y_train, etc.
    """
    n = len(residuals)
    if n <= sequence_length:
        raise ValueError(f"Not enough data ({n}) for sequence length {sequence_length}.")
        
    # Compute implicit target: divergence momentum (derivative of residual)
    delta_residuals = np.zeros_like(residuals)
    delta_residuals[1:] = residuals[1:] - residuals[:-1]
    
    std_delta = np.std(delta_residuals)
    mean_delta = np.mean(delta_residuals) if config.MEAN_CENTER_OSCILLATOR else 0.0
    
    # Avoid division by zero
    std_delta = std_delta if std_delta > 1e-12 else 1.0
    
    # Target is the momentum of the divergence
    targets = np.tanh((delta_residuals - mean_delta) / std_delta)
    
    if config.MEAN_CENTER_OSCILLATOR:
        logger.info("Mean-centering divergence momentum (subtracted %.4f)", mean_delta)
    
    X_res, X_q, y = [], [], []
    
    for i in range(n - sequence_length):
        # The history of residuals leading up to time t
        window = residuals[i : i + sequence_length]
        # We need to reshape the window to add a feature dimension
        X_res.append(window.reshape(-1, 1))
        
        # The primary model's prediction for time t+1
        target_idx = i + sequence_length
        X_q.append(pred_q[target_idx])
        
        # The actual implicit valuation target at time t+1
        y.append(targets[target_idx])
        
    X_res = np.array(X_res, dtype=np.float32)
    X_q = np.array(X_q, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    split_idx = int(len(X_res) * train_split)
    
    logger.info(
        "Prepared oscillator data — seq_len=%d, total=%d, train=%d",
        sequence_length,
        len(X_res),
        split_idx,
    )
    
    return {
        "X_res_train": X_res[:split_idx],
        "X_q_train": X_q[:split_idx],
        "y_train": y[:split_idx],
        "X_res_test": X_res[split_idx:],
        "X_q_test": X_q[split_idx:],
        "y_test": y[split_idx:],
        "residual_std": std_delta,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Demo with synthetic OHLCV data ---
    np.random.seed(42)
    n_days = 200
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    closes = base_price * np.cumprod(1 + returns)

    demo_df = pd.DataFrame(
        {
            "Open": closes * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "High": closes * (1 + np.random.uniform(0.005, 0.025, n_days)),
            "Low": closes * (1 - np.random.uniform(0.005, 0.025, n_days)),
            "Close": closes,
            "Volume": np.random.uniform(1000, 5000, n_days),
        }
    )

    # Single-sample encoding
    q_single = encode_ohlcv_to_quaternion(100.0, 105.0, 95.0, 1500.0)
    print(f"Single quaternion encoding: {q_single}")
    print(f"  w (open)   = {q_single[0]}")
    print(f"  x (high)   = {q_single[1]}")
    print(f"  y (low)    = {q_single[2]}")
    print(f"  z (volume) = {q_single[3]}")

    # DataFrame encoding
    scaled, scaler = encode_dataframe(demo_df)
    print(f"\nDataFrame encoding shape: {scaled.shape}")
    print(f"Scaled min: {scaled.min(axis=0)}")
    print(f"Scaled max: {scaled.max(axis=0)}")

    # Decode back to prices
    decoded = decode_quaternion_to_price(scaled[:5], scaler)
    print(f"\nDecoded close prices (first 5): {decoded}")
    print(f"Original close prices (first 5): {demo_df['Close'].values[:5]}")

    # Context features
    ctx = compute_context_features(demo_df)
    print(f"\n--- Context Features ---")
    print(f"  Shape: {ctx.shape}")
    print(f"  Features: [norm_volume, range_ratio, vol_momentum, norm_sq, range_asymmetry]")
    print(f"  First 3 rows:\n{ctx[:3]}")

    # Single-stream pipeline (backwards compatible)
    result = prepare_training_data(demo_df, sequence_length=30)
    print(f"\n--- Single-Stream Training Data ---")
    for key, value in result["encoding_info"].items():
        print(f"  {key}: {value}")
    print(f"  X_train shape: {result['X_train'].shape}")
    print(f"  y_train shape: {result['y_train'].shape}")

    # Dual-stream pipeline
    result_ds = prepare_training_data(demo_df, sequence_length=30, dual_stream=True)
    print(f"\n--- Dual-Stream Training Data ---")
    for key, value in result_ds["encoding_info"].items():
        print(f"  {key}: {value}")
    print(f"  X_train shape:     {result_ds['X_train'].shape}")
    print(f"  ctx_X_train shape: {result_ds['ctx_X_train'].shape}")
    print(f"  y_train shape:     {result_ds['y_train'].shape}")
    print(f"  X_test shape:      {result_ds['X_test'].shape}")
    print(f"  ctx_X_test shape:  {result_ds['ctx_X_test'].shape}")
    print(f"  y_test shape:      {result_ds['y_test'].shape}")
    print(f"\n  Alignment check: X and ctx_X have same sample count: "
          f"{result_ds['X_train'].shape[0] == result_ds['ctx_X_train'].shape[0]}")
