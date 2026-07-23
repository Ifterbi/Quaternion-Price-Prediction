"""
Model Analysis Module.

Contains functionalities for evaluating the Quaternion LSTM Price Predictor:
    - Pure Auto-Regressive Simulation
    - Teacher-Forced (Rolling Window) Prediction
    - Error Analysis (RMSE, MAE, MAPE)
    - Visualization (Matplotlib graphing)

Supports both single-stream and dual-stream model architectures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import logging
from typing import Tuple, Dict, Optional

import config
from quaternion_encoder import decode_quaternion_to_price

logger = logging.getLogger(__name__)


def simulate_autoregressive(
    predictor,
    scaler,
    initial_sequence: np.ndarray,
    n_steps: int,
    initial_context: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pure auto-regressive prediction (predicts its own future).

    For dual-stream models, context beyond the known window is handled by
    repeating the last known context timestep forward (stationarity
    assumption).  The model's context dropout training mitigates
    degradation from this stale-context strategy.

    Args:
        predictor: Trained QuaternionLSTMPredictor instance.
        scaler: MinMaxScaler used for encoding.
        initial_sequence: The starting price window, shape (1, seq_len, 4).
        n_steps: Number of steps to forecast into the future.
        initial_context: The starting context window for dual-stream models,
            shape (1, seq_len, n_context_features).  Ignored for
            single-stream models.

    Returns:
        Array of decoded close prices of shape (n_steps,).
    """
    is_dual = getattr(predictor, "dual_stream", False)
    is_extended = isinstance(initial_sequence, list) and len(initial_sequence) == 3
    
    logger.info(
        "Running pure auto-regressive simulation for %d steps "
        "(dual_stream=%s, extended_mtl=%s)",
        n_steps,
        is_dual,
        is_extended,
    )

    if is_extended:
        current_price_seq = initial_sequence[0].copy()
        current_ctx_seq = initial_sequence[1].copy()
        current_q_state = initial_sequence[2].copy()
    else:
        current_price_seq = initial_sequence.copy()
        current_ctx_seq = initial_context.copy() if initial_context is not None else None
        current_q_state = None
        
    predictions_scaled = []

    for _ in range(n_steps):
        # Build model input
        if is_extended:
            model_input = [current_price_seq, current_ctx_seq, current_q_state]
        elif is_dual and current_ctx_seq is not None:
            model_input = [current_price_seq, current_ctx_seq]
        else:
            model_input = current_price_seq

        # Predict next quaternion (shape: (1, 4))
        next_q = predictor.predict(model_input)
        predictions_scaled.append(next_q[0])

        # Slide the price window: drop oldest, append prediction
        next_q_expanded = np.expand_dims(next_q, axis=1)
        current_price_seq = np.concatenate(
            (current_price_seq[:, 1:, :], next_q_expanded), axis=1
        )

        # Slide the context window: repeat last known context forward
        if is_extended:
            last_ctx = current_ctx_seq[:, -1:, :]
            current_ctx_seq = np.concatenate(
                (current_ctx_seq[:, 1:, :], last_ctx), axis=1
            )
            current_q_state = next_q
        elif is_dual and current_ctx_seq is not None:
            last_ctx = current_ctx_seq[:, -1:, :]  # (1, 1, n_ctx)
            current_ctx_seq = np.concatenate(
                (current_ctx_seq[:, 1:, :], last_ctx), axis=1
            )

    predictions_scaled = np.array(predictions_scaled)
    # Decode w component to close prices
    predicted_prices = decode_quaternion_to_price(predictions_scaled, scaler)
    return predicted_prices


def simulate_teacher_forcing(
    predictor,
    scaler,
    X_test: np.ndarray,
    ctx_X_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Teacher-forced prediction (1-step ahead using actual history).

    Args:
        predictor: Trained QuaternionLSTMPredictor instance.
        scaler: MinMaxScaler used for encoding.
        X_test: Price sequences, shape (N, seq_len, 4).
        ctx_X_test: Context sequences for dual-stream, shape
            (N, seq_len, n_context_features).  Ignored for single-stream.

    Returns:
        Array of decoded close prices of shape (N,).
    """
    is_dual = getattr(predictor, "dual_stream", False)
    is_extended = isinstance(X_test, list) and len(X_test) == 3
    
    logger.info(
        "Running teacher-forced prediction on %d sequences (dual_stream=%s, extended_mtl=%s)",
        len(X_test[0]) if is_extended else len(X_test),
        is_dual,
        is_extended
    )

    if is_extended:
        model_input = X_test
    elif is_dual and ctx_X_test is not None:
        model_input = [X_test, ctx_X_test]
    else:
        model_input = X_test

    # Predict all next steps in batch
    predictions_scaled = predictor.predict(model_input)

    # Decode w component to close prices
    predicted_prices = decode_quaternion_to_price(predictions_scaled, scaler)
    return predicted_prices


def analyze_errors(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate standard error metrics between actual and predicted prices.

    Args:
        actual: 1D array of actual prices.
        predicted: 1D array of predicted prices.

    Returns:
        Dict containing MAE, RMSE, and MAPE.
    """
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual({len(actual)}) vs predicted({len(predicted)})")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean(np.square(actual - predicted)))

    # Avoid division by zero in MAPE
    nonzero_idx = actual != 0
    if np.any(nonzero_idx):
        mape = np.mean(np.abs((actual[nonzero_idx] - predicted[nonzero_idx]) / actual[nonzero_idx])) * 100
    else:
        mape = 0.0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    logger.info("Error Metrics: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%", mae, rmse, mape)
    return metrics


def plot_results(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted_ar: np.ndarray,
    predicted_tf: np.ndarray,
    title_suffix: str = ""
):
    """Generate and save comparative plots.

    Args:
        dates: DatetimeIndex for the x-axis.
        actual: Array of actual prices.
        predicted_ar: Array of pure auto-regressive predictions.
        predicted_tf: Array of teacher-forced predictions.
        title_suffix: Optional suffix for the plot title.
    """
    # 1. Main Price Overlay
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Price', color='black', linewidth=2)
    plt.plot(dates, predicted_tf, label='Predicted (Teacher Forcing)', color='blue', alpha=0.7)
    plt.plot(dates, predicted_ar, label='Predicted (Auto-Regressive)', color='red', alpha=0.7)

    plt.title(f"Quaternion LSTM Price Prediction {title_suffix}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (AUD/USD)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    overlay_path = os.path.join(config.VISUALIZATION_DIR, "prediction_overlay.png")
    plt.savefig(overlay_path, bbox_inches='tight')
    plt.close()
    logger.info("Saved price overlay plot to %s", overlay_path)

    # 2. Residuals (Errors) Plot
    plt.figure(figsize=(14, 5))
    errors_tf = predicted_tf - actual
    errors_ar = predicted_ar - actual

    plt.plot(dates, errors_tf, label='Error (Teacher Forcing)', color='blue', alpha=0.5)
    plt.plot(dates, errors_ar, label='Error (Auto-Regressive)', color='red', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.title("Prediction Errors (Predicted - Actual)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Error Magnitude", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    residuals_path = os.path.join(config.VISUALIZATION_DIR, "prediction_residuals.png")
    plt.savefig(residuals_path, bbox_inches='tight')
    plt.close()
    logger.info("Saved residuals plot to %s", residuals_path)
    
    # Export data to JSON for interactive UI
    json_path = os.path.join(config.VISUALIZATION_DIR, "chart_data.json")
    chart_data = {
        "dates_results": [str(d) for d in dates],
        "actual": actual.tolist(),
        "predicted_tf": predicted_tf.tolist(),
        "predicted_ar": predicted_ar.tolist(),
        "errors_tf": errors_tf.tolist(),
        "errors_ar": errors_ar.tolist()
    }
    with open(json_path, 'w') as f:
        json.dump(chart_data, f)
    logger.info("Saved JSON chart data to %s", json_path)


def plot_oscillator_signals(
    dates: pd.DatetimeIndex,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    signals: np.ndarray,
    title_suffix: str = ""
):
    """Plot the price with the oscillator signals on a secondary axis.
    
    Args:
        dates: DatetimeIndex for the x-axis.
        actual_prices: Array of actual prices.
        predicted_prices: Array of predicted prices from the primary model.
        signals: Array of oscillator signals in [-1, 1].
        title_suffix: Optional suffix for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color1 = 'black'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (AUD/USD)', color=color1, fontsize=12)
    ax1.plot(dates, actual_prices, color=color1, label='Actual Price', linewidth=2)
    ax1.plot(dates, predicted_prices, color='blue', label='Predicted Price', linewidth=1.5, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'purple'
    ax2.set_ylabel('Divergence Momentum [-1, 1]', color=color2, fontsize=12)
    ax2.plot(dates, signals, color=color2, label='Momentum Signal', alpha=0.7, linestyle='-')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.8)
    
    # Highlight momentum zones based on the signal across the entire vertical axis
    ax1.fill_between(
        dates, 0, 1, 
        where=(signals.flatten() > 0), 
        color='red', alpha=0.1, 
        label='Upward Divergence', 
        transform=ax1.get_xaxis_transform()
    )
    ax1.fill_between(
        dates, 0, 1, 
        where=(signals.flatten() < 0), 
        color='green', alpha=0.1, 
        label='Downward Divergence', 
        transform=ax1.get_xaxis_transform()
    )
    
    ax2.set_ylim(-1.1, 1.1)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"Price vs Divergence Momentum Oscillator {title_suffix}", fontsize=16)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

    plot_path = os.path.join(config.VISUALIZATION_DIR, "oscillator_signals.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.info("Saved oscillator signals plot to %s", plot_path)

    # Export data to JSON for interactive UI
    json_path = os.path.join(config.VISUALIZATION_DIR, "chart_data.json")
    chart_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            chart_data = json.load(f)
            
    chart_data.update({
        "dates_oscillator": [str(d) for d in dates],
        "osc_actual": actual_prices.tolist(),
        "osc_predicted": predicted_prices.tolist(),
        "osc_signals": signals.flatten().tolist()
    })
    
    with open(json_path, 'w') as f:
        json.dump(chart_data, f)
    logger.info("Updated JSON chart data at %s", json_path)
