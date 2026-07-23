"""
Quaternion Path-Based LSTM Price Predictor — Main Entry Point.

This script ties together all modules to demonstrate the full pipeline:
    1. Fetch Bitcoin OHLCV data via yfinance
    2. Encode OHLCV data into quaternion representation
    3. Optionally compute quaternion path deltas
    4. Prepare sequences for LSTM input (single- or dual-stream)
    5. Build the LSTM model and display its summary

Training is NOT auto-executed — call ``run_training()`` explicitly or
set ``TRAIN_ON_RUN = True`` to train when running the script directly.
"""

import logging
import sys
import numpy as np
import pandas as pd

# Project modules
import config
from data_fetcher import fetch_bitcoin_data, get_ohlcv
from quaternion_encoder import (
    encode_dataframe,
    decode_quaternion_to_price,
    compute_quaternion_path,
    compute_context_features,
    prepare_training_data,
    compute_residuals,
    prepare_oscillator_data,
)
from model_factory import build_primary_model
from signal_model import ResidualOscillator
import extended_encoder
from extended_signal_model import FeedbackOscillator, ExtendedPipeline
from model_analysis import (
    simulate_autoregressive,
    simulate_teacher_forcing,
    analyze_errors,
    plot_results,
    plot_oscillator_signals,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_ON_RUN: bool = True  # Set True to train when running main.py directly


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a human-readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def fetch_data():
    """Fetch and display Bitcoin OHLCV data."""
    print("\n" + "=" * 60)
    print("  Step 1: Fetching Bitcoin OHLCV Data")
    print("=" * 60)

    df = fetch_bitcoin_data(
        ticker_primary=config.TICKER_PRIMARY,
        ticker_fallback=config.TICKER_FALLBACK,
        start=config.DEFAULT_START_DATE,
    )

    ohlcv = get_ohlcv(df)
    print(f"\n  Ticker:     {config.TICKER_PRIMARY} (fallback: {config.TICKER_FALLBACK})")
    print(f"  Shape:      {ohlcv.shape}")
    print(f"  Date range: {ohlcv.index.min()} → {ohlcv.index.max()}")
    print(f"\n  Head:\n{ohlcv.head().to_string(max_cols=5)}")
    print(f"\n  Tail:\n{ohlcv.tail().to_string(max_cols=5)}")

    return ohlcv


def encode_data(df):
    """Encode OHLCV data into quaternion representation."""
    print("\n" + "=" * 60)
    print("  Step 2: Encoding OHLCV → Quaternion Representation")
    print("=" * 60)

    scaled_data, scaler = encode_dataframe(df)

    print(f"\n  Quaternion data shape: {scaled_data.shape}")
    print(f"  Components: [w=open, x=high, y=low, z=volume]")
    print(f"  Scaled range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
    print(f"\n  First 3 quaternion samples:")
    for i in range(min(3, len(scaled_data))):
        print(f"    q[{i}] = {scaled_data[i]}")

    return scaled_data, scaler


def compute_path(scaled_data):
    """Compute quaternion path deltas."""
    print("\n" + "=" * 60)
    print("  Step 3: Computing Quaternion Path Deltas")
    print("=" * 60)

    path_deltas = compute_quaternion_path(scaled_data)

    print(f"\n  Path deltas shape: {path_deltas.shape}")
    print(f"  (Each delta represents the relative path step q[i]⁻¹ * q[i+1])")
    print(f"\n  First 3 path deltas:")
    for i in range(min(3, len(path_deltas))):
        print(f"    δ[{i}] = {path_deltas[i]}")

    return path_deltas


def prepare_data(df, model_type="lstm"):
    """Run the full data preparation pipeline."""
    use_dual = config.DUAL_STREAM

    print("\n" + "=" * 60)
    print(f"  Step 4: Preparing Training Sequences ({'dual-stream' if use_dual else 'single-stream'})")
    print("=" * 60)

    if model_type == "extended_mtl":
        data = extended_encoder.prepare_extended_training_data(
            df,
            sequence_length=config.SEQUENCE_LENGTH,
            train_split=config.TRAIN_TEST_SPLIT,
            volume_ma_window=config.VOLUME_MA_WINDOW if use_dual else 20,
        )
        info = {
            "sequence_length": config.SEQUENCE_LENGTH,
            "total_samples": len(data["X_train"][0]) + len(data["X_test"][0]),
            "train_samples": len(data["X_train"][0]),
            "test_samples": len(data["X_test"][0]),
        }
        data["encoding_info"] = info
    else:
        data = prepare_training_data(
            df,
            sequence_length=config.SEQUENCE_LENGTH,
            train_split=config.TRAIN_TEST_SPLIT,
            use_path_deltas=False,
            dual_stream=use_dual,
            volume_ma_window=config.VOLUME_MA_WINDOW if use_dual else 20,
        )

    info = data["encoding_info"]
    print(f"\n  Sequence length:  {info['sequence_length']}")
    print(f"  Dual stream:      {info.get('dual_stream', False)}")
    print(f"  Total sequences:  {info['total_samples']}")
    print(f"  Training samples: {info['train_samples']}")
    print(f"  Test samples:     {info['test_samples']}")
    
    if model_type == "extended_mtl":
        print(f"  X_train shape:    {data['X_train'][0].shape}")
        print(f"  y_train shape:    {data['y_train']['out_main'].shape}")
        print(f"  X_test shape:     {data['X_test'][0].shape}")
        print(f"  y_test shape:     {data['y_test']['out_main'].shape}")
    else:
        print(f"  X_train shape:    {data['X_train'].shape}")
        print(f"  y_train shape:    {data['y_train'].shape}")
        print(f"  X_test shape:     {data['X_test'].shape}")
        print(f"  y_test shape:     {data['y_test'].shape}")

    if use_dual and model_type != "extended_mtl":
        print(f"  ctx_X_train shape: {data['ctx_X_train'].shape}")
        print(f"  ctx_X_test shape:  {data['ctx_X_test'].shape}")

    return data


def build_model(model_type="lstm"):
    """Build the LSTM model and display its summary."""
    use_dual = config.DUAL_STREAM

    print("\n" + "=" * 60)
    print(f"  Step 5: Building {'Dual-Stream' if use_dual else 'Single-Stream'} {model_type.upper()} Model")
    print("=" * 60)

    predictor = build_primary_model(model_type)

    print(f"\n  Model Summary:")
    print(f"  {'-' * 50}")
    summary = predictor.get_summary()
    for line in summary.strip().split("\n"):
        print(f"  {line}")

    return predictor


def run_training(predictor, data):
    """Train the model on prepared data."""
    use_dual = config.DUAL_STREAM

    print("\n" + "=" * 60)
    print("  Step 6: Training")
    print("=" * 60)

    # Pack inputs for dual-stream or extended_mtl
    if isinstance(data["X_train"], list):
        X_train = data["X_train"]
        X_test = data["X_test"]
    elif use_dual and "ctx_X_train" in data:
        X_train = [data["X_train"], data["ctx_X_train"]]
        X_test = [data["X_test"], data["ctx_X_test"]]
    else:
        X_train = data["X_train"]
        X_test = data["X_test"]

    history = predictor.train(
        X_train,
        data["y_train"],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
    )

    # Evaluate on test set
    metrics = predictor.evaluate(X_test, data["y_test"])
    print(f"\n  Test Loss (MSE): {metrics['loss']:.6f}")
    if "main_loss" in metrics:
        print(f"  Test Main Loss:  {metrics['main_loss']:.6f}")
    print(f"  Test MAE:        {metrics['mae']:.6f}")

    return history, metrics


def run_analysis(predictor, data, df):
    """Run simulations and evaluate the model."""
    use_dual = config.DUAL_STREAM

    print("\n" + "=" * 60)
    print("  Step 7: Model Analysis & Simulation")
    print("=" * 60)

    scaler = data["scaler"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    ctx_X_test = data.get("ctx_X_test") if use_dual else None

    # 1. Get actual prices for test set
    if isinstance(y_test, dict):
        actual_prices = decode_quaternion_to_price(y_test["out_main"], scaler)
    else:
        actual_prices = decode_quaternion_to_price(y_test, scaler)

    # 2. Teacher-forcing simulation
    preds_tf = simulate_teacher_forcing(
        predictor, scaler, X_test, ctx_X_test=ctx_X_test,
    )

    # 3. Pure auto-regressive simulation
    if isinstance(X_test, list) and len(X_test) == 3:
        initial_seq = [
            X_test[0][0:1],
            X_test[1][0:1],
            X_test[2][0:1]
        ]
        initial_ctx = None
    else:
        # Use the first sequence of X_test as the starting point
        initial_seq = X_test[0:1]  # shape (1, seq_len, 4)
        initial_ctx = ctx_X_test[0:1] if ctx_X_test is not None else None
        
    preds_ar = simulate_autoregressive(
        predictor, scaler, initial_seq, n_steps=len(X_test[0]) if isinstance(X_test, list) else len(X_test),
        initial_context=initial_ctx,
    )

    # 4. Error Analysis
    print("\n  Teacher-Forcing Errors:")
    metrics_tf = analyze_errors(actual_prices, preds_tf)
    for k, v in metrics_tf.items():
        print(f"    {k}: {v:.4f}")

    print("\n  Auto-Regressive Errors:")
    metrics_ar = analyze_errors(actual_prices, preds_ar)
    for k, v in metrics_ar.items():
        print(f"    {k}: {v:.4f}")

    # 5. Plot results
    # Get the corresponding dates for the test set
    n_test_samples = len(X_test[0]) if isinstance(X_test, list) else len(X_test)
    test_dates = df.index[-n_test_samples:]

    plot_results(test_dates, actual_prices, preds_ar, preds_tf)
    print(f"\n  Visualizations saved to '{config.VISUALIZATION_DIR}/'")


def train_custom_model(epochs: int, model_type: str = "lstm", save_path: str = "saved_models/custom_model.keras"):
    """Train the model for a user-specified number of epochs and save it."""
    setup_logging()
    print("\n" + "=" * 60)
    print(f"  Training Custom Model for {epochs} Epochs")
    print("=" * 60)

    # Fetch and prepare data
    df = fetch_data()
    data = prepare_data(df, model_type)

    # Build model
    predictor = build_model(model_type=model_type)

    # Train
    print("\n  Starting training...")

    use_dual = config.DUAL_STREAM
    if use_dual and "ctx_X_train" in data:
        X_train = [data["X_train"], data["ctx_X_train"]]
    else:
        X_train = data["X_train"]

    predictor.train(
        X_train,
        data["y_train"],
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        save_best=True,
        model_path=save_path
    )

    # Force save the final epoch model as well
    predictor.save_model(save_path.replace('.keras', '_final.keras'))

    # Run analysis
    run_analysis(predictor, data, df)

    return predictor


def train_oscillator_model(primary_predictor, data, ohlcv_df):
    """Train the complementary Residual Oscillator model."""
    print("\n" + "=" * 60)
    print("  PHASE 4: Complementary Oscillator Training")
    print("=" * 60)
def train_oscillator_model(primary_predictor, data, ohlcv_df, osc_arch_type="residual"):
    """Train the complementary residual oscillator."""
    print("\n" + "=" * 60)
    print("  Step 5: Training Complementary Oscillator")
    print("=" * 60)
    
    # 1. Compute residuals over the entire dataset
    # The oscillator learns to predict the error of the primary model
    if isinstance(data["X_train"], list) and len(data["X_train"]) == 3:
        # Extended MTL uses a list of 3 inputs
        X_all = [
            np.concatenate([data["X_train"][0], data["X_test"][0]], axis=0),
            np.concatenate([data["X_train"][1], data["X_test"][1]], axis=0),
            np.concatenate([data["X_train"][2], data["X_test"][2]], axis=0)
        ]
        y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
        ctx_X_all = None
        is_extended = True
    else:
        X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
        if isinstance(data["y_train"], dict):
            y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
        else:
            y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
        ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0) if config.DUAL_STREAM else None
        is_extended = False
    
    residuals, pred_q = compute_residuals(primary_predictor, X_all, y_all, data["scaler"], ctx_X_test=ctx_X_all)
    
    # 2. Prepare sequences of residuals
    osc_data = prepare_oscillator_data(
        residuals, pred_q,
        sequence_length=config.OSCILLATOR_SEQ_LEN,
        train_split=config.TRAIN_TEST_SPLIT,
        oscillator_type=osc_arch_type,
    )
    
    # 3. Build and train oscillator
    if is_extended and osc_arch_type == "residual":
        oscillator = FeedbackOscillator(
            sequence_length=config.OSCILLATOR_SEQ_LEN,
            lstm_units=config.OSCILLATOR_LSTM_UNITS,
            dense_units=config.OSCILLATOR_DENSE_UNITS,
            learning_rate=config.OSCILLATOR_LEARNING_RATE
        )
    elif osc_arch_type == "classification":
        from signal_model import ClassificationOscillator
        oscillator = ClassificationOscillator(
            sequence_length=config.OSCILLATOR_SEQ_LEN,
            lstm_units=config.OSCILLATOR_LSTM_UNITS,
            dense_units=config.OSCILLATOR_DENSE_UNITS,
            learning_rate=config.OSCILLATOR_LEARNING_RATE
        )
    elif osc_arch_type == "threshold":
        from signal_model import ThresholdOscillator
        oscillator = ThresholdOscillator(
            sequence_length=config.OSCILLATOR_SEQ_LEN,
            lstm_units=config.OSCILLATOR_LSTM_UNITS,
            dense_units=config.OSCILLATOR_DENSE_UNITS,
            learning_rate=config.OSCILLATOR_LEARNING_RATE
        )
    else:
        oscillator = ResidualOscillator(
            sequence_length=config.OSCILLATOR_SEQ_LEN,
            lstm_units=config.OSCILLATOR_LSTM_UNITS,
            dense_units=config.OSCILLATOR_DENSE_UNITS,
            learning_rate=config.OSCILLATOR_LEARNING_RATE
        )
    
    oscillator.build_model()
    print("\nResidual Oscillator Summary:")
    print(oscillator.get_summary())
    
    oscillator.train(
        osc_data,
        epochs=config.OSCILLATOR_EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        save_best=True
    )
    
    # 4. Evaluate
    print("\nEvaluating Oscillator...")
    oscillator.evaluate(osc_data)
    
    # 5. Generate Signals and Plot
    test_signals = oscillator.predict(osc_data["X_res_test"], osc_data["X_q_test"])
    
    # Extract the test dates and actual prices to align with the signals
    # The oscillator test data aligns with the END of the dataset
    test_dates = ohlcv_df.index[-len(test_signals):]
    actual_test_prices = ohlcv_df['Close'].values[-len(test_signals):]
    predicted_test_prices = decode_quaternion_to_price(osc_data["X_q_test"], data["scaler"])
    
    plot_oscillator_signals(
        test_dates, 
        actual_test_prices,
        predicted_test_prices,
        test_signals, 
        title_suffix="(Test Set)"
    )
    
    # 6. Predict the next unknown signal
    next_signal = get_next_oscillator_signal(primary_predictor, oscillator, data, ohlcv_df)
    print(f"\n[!] FUTURE DIVERGENCE MOMENTUM SIGNAL for {ohlcv_df.index[-1] + pd.Timedelta(days=1)}: {next_signal:.4f}")
    if next_signal > 0.5:
        print("    -> Indicates STRONG UPWARD DIVERGENCE (Price breaking away up)")
    elif next_signal < -0.5:
        print("    -> Indicates STRONG DOWNWARD DIVERGENCE (Price breaking away down)")
    elif abs(next_signal) < 0.1:
        print("    -> Indicates POTENTIAL INFLECTION POINT (Momentum flipping - Reversal Imminent)")
    else:
        print("    -> Indicates WEAK/NEUTRAL MOMENTUM")

    return oscillator


def get_next_oscillator_signal(primary_predictor, oscillator, data, df):
    """Calculate the oscillator signal for the immediate unseen future step."""
    # To predict tomorrow's signal, we need:
    # 1. The last OSCILLATOR_SEQ_LEN residuals
    # 2. The primary model's prediction for tomorrow
    
    # Run the last available window through the primary model
    if isinstance(data["X_test"], list) and len(data["X_test"]) == 3:
        model_input = [
            data["X_test"][0][-1:],
            data["X_test"][1][-1:],
            data["X_test"][2][-1:]
        ]
    else:
        last_x = data["X_test"][-1:] # Shape (1, 60, 4)
        last_ctx = data["ctx_X_test"][-1:] if config.DUAL_STREAM else None
        model_input = [last_x, last_ctx] if last_ctx is not None else last_x
        
    next_q = primary_predictor.predict(model_input) # Shape (1, 4)
    
    # Get recent residuals
    # We'll just grab the most recent residuals from the training/test set
    # Using the test set residuals we already computed
    if isinstance(data["X_train"], list) and len(data["X_train"]) == 3:
        X_all = [
            np.concatenate([data["X_train"][0], data["X_test"][0]], axis=0),
            np.concatenate([data["X_train"][1], data["X_test"][1]], axis=0),
            np.concatenate([data["X_train"][2], data["X_test"][2]], axis=0)
        ]
        y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
        ctx_X_all = None
    else:
        X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
        if isinstance(data["y_train"], dict):
            y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
        else:
            y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
        ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0) if config.DUAL_STREAM else None
    
    residuals, _ = compute_residuals(primary_predictor, X_all, y_all, data["scaler"], ctx_X_test=ctx_X_all)
    
    recent_residuals = residuals[-config.OSCILLATOR_SEQ_LEN:]
    recent_residuals = recent_residuals.reshape(1, config.OSCILLATOR_SEQ_LEN, 1).astype(np.float32)
    
    next_signal = oscillator.predict(recent_residuals, next_q)
    return next_signal[0, 0]


def main(train: bool = TRAIN_ON_RUN, epochs: int = config.EPOCHS, model_type: str = "lstm"):
    """Run the full Quaternion LSTM Price Predictor pipeline."""
    setup_logging()

    mode = "Dual-Stream" if config.DUAL_STREAM else "Single-Stream"
    fusion = f" ({config.FUSION_STRATEGY.upper()} fusion)" if config.DUAL_STREAM else ""

    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  Quaternion Path-Based LSTM Price Predictor              ║")
    print(f"║  {mode}{fusion}{'  ' * ((48 - len(mode) - len(fusion)) // 2)}║")
    print("╚" + "═" * 58 + "╝")

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    # --- Pipeline ---
    # 1. Fetch data
    ohlcv_df = fetch_data()

    # 2. Encode to quaternions (demo)
    scaled_data, scaler = encode_data(ohlcv_df)

    # 3. Compute path deltas (demo)
    path_deltas = compute_path(scaled_data)

    # 4. Prepare training sequences
    data = prepare_data(ohlcv_df, model_type)

    # 5. Build model
    predictor = build_model(model_type=model_type)

    # 6. Train (optional)
    if train:
        # Override config epochs if a custom amount is passed
        original_epochs = config.EPOCHS
        config.EPOCHS = epochs

        history, metrics = run_training(predictor, data)
        run_analysis(predictor, data, ohlcv_df)
        
        # Train complementary model
        train_oscillator_model(predictor, data, ohlcv_df)

        config.EPOCHS = original_epochs
    else:
        print("\n" + "=" * 60)
        print("  Training and Analysis skipped")
        print("  Run with --train to execute the training loop")
        print("=" * 60)

    # --- Summary ---
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  Pipeline Complete ✓                                     ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Data points:       {len(ohlcv_df)}")
    print(f"  Quaternion shape:  {scaled_data.shape}")
    print(f"  Path deltas:       {path_deltas.shape}")
    print(f"  Training samples:  {data['encoding_info']['train_samples']}")
    print(f"  Test samples:      {data['encoding_info']['test_samples']}")
    print(f"  Model parameters:  {predictor.model.count_params():,}")
    if getattr(predictor, "dual_stream", False):
        print(f"  Architecture:      Dual-Stream ({config.FUSION_STRATEGY.upper()} fusion)")
        print(f"  Context dropout:   {config.CONTEXT_DROPOUT_RATE:.0%}")
    elif model_type == "mtl":
        print(f"  Architecture:      Multi-Task Learning (MTL)")
    elif model_type == "extended_mtl":
        print(f"  Architecture:      Extended Multi-Task Learning (Momentum-Rotation)")
    else:
        print(f"  Architecture:      Single-Stream LSTM")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quaternion LSTM Price Predictor")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs to train")
    parser.add_argument("--custom_train", action="store_true", help="Run the custom interactive training function")
    parser.add_argument("--model", type=str, choices=["lstm", "mtl", "extended_mtl"], default="lstm", help="Choose model architecture to use")
    
    parser.add_argument("--train-oscillator-only", action="store_true", help="Train only the oscillator on top of an existing primary model")
    parser.add_argument("--primary-model", type=str, help="Filename of the primary model to attach the oscillator to")
    parser.add_argument("--oscillator-type", type=str, choices=["residual", "classification", "threshold"], default="residual", help="Type of oscillator to train")

    args = parser.parse_args()

    if args.train_oscillator_only:
        if not args.primary_model:
            print("Error: --primary-model must be provided when using --train-oscillator-only")
            sys.exit(1)
            
        import os
        import tensorflow as tf
        model_path = os.path.join(config.MODEL_SAVE_DIR, args.primary_model)
        if not os.path.exists(model_path):
            print(f"Error: Primary model not found at {model_path}")
            sys.exit(1)
            
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        
        if temp_model.name == "MTLQuaternionPredictor":
            arch_type = "mtl"
        elif temp_model.name == "ExtendedMTLPredictor":
            arch_type = "extended_mtl"
        else:
            arch_type = "lstm"
            
        primary_predictor = build_primary_model(arch_type)
        primary_predictor.model = temp_model
        
        ohlcv_df = fetch_data()
        scaled_data, scaler = encode_data(ohlcv_df)
        data = prepare_data(ohlcv_df, arch_type)
        
        # Override config.OSCILLATOR_TYPE so it trains the requested one
        # Note: train_oscillator_model doesn't take oscillator_type currently, so we'll need to patch it or just set config
        # Actually wait, I'll update train_oscillator_model signature to accept osc_type
        train_oscillator_model(primary_predictor, data, ohlcv_df, osc_arch_type=args.oscillator_type)
        print("\nStandalone oscillator training completed.")
        
    elif args.custom_train:
        # Run the standalone function requested by the user
        train_custom_model(epochs=args.epochs, model_type=args.model)
    else:
        # Use CLI args to override defaults if provided, else use TRAIN_ON_RUN
        should_train = args.train or TRAIN_ON_RUN
        main(train=should_train, epochs=args.epochs, model_type=args.model)
