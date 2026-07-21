"""
FastAPI Web Server for the Quaternion LSTM Price Predictor Dashboard.

Provides REST endpoints for:
  - Running inference and returning prediction data
  - Background training with real-time progress polling
  - Full configuration management (read/write config.py)
  - Model file management (upload, download, list, activate, delete)
  - Yahoo Finance ticker autocomplete
"""

import os
import re
import json
import threading
import time
import logging
import importlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Project modules
import config
from data_fetcher import fetch_bitcoin_data, get_ohlcv
from quaternion_encoder import (
    encode_dataframe,
    decode_quaternion_to_price,
    compute_quaternion_path,
    prepare_training_data,
    compute_residuals,
    prepare_oscillator_data,
)
from lstm_model import QuaternionLSTMPredictor
from mtl_model import MultiTaskQuaternionPredictor
from signal_model import ResidualOscillator
from model_analysis import (
    simulate_autoregressive,
    simulate_teacher_forcing,
    analyze_errors,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global training state (shared with polling endpoint)
# ──────────────────────────────────────────────
TRAINING_STATE = {
    "is_training": False,
    "phase": "idle",           # idle | primary | oscillator | complete | error
    "epoch": 0,
    "total_epochs": 0,
    "history": {
        "loss": [],
        "val_loss": [],
        "mae": [],
        "val_mae": [],
    },
    "logs": [],
    "progress_log": "",
    "error": None,
}

# Track which models are active
ACTIVE_MODELS = {
    "primary": "best_model.keras",
    "oscillator": "oscillator_model.keras",
}


# ──────────────────────────────────────────────
# Keras Callback for real-time training progress
# ──────────────────────────────────────────────
class TrainingStateCallback(tf.keras.callbacks.Callback):
    """Writes training metrics to TRAINING_STATE for real-time UI polling."""

    def __init__(self, phase: str = "primary"):
        super().__init__()
        self.phase = phase

    def on_train_begin(self, logs=None):
        TRAINING_STATE["phase"] = self.phase
        TRAINING_STATE["is_training"] = True
        TRAINING_STATE["error"] = None
        if self.phase == "primary":
            # Reset history only for primary phase
            TRAINING_STATE["history"] = {
                "loss": [], "val_loss": [], "mae": [], "val_mae": [],
            }
            TRAINING_STATE["logs"] = []

    def on_epoch_begin(self, epoch, logs=None):
        TRAINING_STATE["epoch"] = epoch + 1
        total = self.params.get("epochs", 0)
        TRAINING_STATE["total_epochs"] = total
        TRAINING_STATE["logs"].append(
            f"[{self.phase.upper()}] Epoch {epoch + 1}/{total} starting..."
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        mae = logs.get("mae", 0)
        val_mae = logs.get("val_mae", 0)

        TRAINING_STATE["history"]["loss"].append(round(float(loss), 6))
        TRAINING_STATE["history"]["val_loss"].append(round(float(val_loss), 6))
        TRAINING_STATE["history"]["mae"].append(round(float(mae), 6))
        TRAINING_STATE["history"]["val_mae"].append(round(float(val_mae), 6))

        total = self.params.get("epochs", 0)
        pct = ((epoch + 1) / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * (epoch + 1) / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        TRAINING_STATE["progress_log"] = f"[{bar}] {pct:.0f}%"

        TRAINING_STATE["logs"].append(
            f"[{self.phase.upper()}] Epoch {epoch + 1}/{total} — "
            f"loss: {loss:.6f}, val_loss: {val_loss:.6f}, "
            f"mae: {mae:.6f}, val_mae: {val_mae:.6f}"
        )

    def on_train_end(self, logs=None):
        TRAINING_STATE["logs"].append(
            f"[{self.phase.upper()}] Training complete."
        )


# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────
app = FastAPI(title="Quaternion AI Predictor", version="1.0.0")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Config Endpoints ──────────────────────────

class ConfigUpdate(BaseModel):
    """Accepts a partial or full config update."""
    # Data source
    ticker_primary: Optional[str] = None
    ticker_fallback: Optional[str] = None
    default_start_date: Optional[str] = None
    default_interval: Optional[str] = None
    # Sequence
    sequence_length: Optional[int] = None
    # Dual-stream
    dual_stream: Optional[bool] = None
    context_lstm_units: Optional[int] = None
    volume_ma_window: Optional[int] = None
    fusion_strategy: Optional[str] = None
    fusion_dense_units: Optional[int] = None
    context_dropout_rate: Optional[float] = None
    # Model architecture
    model_type: Optional[str] = None
    lstm_units: Optional[int] = None
    dense_units: Optional[int] = None
    dropout_rate: Optional[float] = None
    # Training
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    validation_split: Optional[float] = None
    train_test_split: Optional[float] = None
    # Oscillator
    oscillator_seq_len: Optional[int] = None
    oscillator_lstm_units: Optional[int] = None
    oscillator_dense_units: Optional[int] = None
    oscillator_epochs: Optional[int] = None
    oscillator_learning_rate: Optional[float] = None
    mean_center_oscillator: Optional[bool] = None
    # Reproducibility
    random_seed: Optional[int] = None


# Map ConfigUpdate field names → config.py variable names
_CONFIG_FIELD_MAP = {
    "ticker_primary": "TICKER_PRIMARY",
    "ticker_fallback": "TICKER_FALLBACK",
    "default_start_date": "DEFAULT_START_DATE",
    "default_interval": "DEFAULT_INTERVAL",
    "sequence_length": "SEQUENCE_LENGTH",
    "dual_stream": "DUAL_STREAM",
    "context_lstm_units": "CONTEXT_LSTM_UNITS",
    "volume_ma_window": "VOLUME_MA_WINDOW",
    "fusion_strategy": "FUSION_STRATEGY",
    "fusion_dense_units": "FUSION_DENSE_UNITS",
    "context_dropout_rate": "CONTEXT_DROPOUT_RATE",
    "model_type": "MODEL_TYPE",
    "lstm_units": "LSTM_UNITS",
    "dense_units": "DENSE_UNITS",
    "dropout_rate": "DROPOUT_RATE",
    "learning_rate": "LEARNING_RATE",
    "batch_size": "BATCH_SIZE",
    "epochs": "EPOCHS",
    "validation_split": "VALIDATION_SPLIT",
    "train_test_split": "TRAIN_TEST_SPLIT",
    "oscillator_seq_len": "OSCILLATOR_SEQ_LEN",
    "oscillator_lstm_units": "OSCILLATOR_LSTM_UNITS",
    "oscillator_dense_units": "OSCILLATOR_DENSE_UNITS",
    "oscillator_epochs": "OSCILLATOR_EPOCHS",
    "oscillator_learning_rate": "OSCILLATOR_LEARNING_RATE",
    "mean_center_oscillator": "MEAN_CENTER_OSCILLATOR",
    "random_seed": "RANDOM_SEED",
}


@app.get("/api/config")
async def get_config():
    """Return all configuration parameters grouped by category."""
    return {
        "data_source": {
            "ticker_primary": config.TICKER_PRIMARY,
            "ticker_fallback": config.TICKER_FALLBACK,
            "default_start_date": config.DEFAULT_START_DATE,
            "default_interval": config.DEFAULT_INTERVAL,
        },
        "sequence": {
            "sequence_length": config.SEQUENCE_LENGTH,
            "n_features": config.N_FEATURES,
        },
        "dual_stream": {
            "dual_stream": config.DUAL_STREAM,
            "n_context_features": config.N_CONTEXT_FEATURES,
            "context_lstm_units": config.CONTEXT_LSTM_UNITS,
            "volume_ma_window": config.VOLUME_MA_WINDOW,
            "fusion_strategy": config.FUSION_STRATEGY,
            "fusion_dense_units": config.FUSION_DENSE_UNITS,
            "context_dropout_rate": config.CONTEXT_DROPOUT_RATE,
        },
        "model_architecture": {
            "model_type": config.MODEL_TYPE,
            "lstm_units": config.LSTM_UNITS,
            "dense_units": config.DENSE_UNITS,
            "dropout_rate": config.DROPOUT_RATE,
        },
        "training": {
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "validation_split": config.VALIDATION_SPLIT,
            "train_test_split": config.TRAIN_TEST_SPLIT,
        },
        "oscillator": {
            "oscillator_seq_len": config.OSCILLATOR_SEQ_LEN,
            "oscillator_lstm_units": config.OSCILLATOR_LSTM_UNITS,
            "oscillator_dense_units": config.OSCILLATOR_DENSE_UNITS,
            "oscillator_epochs": config.OSCILLATOR_EPOCHS,
            "oscillator_learning_rate": config.OSCILLATOR_LEARNING_RATE,
            "mean_center_oscillator": config.MEAN_CENTER_OSCILLATOR,
        },
        "reproducibility": {
            "random_seed": config.RANDOM_SEED,
        },
        "active_models": ACTIVE_MODELS,
    }


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update config values in-memory and persist to config.py on disk."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    updates = update.model_dump(exclude_none=True)
    if not updates:
        return {"status": "no changes"}

    for field_name, value in updates.items():
        config_var = _CONFIG_FIELD_MAP.get(field_name)
        if not config_var:
            continue

        # Update in-memory
        setattr(config, config_var, value)

        # Update on disk — match the variable assignment line
        if isinstance(value, str):
            replacement = f'{config_var}: str = "{value}"'
            pattern = rf'^{config_var}:\s*str\s*=\s*".*?"'
        elif isinstance(value, bool):
            replacement = f"{config_var}: bool = {value}"
            pattern = rf"^{config_var}:\s*bool\s*=\s*(True|False)"
        elif isinstance(value, int):
            replacement = f"{config_var}: int = {value}"
            pattern = rf"^{config_var}:\s*int\s*=\s*\d+"
        elif isinstance(value, float):
            replacement = f"{config_var}: float = {value}"
            pattern = rf"^{config_var}:\s*float\s*=\s*[\d.]+"
        else:
            continue

        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "ok", "updated": list(updates.keys())}


# ── Ticker Search ─────────────────────────────

@app.get("/api/search")
async def search_ticker(q: str = Query(..., min_length=1)):
    """Yahoo Finance ticker autocomplete."""
    try:
        import yfinance as yf
        search = yf.Search(q, max_results=6)
        quotes = []
        for quote in (search.quotes or []):
            quotes.append({
                "symbol": quote.get("symbol", ""),
                "shortname": quote.get("shortname", quote.get("longname", "")),
                "exchange": quote.get("exchange", ""),
                "type": quote.get("quoteType", ""),
            })
        return {"results": quotes[:5]}
    except Exception as e:
        logger.warning("Ticker search failed: %s", e)
        return {"results": [], "error": str(e)}


# ── Inference / Data Endpoint ─────────────────

@app.get("/api/data")
async def get_prediction_data():
    """Run full inference pipeline and return prediction data for charts."""
    try:
        # 1. Fetch OHLCV data
        ohlcv_df = get_ohlcv(fetch_bitcoin_data(
            ticker_primary=config.TICKER_PRIMARY,
            ticker_fallback=config.TICKER_FALLBACK,
            start=config.DEFAULT_START_DATE,
            interval=config.DEFAULT_INTERVAL,
        ))

        # 2. Prepare data
        data = prepare_training_data(
            ohlcv_df,
            sequence_length=config.SEQUENCE_LENGTH,
            train_split=config.TRAIN_TEST_SPLIT,
            use_path_deltas=False,
            dual_stream=config.DUAL_STREAM,
            volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20,
        )

        # 3. Build and load primary model
        if config.MODEL_TYPE == "mtl":
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

        model_path = os.path.join(config.MODEL_SAVE_DIR, ACTIVE_MODELS["primary"])
        if os.path.exists(model_path):
            predictor.load_model(model_path)
        else:
            return JSONResponse(status_code=404, content={
                "error": f"Primary model not found: {ACTIVE_MODELS['primary']}. Train or upload a model first."
            })

        scaler = data["scaler"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        ctx_X_test = data.get("ctx_X_test") if config.DUAL_STREAM else None

        # 4. Get actual prices
        actual_prices = decode_quaternion_to_price(y_test, scaler)

        # 5. Teacher-forcing predictions
        predicted_prices = simulate_teacher_forcing(
            predictor, scaler, X_test, ctx_X_test=ctx_X_test,
        )

        # 6. Error metrics
        metrics = analyze_errors(actual_prices, predicted_prices)

        # 7. Oscillator signals
        signals = []
        next_signal = 0.0
        osc_path = os.path.join(config.MODEL_SAVE_DIR, ACTIVE_MODELS["oscillator"])
        if os.path.exists(osc_path):
            try:
                X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
                y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
                ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0) if config.DUAL_STREAM else None

                residuals, pred_q = compute_residuals(
                    predictor, X_all, y_all, scaler, ctx_X_test=ctx_X_all
                )

                osc_data = prepare_oscillator_data(
                    residuals, pred_q,
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    train_split=config.TRAIN_TEST_SPLIT,
                )

                oscillator = ResidualOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )
                oscillator.build_model()
                oscillator.model.load_weights(osc_path)

                test_signals = oscillator.predict(
                    osc_data["X_res_test"], osc_data["X_q_test"]
                )
                signals = test_signals.flatten().tolist()

                # Next signal
                last_x = data["X_test"][-1:]
                last_ctx = data["ctx_X_test"][-1:] if config.DUAL_STREAM else None
                model_input = [last_x, last_ctx] if last_ctx is not None else last_x
                next_q = predictor.predict(model_input)

                recent_residuals = residuals[-config.OSCILLATOR_SEQ_LEN:]
                recent_residuals = recent_residuals.reshape(1, config.OSCILLATOR_SEQ_LEN, 1).astype(np.float32)
                next_signal = float(oscillator.predict(recent_residuals, next_q)[0, 0])
            except Exception as e:
                logger.warning("Oscillator inference failed: %s", e)
                signals = []
                next_signal = 0.0

        # 8. Prepare dates
        test_dates = ohlcv_df.index[-len(actual_prices):]
        dates = [d.strftime("%Y-%m-%d %H:%M") if hasattr(d, "strftime") else str(d)
                 for d in test_dates]

        return {
            "ticker": config.TICKER_PRIMARY,
            "interval": config.DEFAULT_INTERVAL,
            "dates": dates,
            "actual_prices": [round(float(p), 4) for p in actual_prices],
            "predicted_prices": [round(float(p), 4) for p in predicted_prices],
            "signals": [round(float(s), 4) for s in signals],
            "next_signal": round(float(next_signal), 4),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "data_points": len(ohlcv_df),
            "test_points": len(actual_prices),
        }

    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Training Endpoints ────────────────────────

@app.post("/api/train")
async def start_training(
    epochs: Optional[int] = None,
    oscillator_epochs: Optional[int] = None,
    model_name: Optional[str] = Query("best_model", description="Base name for saved models"),
):
    """Start background training for both primary and oscillator models."""
    if TRAINING_STATE["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    train_epochs = epochs or config.EPOCHS
    osc_epochs = oscillator_epochs or config.OSCILLATOR_EPOCHS

    def _train_background():
        try:
            TRAINING_STATE["is_training"] = True
            TRAINING_STATE["phase"] = "primary"
            TRAINING_STATE["epoch"] = 0
            TRAINING_STATE["total_epochs"] = train_epochs
            TRAINING_STATE["history"] = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
            TRAINING_STATE["logs"] = [f"[SYSTEM] Starting training pipeline..."]
            TRAINING_STATE["progress_log"] = ""
            TRAINING_STATE["error"] = None

            # 1. Fetch data
            TRAINING_STATE["logs"].append("[SYSTEM] Fetching market data...")
            ohlcv_df = get_ohlcv(fetch_bitcoin_data(
                ticker_primary=config.TICKER_PRIMARY,
                ticker_fallback=config.TICKER_FALLBACK,
                start=config.DEFAULT_START_DATE,
                interval=config.DEFAULT_INTERVAL,
            ))
            TRAINING_STATE["logs"].append(f"[SYSTEM] Data loaded: {len(ohlcv_df)} data points")

            # 2. Prepare training data
            TRAINING_STATE["logs"].append("[SYSTEM] Preparing training sequences...")
            data = prepare_training_data(
                ohlcv_df,
                sequence_length=config.SEQUENCE_LENGTH,
                train_split=config.TRAIN_TEST_SPLIT,
                use_path_deltas=False,
                dual_stream=config.DUAL_STREAM,
                volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20,
            )
            info = data["encoding_info"]
            TRAINING_STATE["logs"].append(
                f"[SYSTEM] Sequences ready — Train: {info['train_samples']}, Test: {info['test_samples']}"
            )

            # 3. Build primary model
            TRAINING_STATE["logs"].append("[SYSTEM] Building primary model...")
            if config.MODEL_TYPE == "mtl":
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
            params = predictor.model.count_params()
            TRAINING_STATE["logs"].append(f"[SYSTEM] Model built — {params:,} parameters")

            # 4. Train primary model
            use_dual = config.DUAL_STREAM
            if use_dual and "ctx_X_train" in data:
                X_train = [data["X_train"], data["ctx_X_train"]]
                X_test = [data["X_test"], data["ctx_X_test"]]
            else:
                X_train = data["X_train"]
                X_test = data["X_test"]

            TRAINING_STATE["logs"].append(f"[PRIMARY] Starting training — {train_epochs} epochs")
            primary_cb = TrainingStateCallback(phase="primary")

            save_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.keras")
            predictor.train(
                X_train, data["y_train"],
                epochs=train_epochs,
                batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT,
                save_best=True,
                model_path=save_path,
                callbacks=[primary_cb],
            )

            # Evaluate primary
            metrics = predictor.evaluate(X_test, data["y_test"])
            TRAINING_STATE["logs"].append(
                f"[PRIMARY] Test Loss: {metrics['loss']:.6f}, Test MAE: {metrics['mae']:.6f}"
            )

            # 5. Train oscillator
            TRAINING_STATE["phase"] = "oscillator"
            TRAINING_STATE["epoch"] = 0
            TRAINING_STATE["total_epochs"] = osc_epochs
            TRAINING_STATE["logs"].append("[OSCILLATOR] Computing residuals...")

            X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
            y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
            ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0) if use_dual else None

            residuals, pred_q = compute_residuals(
                predictor, X_all, y_all, data["scaler"], ctx_X_test=ctx_X_all
            )

            osc_data = prepare_oscillator_data(
                residuals, pred_q,
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                train_split=config.TRAIN_TEST_SPLIT,
            )

            oscillator = ResidualOscillator(
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                lstm_units=config.OSCILLATOR_LSTM_UNITS,
                dense_units=config.OSCILLATOR_DENSE_UNITS,
                learning_rate=config.OSCILLATOR_LEARNING_RATE,
            )
            oscillator.build_model()

            TRAINING_STATE["logs"].append(f"[OSCILLATOR] Starting training — {osc_epochs} epochs")
            osc_cb = TrainingStateCallback(phase="oscillator")
            osc_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}_oscillator.keras")

            oscillator.train(
                osc_data,
                epochs=osc_epochs,
                batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT,
                save_best=True,
                model_path=osc_save_path,
                callbacks=[osc_cb],
            )

            osc_metrics = oscillator.evaluate(osc_data)
            TRAINING_STATE["logs"].append(
                f"[OSCILLATOR] Test Loss: {osc_metrics.get('loss', 0):.6f}"
            )

            # 6. Done
            TRAINING_STATE["phase"] = "complete"
            TRAINING_STATE["is_training"] = False
            TRAINING_STATE["logs"].append("[SYSTEM] ✓ Training pipeline complete!")

        except Exception as e:
            TRAINING_STATE["phase"] = "error"
            TRAINING_STATE["is_training"] = False
            TRAINING_STATE["error"] = str(e)
            TRAINING_STATE["logs"].append(f"[ERROR] {str(e)}")
            logger.exception("Background training failed")

    thread = threading.Thread(target=_train_background, daemon=True)
    thread.start()
    return {"status": "started", "epochs": train_epochs, "oscillator_epochs": osc_epochs}


@app.get("/api/training_status")
async def training_status():
    """Return current training state for real-time UI polling."""
    return TRAINING_STATE


# ── Model Management ──────────────────────────

@app.get("/api/models")
async def list_models():
    """List all saved model files."""
    models = []
    model_dir = config.MODEL_SAVE_DIR
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if fname.endswith(".keras"):
                fpath = os.path.join(model_dir, fname)
                stat = os.stat(fpath)
                models.append({
                    "filename": fname,
                    "size_bytes": stat.st_size,
                    "size_human": _human_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_active_primary": fname == ACTIVE_MODELS["primary"],
                    "is_active_oscillator": fname == ACTIVE_MODELS["oscillator"],
                })
    return {"models": models, "active": ACTIVE_MODELS}


@app.post("/api/upload_model")
async def upload_model(
    file: UploadFile = File(...),
    model_type: str = Query("primary", pattern="^(primary|oscillator)$"),
):
    """Upload a .keras model file."""
    if not file.filename.endswith(".keras"):
        raise HTTPException(status_code=400, detail="Only .keras files are accepted")

    dest = os.path.join(config.MODEL_SAVE_DIR, file.filename)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # Optionally set as active
    ACTIVE_MODELS[model_type] = file.filename

    return {
        "status": "ok",
        "filename": file.filename,
        "size_bytes": len(content),
        "model_type": model_type,
    }


@app.post("/api/activate_model")
async def activate_model(
    filename: str = Query(...),
    model_type: str = Query("primary", pattern="^(primary|oscillator)$"),
):
    """Set a saved model as the active primary or oscillator model."""
    fpath = os.path.join(config.MODEL_SAVE_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Model not found: {filename}")

    ACTIVE_MODELS[model_type] = filename
    return {"status": "ok", "model_type": model_type, "filename": filename}


@app.delete("/api/models/{filename}")
async def delete_model(filename: str):
    """Delete a saved model file."""
    fpath = os.path.join(config.MODEL_SAVE_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Model not found: {filename}")

    # Don't allow deleting the active model
    if filename == ACTIVE_MODELS["primary"] or filename == ACTIVE_MODELS["oscillator"]:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete active model '{filename}'. Activate a different model first."
        )

    os.remove(fpath)
    return {"status": "deleted", "filename": filename}


@app.get("/api/models/{filename}/download")
async def download_model(filename: str):
    """Download a saved model file."""
    fpath = os.path.join(config.MODEL_SAVE_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Model not found: {filename}")
    return FileResponse(fpath, filename=filename, media_type="application/octet-stream")


# ── Helpers ───────────────────────────────────

def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ──────────────────────────────────────────────
# Run with: uvicorn api:app --host 127.0.0.1 --port 8000
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
