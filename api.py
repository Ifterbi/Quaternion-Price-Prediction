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
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
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
from extended_encoder import prepare_extended_training_data
from oscillator_data import prepare_oscillator_training_data
from model_factory import build_primary_model
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

# Serve static files (React build)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web", "dist")
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

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
async def get_prediction_data(
    models: List[str] = Query(default=[]),
    oscillator: Optional[str] = Query(default=None),
    ticker: Optional[str] = Query(default=None)
):
    """Run full inference pipeline and return prediction data for charts."""
    try:
        # 1. Fetch OHLCV data
        ohlcv_df = get_ohlcv(fetch_bitcoin_data(
            ticker_primary=ticker if ticker else config.TICKER_PRIMARY,
            ticker_fallback=config.TICKER_FALLBACK,
            start=config.DEFAULT_START_DATE,
            interval=config.DEFAULT_INTERVAL,
        ))

        if not models:
            models = [ACTIVE_MODELS["primary"]]

        import tensorflow as tf
        from lstm_model import ContextDropout
        
        loaded_models = []
        min_seq_len = 9999
        
        # Phase 1: Load all models and determine sequence lengths
        for model_filename in models:
            model_path = os.path.join(config.MODEL_SAVE_DIR, model_filename)
            if not os.path.exists(model_path):
                logger.warning("Requested model not found: %s", model_filename)
                continue
                
            temp_model = tf.keras.models.load_model(
                model_path,
                custom_objects={"ContextDropout": ContextDropout}
            )
            
            # Safely infer expected sequence length from model inputs
            seq_len = config.SEQUENCE_LENGTH
            try:
                # e.g., shape is (None, 60, 4)
                input_shape = temp_model.inputs[0].shape
                if len(input_shape) >= 2 and input_shape[1] is not None:
                    seq_len = int(input_shape[1])
            except Exception as e:
                logger.warning("Could not infer sequence length for %s: %s", model_filename, e)
                
            min_seq_len = min(min_seq_len, seq_len)
            
            if temp_model.name == "MTLQuaternionPredictor":
                mod_type = "mtl"
            elif temp_model.name == "ExtendedMTLPredictor":
                mod_type = "extended_mtl"
            else:
                mod_type = "lstm"
                
            predictor = build_primary_model(mod_type)
            predictor.model = temp_model
            
            loaded_models.append({
                "filename": model_filename,
                "predictor": predictor,
                "seq_len": seq_len,
                "mod_type": mod_type
            })
            
        if not loaded_models:
            return JSONResponse(status_code=404, content={"error": "No valid models found to load."})

        # Phase 2: Prepare global reference data (using the minimum sequence length to get the maximum number of test points)
        reference_data = prepare_training_data(
            ohlcv_df,
            sequence_length=min_seq_len,
            train_split=config.TRAIN_TEST_SPLIT,
            use_path_deltas=False,
            dual_stream=True,
            volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20,
        )
        
        global_scaler = reference_data["scaler"]
        actual_prices = decode_quaternion_to_price(reference_data["y_test"], global_scaler)
        max_test_points = len(actual_prices)
        
        model_predictions = []
        
        # Phase 3: Run inference individually with the correct sequence lengths
        for m_info in loaded_models:
            model_filename = m_info["filename"]
            predictor = m_info["predictor"]
            seq_len = m_info["seq_len"]
            m_mod_type = m_info.get("mod_type", "lstm")
            
            if m_mod_type == "extended_mtl":
                m_data = prepare_extended_training_data(
                    ohlcv_df,
                    sequence_length=seq_len,
                    train_split=config.TRAIN_TEST_SPLIT,
                )
            else:
                m_data = prepare_training_data(
                    ohlcv_df,
                    sequence_length=seq_len,
                    train_split=config.TRAIN_TEST_SPLIT,
                    use_path_deltas=False,
                    dual_stream=True,
                    volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20,
                )
            
            X_test_m = m_data["X_test"]
            ctx_X_test_m = m_data.get("ctx_X_test")
            
            # Predict
            if m_mod_type == "extended_mtl":
                predicted_prices = simulate_teacher_forcing(
                    predictor, global_scaler, X_test_m
                )
            else:
                predicted_prices = simulate_teacher_forcing(
                    predictor, global_scaler, X_test_m, ctx_X_test=ctx_X_test_m,
                )
            
            # Calculate alignment padding (how many points this model is missing compared to the max_test_points)
            padding_needed = max_test_points - len(predicted_prices)
            padded_predictions = ([None] * padding_needed) + [round(float(p), 4) for p in predicted_prices]
            
            # Error metrics (only calculate on the valid non-padded portion)
            model_actual = actual_prices[padding_needed:]
            metrics = analyze_errors(model_actual, predicted_prices)
            
            model_predictions.append({
                "name": model_filename,
                "predicted_prices": padded_predictions,
                "metrics": {k: round(float(v), 4) for k, v in metrics.items()}
            })
            
            if model_filename == models[0]:
                primary_predictor = predictor
                data = m_data  # Save for oscillator

        # 7. Oscillator signals (using the first loaded model's residuals)
        signals = []
        next_signal = 0.0
        if "primary_predictor" in locals():
            osc_name = oscillator if oscillator else ACTIVE_MODELS.get("oscillator", "")
            osc_path = os.path.join(config.MODEL_SAVE_DIR, osc_name)
            if os.path.exists(osc_path):
                try:
                    if m_mod_type == "extended_mtl":
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
                        ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0)
    
                    residuals, pred_q = compute_residuals(
                        primary_predictor, X_all, y_all, global_scaler, ctx_X_test=ctx_X_all
                    )
    
                    temp_osc = tf.keras.models.load_model(osc_path, compile=False)
                    osc_type_name = temp_osc.name

                    # Determine internal type string based on model name
                    if osc_type_name == "ClassificationOscillator":
                        internal_osc_type = "classification"
                    elif osc_type_name == "ThresholdOscillator":
                        internal_osc_type = "threshold"
                    elif osc_type_name == "FeedbackOscillator":
                        internal_osc_type = "residual" # fallback for feedback
                    elif osc_type_name == "SelfLearningOscillator":
                        internal_osc_type = "self_learning"
                    else:
                        internal_osc_type = "residual"

                    if internal_osc_type == "self_learning":
                        osc_data_full = prepare_oscillator_training_data(
                            ohlcv_df,
                            sequence_length=config.OSCILLATOR_SEQ_LEN,
                            train_split=config.TRAIN_TEST_SPLIT
                        )
                        osc_data = {
                            "X_res_test": osc_data_full["X_price_test"],
                            "X_q_test": osc_data_full["X_q_test"]
                        }
                        
                        from signal_model import SelfLearningOscillator
                        oscillator = SelfLearningOscillator(
                            sequence_length=config.OSCILLATOR_SEQ_LEN,
                            lstm_units=config.OSCILLATOR_LSTM_UNITS,
                            dense_units=config.OSCILLATOR_DENSE_UNITS,
                        )
                    else:
                        osc_data = prepare_oscillator_data(
                            residuals, pred_q,
                            sequence_length=config.OSCILLATOR_SEQ_LEN,
                            train_split=config.TRAIN_TEST_SPLIT,
                            oscillator_type=internal_osc_type,
                        )
        
                        if internal_osc_type == "classification":
                            from signal_model import ClassificationOscillator
                            oscillator = ClassificationOscillator(
                                sequence_length=config.OSCILLATOR_SEQ_LEN,
                                lstm_units=config.OSCILLATOR_LSTM_UNITS,
                                dense_units=config.OSCILLATOR_DENSE_UNITS,
                            )
                        elif internal_osc_type == "threshold":
                            from signal_model import ThresholdOscillator
                            oscillator = ThresholdOscillator(
                                sequence_length=config.OSCILLATOR_SEQ_LEN,
                                lstm_units=config.OSCILLATOR_LSTM_UNITS,
                                dense_units=config.OSCILLATOR_DENSE_UNITS,
                            )
                        elif osc_type_name == "FeedbackOscillator":
                            from extended_signal_model import FeedbackOscillator
                            oscillator = FeedbackOscillator(
                                sequence_length=config.OSCILLATOR_SEQ_LEN,
                                lstm_units=config.OSCILLATOR_LSTM_UNITS,
                                dense_units=config.OSCILLATOR_DENSE_UNITS,
                            )
                        else:
                            oscillator = ResidualOscillator(
                                sequence_length=config.OSCILLATOR_SEQ_LEN,
                                lstm_units=config.OSCILLATOR_LSTM_UNITS,
                                dense_units=config.OSCILLATOR_DENSE_UNITS,
                            )
                    
                    oscillator.model = temp_osc
    
                    test_signals = oscillator.predict(
                        osc_data["X_res_test"], osc_data["X_q_test"]
                    )
                    
                    if internal_osc_type == "self_learning" or internal_osc_type == "threshold":
                        signals = {
                            "type": "threshold",
                            "values": [round(float(p[0]), 4) for p in test_signals],
                            "buy_threshold": round(float(test_signals[-1, 1]), 4) if len(test_signals) > 0 else 0.5,
                            "sell_threshold": round(float(test_signals[-1, 2]), 4) if len(test_signals) > 0 else -0.5,
                        }
                    elif internal_osc_type == "classification":
                        # Output format: {"type": "classification", "values": [...], "p_buy": [...], "p_sell": [...]}
                        signals = {
                            "type": "classification",
                            "values": [round(float(p[2] - p[0]), 4) for p in test_signals], # P(Buy) - P(Sell)
                            "p_buy": [round(float(p[2]), 4) for p in test_signals],
                            "p_sell": [round(float(p[0]), 4) for p in test_signals],
                            "p_hold": [round(float(p[1]), 4) for p in test_signals]
                        }

                    else:
                        signals = {
                            "type": "residual",
                            "values": [round(float(s), 4) for s in test_signals.flatten()]
                        }
    
                    # Next signal
                    if m_mod_type == "extended_mtl":
                        model_input = [
                            data["X_test"][0][-1:],
                            data["X_test"][1][-1:],
                            data["X_test"][2][-1:]
                        ]
                    else:
                        last_x = data["X_test"][-1:]
                        if "ctx_X_test" in data:
                            last_ctx = data["ctx_X_test"][-1:]
                            model_input = [last_x, last_ctx]
                        else:
                            model_input = last_x

                    next_q = primary_predictor.predict(model_input)
    
                    if internal_osc_type == "self_learning":
                        recent_inputs = osc_data_full["X_price_test"][-1:]
                        next_raw = oscillator.predict(recent_inputs, next_q)
                        signals["type"] = "threshold" # Ensure we use threshold dict format
                    else:
                        recent_residuals = residuals[-config.OSCILLATOR_SEQ_LEN:]
                        recent_residuals = recent_residuals.reshape(1, config.OSCILLATOR_SEQ_LEN, 1).astype(np.float32)
                        next_raw = oscillator.predict(recent_residuals, next_q)
                        
                    if isinstance(signals, dict) and signals["type"] == "classification":
                        next_signal = {
                            "value": float(next_raw[0, 2] - next_raw[0, 0]),
                            "p_buy": float(next_raw[0, 2]),
                            "p_sell": float(next_raw[0, 0]),
                            "p_hold": float(next_raw[0, 1]),
                        }
                    elif isinstance(signals, dict) and signals["type"] == "threshold":
                        next_signal = {
                            "value": float(next_raw[0, 0]),
                            "buy_threshold": float(next_raw[0, 1]),
                            "sell_threshold": float(next_raw[0, 2]),
                        }
                    else:
                        next_signal = float(next_raw[0, 0])
                except Exception as e:
                    logger.warning("Oscillator inference failed: %s", e, exc_info=True)
                    signals = {"type": "residual", "values": []}
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
            "model_predictions": model_predictions,
            "signals": signals,
            "next_signal": next_signal,
            "data_points": len(ohlcv_df),
            "test_points": len(actual_prices),
        }

    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Training Endpoints ────────────────────────

@app.post("/api/train_oscillator")
async def start_oscillator_training(
    epochs: Optional[int] = Query(None, description="Number of epochs to train"),
    primary_model_name: str = Query(..., description="Name of the primary model to attach to"),
    oscillator_type: str = Query("classification", description="Type of oscillator to train"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Start background training for an oscillator standalone."""
    if TRAINING_STATE["is_training"]:
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})

    osc_epochs = epochs or config.OSCILLATOR_EPOCHS

    def _train_osc_background():
        try:
            TRAINING_STATE["is_training"] = True
            TRAINING_STATE["phase"] = "oscillator"
            TRAINING_STATE["epoch"] = 0
            TRAINING_STATE["total_epochs"] = osc_epochs
            TRAINING_STATE["history"] = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
            TRAINING_STATE["logs"] = [f"[SYSTEM] Starting oscillator-only pipeline..."]
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
            
            # Load Primary Model
            TRAINING_STATE["logs"].append(f"[SYSTEM] Loading primary model: {primary_model_name}...")
            model_path = os.path.join(config.MODEL_SAVE_DIR, primary_model_name)
            temp_model = tf.keras.models.load_model(model_path, compile=False)
            
            seq_len = config.SEQUENCE_LENGTH
            try:
                input_shape = temp_model.inputs[0].shape
                if len(input_shape) >= 2 and input_shape[1] is not None:
                    seq_len = int(input_shape[1])
            except Exception as e:
                pass
                
            if temp_model.name == "MTLQuaternionPredictor":
                arch_type = "mtl"
            elif temp_model.name == "ExtendedMTLPredictor":
                arch_type = "extended_mtl"
            else:
                arch_type = "lstm"
                
            primary_predictor = build_primary_model(arch_type)
            primary_predictor.model = temp_model
            
            # Prepare data
            TRAINING_STATE["logs"].append("[SYSTEM] Preparing data for residuals...")
            if arch_type == "extended_mtl":
                data = prepare_extended_training_data(
                    ohlcv_df, sequence_length=seq_len, train_split=config.TRAIN_TEST_SPLIT
                )
                X_all = [
                    np.concatenate([data["X_train"][0], data["X_test"][0]], axis=0),
                    np.concatenate([data["X_train"][1], data["X_test"][1]], axis=0),
                    np.concatenate([data["X_train"][2], data["X_test"][2]], axis=0)
                ]
                y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
                ctx_X_all = None
            else:
                data = prepare_training_data(
                    ohlcv_df, sequence_length=seq_len, train_split=config.TRAIN_TEST_SPLIT,
                    use_path_deltas=False, dual_stream=True,
                    volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20
                )
                X_all = np.concatenate([data["X_train"], data["X_test"]], axis=0)
                if isinstance(data["y_train"], dict):
                    y_all = np.concatenate([data["y_train"]["out_main"], data["y_test"]["out_main"]], axis=0)
                else:
                    y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
                ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0)
                
            # Compute Residuals
            TRAINING_STATE["logs"].append("[OSCILLATOR] Computing residuals...")
            residuals, pred_q = compute_residuals(
                primary_predictor, X_all, y_all, data["scaler"], ctx_X_test=ctx_X_all
            )
            
            osc_data = prepare_oscillator_data(
                residuals, pred_q,
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                train_split=config.TRAIN_TEST_SPLIT,
                oscillator_type=oscillator_type,
            )
            
            TRAINING_STATE["logs"].append(f"[OSCILLATOR] Building {oscillator_type} oscillator...")
            if arch_type == "extended_mtl" and oscillator_type == "residual":
                from extended_signal_model import FeedbackOscillator
                oscillator = FeedbackOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )
            elif oscillator_type == "classification":
                from signal_model import ClassificationOscillator
                oscillator = ClassificationOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )
            elif oscillator_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )
            else:
                oscillator = ResidualOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )
                
            oscillator.build_model()
            
            TRAINING_STATE["logs"].append(f"[OSCILLATOR] Starting training — {osc_epochs} epochs")
            osc_cb = TrainingStateCallback(phase="oscillator")
            
            import datetime
            dateStr = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"osc_{oscillator_type}_{dateStr}.keras")
            
            oscillator.train(
                osc_data,
                epochs=osc_epochs,
                batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT,
                save_best=True,
                model_path=save_path,
                callbacks=[osc_cb]
            )
            
            # Evaluate
            metrics = oscillator.evaluate(osc_data)
            metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
            TRAINING_STATE["logs"].append(f"[OSCILLATOR] Test Metrics: {metrics_str}")
            
            ACTIVE_MODELS["primary"] = primary_model_name
            ACTIVE_MODELS["oscillator"] = os.path.basename(save_path)
            
            TRAINING_STATE["is_training"] = False
            TRAINING_STATE["logs"].append("[SYSTEM] Background oscillator training complete.")
            
        except Exception as e:
            logger.exception("Oscillator training pipeline failed")
            TRAINING_STATE["is_training"] = False
            TRAINING_STATE["error"] = str(e)
            TRAINING_STATE["logs"].append(f"[ERROR] {str(e)}")
            
    background_tasks.add_task(_train_osc_background)
    return {"status": "ok", "message": "Oscillator training started in background"}


@app.post("/api/train")
async def start_training(
    epochs: Optional[int] = None,
    oscillator_epochs: Optional[int] = None,
    model_name: Optional[str] = Query("best_model", description="Base name for saved models"),
    model_type: Optional[str] = Query(None, description="Model architecture type"),
    oscillator_type: Optional[str] = Query("residual", description="Oscillator architecture type"),
):
    """Start background training for both primary and oscillator models."""
    if TRAINING_STATE["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    train_epochs = epochs or config.EPOCHS
    osc_epochs = oscillator_epochs or config.OSCILLATOR_EPOCHS
    arch_type = model_type or config.MODEL_TYPE
    osc_arch_type = oscillator_type or "residual"

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
            if arch_type == "extended_mtl":
                data = prepare_extended_training_data(
                    ohlcv_df,
                    sequence_length=config.SEQUENCE_LENGTH,
                    train_split=config.TRAIN_TEST_SPLIT,
                )
            else:
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
            predictor = build_primary_model(arch_type)
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

            if arch_type == "extended_mtl":
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
                ctx_X_all = np.concatenate([data["ctx_X_train"], data["ctx_X_test"]], axis=0) if use_dual else None

            residuals, pred_q = compute_residuals(
                predictor, X_all, y_all, data["scaler"], ctx_X_test=ctx_X_all
            )

            osc_data = prepare_oscillator_data(
                residuals, pred_q,
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                train_split=config.TRAIN_TEST_SPLIT,
                oscillator_type=osc_arch_type,
            )

            if arch_type == "extended_mtl" and osc_arch_type == "residual":
                from extended_signal_model import FeedbackOscillator
                oscillator = FeedbackOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )
            elif osc_arch_type == "classification":
                from signal_model import ClassificationOscillator
                oscillator = ClassificationOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )
            elif osc_arch_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )
            else:
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

from fastapi import Request

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(request: Request, full_path: str):
    """Serve the React app for all non-API routes."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
        
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="React build not found. Run 'npm run build' in web directory.")


# ──────────────────────────────────────────────
# Run with: uvicorn api:app --host 127.0.0.1 --port 8000
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
