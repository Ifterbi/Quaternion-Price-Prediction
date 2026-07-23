import os
import re

api_path = '/home/ifterbi/OneDrive/My World/Quaternion Price Predictor/api.py'

with open(api_path, 'r') as f:
    content = f.read()

# 1. Add import
import_str = "from extended_encoder import prepare_extended_training_data\n"
if "from oscillator_data import prepare_oscillator_training_data" not in content:
    content = content.replace(
        import_str, 
        import_str + "from oscillator_data import prepare_oscillator_training_data\n"
    )

# 2. Update model type detection
if 'elif osc_type_name == "SelfLearningOscillator":' not in content:
    old_type_det = """                    elif osc_type_name == "FeedbackOscillator":
                        internal_osc_type = "residual" # fallback for feedback
                    else:
                        internal_osc_type = "residual\""""
    new_type_det = """                    elif osc_type_name == "FeedbackOscillator":
                        internal_osc_type = "residual" # fallback for feedback
                    elif osc_type_name == "SelfLearningOscillator":
                        internal_osc_type = "self_learning"
                    else:
                        internal_osc_type = "residual\""""
    content = content.replace(old_type_det, new_type_det)

# 3. Update oscillator data preparation and instantiation
old_inst_block = """                    osc_data = prepare_oscillator_data(
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
                        )"""

new_inst_block = """                    if internal_osc_type == "self_learning":
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
                            )"""

content = content.replace(old_inst_block, new_inst_block)


# 4. Update next signal processing
old_next_block = """                    recent_residuals = residuals[-config.OSCILLATOR_SEQ_LEN:]
                    recent_residuals = recent_residuals.reshape(1, config.OSCILLATOR_SEQ_LEN, 1).astype(np.float32)
                    next_raw = oscillator.predict(recent_residuals, next_q)
                    if isinstance(signals, dict) and signals["type"] == "classification":"""

new_next_block = """                    if internal_osc_type == "self_learning":
                        recent_inputs = osc_data_full["X_price_test"][-1:]
                        next_raw = oscillator.predict(recent_inputs, next_q)
                        signals["type"] = "threshold" # Ensure we use threshold dict format
                    else:
                        recent_residuals = residuals[-config.OSCILLATOR_SEQ_LEN:]
                        recent_residuals = recent_residuals.reshape(1, config.OSCILLATOR_SEQ_LEN, 1).astype(np.float32)
                        next_raw = oscillator.predict(recent_residuals, next_q)
                        
                    if isinstance(signals, dict) and signals["type"] == "classification":"""
content = content.replace(old_next_block, new_next_block)


# 5. Fix signals format for self_learning
old_signals_block = """                    if internal_osc_type == "classification":
                        # Output format: {"type": "classification", "values": [...], "p_buy": [...], "p_sell": [...]}"""

new_signals_block = """                    if internal_osc_type == "self_learning" or internal_osc_type == "threshold":
                        signals = {
                            "type": "threshold",
                            "values": [round(float(p[0]), 4) for p in test_signals],
                            "buy_threshold": round(float(test_signals[-1, 1]), 4) if len(test_signals) > 0 else 0.5,
                            "sell_threshold": round(float(test_signals[-1, 2]), 4) if len(test_signals) > 0 else -0.5,
                        }
                    elif internal_osc_type == "classification":
                        # Output format: {"type": "classification", "values": [...], "p_buy": [...], "p_sell": [...]}"""

if 'internal_osc_type == "self_learning" or internal_osc_type == "threshold"' not in content:
    # First, replace the threshold elif
    content = content.replace("""                    elif internal_osc_type == "threshold":
                        signals = {
                            "type": "threshold",
                            "values": [round(float(p[0]), 4) for p in test_signals],
                            "buy_threshold": round(float(test_signals[-1, 1]), 4) if len(test_signals) > 0 else 0.5,
                            "sell_threshold": round(float(test_signals[-1, 2]), 4) if len(test_signals) > 0 else -0.5,
                        }""", "")
    content = content.replace(old_signals_block, new_signals_block)


with open(api_path, 'w') as f:
    f.write(content)
print("api.py patched successfully!")
