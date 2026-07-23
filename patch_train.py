import os

api_path = '/home/ifterbi/OneDrive/My World/Quaternion Price Predictor/api.py'

with open(api_path, 'r') as f:
    content = f.read()

# 1. Patch _train_osc_background
old_osc_data_1 = """            osc_data = prepare_oscillator_data(
                residuals, pred_q,
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                train_split=config.TRAIN_TEST_SPLIT,
                oscillator_type=oscillator_type,
            )"""

new_osc_data_1 = """            if oscillator_type == "self_learning":
                osc_data = prepare_oscillator_training_data(
                    ohlcv_df,
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    train_split=config.TRAIN_TEST_SPLIT
                )
            else:
                osc_data = prepare_oscillator_data(
                    residuals, pred_q,
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    train_split=config.TRAIN_TEST_SPLIT,
                    oscillator_type=oscillator_type,
                )"""

content = content.replace(old_osc_data_1, new_osc_data_1)

old_osc_model_1 = """            elif oscillator_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )"""

new_osc_model_1 = """            elif oscillator_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )
            elif oscillator_type == "self_learning":
                from signal_model import SelfLearningOscillator
                oscillator = SelfLearningOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                )"""
if 'elif oscillator_type == "self_learning":' not in content:
    content = content.replace(old_osc_model_1, new_osc_model_1)

# 2. Patch _train_background
old_osc_data_2 = """            osc_data = prepare_oscillator_data(
                residuals, pred_q,
                sequence_length=config.OSCILLATOR_SEQ_LEN,
                train_split=config.TRAIN_TEST_SPLIT,
                oscillator_type=osc_arch_type,
            )"""

new_osc_data_2 = """            if osc_arch_type == "self_learning":
                osc_data = prepare_oscillator_training_data(
                    ohlcv_df,
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    train_split=config.TRAIN_TEST_SPLIT
                )
            else:
                osc_data = prepare_oscillator_data(
                    residuals, pred_q,
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    train_split=config.TRAIN_TEST_SPLIT,
                    oscillator_type=osc_arch_type,
                )"""
content = content.replace(old_osc_data_2, new_osc_data_2)

old_osc_model_2 = """            elif osc_arch_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )"""

new_osc_model_2 = """            elif osc_arch_type == "threshold":
                from signal_model import ThresholdOscillator
                oscillator = ThresholdOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )
            elif osc_arch_type == "self_learning":
                from signal_model import SelfLearningOscillator
                oscillator = SelfLearningOscillator(
                    sequence_length=config.OSCILLATOR_SEQ_LEN,
                    lstm_units=config.OSCILLATOR_LSTM_UNITS,
                    dense_units=config.OSCILLATOR_DENSE_UNITS,
                    learning_rate=config.OSCILLATOR_LEARNING_RATE,
                )"""
if 'elif osc_arch_type == "self_learning":' not in content:
    content = content.replace(old_osc_model_2, new_osc_model_2)


with open(api_path, 'w') as f:
    f.write(content)
print("api.py patched successfully!")
