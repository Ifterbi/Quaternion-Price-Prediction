"""
Residual Oscillator Model for Trading Signals.

This complementary model takes the residuals (prediction errors) from the 
primary Quaternion LSTM model, alongside its next predicted quaternion,
to forecast an implicit valuation signal between -1 (undervalued) and 1 (overvalued).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import logging
import os
from typing import Dict, Optional, List
from io import StringIO

logger = logging.getLogger(__name__)

import keras

@keras.saving.register_keras_serializable()
def directional_momentum_loss(y_true, y_pred):
    """
    Custom loss combining MSE with a Soft-Sign Momentum matching penalty.
    Loss = MSE(y_true, y_pred) - 0.5 * (y_true * y_pred)
    """
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
    # Matching signs yield positive product (subtracts from loss).
    # Opposing signs yield negative product (adds to loss).
    lambda_weight = 0.5
    directional_reward = tf.reduce_mean(lambda_weight * (y_true * y_pred), axis=-1)
    
    return mse_loss - directional_reward

class ResidualOscillator:
    """Predicts a valuation signal in [-1, 1].

    The model processes a sequence of historical residuals using an LSTM
    and combines it with the primary model's prediction for the next step.
    
    Attributes:
        sequence_length: Length of the residual history window.
        lstm_units: Units in the residual LSTM layer.
        dense_units: Units in the combined Dense layer.
        learning_rate: Adam learning rate.
        model: Compiled tf.keras.Model.
    """

    def __init__(
        self,
        sequence_length: int = 14,
        lstm_units: int = 16,
        dense_units: int = 8,
        learning_rate: float = 0.005,
    ):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model: Optional[Model] = None

        logger.info(
            "ResidualOscillator initialised — seq_len=%d, lstm_units=%d, "
            "dense_units=%d, lr=%.4f",
            sequence_length,
            lstm_units,
            dense_units,
            learning_rate,
        )

    def build_model(self) -> Model:
        """Build and compile the Keras Functional model."""
        
        # Input 1: Sequence of past residuals
        # Shape: (batch_size, sequence_length, 1)
        res_input = Input(shape=(self.sequence_length, 1), name="residual_seq")
        
        # Process residuals with LSTM to capture momentum/trend
        h_res = LSTM(self.lstm_units, return_sequences=False, name="residual_lstm")(res_input)
        h_res = Dropout(0.2, name="residual_dropout")(h_res)
        
        # Input 2: Next predicted quaternion from primary model
        # Shape: (batch_size, 4)
        q_input = Input(shape=(4,), name="next_quaternion")
        
        # Combine residual momentum with future fundamental prediction
        merged = Concatenate(name="concat_features")([h_res, q_input])
        
        # Fully connected processing
        dense = Dense(self.dense_units, activation="relu", name="dense_1")(merged)
        
        # Output strictly in [-1, 1] using tanh
        output = Dense(1, activation="tanh", name="oscillator_out")(dense)
        
        model = Model(inputs=[res_input, q_input], outputs=output, name="ResidualOscillator")
        
        optimizer = Adam(learning_rate=self.learning_rate)
        # We use the custom directional momentum loss
        model.compile(optimizer=optimizer, loss=directional_momentum_loss, metrics=["mae"])
        
        self.model = model
        logger.info("ResidualOscillator built and compiled.")
        return model

    def train(
        self,
        data_dict: Dict[str, np.ndarray],
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.1,
        save_best: bool = True,
        model_path: str = "saved_models/oscillator_model.keras",
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        """Train the oscillator model."""
        if self.model is None:
            raise RuntimeError("Model not built.")

        internal_callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        ]

        if save_best:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            internal_callbacks.append(
                ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1)
            )

        if callbacks:
            internal_callbacks.extend(callbacks)

        X_train = [data_dict["X_res_train"], data_dict["X_q_train"]]
        y_train = data_dict["y_train"]

        logger.info("Training oscillator for %d epochs...", epochs)
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=internal_callbacks,
            verbose=1,
        )
        return history

    def evaluate(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise RuntimeError("Model not built.")
            
        X_test = [data_dict["X_res_test"], data_dict["X_q_test"]]
        y_test = data_dict["y_test"]
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {"loss": results[0], "mae": results[1]}
        
        logger.info("Oscillator Evaluation — Loss (MSE): %.6f, MAE: %.6f", metrics["loss"], metrics["mae"])
        return metrics

    def predict(self, res_seq: np.ndarray, next_q: np.ndarray) -> np.ndarray:
        """Generate signals.
        
        Args:
            res_seq: Array of shape (N, seq_len, 1)
            next_q: Array of shape (N, 4)
            
        Returns:
            Signals of shape (N, 1) in [-1, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not built.")
            
        return self.model.predict([res_seq, next_q], verbose=0)

    def get_summary(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not built.")
        buffer = StringIO()
        self.model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        return buffer.getvalue()


class ClassificationOscillator(ResidualOscillator):
    """Predicts probabilities of [SELL, HOLD, BUY]."""
    
    def build_model(self) -> Model:
        res_input = Input(shape=(self.sequence_length, 1), name="residual_seq")
        h_res = LSTM(self.lstm_units, return_sequences=False, name="residual_lstm")(res_input)
        h_res = Dropout(0.2, name="residual_dropout")(h_res)
        
        q_input = Input(shape=(4,), name="next_quaternion")
        merged = Concatenate(name="concat_features")([h_res, q_input])
        dense = Dense(self.dense_units, activation="relu", name="dense_1")(merged)
        
        # 3 classes: 0 (Sell), 1 (Hold), 2 (Buy)
        output = Dense(3, activation="softmax", name="oscillator_out")(dense)
        
        model = Model(inputs=[res_input, q_input], outputs=output, name="ClassificationOscillator")
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model = model
        return model

@keras.saving.register_keras_serializable()
class DynamicThresholdLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.buy_threshold = self.add_weight(name="buy_threshold", shape=(), initializer=tf.constant_initializer(0.5), trainable=True)
        self.sell_threshold = self.add_weight(name="sell_threshold", shape=(), initializer=tf.constant_initializer(-0.5), trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        osc_val = inputs
        # We output the osc_val along with the thresholds repeated for the batch
        batch_size = tf.shape(osc_val)[0]
        buy_t = tf.fill([batch_size, 1], self.buy_threshold)
        sell_t = tf.fill([batch_size, 1], self.sell_threshold)
        return tf.concat([osc_val, buy_t, sell_t], axis=-1)

@keras.saving.register_keras_serializable()
def profit_loss(y_true, y_pred):
    """
    Custom loss that penalizes missed profit.
    y_true is the continuous future change.
    y_pred is [osc_val, buy_thresh, sell_thresh]
    """
    osc_val = y_pred[:, 0:1]
    buy_t = y_pred[:, 1:2]
    sell_t = y_pred[:, 2:3]
    
    # Soft trade signals using sigmoid to keep it differentiable
    # 1 if osc_val > buy_t, 0 otherwise
    buy_signal = tf.sigmoid((osc_val - buy_t) * 10.0)
    # 1 if osc_val < sell_t, 0 otherwise
    sell_signal = tf.sigmoid((sell_t - osc_val) * 10.0)
    
    # Trade signal: 1 (long), -1 (short), 0 (hold)
    trade_signal = buy_signal - sell_signal
    
    # We want to maximize (trade_signal * y_true), so we minimize its negative
    profit = trade_signal * y_true
    
    # We add a small MSE term to keep osc_val bounded
    mse = tf.square(y_true - osc_val)
    
    return -tf.reduce_mean(profit) + 0.1 * tf.reduce_mean(mse)

class ThresholdOscillator(ResidualOscillator):
    """Predicts continuous value and dynamically learned thresholds."""
    
    def build_model(self) -> Model:
        res_input = Input(shape=(self.sequence_length, 1), name="residual_seq")
        h_res = LSTM(self.lstm_units, return_sequences=False, name="residual_lstm")(res_input)
        h_res = Dropout(0.2, name="residual_dropout")(h_res)
        
        q_input = Input(shape=(4,), name="next_quaternion")
        merged = Concatenate(name="concat_features")([h_res, q_input])
        dense = Dense(self.dense_units, activation="relu", name="dense_1")(merged)
        
        osc_val = Dense(1, activation="tanh", name="raw_oscillator")(dense)
        output = DynamicThresholdLayer(name="oscillator_out")(osc_val)
        
        model = Model(inputs=[res_input, q_input], outputs=output, name="ThresholdOscillator")
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=profit_loss,
            metrics=["mae"]
        )
        self.model = model
        return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Smoke test
    print("Running Oscillator Smoke Test...")
    osc = ResidualOscillator(sequence_length=14)
    osc.build_model()
    print(osc.get_summary())
    
    # Dummy data
    n_samples = 100
    res_seq = np.random.randn(n_samples, 14, 1).astype(np.float32)
    next_q = np.random.randn(n_samples, 4).astype(np.float32)
    targets = np.random.uniform(-1, 1, size=(n_samples, 1)).astype(np.float32)
    
    data = {
        "X_res_train": res_seq[:80],
        "X_q_train": next_q[:80],
        "y_train": targets[:80],
        "X_res_test": res_seq[80:],
        "X_q_test": next_q[80:],
        "y_test": targets[80:],
    }
    
    osc.train(data, epochs=2, save_best=False)
    metrics = osc.evaluate(data)
    
    preds = osc.predict(res_seq[:5], next_q[:5])
    print(f"\nSample Predictions (should be in [-1, 1]):\n{preds}")
