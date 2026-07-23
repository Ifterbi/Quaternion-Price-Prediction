"""
Extended Multi-Task Learning (MTL) Model.

This model extends the base MultiTaskQuaternionPredictor by predicting
momentum quaternions, and applying them to the current state to predict the next.
It also includes an integrated feedback loop mechanism for trading signals.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer
import logging
from typing import Optional, Dict
import numpy as np

# Import the base class
import mtl_model

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable()
class QuaternionHamiltonProductLayer(Layer):
    """Custom Keras layer that computes the Hamilton product of two quaternions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Computes Q_next = Q_rot * Q_current
        inputs[0]: Q_rot (batch_size, 4) - The predicted momentum rotation
        inputs[1]: Q_current (batch_size, 4) - The current state
        """
        q1 = inputs[0]
        q2 = inputs[1]

        # q = [w, x, y, z]
        w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
        w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]

        # Hamilton product rules
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return tf.concat([w, x, y, z], axis=-1)


class ExtendedMTLPredictor(mtl_model.MultiTaskQuaternionPredictor):
    """
    Extended MTL Model that predicts OHLV momentum as a rotation quaternion,
    and applies it to the current state.
    """

    def __init__(self, *args, **kwargs):
        # Update default loss weights for momentum targets
        loss_weights = kwargs.pop("loss_weights", {
            "mom_w": 0.1,
            "mom_x": 0.1,
            "mom_y": 0.1,
            "mom_z": 0.1,
            "out_main": 1.0
        })
        kwargs["loss_weights"] = loss_weights
        super().__init__(*args, **kwargs)

    def build_model(self) -> Model:
        """Override to build the momentum rotation architecture."""
        logger.info("Building Extended MTL architecture model...")

        # --- 1. Inputs ---
        seq_input = Input(shape=(self.sequence_length, self.n_features), name="quaternion_seq_input")
        ctx_input = Input(shape=(self.sequence_length, 5), name="context_input")
        q_current = Input(shape=(4,), name="current_quaternion")

        # --- 2. Shared Trunk ---
        # Concatenate seq and context along the feature dimension
        merged_input = Concatenate(axis=-1)([seq_input, ctx_input])
        
        shared = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, name="shared_lstm_1")(merged_input)
        shared = tf.keras.layers.Dropout(self.dropout_rate, name="shared_dropout_1")(shared)
        shared_state = tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, name="shared_lstm_2")(shared)
        shared_state = tf.keras.layers.Dropout(self.dropout_rate, name="shared_dropout_2")(shared_state)

        # --- 3. Auxiliary Heads (Momentum Prediction) ---
        # Predict momentums instead of raw values
        h_w = Dense(self.aux_dense_units, activation="relu", name="dense_mom_w")(shared_state)
        mom_w = Dense(1, activation="linear", name="mom_w")(h_w)
        
        h_x = Dense(self.aux_dense_units, activation="relu", name="dense_mom_x")(shared_state)
        mom_x = Dense(1, activation="linear", name="mom_x")(h_x)
        
        h_y = Dense(self.aux_dense_units, activation="relu", name="dense_mom_y")(shared_state)
        mom_y = Dense(1, activation="linear", name="mom_y")(h_y)
        
        h_z = Dense(self.aux_dense_units, activation="relu", name="dense_mom_z")(shared_state)
        mom_z = Dense(1, activation="linear", name="mom_z")(h_z)

        # --- 4. Rotation Synthesis ---
        # Form the rotation quaternion Q_rot
        q_rot = Concatenate(name="q_rot_synthesis")([mom_w, mom_x, mom_y, mom_z])
        
        # Calculate Q_next = Q_current * Q_rot
        out_main = QuaternionHamiltonProductLayer(name="out_main")([q_current, q_rot])

        # --- Model Assembly & Compilation ---
        model = Model(
            inputs=[seq_input, ctx_input, q_current],
            outputs=[mom_w, mom_x, mom_y, mom_z, out_main],
            name="ExtendedMTLPredictor"
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss={
                "mom_w": "mse",
                "mom_x": "mse",
                "mom_y": "mse",
                "mom_z": "mse",
                "out_main": "mse"
            },
            loss_weights=self.loss_weights,
            metrics={
                "mom_w": "mae",
                "mom_x": "mae",
                "mom_y": "mae",
                "mom_z": "mae",
                "out_main": "mae"
            }
        )

        self.model = model
        logger.info("Extended MTL Model built and compiled successfully.")
        return model
        
    def predict_next_q(self, X_seq: np.ndarray, X_ctx: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """
        Helper method to get just the final predicted quaternion.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")
            
        preds = self.model.predict([X_seq, X_ctx, q_current], verbose=0)
        return preds[4] # out_main is index 4

    def predict(self, X) -> np.ndarray:
        """Override to properly handle the list of inputs for extended MTL."""
        if self.model is None:
            raise RuntimeError("Model not available.")
            
        # X should be [X_seq, X_ctx, q_current]
        predictions_list = self.model.predict(X, verbose=0)
        return predictions_list[4]

    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, validation_data=None, callbacks=None, save_best=True, model_path="saved_models/extended_mtl_model.keras"):
        """Override train to handle pre-formatted dict targets."""
        if self.model is None:
            raise RuntimeError("Model not built.")
            
        import os
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        fit_callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ]
        
        if callbacks:
            fit_callbacks.extend(callbacks)

        if save_best:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            fit_callbacks.append(
                ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1)
            )

        logger.info(f"Starting Extended MTL training — epochs={epochs}, batch_size={batch_size}")
        
        fit_kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "callbacks": fit_callbacks,
            "verbose": 1,
        }
        
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        else:
            fit_kwargs["validation_split"] = validation_split
            
        history = self.model.fit(
            X_train,
            y_train,
            **fit_kwargs
        )
        return history

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate the model using pre-formatted dict targets."""
        if self.model is None:
            raise RuntimeError("Model not built.")
            
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Keras returns a list of metrics. Usually:
        # [loss, mom_w_loss, mom_x_loss, mom_y_loss, mom_z_loss, out_main_loss, mom_w_mae, mom_x_mae, mom_y_mae, mom_z_mae, out_main_mae]
        # We'll just map the first few to keep it consistent.
        metrics = {
            "loss": results[0],
            "main_loss": results[5] if len(results) > 5 else results[0],
            "mae": results[-1] if len(results) > 0 else 0.0
        }
        
        logger.info(
            f"Evaluation — Total Loss: {metrics['loss']:.6f}, Main Loss: {metrics['main_loss']:.6f}, Main MAE: {metrics['mae']:.6f}"
        )
        return metrics
