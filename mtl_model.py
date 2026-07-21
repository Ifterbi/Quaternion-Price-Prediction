"""
Multi-Task Learning (MTL) Quaternion LSTM Price Predictor.

This module provides the ``MultiTaskQuaternionPredictor`` class, a neural network
that leverages Multi-Task Learning to independently optimize the prediction of 
individual quaternion components (e.g., Open, High, Low, Volume) while simultaneously
learning their cross-correlations to predict a final quaternion rotation.

Architecture:
    Shared Trunk (LSTM) -> Maintains early cross-correlation
        ├── Head W (Dense) -> Predicts scalar component (e.g., Open price)
        ├── Head X (Dense) -> Predicts 1st imaginary component
        ├── Head Y (Dense) -> Predicts 2nd imaginary component
        ├── Head Z (Dense) -> Predicts 3rd imaginary component
        └── Main Head (Concatenate + Dense) -> Predicts final 4D quaternion/rotation
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict
from io import StringIO

logger = logging.getLogger(__name__)

class MultiTaskQuaternionPredictor:
    """Multi-Task Learning model for quaternion-encoded price prediction.

    Attributes:
        sequence_length: Number of time-steps in each input window.
        n_features: Number of input features per time-step (default 4).
        lstm_units: Number of units in the shared LSTM trunk layers.
        aux_dense_units: Number of units in the auxiliary Dense heads.
        dropout_rate: Dropout rate applied after each LSTM layer.
        learning_rate: Learning rate for the Adam optimiser.
        loss_weights: Dictionary of weights for the multi-task loss function.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 4,
        lstm_units: int = 64,
        aux_dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.aux_dense_units = aux_dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Default loss weights favor the main rotation prediction while still 
        # heavily regularizing via the individual component mechanics.
        self.loss_weights = loss_weights or {
            "out_w": 0.1,  # Scalar component (e.g., Open)
            "out_x": 0.1,  # 1st Imaginary (e.g., High)
            "out_y": 0.1,  # 2nd Imaginary (e.g., Low)
            "out_z": 0.1,  # 3rd Imaginary (e.g., Volume)
            "out_main": 1.0 # Primary 4D Quaternion prediction
        }
        
        self.model: Optional[Model] = None

        logger.info(
            "MultiTaskQuaternionPredictor initialised — seq_len=%d, features=%d, "
            "lstm_units=%d, aux_dense_units=%d, dropout=%.2f, lr=%.4f",
            sequence_length,
            n_features,
            lstm_units,
            aux_dense_units,
            dropout_rate,
            learning_rate,
        )

    def build_model(self) -> Model:
        """Build and compile the Multi-Task Functional API model."""
        logger.info("Building MTL architecture model...")

        # --- 1. Shared Trunk (Cross-Correlation) ---
        inputs = Input(shape=(self.sequence_length, self.n_features), name="quaternion_input")
        
        shared = LSTM(self.lstm_units, return_sequences=True, name="shared_lstm_1")(inputs)
        shared = Dropout(self.dropout_rate, name="shared_dropout_1")(shared)
        shared_state = LSTM(self.lstm_units, return_sequences=False, name="shared_lstm_2")(shared)
        shared_state = Dropout(self.dropout_rate, name="shared_dropout_2")(shared_state)

        # --- 2. Auxiliary Heads (Component Specialization) ---
        # W Head (Scalar / Open Price)
        h_w = Dense(self.aux_dense_units, activation="relu", name="dense_w")(shared_state)
        out_w = Dense(1, activation="linear", name="out_w")(h_w)
        
        # X Head
        h_x = Dense(self.aux_dense_units, activation="relu", name="dense_x")(shared_state)
        out_x = Dense(1, activation="linear", name="out_x")(h_x)
        
        # Y Head
        h_y = Dense(self.aux_dense_units, activation="relu", name="dense_y")(shared_state)
        out_y = Dense(1, activation="linear", name="out_y")(h_y)
        
        # Z Head
        h_z = Dense(self.aux_dense_units, activation="relu", name="dense_z")(shared_state)
        out_z = Dense(1, activation="linear", name="out_z")(h_z)

        # --- 3. Primary Head (Quaternion Synthesis) ---
        # Recombine the cleanly predicted individual components
        fused_state = Concatenate(name="fusion_concat")([out_w, out_x, out_y, out_z])
        
        # Final prediction for the composed 4D quaternion rotation
        out_main = Dense(4, activation="linear", name="out_main")(fused_state)

        # --- Model Assembly & Compilation ---
        model = Model(
            inputs=inputs,
            outputs=[out_w, out_x, out_y, out_z, out_main],
            name="MTLQuaternionPredictor"
        )

        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss={
                "out_w": "mse",
                "out_x": "mse",
                "out_y": "mse",
                "out_z": "mse",
                "out_main": "mse"
            },
            loss_weights=self.loss_weights,
            metrics={
                "out_w": "mae",
                "out_x": "mae",
                "out_y": "mae",
                "out_z": "mae",
                "out_main": "mae"
            }
        )

        self.model = model
        logger.info("MTL Model built and compiled successfully.")
        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create sliding-window input and multi-task target dictionaries.

        Args:
            data: 2-D array of shape ``(N, 4)`` representing quaternion sequence.
            sequence_length: Window size; defaults to ``self.sequence_length``.

        Returns:
            Tuple ``(X, y_dict)`` where:
                ``X`` has shape ``(N - seq_len, seq_len, 4)``.
                ``y_dict`` contains the 5 targeted outputs mapped to their output layers.
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        # Prepare the dictionary of targets matching the layer names
        y_dict = {
            "out_w": y[:, 0:1], # Scalar component (Open)
            "out_x": y[:, 1:2], # Imaginary x
            "out_y": y[:, 2:3], # Imaginary y
            "out_z": y[:, 3:4], # Imaginary z
            "out_main": y       # Full 4D Quaternion
        }

        logger.info(
            "Prepared %d MTL sequences — X shape: %s",
            len(X),
            X.shape,
        )
        return X, y_dict

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        save_best: bool = True,
        model_path: str = "saved_models/mtl_best_model.keras",
    ) -> tf.keras.callbacks.History:
        """Train the MTL LSTM model."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # Ignore context features if passed as dual-stream list
        if isinstance(X_train, list):
            X_train = X_train[0]

        logger.info(
            "Starting MTL training — epochs=%d, batch_size=%d",
            epochs, batch_size
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ]

        if save_best:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1)
            )

        y_train_dict = {
            "out_w": y_train[:, 0:1],
            "out_x": y_train[:, 1:2],
            "out_y": y_train[:, 2:3],
            "out_z": y_train[:, 3:4],
            "out_main": y_train
        }

        history = self.model.fit(
            X_train,
            y_train_dict,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input sequences.

        Args:
            X: Input array of shape ``(n_samples, sequence_length, 4)``.

        Returns:
            The final predicted 4D quaternion values of shape ``(n_samples, 4)``.
            (Ignores the auxiliary component predictions during normal inference).
        """
        if self.model is None:
            raise RuntimeError("Model not available.")

        # Ignore context features if passed as dual-stream list
        if isinstance(X, list):
            X = X[0]

        # predict returns a list of 5 arrays corresponding to the 5 output layers
        predictions_list = self.model.predict(X, verbose=0)
        
        # The main output is the last one in the list (index 4)
        main_predictions = predictions_list[4]
        return main_predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Returns:
            Dictionary with comprehensive metrics including total loss and main head MAE.
        """
        if self.model is None:
            raise RuntimeError("Model not available.")

        # Ignore context features if passed as dual-stream list
        if isinstance(X_test, list):
            X_test = X_test[0]

        y_test_dict = {
            "out_w": y_test[:, 0:1],
            "out_x": y_test[:, 1:2],
            "out_y": y_test[:, 2:3],
            "out_z": y_test[:, 3:4],
            "out_main": y_test
        }

        results = self.model.evaluate(X_test, y_test_dict, verbose=0)
        
        # Keras returns a flat list of: [total_loss, loss1, loss2..., mae1, mae2...]
        # We'll map the total loss and the main head's loss/mae for clarity
        metrics = {
            "loss": results[0],
            "main_loss": results[5], # Assuming order: total, 4 aux losses, 1 main loss, then metrics
            "mae": results[-1]  # Last metric is typically the main head's MAE
        }

        logger.info("Evaluation — Total Loss: %.6f, Main Head MAE: %.6f", metrics["loss"], metrics["mae"])
        return metrics

    def get_summary(self) -> str:
        """Return the model summary as a string."""
        if self.model is None:
            raise RuntimeError("Model not available.")
        buffer = StringIO()
        self.model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        return buffer.getvalue()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Multi-Task Learning (MTL) Architecture - Smoke Test")
    print("=" * 60)

    SEQ_LEN = 30
    N_FEATURES = 4
    N_SAMPLES = 200

    np.random.seed(42)
    dummy_data = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)

    # 1. Init & Build
    mtl_predictor = MultiTaskQuaternionPredictor(
        sequence_length=SEQ_LEN,
        lstm_units=32,
        aux_dense_units=16
    )
    mtl_predictor.build_model()
    
    # 2. Data Preparation
    X, y_dict = mtl_predictor.prepare_sequences(dummy_data)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train_dict = {k: v[:split] for k, v in y_dict.items()}
    y_test_dict = {k: v[split:] for k, v in y_dict.items()}

    # 3. Train
    mtl_predictor.train(X_train, y_train_dict, epochs=2, batch_size=16, save_best=False)

    # 4. Evaluate
    metrics = mtl_predictor.evaluate(X_test, y_test_dict)
    
    # 5. Predict
    sample_preds = mtl_predictor.predict(X_test[:3])
    print(f"\nSample main predictions shape (should be (3, 4)): {sample_preds.shape}")
    print(f"Sample prediction outputs:\n{sample_preds}")
    
    print("\nSmoke test completed successfully ✓")
