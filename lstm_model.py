"""
Quaternion LSTM Price Predictor Model.

This module provides the ``QuaternionLSTMPredictor`` class, a stacked LSTM
neural network built with TensorFlow/Keras for predicting financial prices
from quaternion-encoded OHLCV data.

Supports two modes:
    **Single-stream** (original):
        LSTM → Dropout → LSTM → Dropout → Dense (ReLU) → Dense (Linear)

    **Dual-stream** (new):
        Price stream:   LSTM → Dropout → LSTM → Dropout → h_price
        Context stream: LSTM → Dropout → h_context
        Fusion:         FiLM / Gate / Concat → Dense (ReLU) → Dense (Linear)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, Multiply, Add, Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict, Union, List
from io import StringIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom layer: Context Dropout
# ---------------------------------------------------------------------------
class ContextDropout(Layer):
    """Replace random context timesteps with stale (repeated) values.

    During training, for each timestep in a context sequence, with
    probability ``rate`` that timestep's features are replaced with the
    features from the *previous* timestep (or the first timestep if at
    position 0).  This simulates the degraded-context scenario the model
    will encounter during live multi-step autoregressive inference.

    During inference (``training=False``) this layer is a no-op.
    """

    def __init__(self, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if not training or self.rate <= 0.0:
            return inputs

        # inputs shape: (batch, seq_len, n_features)
        shape = tf.shape(inputs)
        batch_size = shape[0]
        seq_len = shape[1]

        # Generate a random mask: True = replace with stale value
        mask = tf.random.uniform((batch_size, seq_len, 1)) < self.rate

        # Build "stale" version: shift the sequence right by 1, repeating
        # the first timestep at position 0.
        first_step = inputs[:, :1, :]  # (batch, 1, features)
        shifted = tf.concat([first_step, inputs[:, :-1, :]], axis=1)

        # Where mask is True, use the shifted (stale) value
        return tf.where(mask, shifted, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class QuaternionLSTMPredictor:
    """Stacked LSTM model for quaternion-encoded price prediction.

    Attributes:
        sequence_length: Number of time-steps in each input window.
        n_features: Number of input features per time-step (default 4 for
            pathfinder quaternion components).
        lstm_units: Number of units in each LSTM layer.
        dense_units: Number of units in the intermediate Dense layer.
        dropout_rate: Dropout rate applied after each LSTM layer.
        learning_rate: Learning rate for the Adam optimiser.
        output_size: Number of output values (default 4 for full quaternion
            prediction).
        dual_stream: Whether to use the dual-stream architecture.
        n_context_features: Number of context features (dual-stream only).
        context_lstm_units: LSTM units for the context stream.
        fusion_strategy: How to fuse price and context streams.
        fusion_dense_units: Dense units after fusion.
        context_dropout_rate: Rate for context dropout during training.
        model: The compiled ``tf.keras.Model`` instance (built lazily via
            :meth:`build_model`).
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 4,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        output_size: int = 4,
        dual_stream: bool = False,
        n_context_features: int = 5,
        context_lstm_units: int = 32,
        fusion_strategy: str = "film",
        fusion_dense_units: int = 48,
        context_dropout_rate: float = 0.15,
    ) -> None:
        """Initialise the predictor with hyper-parameters.

        Args:
            sequence_length: Number of time-steps per input window.
            n_features: Number of input features per time-step.
            lstm_units: Units in each LSTM layer.
            dense_units: Units in the intermediate Dense layer (single-stream).
            dropout_rate: Dropout fraction (0–1).
            learning_rate: Adam optimiser learning rate.
            output_size: Dimensionality of the output prediction.
            dual_stream: Enable the dual-stream architecture.
            n_context_features: Number of context features per time-step.
            context_lstm_units: Units in the context stream LSTM.
            fusion_strategy: ``'concat'``, ``'gate'``, or ``'film'``.
            fusion_dense_units: Units in the Dense layer after fusion.
            context_dropout_rate: Probability of replacing a context timestep
                with stale data during training.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.dual_stream = dual_stream
        self.n_context_features = n_context_features
        self.context_lstm_units = context_lstm_units
        self.fusion_strategy = fusion_strategy
        self.fusion_dense_units = fusion_dense_units
        self.context_dropout_rate = context_dropout_rate
        self.model: Optional[tf.keras.Model] = None

        logger.info(
            "QuaternionLSTMPredictor initialised — seq_len=%d, features=%d, "
            "lstm_units=%d, dense_units=%d, dropout=%.2f, lr=%.4f, output=%d, "
            "dual_stream=%s",
            sequence_length,
            n_features,
            lstm_units,
            dense_units,
            dropout_rate,
            learning_rate,
            output_size,
            dual_stream,
        )
        if dual_stream:
            logger.info(
                "  Dual-stream config — context_features=%d, "
                "context_lstm=%d, fusion=%s, fusion_dense=%d, "
                "context_dropout=%.2f",
                n_context_features,
                context_lstm_units,
                fusion_strategy,
                fusion_dense_units,
                context_dropout_rate,
            )

    def _build_single_stream(self) -> tf.keras.Model:
        """Build the original Sequential model (backwards-compatible)."""
        model = Sequential(
            [
                LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.n_features),
                ),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(self.dense_units, activation="relu"),
                Dense(self.output_size, activation="linear"),
            ]
        )
        return model

    def _build_dual_stream(self) -> tf.keras.Model:
        """Build the Functional API dual-stream model.

        Architecture:
            Price stream:   LSTM(lstm_units) → Dropout → LSTM(lstm_units) → Dropout → h_price
            Context stream: ContextDropout → LSTM(context_lstm_units) → Dropout → h_context
            Fusion:         FiLM | Gate | Concat → Dense(fusion_dense_units, relu)
            Output:         Dense(output_size, linear)
        """
        # --- Price stream ---
        price_input = Input(
            shape=(self.sequence_length, self.n_features),
            name="price_input",
        )
        p = LSTM(self.lstm_units, return_sequences=True, name="price_lstm_1")(
            price_input
        )
        p = Dropout(self.dropout_rate, name="price_dropout_1")(p)
        p = LSTM(self.lstm_units, return_sequences=False, name="price_lstm_2")(p)
        h_price = Dropout(self.dropout_rate, name="price_dropout_2")(p)

        # --- Context stream ---
        context_input = Input(
            shape=(self.sequence_length, self.n_context_features),
            name="context_input",
        )
        c = ContextDropout(
            rate=self.context_dropout_rate, name="context_dropout_input"
        )(context_input)
        c = LSTM(
            self.context_lstm_units, return_sequences=False, name="context_lstm"
        )(c)
        h_context = Dropout(self.dropout_rate, name="context_dropout")(c)

        # --- Fusion ---
        if self.fusion_strategy == "concat":
            fused = Concatenate(name="fusion_concat")([h_price, h_context])
            fused = Dense(
                self.fusion_dense_units, activation="relu", name="fusion_dense"
            )(fused)

        elif self.fusion_strategy == "gate":
            gate = Dense(
                self.lstm_units, activation="sigmoid", name="gate"
            )(h_context)
            fused = Multiply(name="fusion_gate")([h_price, gate])
            fused = Dense(
                self.fusion_dense_units, activation="relu", name="fusion_dense"
            )(fused)

        elif self.fusion_strategy == "film":
            gamma = Dense(
                self.lstm_units,
                activation="linear",
                name="film_gamma",
                kernel_initializer="ones",
                bias_initializer="zeros",
            )(h_context)
            beta = Dense(
                self.lstm_units,
                activation="linear",
                name="film_beta",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )(h_context)
            scaled = Multiply(name="film_scale")([h_price, gamma])
            fused = Add(name="film_shift")([scaled, beta])
            fused = Dense(
                self.fusion_dense_units, activation="relu", name="fusion_dense"
            )(fused)

        else:
            raise ValueError(
                f"Unknown fusion strategy '{self.fusion_strategy}'. "
                f"Choose from 'concat', 'gate', or 'film'."
            )

        # --- Output ---
        output = Dense(
            self.output_size, activation="linear", name="output"
        )(fused)

        model = Model(
            inputs=[price_input, context_input],
            outputs=output,
            name="DualStreamQuaternionLSTM",
        )
        return model

    def build_model(self) -> tf.keras.Model:
        """Build and compile the model.

        Builds the single-stream Sequential model when ``dual_stream=False``,
        or the Functional API dual-stream model when ``dual_stream=True``.

        Returns:
            The compiled ``tf.keras.Model``.
        """
        logger.info(
            "Building %s model …",
            "dual-stream" if self.dual_stream else "single-stream",
        )

        if self.dual_stream:
            model = self._build_dual_stream()
        else:
            model = self._build_single_stream()

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        self.model = model
        logger.info("Model built and compiled successfully.")
        logger.info("\n%s", self.get_summary())
        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window input/target pairs.

        Args:
            data: 2-D array of shape ``(N, n_features)``.
            sequence_length: Window size; defaults to ``self.sequence_length``.

        Returns:
            Tuple ``(X, y)`` where ``X`` has shape
            ``(N - sequence_length, sequence_length, n_features)`` and ``y``
            has shape ``(N - sequence_length, n_features)``.
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(
            "Prepared %d sequences — X shape: %s, y shape: %s",
            len(X),
            X.shape,
            y.shape,
        )
        return X, y

    def train(
        self,
        X_train: Union[np.ndarray, List[np.ndarray]],
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        save_best: bool = True,
        model_path: str = "saved_models/best_model.keras",
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        """Train the LSTM model.

        Args:
            X_train: Training input sequences.  For single-stream, a single
                array of shape ``(N, seq_len, 4)``.  For dual-stream, a list
                ``[price_X, ctx_X]`` of two arrays.
            y_train: Training target values.
            epochs: Maximum number of training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction of training data used for validation.
            save_best: Whether to save the best model via ``ModelCheckpoint``.
            model_path: File path for the saved model checkpoint.

        Returns:
            The Keras ``History`` object from ``model.fit()``.

        Raises:
            RuntimeError: If the model has not been built yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built. Call build_model() before training."
            )

        logger.info(
            "Starting training — epochs=%d, batch_size=%d, "
            "validation_split=%.2f, save_best=%s",
            epochs,
            batch_size,
            validation_split,
            save_best,
        )

        # --- Callbacks ---
        internal_callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if save_best:
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            internal_callbacks.append(
                ModelCheckpoint(
                    filepath=model_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        if callbacks:
            internal_callbacks.extend(callbacks)

        # --- Fit ---
        # Keras model.fit() natively accepts a list of arrays for
        # multi-input models (Functional API), so no special handling
        # is needed for dual-stream vs single-stream.
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=internal_callbacks,
            verbose=1,
        )

        logger.info(
            "Training complete — final loss: %.6f, final MAE: %.6f",
            history.history["loss"][-1],
            history.history["mae"][-1],
        )
        return history

    def predict(
        self, X: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """Generate predictions for the given input sequences.

        Args:
            X: Input array of shape ``(n_samples, sequence_length, n_features)``
               for single-stream, or ``[price_X, ctx_X]`` for dual-stream.

        Returns:
            Predicted values of shape ``(n_samples, output_size)``.

        Raises:
            RuntimeError: If the model has not been built or loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not available. Build or load a model first."
            )

        predictions = self.model.predict(X, verbose=0)
        logger.info("Generated %d predictions (shape %s)", len(predictions), predictions.shape)
        return predictions

    def evaluate(
        self,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test input sequences (single array or list for dual-stream).
            y_test: Test target values.

        Returns:
            Dictionary with ``'loss'`` (MSE) and ``'mae'`` keys.

        Raises:
            RuntimeError: If the model has not been built or loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not available. Build or load a model first."
            )

        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {"loss": results[0], "mae": results[1]}

        logger.info("Evaluation — Loss (MSE): %.6f, MAE: %.6f", metrics["loss"], metrics["mae"])
        return metrics

    def save_model(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Destination file path (e.g. ``'models/my_model.keras'``).

        Raises:
            RuntimeError: If no model is available to save.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Build or train a model first.")

        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        self.model.save(path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Path to the saved model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = keras_load_model(
            path,
            custom_objects={"ContextDropout": ContextDropout},
        )
        logger.info("Model loaded from %s", path)

    def get_summary(self) -> str:
        """Return the model summary as a string.

        Returns:
            Human-readable model summary.

        Raises:
            RuntimeError: If the model has not been built or loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not available. Build or load a model first."
            )

        buffer = StringIO()
        self.model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        return buffer.getvalue()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Quick smoke test with dummy data ---
    SEQ_LEN = 30
    N_FEATURES = 4
    N_CONTEXT = 5
    N_SAMPLES = 200

    np.random.seed(42)

    # ==========================================
    # Test 1: Single-stream (backwards compat)
    # ==========================================
    print("=" * 60)
    print("Test 1: Single-Stream (original architecture)")
    print("=" * 60)

    dummy_data = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)

    predictor_ss = QuaternionLSTMPredictor(
        sequence_length=SEQ_LEN,
        n_features=N_FEATURES,
        lstm_units=32,
        dense_units=16,
        dropout_rate=0.1,
        learning_rate=0.001,
        output_size=N_FEATURES,
        dual_stream=False,
    )
    predictor_ss.build_model()

    X, y = predictor_ss.prepare_sequences(dummy_data, sequence_length=SEQ_LEN)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    history = predictor_ss.train(
        X_train, y_train, epochs=2, batch_size=16,
        validation_split=0.1, save_best=False,
    )
    metrics = predictor_ss.evaluate(X_test, y_test)
    print(f"\nSingle-stream — Test MSE: {metrics['loss']:.6f}, MAE: {metrics['mae']:.6f}")

    # ==========================================
    # Test 2: Dual-stream with FiLM fusion
    # ==========================================
    print("\n" + "=" * 60)
    print("Test 2: Dual-Stream (FiLM fusion + context dropout)")
    print("=" * 60)

    dummy_price = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    dummy_context = np.random.rand(N_SAMPLES, N_CONTEXT).astype(np.float32)

    predictor_ds = QuaternionLSTMPredictor(
        sequence_length=SEQ_LEN,
        n_features=N_FEATURES,
        lstm_units=32,
        dense_units=16,
        dropout_rate=0.1,
        learning_rate=0.001,
        output_size=N_FEATURES,
        dual_stream=True,
        n_context_features=N_CONTEXT,
        context_lstm_units=16,
        fusion_strategy="film",
        fusion_dense_units=24,
        context_dropout_rate=0.15,
    )
    predictor_ds.build_model()

    # Create windowed sequences for both streams
    price_X, price_y = [], []
    ctx_X = []
    for i in range(len(dummy_price) - SEQ_LEN):
        price_X.append(dummy_price[i : i + SEQ_LEN])
        price_y.append(dummy_price[i + SEQ_LEN])
        ctx_X.append(dummy_context[i : i + SEQ_LEN])

    price_X = np.array(price_X)
    price_y = np.array(price_y)
    ctx_X = np.array(ctx_X)

    split = int(len(price_X) * 0.8)
    train_input = [price_X[:split], ctx_X[:split]]
    test_input = [price_X[split:], ctx_X[split:]]

    history = predictor_ds.train(
        train_input, price_y[:split], epochs=2, batch_size=16,
        validation_split=0.1, save_best=False,
    )
    metrics = predictor_ds.evaluate(test_input, price_y[split:])
    print(f"\nDual-stream — Test MSE: {metrics['loss']:.6f}, MAE: {metrics['mae']:.6f}")

    preds = predictor_ds.predict(test_input[:3] if isinstance(test_input, np.ndarray) else [ti[:3] for ti in test_input])
    print(f"Sample predictions shape: {preds.shape}")

    print(f"\nModel Summary:\n{predictor_ds.get_summary()}")

    print("=" * 60)
    print("All smoke tests complete ✓")
    print("=" * 60)
