"""
Smoke Test for Extended MTL Model and Pipeline.
"""

import numpy as np
import pandas as pd
import logging

from extended_encoder import prepare_extended_training_data
from extended_mtl_model import ExtendedMTLPredictor
from extended_signal_model import FeedbackOscillator, ExtendedPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("=" * 60)
    print("Testing Extended MTL Model & Pipeline")
    print("=" * 60)

    # 1. Generate Dummy Data
    np.random.seed(42)
    n_days = 200
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    closes = base_price * np.cumprod(1 + returns)

    demo_df = pd.DataFrame(
        {
            "Open": closes * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "High": closes * (1 + np.random.uniform(0.005, 0.025, n_days)),
            "Low": closes * (1 - np.random.uniform(0.005, 0.025, n_days)),
            "Close": closes,
            "Volume": np.random.uniform(1000, 5000, n_days),
        }
    )

    SEQ_LEN = 30

    # 2. Prepare Data
    print("\n--- 1. Data Preparation ---")
    data_dict = prepare_extended_training_data(demo_df, sequence_length=SEQ_LEN)
    
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    
    print(f"X_seq shape: {X_train[0].shape}")
    print(f"X_ctx shape: {X_train[1].shape}")
    print(f"X_current shape: {X_train[2].shape}")
    print(f"y_mom_w shape: {y_train['mom_w'].shape}")
    print(f"y_out_main shape: {y_train['out_main'].shape}")

    # 3. Build & Train Predictor
    print("\n--- 2. Building Predictor ---")
    predictor = ExtendedMTLPredictor(sequence_length=SEQ_LEN, lstm_units=32, aux_dense_units=16)
    predictor.build_model()
    
    print("\n--- 3. Training Predictor ---")
    predictor.model.fit(
        x=X_train,
        y=y_train,
        epochs=1,
        batch_size=16,
        verbose=1
    )

    # 4. Build Oscillator
    print("\n--- 4. Building Oscillator ---")
    oscillator = FeedbackOscillator(sequence_length=14)
    oscillator.build_model()

    # Dummy residual data for training the oscillator
    n_samples = len(X_train[0])
    res_seq = np.random.randn(n_samples, 14, 1).astype(np.float32)
    next_q = predictor.predict_next_q(X_train[0], X_train[1], X_train[2])
    
    # We use a dummy target for the signal [-1, 1]
    osc_targets = np.random.uniform(-1, 1, size=(n_samples, 1)).astype(np.float32)

    print("\n--- 5. Training Oscillator ---")
    oscillator.model.fit(
        x=[res_seq, next_q],
        y=osc_targets,
        epochs=1,
        batch_size=16,
        verbose=1
    )

    # 5. Test the Integrated Pipeline
    print("\n--- 6. Integrated Pipeline Inference ---")
    pipeline = ExtendedPipeline(predictor, oscillator, data_dict["scaler"])
    
    # Test on a few samples
    sample_res = res_seq[:3]
    result = pipeline.predict_with_feedback(X_train[0][:3], X_train[1][:3], X_train[2][:3], sample_res)
    
    print("Pipeline Results:")
    print(f"Predicted Q shape: {result['predicted_quaternion'].shape}")
    print(f"Predicted Close Price: {result['predicted_close_price']}")
    print(f"Trading Signal: {result['trading_signal'].flatten()}")
    
    print("\nSmoke test completed successfully ✓")

if __name__ == "__main__":
    main()
