import re

with open("api.py", "r") as f:
    content = f.read()

old_logic = """
        # 2. Prepare data
        data = prepare_training_data(
            ohlcv_df,
            sequence_length=config.SEQUENCE_LENGTH,
            train_split=config.TRAIN_TEST_SPLIT,
            use_path_deltas=False,
            dual_stream=True,  # Force dual_stream=True to guarantee ctx_X is available if any model needs it
            volume_ma_window=config.VOLUME_MA_WINDOW if config.DUAL_STREAM else 20,
        )
        
        scaler = data["scaler"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        ctx_X_test = data.get("ctx_X_test")
        
        # 4. Get actual prices
        actual_prices = decode_quaternion_to_price(y_test, scaler)
        
        # If no models specified, fallback to active primary
        if not models:
            models = [ACTIVE_MODELS["primary"]]
            
        model_predictions = []
        import tensorflow as tf
        from lstm_model import ContextDropout
        
        # 3. Iterate over requested models and get predictions
        for model_filename in models:
            model_path = os.path.join(config.MODEL_SAVE_DIR, model_filename)
            if not os.path.exists(model_path):
                logger.warning("Requested model not found: %s", model_filename)
                continue
                
            temp_model = tf.keras.models.load_model(
                model_path,
                custom_objects={"ContextDropout": ContextDropout}
            )
            
            # Dynamically infer architecture
            if temp_model.name == "MTLQuaternionPredictor":
                mod_type = "mtl"
            else:
                mod_type = "lstm"
                
            predictor = build_primary_model(mod_type)
            predictor.model = temp_model
            
            # Predict
            predicted_prices = simulate_teacher_forcing(
                predictor, scaler, X_test, ctx_X_test=ctx_X_test,
            )
            
            # Error metrics
            metrics = analyze_errors(actual_prices, predicted_prices)
            
            model_predictions.append({
                "name": model_filename,
                "predicted_prices": [round(float(p), 4) for p in predicted_prices],
                "metrics": {k: round(float(v), 4) for k, v in metrics.items()}
            })
            
            # We will use the first successfully loaded model to run the oscillator
            if "primary_predictor" not in locals():
                primary_predictor = predictor
"""

new_logic = """
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
            else:
                mod_type = "lstm"
                
            predictor = build_primary_model(mod_type)
            predictor.model = temp_model
            
            loaded_models.append({
                "filename": model_filename,
                "predictor": predictor,
                "seq_len": seq_len
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
            
            if "primary_predictor" not in locals():
                primary_predictor = predictor
                data = reference_data  # Save for oscillator
"""

content = content.replace(old_logic.strip(), new_logic.strip())

with open("api.py", "w") as f:
    f.write(content)
print("api.py patched successfully")
