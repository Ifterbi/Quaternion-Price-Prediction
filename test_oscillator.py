import logging
import numpy as np
import tensorflow as tf
from data_fetcher import fetch_bitcoin_data
from oscillator_data import prepare_oscillator_training_data
from signal_model import SelfLearningOscillator

def run_test():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Fetching data...")
    df = fetch_bitcoin_data(start="2022-01-01", interval="1d")
    
    logger.info("Preparing data...")
    seq_len = 14
    data_dict = prepare_oscillator_training_data(df, sequence_length=seq_len)
    
    logger.info("Building SelfLearningOscillator...")
    osc = SelfLearningOscillator(sequence_length=seq_len)
    osc.build_model()
    
    # Optional: view summary
    logger.info(osc.get_summary())
    
    logger.info("Training oscillator...")
    osc.train(data_dict, epochs=10, batch_size=32)
    
    logger.info("Evaluating oscillator...")
    osc.evaluate(data_dict)
    
    logger.info("Running simulation on test set...")
    # Evaluate signals on a small sample
    sample_price = data_dict["X_price_test"][:10]
    sample_q = data_dict["X_q_test"][:10]
    sample_y = data_dict["y_test"][:10]
    
    predictions = osc.predict(sample_price, sample_q)
    
    logger.info("Simulation Results:")
    for i in range(10):
        osc_val, buy_t, sell_t = predictions[i]
        actual_profit = sample_y[i][0]
        
        signal = "HOLD"
        if osc_val > buy_t:
            signal = "BUY "
        elif osc_val < sell_t:
            signal = "SELL"
            
        profit = 0
        if signal == "BUY ":
            profit = actual_profit
        elif signal == "SELL":
            profit = -actual_profit
            
        logger.info(f"Step {i+1}: Sig={signal} | Osc={osc_val:+.4f} (B:{buy_t:+.4f}, S:{sell_t:+.4f}) | Actual Change={actual_profit:+.4f} | Profit={profit:+.4f}")
        
if __name__ == "__main__":
    run_test()
