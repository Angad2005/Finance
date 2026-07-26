import pandas as pd
import numpy as np
import logging
import torch
import os
from pathlib import Path

from src.baselines import run_ubh, run_mvo, train_mlp_rl, evaluate_rl
from src.lstm_rl_agent import train_lstm_rl, evaluate_lstm_rl

logger = logging.getLogger(__name__)

def run_wfv_backtest(log_returns, rf_rate, splits, config):
    results = {
        "UBH": [],
        "MVO": [],
        "MLP-RL": [],
        "LSTM-RL": []
    }
    
    weights_history = {
        "UBH": [],
        "MVO": [],
        "MLP-RL": [],
        "LSTM-RL": []
    }
    
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for i, split in enumerate(splits):
        logger.info(f"--- Starting Fold {i+1}/{len(splits)} ---")
        
        train_start, train_end = split["train"]
        val_start, val_end = split["val"]
        test_start, test_end = split["test"]
        
        train_ret = log_returns.iloc[train_start:train_end]
        val_ret = log_returns.iloc[val_start:val_end]
        test_ret = log_returns.iloc[test_start:test_end]
        
        train_rf = rf_rate.iloc[train_start:train_end]
        val_rf = rf_rate.iloc[val_start:val_end]
        test_rf = rf_rate.iloc[test_start:test_end]
        
        env_kwargs_train = {
            "log_returns": train_ret,
            "rf_rate": train_rf,
            "window_size": config["environment"]["window_size"],
            "transaction_cost_bps": config["environment"]["transaction_cost_bps"],
            "reward_type": config["environment"].get("reward_type", "return")
        }
        
        env_kwargs_val = {
            "log_returns": val_ret,
            "rf_rate": val_rf,
            "window_size": config["environment"]["window_size"],
            "transaction_cost_bps": config["environment"]["transaction_cost_bps"],
            "reward_type": config["environment"].get("reward_type", "return")
        }
        
        env_kwargs_test = {
            "log_returns": test_ret,
            "rf_rate": test_rf,
            "window_size": config["environment"]["window_size"],
            "transaction_cost_bps": config["environment"]["transaction_cost_bps"],
            "reward_type": config["environment"].get("reward_type", "return")
        }
        
        # 1. UBH
        logger.info("Running UBH baseline...")
        ubh_returns = run_ubh(test_ret, config["environment"]["transaction_cost_bps"])
        
        # 2. MVO
        logger.info("Running MVO baseline...")
        mvo_returns, mvo_weights = run_mvo(train_ret, test_ret, config["environment"]["transaction_cost_bps"])
        np.save(models_dir / f"mvo_weights_fold_{i}.npy", mvo_weights)
        
        # 3. MLP-RL
        logger.info("Training MLP-RL...")
        mlp_model = train_mlp_rl(
            env_kwargs_train, 
            config["rl"]["total_timesteps"], 
            config["rl"]["net_arch"],
            config["experiment"]["seed"]
        )
        mlp_model.save(models_dir / f"mlp_fold_{i}.zip")
        
        mlp_returns, mlp_weights = evaluate_rl(mlp_model, env_kwargs_test)
        
        del mlp_model
        torch.cuda.empty_cache()
        
        # 4. LSTM-RL
        logger.info("Training LSTM-RL...")
        lstm_model = train_lstm_rl(
            env_kwargs_train,
            env_kwargs_val,
            config["rl"]["total_timesteps"],
            config["rl"]["net_arch"],
            config["rl"]["lstm_hidden_size"],
            config["rl"]["n_lstm_layers"],
            config["experiment"]["seed"]
        )
        lstm_model.save(models_dir / f"lstm_fold_{i}.zip")
        
        lstm_returns, lstm_weights = evaluate_lstm_rl(lstm_model, env_kwargs_test)
        
        del lstm_model
        torch.cuda.empty_cache()
        
        # ------------------------------------------------------------------ #
        # Align all fold return series to the same length before accumulating.
        # UBH/MVO iterate over all test_ret rows (T steps), while RL agents
        # use a sliding observation window and therefore produce only
        # (T - window_size) steps.  Trimming to the minimum ensures the
        # global results lists stay equal-length and pd.DataFrame(results)
        # never raises "All arrays must be of the same length".
        # ------------------------------------------------------------------ #
        fold_lengths = {
            "UBH": len(ubh_returns),
            "MVO": len(mvo_returns),
            "MLP-RL": len(mlp_returns),
            "LSTM-RL": len(lstm_returns),
        }
        min_len = min(fold_lengths.values())
        if len(set(fold_lengths.values())) > 1:
            logger.warning(
                f"Fold {i+1}: strategy return lengths differ {fold_lengths}. "
                f"Trimming all to {min_len} steps (the minimum)."
            )
        
        results["UBH"].extend(ubh_returns[-min_len:])
        results["MVO"].extend(mvo_returns[-min_len:])
        results["MLP-RL"].extend(mlp_returns[:min_len])
        weights_history["MLP-RL"].extend(mlp_weights[:min_len])
        results["LSTM-RL"].extend(lstm_returns[:min_len])
        weights_history["LSTM-RL"].extend(lstm_weights[:min_len])
        
    return results, weights_history
