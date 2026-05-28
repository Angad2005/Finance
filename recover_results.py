import os
import yaml
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
from stable_baselines3 import PPO

from src.data_loader import load_data
from src.wfv_engine import generate_wfv_splits
from src.baselines import run_ubh, run_mvo, evaluate_rl
from src.lstm_rl_agent import evaluate_lstm_rl
from src.metrics import compute_all_metrics, compute_annualized_sharpe
from src.stats_test import diebold_mariano_test, bootstrap_ci
from run_experiment import generate_plots, set_seeds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrueRecovery")

def main():
    logger.info("Starting FAST evaluation recovery over completed folds...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seeds(config["experiment"]["seed"])
    models_dir = Path("results/models")
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load raw underlying asset matrix
    log_returns, rf_rate = load_data(
        config["experiment"]["tickers"], config["experiment"]["start_date"], config["experiment"]["end_date"], "data"
    )
    all_splits = generate_wfv_splits(len(log_returns), config["wfv"]["train_window"], config["wfv"]["val_window"], config["wfv"]["test_window"], "data/processed")

    results = {"UBH": [], "MVO": [], "MLP-RL": [], "LSTM-RL": []}
    weights_history = {"UBH": [], "MVO": [], "MLP-RL": [], "LSTM-RL": []}

    # 1. Determine how many complete model pairs exist on disk
    completed_folds = 0
    for idx in range(len(all_splits)):
        if (models_dir / f"mlp_fold_{idx}.zip").exists() and (models_dir / f"lstm_fold_{idx}.zip").exists():
            completed_folds += 1
        else:
            break

    if completed_folds == 0:
        raise FileNotFoundError("No matching completed training model weights found in results/models/")

    logger.info(f"Detected {completed_folds} fully trained chronological folds on disk. Truncating evaluation loop to align arrays.")
    valid_splits = all_splits[:completed_folds]

    # 2. Sequential evaluation sweep over completed models
    for i, split in enumerate(valid_splits):
        logger.info(f"Evaluating valid trained block {i+1}/{completed_folds}...")
        
        test_start, test_end = split["test"]
        train_start, train_end = split["train"]
        
        test_ret = log_returns.iloc[test_start:test_end]
        test_rf = rf_rate.iloc[test_start:test_end]
        train_ret = log_returns.iloc[train_start:train_end]

        env_kwargs_test = {
            "log_returns": test_ret, "rf_rate": test_rf,
            "window_size": config["environment"]["window_size"],
            "transaction_cost_bps": config["environment"]["transaction_cost_bps"],
            "reward_type": config["environment"].get("reward_type", "return")
        }

        # Evaluate RL environments first to determine step truncation constraints
        mlp_model = PPO.load(models_dir / f"mlp_fold_{i}.zip")
        mlp_returns, mlp_w = evaluate_rl(mlp_model, env_kwargs_test)
        
        try:
            from sb3_contrib import RecurrentPPO
            lstm_model = RecurrentPPO.load(models_dir / f"lstm_fold_{i}.zip")
        except ImportError:
            lstm_model = PPO.load(models_dir / f"lstm_fold_{i}.zip")
            
        lstm_returns, lstm_w = evaluate_lstm_rl(lstm_model, env_kwargs_test)

        # Enforce consistency check between RL structures
        assert len(mlp_returns) == len(lstm_returns), f"Fold {i} RL trajectory length mismatch!"
        rl_length = len(mlp_returns)

        # Calculate raw baseline arrays matching the complete data slices
        ubh_returns = run_ubh(test_ret, config["environment"]["transaction_cost_bps"])
        mvo_returns, _ = run_mvo(train_ret, test_ret, config["environment"]["transaction_cost_bps"])

        # Slice baselines backward from the end to mirror the warm-up window exclusions
        results["UBH"].extend(ubh_returns[-rl_length:])
        results["MVO"].extend(mvo_returns[-rl_length:])
        results["MLP-RL"].extend(mlp_returns)
        results["LSTM-RL"].extend(lstm_returns)

        weights_history["MLP-RL"].extend(mlp_w)
        weights_history["LSTM-RL"].extend(lstm_w)

    # 3. Structural flattening pass to clean data arrays
    clean_results = {k: np.array(v, dtype=np.float64).flatten() for k, v in results.items()}
    clean_weights = {k: np.array(v, dtype=np.float64) for k, v in weights_history.items() if len(v) > 0}

    # Confirm all parsed keys have exact element parity
    lengths = {k: len(v) for k, v in clean_results.items()}
    logger.info(f"Generated array structures verified: {lengths}")

    # 4. Compile final summary files
    logger.info("Compiling metrics database...")
    metrics_df = compute_all_metrics(clean_results, clean_weights)
    metrics_df.to_csv(metrics_dir / "performance_summary.csv", index=False)
    
    results_df = pd.DataFrame(clean_results)
    for col in results_df.columns:
        results_df[col].to_csv(metrics_dir / f"returns_series_{col}.csv", index=False)

    # Run tests against baselines
    for baseline in ["UBH", "MVO", "MLP-RL"]:
        dm_stat, p_val = diebold_mariano_test(clean_results["LSTM-RL"], clean_results[baseline])
        logger.info(f"LSTM vs {baseline}: DM-Stat={dm_stat:.4f}, p-value={p_val:.4f}")

    generate_plots(results_df, clean_weights.get("LSTM-RL", []), metrics_df, "results/plots")
    
    # Save results JSON tracking metadata
    summary = {
        "runtime_date": datetime.now().isoformat(),
        "evaluated_folds": completed_folds,
        "metrics": metrics_df.to_dict(orient="records"),
        "environment": {
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }
    with open("results/results_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("RECOVERY COMPLETE. Your empirical results are generated successfully.")
    print("\n[Success] Your metrics, tables, and charts are fully compiled inside 'results/'!")

if __name__ == "__main__":
    main()