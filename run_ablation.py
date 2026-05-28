import os
import yaml
import logging
import random
import numpy as np
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_data
from src.wfv_engine import generate_wfv_splits
from src.backtest_runner import run_wfv_backtest
from src.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ablation_runner")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    logger.info("Initializing ablation and sensitivity studies...")
    set_seeds(42)
    
    # Create output directories
    ablation_dir = Path("results/ablation")
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    # We override tickers, total_timesteps, and reward_type for ablation
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    total_timesteps = 10000
    reward_type = "sortino"
    
    logger.info(f"Downloading data for ablation tickers: {tickers}")
    log_returns, rf_rate = load_data(
        tickers,
        base_config["experiment"]["start_date"],
        base_config["experiment"]["end_date"],
        "data/processed"
    )
    
    splits = generate_wfv_splits(
        len(log_returns),
        base_config["wfv"]["train_window"],
        base_config["wfv"]["val_window"],
        base_config["wfv"]["test_window"],
        "data/processed"
    )
    # Speedup: only run the last 2 splits for ablation/sensitivity verification
    splits = splits[-2:]
    
    # Define experiment config templates
    def get_config(window_size=60, cost_bps=10):
        config = {
            "experiment": {
                "seed": 42,
                "start_date": base_config["experiment"]["start_date"],
                "end_date": base_config["experiment"]["end_date"],
                "tickers": tickers
            },
            "wfv": base_config["wfv"],
            "environment": {
                "window_size": window_size,
                "transaction_cost_bps": cost_bps,
                "reward_type": reward_type
            },
            "rl": {
                "total_timesteps": total_timesteps,
                "net_arch": base_config["rl"]["net_arch"],
                "lstm_hidden_size": base_config["rl"]["lstm_hidden_size"],
                "n_lstm_layers": base_config["rl"]["n_lstm_layers"]
            }
        }
        return config

    # ==========================================
    # Study 1: Look-Back Window Size Ablation
    # ==========================================
    logger.info("==========================================")
    logger.info("Study 1: Look-back window size ablation (10 vs 60 days)...")
    logger.info("==========================================")
    
    window_sizes = [10, 60]
    window_results = {}
    
    for ws in window_sizes:
        logger.info(f"Running backtest for look-back window size: {ws}")
        config = get_config(window_size=ws, cost_bps=10)
        results, weights_history = run_wfv_backtest(log_returns, rf_rate, splits, config)
        metrics_df = compute_all_metrics(results, weights_history)
        
        # Save metrics
        metrics_df.to_csv(ablation_dir / f"metrics_window_{ws}.csv", index=False)
        window_results[ws] = {
            "returns": results,
            "metrics": metrics_df.to_dict(orient="records"),
            "weights": weights_history
        }

    # Plot Study 1
    plt.figure(figsize=(12, 6))
    # UBH and MVO are relatively stable across look-backs, we plot LSTM-RL and baselines from the ws=60 run
    # and LSTM-RL from ws=10 run
    r60 = window_results[60]["returns"]
    r10 = window_results[10]["returns"]
    
    plt.plot((1 + pd.Series(r60["UBH"])).cumprod(), label="UBH", alpha=0.7)
    plt.plot((1 + pd.Series(r60["MVO"])).cumprod(), label="MVO", alpha=0.7)
    plt.plot((1 + pd.Series(r10["LSTM-RL"])).cumprod(), label="LSTM-RL (10-day lookback)", style="--")
    plt.plot((1 + pd.Series(r60["LSTM-RL"])).cumprod(), label="LSTM-RL (60-day lookback)")
    
    plt.yscale('log')
    plt.title('Look-Back Window Ablation: Equity Curves (Log Scale)')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(ablation_dir / "window_ablation_equity.png")
    plt.close()
    
    # Compare key metrics
    comparison = []
    for ws in window_sizes:
        lstm_row = next(item for item in window_results[ws]["metrics"] if item["Strategy"] == "LSTM-RL")
        comparison.append({
            "Window Size": f"{ws} Days",
            "Cumulative Return": lstm_row["Cumulative Return"],
            "Max Drawdown": lstm_row["Max Drawdown"],
            "Annualized Sharpe": lstm_row["Annualized Sharpe"],
            "Annualized Turnover": lstm_row["Annualized Turnover"]
        })
    pd.DataFrame(comparison).to_csv(ablation_dir / "window_ablation_summary.csv", index=False)

    # ==========================================
    # Study 2: Transaction Cost Sensitivity
    # ==========================================
    logger.info("==========================================")
    logger.info("Study 2: Transaction cost sensitivity (0, 10, 50, 100 bps)...")
    logger.info("==========================================")
    
    cost_levels = [0, 10, 50, 100]
    cost_results = {}
    
    for bps in cost_levels:
        logger.info(f"Running backtest for transaction cost: {bps} bps")
        config = get_config(window_size=60, cost_bps=bps)
        results, weights_history = run_wfv_backtest(log_returns, rf_rate, splits, config)
        metrics_df = compute_all_metrics(results, weights_history)
        
        metrics_df.to_csv(ablation_dir / f"metrics_cost_{bps}bps.csv", index=False)
        cost_results[bps] = {
            "returns": results,
            "metrics": metrics_df.to_dict(orient="records"),
            "weights": weights_history
        }
        
    # Summarize and plot cost sensitivity
    cost_summary = []
    for bps in cost_levels:
        lstm_row = next(item for item in cost_results[bps]["metrics"] if item["Strategy"] == "LSTM-RL")
        cost_summary.append({
            "Cost (bps)": bps,
            "Cumulative Return": lstm_row["Cumulative Return"],
            "Max Drawdown": lstm_row["Max Drawdown"],
            "Annualized Sharpe": lstm_row["Annualized Sharpe"],
            "Annualized Turnover": lstm_row["Annualized Turnover"]
        })
    cost_summary_df = pd.DataFrame(cost_summary)
    cost_summary_df.to_csv(ablation_dir / "cost_sensitivity_summary.csv", index=False)
    
    # Plot Cost vs Sharpe & Turnover
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Transaction Cost (bps)')
    ax1.set_ylabel('Annualized Sharpe Ratio', color=color)
    ax1.plot(cost_summary_df["Cost (bps)"], cost_summary_df["Annualized Sharpe"], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Annualized Turnover Rate', color=color)
    ax2.plot(cost_summary_df["Cost (bps)"], cost_summary_df["Annualized Turnover"], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Transaction Cost Sensitivity: Sharpe Ratio vs. Portfolio Turnover')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(ablation_dir / "cost_sensitivity_curves.png")
    plt.close()
    
    logger.info("Ablation and sensitivity studies completed successfully. Plots and summaries saved to results/ablation/.")

if __name__ == "__main__":
    main()
