import os
import yaml
import logging
import random
import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from src.data_loader import load_data
from src.wfv_engine import generate_wfv_splits
from src.backtest_runner import run_wfv_backtest
from src.metrics import compute_all_metrics
from src.stats_test import diebold_mariano_test, bootstrap_ci

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging():
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if script is rerun interactively
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(log_dir / "experiment.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def generate_plots(results_df, lstm_weights, metrics_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Equity Curves
    plt.figure(figsize=(12, 6))
    for col in results_df.columns:
        cumulative = (1 + results_df[col]).cumprod()
        plt.plot(cumulative.index, cumulative, label=col)
    plt.yscale('log')
    plt.title('Equity Curves (Log Scale)')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(out_dir / "equity_curves.png")
    plt.close()
    
    # 2. Rolling Sharpe (60 days)
    plt.figure(figsize=(12, 6))
    for col in results_df.columns:
        rolling_mean = results_df[col].rolling(60).mean()
        rolling_std = results_df[col].rolling(60).std()
        # Prevent zero-division warnings on initial flat blocks
        rolling_std = rolling_std.replace(0, np.nan)
        rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)
        plt.plot(rolling_sharpe.index, rolling_sharpe, label=col, alpha=0.7)
    plt.title('Rolling 60-Day Annualized Sharpe Ratio')
    plt.xlabel('Days')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "rolling_sharpe_60d.png")
    plt.close()
    
    # 3. LSTM Weights Stacked Area Plot
    if len(lstm_weights) > 0:
        plt.figure(figsize=(12, 6))
        w = np.array(lstm_weights)
        plt.stackplot(range(len(w)), w.T)
        plt.title('LSTM Agent Weight Allocation over Time')
        plt.xlabel('Test Days')
        plt.ylabel('Weights')
        plt.margins(0, 0)
        plt.savefig(out_dir / "weight_allocation_lstm.png")
        plt.close()

def main():
    logger = setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info("Setting seeds...")
        set_seeds(config["experiment"]["seed"])
        
        logger.info("Loading Data...")
        
        # Fixed: Safely pass named parameter structure and attempt disk fallback if Yahoo is blocked
        try:
            log_returns, rf_rate = load_data(
                tickers=config["experiment"]["tickers"],
                start_date=config["experiment"]["start_date"],
                end_date=config["experiment"]["end_date"],
                output_dir="data"
            )
        except Exception as data_err:
            logger.warning(f"Live yfinance pipeline rejected connection ({data_err}). Inspecting local backup caches...")
            
            # Paths where verified cached returns might reside from previous runs
            cache_path_1 = Path("data/processed_returns.csv")
            cache_path_2 = Path("data/processed/processed_returns.csv")
            rf_path_1 = Path("data/risk_free_rate.csv")
            rf_path_2 = Path("data/processed/risk_free_rate.csv")
            
            target_cache = cache_path_1 if cache_path_1.exists() else cache_path_2
            target_rf = rf_path_1 if rf_path_1.exists() else rf_path_2
            
            if target_cache.exists() and target_rf.exists():
                logger.info(f"Verified structural disk arrays located. Loading cached file: {target_cache}")
                log_returns = pd.read_csv(target_cache, index_col=0, parse_dates=True)
                rf_rate = pd.read_csv(target_rf, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                logger.critical("Data layer isolated: Internet connection blocked by Yahoo and no local historical CSV cache exists.")
                raise FileNotFoundError("Real-world asset matrix completely unavailable. Data download halted.")

        logger.info("Generating WFV Splits...")
        splits = generate_wfv_splits(
            len(log_returns),
            config["wfv"]["train_window"],
            config["wfv"]["val_window"],
            config["wfv"]["test_window"],
            "data/processed"
        )
        
        logger.info("Starting Backtest...")
        results, weights_history = run_wfv_backtest(log_returns, rf_rate, splits, config)
        
        logger.info("Computing Metrics...")
        metrics_df = compute_all_metrics(results, weights_history)
        metrics_dir = Path("results/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_dir / "performance_summary.csv", index=False)
        
        results_df = pd.DataFrame(results)
        for col in results_df.columns:
            results_df[col].to_csv(metrics_dir / f"returns_series_{col}.csv", index=False)
            
        logger.info("Running Statistical Tests...")
        lstm_ret = results["LSTM-RL"]
        stat_results = {}
        from src.metrics import compute_annualized_sharpe
        
        for baseline in ["UBH", "MVO", "MLP-RL"]:
            dm_stat, p_val = diebold_mariano_test(lstm_ret, results[baseline])
            lower, upper = bootstrap_ci(lstm_ret, compute_annualized_sharpe)
            stat_results[f"LSTM vs {baseline}"] = {
                "DM_Stat": float(dm_stat) if not np.isnan(dm_stat) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "Bootstrap_95CI_Sharpe": [float(lower), float(upper)]
            }
            
        with open(metrics_dir / "stat_tests.json", "w") as f:
            json.dump(stat_results, f, indent=4)
            
        logger.info("Generating Plots...")
        generate_plots(results_df, weights_history.get("LSTM-RL", []), metrics_df, "results/plots")
        
        logger.info("Creating Final Summary...")
        summary = {
            "runtime_date": datetime.now().isoformat(),
            "config_hash": hash(json.dumps(config, sort_keys=True)),
            "metrics": metrics_df.to_dict(orient="records"),
            "stat_tests": stat_results,
            "environment": {
                "python_version": os.sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        }
        with open("results/results_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        logger.info("Experiment completed successfully.")
        print("\nAll tasks completed. Results saved to results/ directory.")
        
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        error_log_dir = Path("results/logs")
        error_log_dir.mkdir(parents=True, exist_ok=True)
        with open(error_log_dir / "error_tracebacks.log", "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    main()