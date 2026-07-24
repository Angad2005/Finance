import numpy as np
import pandas as pd

def compute_cumulative_return(returns):
    returns = np.asarray(returns, dtype=float)
    return np.prod(1 + returns) - 1.0

def compute_max_drawdown(returns):
    returns = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)

def compute_annualized_sharpe(returns, risk_free_rate=0.0):
    returns_arr = np.array(returns)
    excess_returns = returns_arr - risk_free_rate
    mean_er = np.mean(excess_returns)
    std_er = np.std(excess_returns)
    if std_er == 0:
        return 0.0
    return np.sqrt(252) * mean_er / std_er

def compute_turnover(weights_history):
    # Annualized turnover
    # weights_history shape: (T, N)
    if len(weights_history) <= 1:
        return 0.0
    diffs = np.sum(np.abs(np.diff(weights_history, axis=0)), axis=1)
    mean_daily_turnover = np.mean(diffs)
    return mean_daily_turnover * 252

def compute_all_metrics(returns_series_dict, weights_dict=None):
    metrics = []
    for strategy, returns in returns_series_dict.items():
        cr = compute_cumulative_return(returns)
        mdd = compute_max_drawdown(returns)
        sharpe = compute_annualized_sharpe(returns)
        
        turnover = np.nan
        if weights_dict and strategy in weights_dict:
            turnover = compute_turnover(weights_dict[strategy])
            
        metrics.append({
            "Strategy": strategy,
            "Cumulative Return": cr,
            "Max Drawdown": mdd,
            "Annualized Sharpe": sharpe,
            "Annualized Turnover": turnover
        })
    return pd.DataFrame(metrics)
