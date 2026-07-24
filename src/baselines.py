import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns, EfficientFrontier
from stable_baselines3 import PPO
from src.env_portfolio import PortfolioEnv
import logging

logger = logging.getLogger(__name__)

def run_ubh(log_returns, transaction_cost_bps=10):
    n_assets = log_returns.shape[1]
    transaction_cost = transaction_cost_bps / 10000.0
    
    weights = np.ones(n_assets) / n_assets
    current_weights = weights.copy()
    
    returns = []
    simple_returns = np.exp(log_returns) - 1.0
    
    for i in range(len(simple_returns)):
        # Rebalance quarterly (approx 63 days)
        if i % 63 == 0 and i > 0:
            cost = np.sum(np.abs(weights - current_weights)) * transaction_cost
            current_weights = weights.copy()
        else:
            cost = 0.0
            
        step_ret = np.sum(current_weights * simple_returns.iloc[i].values) - cost
        returns.append(step_ret)
        
        # Weight drift
        current_weights = current_weights * (1 + simple_returns.iloc[i].values)
        current_weights /= np.sum(current_weights)
        
    return np.array(returns)

def run_mvo(train_returns, test_returns, transaction_cost_bps=10):
    # Correct price construction: cumulative log‑returns → price series starting at 1.0
    train_prices = np.exp(train_returns.cumsum())
    train_prices.iloc[0] = 1.0

    n_assets = train_returns.shape[1]
    weights = np.ones(n_assets) / n_assets  # default: equal weight fallback

    # Drop near-duplicate or near-constant columns to avoid singular covariance matrix
    def _clean_prices(prices):
        """Remove columns with near-zero variance or near-perfect correlation."""
        # Drop constant / near-constant columns (std < 1e-8)
        std = prices.std()
        prices = prices.loc[:, std > 1e-8]
        if prices.shape[1] < 2:
            return prices
        # Drop columns that are near-perfect duplicates (corr > 0.9999)
        corr = prices.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9999)]
        return prices.drop(columns=to_drop)

    clean_prices = _clean_prices(train_prices)
    active_cols = clean_prices.columns.tolist()

    try:
        S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()
        mu = expected_returns.mean_historical_return(clean_prices)

        # Tier 1: max_sharpe with tight per-asset cap
        upper_bound = max(0.2, 1.0 / len(active_cols))  # at least 1/n to stay feasible
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, upper_bound))
            ef.max_sharpe()
            weights_dict = ef.clean_weights()
            logger.info("MVO: max_sharpe succeeded (tight bounds).")
        except Exception as e1:
            logger.warning(f"MVO max_sharpe (tight bounds) failed ({e1}), relaxing bounds.")
            # Tier 2: max_sharpe with relaxed bounds
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                ef.max_sharpe()
                weights_dict = ef.clean_weights()
                logger.info("MVO: max_sharpe succeeded (relaxed bounds).")
            except Exception as e2:
                logger.warning(f"MVO max_sharpe (relaxed bounds) failed ({e2}), trying min_volatility.")
                # Tier 3: min_volatility (always feasible for PSD matrices)
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                    ef.min_volatility()
                    weights_dict = ef.clean_weights()
                    logger.info("MVO: min_volatility succeeded.")
                except Exception as e3:
                    logger.warning(f"MVO min_volatility failed ({e3}), using equal weights.")
                    weights_dict = None

        if weights_dict is not None:
            # Map optimised weights back to the full asset universe (zero for dropped cols)
            partial_w = np.array([weights_dict.get(c, 0.0) for c in active_cols])
            full_w = np.zeros(n_assets)
            for i, col in enumerate(train_returns.columns):
                if col in active_cols:
                    full_w[i] = partial_w[active_cols.index(col)]
            # Normalise to sum to 1
            total = full_w.sum()
            if total > 1e-8:
                weights = full_w / total
            else:
                weights = np.ones(n_assets) / n_assets

    except Exception as e:
        logger.warning(f"MVO failed entirely ({e}), falling back to UBH weights.")
        weights = np.ones(n_assets) / n_assets
        
    # Evaluate on test
    transaction_cost = transaction_cost_bps / 10000.0
    current_weights = weights.copy()
    
    returns = []
    simple_returns = np.exp(test_returns) - 1.0
    
    for i in range(len(simple_returns)):
        # Rebalance monthly (approx 21 days)
        if i % 21 == 0 and i > 0:
            cost = np.sum(np.abs(weights - current_weights)) * transaction_cost
            current_weights = weights.copy()
        else:
            cost = 0.0
            
        step_ret = np.sum(current_weights * simple_returns.iloc[i].values) - cost
        returns.append(step_ret)
        
        current_weights = current_weights * (1 + simple_returns.iloc[i].values)
        current_weights /= np.sum(current_weights)
        
    return np.array(returns), weights

def train_mlp_rl(env_kwargs, total_timesteps, net_arch, seed):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return PortfolioEnv(**env_kwargs)
        
    env = make_vec_env(make_env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)
    
    # For SB3 >=2.0, net_arch can be passed directly as a list for both policy and value nets
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    
    return model

def evaluate_rl(model, env_kwargs):
    env = PortfolioEnv(**env_kwargs)
    obs, _ = env.reset()
    done = False
    
    returns = []
    weights_history = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        returns.append(info["portfolio_return"])
        weights_history.append(info["weights"])
        
    return np.array(returns), np.array(weights_history)
