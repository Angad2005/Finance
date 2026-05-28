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
    
    try:
        # Ledoit-Wolf shrinkage
        S = risk_models.CovarianceShrinkage(train_prices).ledoit_wolf()
        mu = expected_returns.mean_historical_return(train_prices)
        
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.2)) # Max weight 20%
        ef.max_sharpe()
        weights_dict = ef.clean_weights()
        weights = np.array([weights_dict.get(c, 0) for c in train_returns.columns])
    except Exception as e:
        logger.warning(f"MVO failed ({e}), falling back to UBH weights.")
        n_assets = train_returns.shape[1]
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
