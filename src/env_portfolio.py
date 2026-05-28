import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    Gymnasium environment for portfolio allocation.
    """
    def __init__(self, log_returns, rf_rate, window_size=60, transaction_cost_bps=10, reward_type="return"):
        super().__init__()
        
        # Ensure DataFrame for rolling operations
        if isinstance(log_returns, pd.DataFrame):
            df_returns = log_returns
        else:
            df_returns = pd.DataFrame(log_returns)
            
        # Ensure numpy arrays for indexing
        self.log_returns = df_returns.values
        self.rf_rate = rf_rate.values if hasattr(rf_rate, 'values') else rf_rate
        
        self.n_assets = self.log_returns.shape[1]
        self.window_size = window_size
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.reward_type = reward_type
        
        self.n_steps = len(self.log_returns) - self.window_size
        
        # Calculate Volatility and Moving Average features using a rolling 20-day window
        # min_periods=1 ensures no NaNs at the beginning of the series
        self.volatility = df_returns.rolling(window=20, min_periods=1).std().fillna(0).values
        self.moving_avg = df_returns.rolling(window=20, min_periods=1).mean().fillna(0).values
        
        # State: log returns (window_size days), volatility (window_size days), moving_avg (window_size days)
        # coupled with the agent's current portfolio allocations w_{t-1} and risk-free rate (1 day)
        # Flattened observation dimension: 3 * n_assets * window_size + n_assets + 1
        obs_dim = 3 * self.n_assets * self.window_size + self.n_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action: n_assets continuous values (will be softmaxed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.history = []
        return self._get_obs(), {}
        
    def _get_obs(self):
        start = self.current_step
        end = self.current_step + self.window_size
        
        ret_window = self.log_returns[start:end]
        vol_window = self.volatility[start:end]
        ma_window = self.moving_avg[start:end]
        
        rf = self.rf_rate[end-1] if self.rf_rate.ndim == 1 else self.rf_rate[end-1, 0]
        
        obs = np.concatenate([
            ret_window.flatten(),
            vol_window.flatten(),
            ma_window.flatten(),
            self.current_weights,
            [rf]
        ])
        return obs.astype(np.float32)
        
    def step(self, action):
        # Softmax
        exp_a = np.exp(action - np.max(action))
        new_weights = exp_a / exp_a.sum()
        
        # Transaction costs
        cost = np.sum(np.abs(new_weights - self.current_weights)) * self.transaction_cost
        
        # Get step returns (convert log returns to simple returns)
        step_returns = np.exp(self.log_returns[self.current_step + self.window_size]) - 1.0
        
        # Portfolio return
        port_return = np.sum(new_weights * step_returns) - cost
        
        self.current_weights = new_weights
        self.portfolio_value *= (1 + port_return)
        self.history.append(port_return)
        
        # Reward calculation
        if self.reward_type == "sortino":
            history_arr = np.array(self.history)
            window = min(len(self.history), 60)
            recent_history = history_arr[-window:]
            downside_returns = recent_history[recent_history < 0]
            if len(downside_returns) > 0:
                downside_std = np.sqrt(np.mean(downside_returns ** 2))
            else:
                downside_std = 1e-4
            reward = port_return / (downside_std + 1e-6)
        elif self.reward_type == "sharpe":
            history_arr = np.array(self.history)
            window = min(len(self.history), 60)
            recent_history = history_arr[-window:]
            std = np.std(recent_history) if len(recent_history) > 1 else 1e-4
            reward = port_return / (std + 1e-6)
        else:  # default is "return"
            reward = port_return
            
        self.current_step += 1
        done = self.current_step >= self.n_steps
        truncated = False
        
        return self._get_obs(), float(reward), done, truncated, {
            "portfolio_return": port_return, 
            "weights": new_weights,
            "cost": cost
        }
