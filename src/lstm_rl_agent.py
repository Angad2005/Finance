import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.env_portfolio import PortfolioEnv
import logging

logger = logging.getLogger(__name__)

class SharpeEarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_env_kwargs, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env_kwargs = eval_env_kwargs
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.best_model = None
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            eval_env = PortfolioEnv(**self.eval_env_kwargs)
            obs, _ = eval_env.reset()
            done = False
            returns = []
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while not done:
                action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                returns.append(info["portfolio_return"])
                episode_starts = np.zeros((1,), dtype=bool)
                
            returns = np.array(returns)
            std = np.std(returns)
            if std == 0:
                sharpe = -1.0
            else:
                sharpe = np.sqrt(252) * np.mean(returns) / std
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                logger.info(f"New best val Sharpe: {sharpe:.4f} at step {self.n_calls}")
                # We could save the model here
        return True

def train_lstm_rl(env_kwargs, val_env_kwargs, total_timesteps, net_arch, lstm_hidden_size, n_lstm_layers, seed):
    def make_env():
        return PortfolioEnv(**env_kwargs)
        
    env = make_vec_env(make_env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)
    
    policy_kwargs = {
        "net_arch": net_arch,
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": n_lstm_layers,
    }
    
    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, seed=seed, verbose=0)
    
    callback = SharpeEarlyStoppingCallback(val_env_kwargs, eval_freq=max(1000, total_timesteps // 5))
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model

def evaluate_lstm_rl(model, env_kwargs):
    env = PortfolioEnv(**env_kwargs)
    obs, _ = env.reset()
    done = False
    
    returns = []
    weights_history = []
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        returns.append(info["portfolio_return"])
        weights_history.append(info["weights"])
        episode_starts = np.zeros((1,), dtype=bool)
        
    return np.array(returns), np.array(weights_history)
