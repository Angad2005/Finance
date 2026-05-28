import numpy as np
import scipy.stats as stats
from scipy.stats import norm

def diebold_mariano_test(returns_a, returns_b, h=1):
    """
    Diebold-Mariano test with HLN correction.
    """
    d = np.array(returns_a) - np.array(returns_b)
    n = len(d)
    
    if n == 0:
        return 0.0, 1.0
        
    mean_d = np.mean(d)
    
    # Autocovariance
    gamma = np.zeros(h)
    for i in range(h):
        if i == 0:
            gamma[i] = np.var(d)
        else:
            gamma[i] = np.cov(d[i:], d[:-i])[0, 1]
            
    v_d = gamma[0] + 2 * np.sum(gamma[1:])
    
    if v_d <= 0:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(v_d / n)
    
    # HLN Correction
    hln_factor = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    dm_stat = dm_stat * hln_factor
    
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)

def bootstrap_ci(returns, statistic, n_resamples=1000, alpha=0.05):
    """
    Bootstrap CI for a statistic.
    """
    n = len(returns)
    if n == 0:
        return 0.0, 0.0
        
    stats_dist = []
    
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        sample = np.array(returns)[indices]
        stats_dist.append(statistic(sample))
        
    lower = np.percentile(stats_dist, alpha / 2 * 100)
    upper = np.percentile(stats_dist, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)
