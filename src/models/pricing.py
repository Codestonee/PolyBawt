import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from math import factorial
import time

class MertonJDCalibrator:
    """
    Calibrate Merton Jump-Diffusion parameters using MLE
    """
    
    def __init__(self):
        self.params = None
        self.last_calibration = None
    
    def calibrate(self, returns, dt=1/96):
        """
        Calibrate to recent returns
        
        Args:
            returns: array of log-returns over 15-min intervals
            dt: time step (1/96 = 15 minutes in daily units)
        """
        
        # Initial guess based on sample statistics
        sample_vol = np.std(returns) / np.sqrt(dt)
        x0 = [
            sample_vol,      # σ: volatility
            5,               # λ: 5 jumps per day (typical)
            0,               # μ_J: expected log jump size
            0.01             # σ_J: volatility of jumps
        ]
        
        # Bounds to prevent unreasonable values
        bounds = [
            (0.001, 0.5),    # σ: 0.1% to 50% daily vol
            (0.01, 50),      # λ: 0.01 to 50 jumps/day
            (-0.2, 0.2),     # μ_J: ±20% jump size
            (0.001, 0.3)     # σ_J: 0.1% to 30% jump vol
        ]
        
        # Optimize
        result = minimize(
            lambda p: self._neg_log_likelihood(p, returns, dt),
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            self.params = {
                'sigma': result.x[0],
                'lambda': result.x[1],
                'mu_j': result.x[2],
                'sigma_j': result.x[3],
                'converged': True
            }
        else:
            self.params = {
                'sigma': x0[0],
                'lambda': x0[1],
                'mu_j': x0[2],
                'sigma_j': x0[3],
                'converged': False
            }
        
        self.last_calibration = time.time()
        return self.params
    
    def _neg_log_likelihood(self, params, returns, dt):
        """
        Calculate negative log-likelihood (to minimize)
        """
        sigma, lambda_, mu_j, sigma_j = params
        
        if sigma <= 0 or lambda_ <= 0 or sigma_j <= 0:
            return 1e10
        
        ll = 0
        n_max = 10
        
        for r in returns:
            prob_total = 0
            for n in range(n_max + 1):
                try:
                    poisson_prob = np.exp(-lambda_ * dt) * (lambda_ * dt) ** n / factorial(n)
                except (OverflowError, ValueError):
                    poisson_prob = 0
                
                if poisson_prob < 1e-10:
                    continue
                
                mean = (lambda_ * mu_j) * dt + n * mu_j
                variance = sigma**2 * dt + n * sigma_j**2
                
                if variance <= 0:
                    continue
                
                gauss_dens = norm.pdf(r, loc=mean, scale=np.sqrt(variance))
                prob_total += poisson_prob * gauss_dens
            
            if prob_total > 1e-10:
                ll -= np.log(prob_total)
            else:
                ll += 1e3
        
        return ll
