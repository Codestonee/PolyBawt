"""
Jump-Diffusion pricing model for binary options.

Implements Merton's Jump-Diffusion model which extends Black-Scholes
to account for discontinuous price jumps common in crypto.

The model calculates P(S_T >= S_0) - the probability that the final price
exceeds the initial price (UP outcome).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    
    Uses the error function for numerical stability.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@lru_cache(maxsize=20)
def factorial(n: int) -> int:
    """Cached factorial computation."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


@dataclass
class JumpDiffusionParams:
    """Parameters for Jump-Diffusion model."""
    
    # Base volatility (annualized)
    sigma: float = 0.60  # 60% default for BTC
    
    # Jump process parameters
    lambda_j: float = 2.0 / 365  # ~2 jumps per day
    mu_j: float = 0.0  # Mean jump size (log)
    sigma_j: float = 0.02  # Jump size std (2%)
    
    # Model settings
    n_terms: int = 10  # Poisson series truncation
    
    @classmethod
    def for_asset(cls, asset: str) -> "JumpDiffusionParams":
        """Get default parameters for a specific asset."""
        volatilities = {
            "BTC": 0.60,
            "ETH": 0.70,
            "SOL": 0.85,
            "XRP": 0.90,
        }
        return cls(sigma=volatilities.get(asset, 0.70))


class JumpDiffusionModel:
    """
    Merton Jump-Diffusion model for binary option pricing.
    
    The model assumes:
    dS_t = μS_t dt + σS_t dW_t + S_t dJ_t
    
    where dJ_t is a compound Poisson process with:
    - λ = jump intensity (jumps per unit time)
    - Jump sizes are log-normally distributed with mean μ_J and std σ_J
    
    For binary options (pays $1 if S_T >= K), we compute P(S_T >= K).
    
    Usage:
        model = JumpDiffusionModel()
        prob = model.prob_up(
            spot=100000,
            initial=99500,
            time_years=15/525600,  # 15 minutes
            params=JumpDiffusionParams.for_asset("BTC")
        )
    """
    
    def prob_up(
        self,
        spot: float,
        initial: float,
        time_years: float,
        params: JumpDiffusionParams | None = None,
    ) -> float:
        """
        Calculate probability of UP outcome (S_T >= S_0).
        
        Args:
            spot: Current spot price
            initial: Initial price at interval start (the "strike")
            time_years: Time to expiry in years
            params: Jump-diffusion parameters
        
        Returns:
            Probability between 0 and 1
        """
        if params is None:
            params = JumpDiffusionParams()
        
        # Handle edge cases
        if time_years <= 0:
            # At or past expiry - deterministic outcome
            return 1.0 if spot >= initial else 0.0
        
        if initial <= 0:
            logger.warning("Invalid initial price", initial=initial)
            return 0.5
        
        if spot <= 0:
            logger.warning("Invalid spot price", spot=spot)
            return 0.0
        
        # Moneyness (log of price ratio)
        moneyness = math.log(spot / initial)
        
        # Sum over Poisson-weighted probabilities
        prob = 0.0
        lambda_t = params.lambda_j * time_years * 365  # Total expected jumps
        
        for n in range(params.n_terms):
            # Poisson weight: P(N = n) = e^(-λT) * (λT)^n / n!
            poisson_weight = (
                math.exp(-lambda_t) * 
                (lambda_t ** n) / 
                factorial(n)
            )
            
            if poisson_weight < 1e-10:
                # Negligible contribution
                continue
            
            # Adjusted volatility for n jumps
            # σ_n² = σ² + n * σ_J² / T
            variance_n = (
                params.sigma ** 2 + 
                (n * params.sigma_j ** 2) / time_years if time_years > 0 else params.sigma ** 2
            )
            sigma_n = math.sqrt(max(variance_n, 1e-10))
            
            # d2 for digital option (probability of finishing ITM)
            # d2 = (ln(S/K) + (μ - σ²/2)T + n*μ_J) / (σ_n * √T)
            # We assume μ = 0 (risk-neutral)
            
            sqrt_t = math.sqrt(time_years)
            
            d2 = (
                moneyness + 
                (-0.5 * sigma_n ** 2) * time_years +
                n * params.mu_j
            ) / (sigma_n * sqrt_t)
            
            # Add Poisson-weighted probability
            prob += poisson_weight * norm_cdf(d2)
        
        # Clamp to valid range
        return max(0.0, min(1.0, prob))
    
    def prob_down(
        self,
        spot: float,
        initial: float,
        time_years: float,
        params: JumpDiffusionParams | None = None,
    ) -> float:
        """
        Calculate probability of DOWN outcome (S_T < S_0).
        
        Simply 1 - P(UP).
        """
        return 1.0 - self.prob_up(spot, initial, time_years, params)


class BlackScholesModel:
    """
    Simple Black-Scholes model for binary options (fallback).
    
    Less accurate than Jump-Diffusion but faster and simpler.
    """
    
    def prob_up(
        self,
        spot: float,
        initial: float,
        time_years: float,
        sigma: float = 0.60,
    ) -> float:
        """
        Calculate probability using standard Black-Scholes.
        
        For digital call: P(up) = N(d2)
        d2 = (ln(S/K) - σ²T/2) / (σ√T)
        """
        if time_years <= 0:
            return 1.0 if spot >= initial else 0.0
        
        if initial <= 0 or spot <= 0:
            return 0.5
        
        sqrt_t = math.sqrt(time_years)
        
        d2 = (
            math.log(spot / initial) - 0.5 * sigma ** 2 * time_years
        ) / (sigma * sqrt_t)
        
        return norm_cdf(d2)
    
    def prob_down(
        self,
        spot: float,
        initial: float,
        time_years: float,
        sigma: float = 0.60,
    ) -> float:
        """Calculate probability of DOWN outcome."""
        return 1.0 - self.prob_up(spot, initial, time_years, sigma)


def seconds_to_years(seconds: float) -> float:
    """Convert seconds to years (for time_years parameter)."""
    return seconds / (365 * 24 * 60 * 60)


def minutes_to_years(minutes: float) -> float:
    """Convert minutes to years."""
    return minutes / (365 * 24 * 60)


# Pre-instantiated models for convenience
jump_diffusion = JumpDiffusionModel()
black_scholes = BlackScholesModel()
