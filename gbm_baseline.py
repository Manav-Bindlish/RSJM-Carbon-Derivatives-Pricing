"""
src/gbm_baseline.py
===================
Geometric Brownian Motion (GBM) baseline — the null hypothesis under which
the Black-Scholes-Merton (BSM) framework is constructed (paper §3).

The GBM SDE:
    dS_t = μ S_t dt + σ S_t dW_t

Log-returns are i.i.d. Normal:
    r_t ~ N((μ − σ²/2)Δt,  σ²Δt)

MLE: closed-form via sample mean and variance of observed log-returns.
Option pricing: standard BSM closed-form formulae.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GBMParams:
    """Maximum-likelihood GBM parameters."""
    mu: float     # annualised drift
    sigma: float  # annualised instantaneous volatility


@dataclass(frozen=True)
class BSMOptionResult:
    """BSM European option prices and Greeks inputs."""
    call: float
    put: float
    d1: float
    d2: float
    sigma: float


# ---------------------------------------------------------------------------
# GBM model
# ---------------------------------------------------------------------------

class GBMModel:
    """
    Geometric Brownian Motion model (paper §3, equation: dS_t = μS_tdt + σS_tdW_t).

    Closed-form MLE:
        σ̂² = Var[r_t] / Δt
        μ̂  = Mean[r_t] / Δt + σ̂²/2

    BSM call:  C = S·Φ(d₁) − K·e^{−rT}·Φ(d₂)
    BSM put:   P = K·e^{−rT}·Φ(−d₂) − S·Φ(−d₁)
    """

    def __init__(self, dt: float = 1.0 / 252.0) -> None:
        self.dt: float = dt
        self.params: GBMParams | None = None
        self._log_likelihood: float = float("-inf")
        self._n_params: int = 2

    # ------------------------------------------------------------------
    # MLE
    # ------------------------------------------------------------------

    def fit(self, log_returns: np.ndarray) -> GBMParams:
        """
        Closed-form maximum likelihood estimation.

        Parameters
        ----------
        log_returns : np.ndarray   1-D array of observed daily log-returns.

        Returns
        -------
        GBMParams
        """
        r: np.ndarray = np.asarray(log_returns, dtype=np.float64).ravel()
        dt = self.dt

        mu_lr    = float(r.mean())
        sigma_sq = float(r.var(ddof=0))

        sigma = np.sqrt(sigma_sq / dt)
        mu    = mu_lr / dt + 0.5 * sigma ** 2

        self.params          = GBMParams(mu=float(mu), sigma=float(sigma))
        self._log_likelihood = self.compute_log_likelihood(r, self.params)
        return self.params

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def compute_log_likelihood(
        self,
        log_returns: np.ndarray,
        params: GBMParams | None = None,
    ) -> float:
        """
        Gaussian log-likelihood:
            LL = Σ_t log N(r_t; (μ − σ²/2)Δt, σ²Δt)
        """
        p  = params if params is not None else self.params
        if p is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        r     = np.asarray(log_returns, dtype=np.float64).ravel()
        loc   = (p.mu - 0.5 * p.sigma ** 2) * self.dt
        scale = p.sigma * np.sqrt(self.dt)
        return float(stats.norm.logpdf(r, loc=loc, scale=scale).sum())

    @property
    def log_likelihood(self) -> float:
        return self._log_likelihood

    @property
    def n_params(self) -> int:
        return self._n_params

    # ------------------------------------------------------------------
    # BSM option pricing
    # ------------------------------------------------------------------

    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float | None = None,
    ) -> BSMOptionResult:
        """
        Black-Scholes-Merton European option prices.

        Parameters
        ----------
        S     : float   Current underlying price.
        K     : float   Strike price.
        T     : float   Time to expiry (years).
        r     : float   Risk-free rate (annualised, continuous).
        sigma : float   Volatility override; uses fitted σ if None.

        Returns
        -------
        BSMOptionResult
        """
        if sigma is None:
            if self.params is None:
                raise RuntimeError("Model not fitted. Call .fit() or supply sigma.")
            sigma = self.params.sigma

        sqrtT = np.sqrt(T)
        d1    = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2    = d1 - sigma * sqrtT

        call = float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
        put  = float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))

        return BSMOptionResult(call=call, put=put, d1=float(d1), d2=float(d2), sigma=sigma)

    # ------------------------------------------------------------------
    # Monte Carlo simulation (fully vectorised, no path-index loops)
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Vectorised Euler–Maruyama GBM simulation.

        Returns
        -------
        paths : np.ndarray, shape (n_steps + 1, n_paths)
                S0 in row 0; terminal prices in row -1.
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        mu, sigma = self.params.mu, self.params.sigma

        # Fully vectorised: (n_steps, n_paths)
        Z           = rng.standard_normal((n_steps, n_paths))
        log_incr    = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_paths   = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_incr, axis=0)])

        return S0 * np.exp(log_paths)

    # ------------------------------------------------------------------
    # RMSE against observed option prices (for backtest comparison)
    # ------------------------------------------------------------------

    def compute_rmse(
        self,
        S: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        observed_calls: np.ndarray,
    ) -> float:
        """Compute RMSE of BSM call prices against observed prices."""
        predicted = np.array([
            self.price_option(S, K, T, r).call for K in strikes
        ])
        return float(np.sqrt(np.mean((predicted - observed_calls) ** 2)))
