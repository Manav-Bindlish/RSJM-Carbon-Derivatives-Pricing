"""
src/likelihood.py
=================
Maximum Likelihood Estimation (MLE) for GBM, JDM, and RSJM models, and
Likelihood Ratio (LR) tests as reported in the paper.

MLE strategy
------------
GBM  : Closed-form (sample mean and variance of log-returns).
JDM  : Bounded L-BFGS-B via scipy.optimize.minimize; enforces λ ≥ 0, σ > 0.
RSJM : Bounded L-BFGS-B wrapping the Hamilton filter; enforces 0 < Pii < 1,
       λ_s ≥ 0, σ_s > 0 in both regimes.

Paper benchmarks (§3.2, §4.2) for EU ETS Phase I & II empirical data:
    JDM  vs GBM  →  LR ≈ 632.38  (df = 3)
    RSJM vs JDM  →  LR ≈ 134.21  (df = 7)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize, Bounds

from src.rsjm_engine import (
    RSJMEngine,
    RSJMParams,
    RegimeParams,
    _K_VALS,
    _LOG_FACT_ARR,
)


# ---------------------------------------------------------------------------
# LR test result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LRTestResult:
    """
    χ² Likelihood Ratio test result.

    LR = −2 · (LL_null − LL_alt)  ~  χ²(df) under H₀.
    """
    statistic: float
    df: int
    p_value: float
    null_model: str
    alt_model: str

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else ("**" if self.p_value < 0.01 else "*")
        return (
            f"LR({self.null_model} → {self.alt_model}):  "
            f"χ²({self.df}) = {self.statistic:.4f},  p = {self.p_value:.3e}  {sig}"
        )


# ---------------------------------------------------------------------------
# MLE estimator
# ---------------------------------------------------------------------------

class LikelihoodEstimator:
    """
    MLE for GBM, JDM, and RSJM, plus χ² likelihood ratio tests.

    All optimisations use scipy.optimize.minimize with L-BFGS-B and explicit
    Bounds objects to prevent negative jump intensities and invalid
    transition probabilities.

    Parameters
    ----------
    dt : float   Daily time step (default 1/252).
    """

    _MAX_K: int = 10   # Poisson mixture truncation for JDM likelihood

    def __init__(self, dt: float = 1.0 / 252.0) -> None:
        self.dt = dt

    # ------------------------------------------------------------------
    # GBM  —  closed-form MLE
    # ------------------------------------------------------------------

    def gbm_log_likelihood(
        self,
        log_returns: np.ndarray,
        mu: float,
        sigma: float,
    ) -> float:
        """Gaussian log-likelihood under GBM."""
        loc   = (mu - 0.5 * sigma ** 2) * self.dt
        scale = sigma * np.sqrt(self.dt)
        return float(stats.norm.logpdf(log_returns, loc=loc, scale=scale).sum())

    def fit_gbm(self, log_returns: np.ndarray) -> dict[str, float]:
        """
        Closed-form MLE for GBM parameters.

        Returns
        -------
        dict with keys: mu, sigma, log_likelihood, n_params
        """
        r  = np.asarray(log_returns, dtype=np.float64).ravel()
        dt = self.dt

        mu_lr    = float(r.mean())
        sigma_sq = float(r.var(ddof=0))

        sigma = np.sqrt(sigma_sq / dt)
        mu    = mu_lr / dt + 0.5 * sigma ** 2
        ll    = self.gbm_log_likelihood(r, mu, sigma)

        return {"mu": mu, "sigma": sigma, "log_likelihood": ll, "n_params": 2}

    # ------------------------------------------------------------------
    # JDM  —  bounded L-BFGS-B MLE
    # ------------------------------------------------------------------

    def _jdm_log_likelihood(
        self,
        log_returns: np.ndarray,
        mu: float,
        sigma: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
    ) -> float:
        """
        Merton (1976) JDM log-likelihood via Poisson mixture of Gaussians.

        f(r_t) = Σ_{k=0}^{K} P(N=k|λΔt) · N(r_t; μ̃_k, ṽ_k)
        μ̃_k = (μ − λκ − σ²/2)Δt + k μ_J
        ṽ_k = σ²Δt + k σ_J²
        """
        r      = np.asarray(log_returns, dtype=np.float64).ravel()
        dt     = self.dt
        K      = self._MAX_K
        k_vals = _K_VALS[: K + 1]
        lf_arr = _LOG_FACT_ARR[: K + 1]

        lam_dt = lam * dt
        kappa  = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

        # Poisson weights  —  shape (K+1,)
        log_pois = k_vals * np.log(max(lam_dt, 1e-300)) - lam_dt - lf_arr
        log_pois -= log_pois.max()
        pois_w   = np.exp(log_pois)
        pois_w  /= pois_w.sum()

        # Component means / stds  —  shape (K+1,)
        mu_k  = (mu - lam * kappa - 0.5 * sigma ** 2) * dt + k_vals * mu_j
        var_k = sigma ** 2 * dt + k_vals * sigma_j ** 2
        std_k = np.sqrt(np.maximum(var_k, 1e-14))

        # Density matrix  —  shape (T, K+1)
        comp = stats.norm.pdf(r[:, np.newaxis], loc=mu_k[np.newaxis], scale=std_k[np.newaxis])
        dens = (pois_w[np.newaxis] * comp).sum(axis=1)   # (T,)

        return float(np.log(np.maximum(dens, 1e-300)).sum())

    def fit_jdm(self, log_returns: np.ndarray) -> dict[str, float]:
        """
        Bounded MLE for JDM (Merton 1976) via L-BFGS-B.

        Warm-started from closed-form GBM estimates.
        Bounds enforce:  σ > 0,  λ ≥ 0,  σ_J > 0.

        Returns
        -------
        dict with keys: mu, sigma, lam, mu_j, sigma_j, log_likelihood, n_params
        """
        r   = np.asarray(log_returns, dtype=np.float64).ravel()
        gbm = self.fit_gbm(r)

        x0 = np.array([
            gbm["mu"],
            gbm["sigma"] * 0.80,   # diffusion component smaller than total
            4.0,                   # λ: moderate jump frequency
            float(r.mean()) / self.dt * 0.1,
            gbm["sigma"] * 0.40,
        ])

        def neg_ll(x: np.ndarray) -> float:
            return -self._jdm_log_likelihood(r, x[0], x[1], x[2], x[3], x[4])

        bounds = Bounds(
            lb=[-10.0, 1e-6, 0.0,  -3.0, 1e-6],
            ub=[ 10.0, 15.0, 80.0,  3.0, 3.0 ],
        )
        res = minimize(
            neg_ll, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8},
        )
        mu, sigma, lam, mu_j, sigma_j = res.x
        ll = -float(res.fun)

        return {
            "mu": float(mu), "sigma": float(sigma), "lam": float(lam),
            "mu_j": float(mu_j), "sigma_j": float(sigma_j),
            "log_likelihood": ll, "n_params": 5,
            "converged": bool(res.success),
        }

    # ------------------------------------------------------------------
    # RSJM  —  Hamilton filter + bounded L-BFGS-B MLE
    # ------------------------------------------------------------------

    def _pack_rsjm_params(self, x: np.ndarray) -> RSJMParams:
        """Unpack optimiser vector → RSJMParams."""
        return RSJMParams(
            regime0=RegimeParams(mu=x[0], sigma=x[1],  lam=x[2],  mu_j=x[3],  sigma_j=x[4]),
            regime1=RegimeParams(mu=x[5], sigma=x[6],  lam=x[7],  mu_j=x[8],  sigma_j=x[9]),
            p11=x[10], p22=x[11],
        )

    def fit_rsjm(self, log_returns: np.ndarray) -> dict:
        """
        Bounded MLE for RSJM via Hamilton filter + L-BFGS-B.

        Parameter vector  x ∈ ℝ¹²:
            [μ₀, σ₀, λ₀, μ_J0, σ_J0,   ← Regime 0 (stable)
             μ₁, σ₁, λ₁, μ_J1, σ_J1,   ← Regime 1 (turbulent)
             P₁₁, P₂₂]                  ← Markov transition diagonal

        Bounds enforce:  σ > 0, λ ≥ 0, 0.50 < Pii < 0.9999.

        Warm-start: JDM estimates split between regime 0 (low params) and
        regime 1 (high params).

        Returns
        -------
        dict with keys: regime0, regime1, p11, p22, log_likelihood, n_params
        """
        r   = np.asarray(log_returns, dtype=np.float64).ravel()
        jdm = self.fit_jdm(r)

        x0 = np.array([
            # Regime 0: stable — lower volatility/jump intensity
            jdm["mu"]    * 0.60, jdm["sigma"]    * 0.55, jdm["lam"]    * 0.30,
            jdm["mu_j"]  * 0.80, jdm["sigma_j"]  * 0.70,
            # Regime 1: turbulent — higher volatility/jump intensity
            jdm["mu"]    * 1.40, jdm["sigma"]    * 1.60, jdm["lam"]    * 2.50,
            jdm["mu_j"]  * 1.20, jdm["sigma_j"]  * 1.40,
            # Transition probabilities (paper §4.2 empirical values as warm-start)
            0.9868, 0.9868,
        ])

        def neg_ll(x: np.ndarray) -> float:
            try:
                params  = self._pack_rsjm_params(x)
                engine  = RSJMEngine(params, dt=self.dt)
                return -engine.log_likelihood(r)
            except Exception:
                return 1e12

        bounds = Bounds(
            lb=[-10.0, 1e-6, 0.0,  -3.0, 1e-6,
                -10.0, 1e-6, 0.0,  -3.0, 1e-6,
                 0.50,  0.50],
            ub=[ 10.0, 15.0, 100.0,  3.0, 3.0,
                 10.0, 15.0, 100.0,  3.0, 3.0,
                 0.9999, 0.9999],
        )

        res = minimize(
            neg_ll, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-8},
        )
        x  = res.x
        ll = -float(res.fun)

        return {
            "regime0": {
                "mu": float(x[0]), "sigma": float(x[1]), "lam": float(x[2]),
                "mu_j": float(x[3]), "sigma_j": float(x[4]),
            },
            "regime1": {
                "mu": float(x[5]), "sigma": float(x[6]), "lam": float(x[7]),
                "mu_j": float(x[8]), "sigma_j": float(x[9]),
            },
            "p11": float(x[10]), "p22": float(x[11]),
            "log_likelihood": ll,
            "n_params": 12,
            "converged": bool(res.success),
        }

    # ------------------------------------------------------------------
    # Likelihood Ratio test  (paper §3.2: 632.38, §4.2: 134.21)
    # ------------------------------------------------------------------

    @staticmethod
    def lr_test(
        ll_null: float,
        ll_alt: float,
        df: int,
        null_name: str,
        alt_name: str,
    ) -> LRTestResult:
        """
        χ² Likelihood Ratio test.

        LR = −2(LL_null − LL_alt)  ~  χ²(df) under H₀: null model is correct.

        Paper benchmarks (EU ETS empirical data):
            §3.2  JDM  vs GBM  →  LR ≈ 632.38
            §4.2  RSJM vs JDM  →  LR ≈ 134.21

        Parameters
        ----------
        ll_null   : float   Log-likelihood of the constrained (null) model.
        ll_alt    : float   Log-likelihood of the unconstrained (alternative).
        df        : int     Degrees of freedom = n_params_alt − n_params_null.
        null_name : str     Label for null model.
        alt_name  : str     Label for alternative model.
        """
        lr_stat = -2.0 * (ll_null - ll_alt)
        p_val   = float(1.0 - stats.chi2.cdf(lr_stat, df=df))
        return LRTestResult(
            statistic=lr_stat,
            df=df,
            p_value=p_val,
            null_model=null_name,
            alt_model=alt_name,
        )
