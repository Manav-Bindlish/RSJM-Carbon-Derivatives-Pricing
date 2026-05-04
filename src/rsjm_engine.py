"""
src/rsjm_engine.py
==================
Regime-Switching Jump Diffusion Model (RSJM) — paper §4.

Exact RSJM SDE (paper §4.1):
    dS_t = μ(X_t) S_t dt + σ(X_t) S_t dW_t + J(X_t) dN_t(X_t) S_t

where X_t ∈ {0, 1} is a hidden two-state continuous-time Markov chain:
    Regime 0: "stable policy"        — low μ, low σ, low λ, light tails.
    Regime 1: "high policy uncertainty" — high σ, high λ, heavy tails.

All parameters (μ, σ, λ, μ_J, σ_J) are functions of the hidden state X_t.
The jump magnitude J ~ N(μ_J(X_t), σ_J(X_t)²).
The drift compensation κ(X_t) = E[e^{J(X_t)} − 1] enforces the martingale.

Empirical transition matrix (MLE on EU ETS Phase I & II, paper §4.2):
    P = [[0.9868, 0.0132],
         [0.0132, 0.9868]]
P11 ≈ P22 ≈ 0.9868 → near-absolute regime persistence (volatility clustering).

CYO Green Equity Option (paper §9.1):
    q_c = E_annual × SCC / M_t          (carbon yield)
    C   = S·e^{−q_c T}·Φ(d₁) − K·e^{−rT}·Φ(d₂)
    d₁  = [ln(S/K) + (r − q_c + σ²/2)T] / (σ√T)
    d₂  = d₁ − σ√T
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Log-factorial helper (cached for Hamilton filter inner loop)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _log_factorial(n: int) -> float:
    """Return log(n!) via recurrence with LRU cache."""
    if n <= 1:
        return 0.0
    return _log_factorial(n - 1) + np.log(float(n))


# Pre-compute array for k = 0..MAX_JUMP_TERMS to vectorise the Poisson sum.
_MAX_K: int = 15
_K_VALS: np.ndarray = np.arange(_MAX_K + 1, dtype=np.float64)
_LOG_FACT_ARR: np.ndarray = np.array([_log_factorial(k) for k in range(_MAX_K + 1)])


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeParams:
    """
    State-dependent parameters for one market regime (paper §4.1).

    Attributes
    ----------
    mu      : annualised drift (physical measure)
    sigma   : annualised Brownian (continuous) volatility
    lam     : Poisson jump arrival rate (per year)
    mu_j    : mean log-jump size  J ~ N(mu_j, sigma_j²)
    sigma_j : std  log-jump size
    """
    mu: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float

    @property
    def kappa(self) -> float:
        """Drift compensation: κ = E[e^J − 1] = exp(μ_J + σ_J²/2) − 1."""
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0)

    def log_density_mixture(
        self,
        log_returns: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Per-observation conditional density f(r_t | X_t = this regime).

        Uses Poisson mixture of Gaussians (Merton, 1976):
            f(r | s) = Σ_{k=0}^{K} P(N=k | λ_s Δt) · N(r; μ̃_k, ṽ_k)

            μ̃_k = (μ_s − λ_s κ_s − σ_s²/2) Δt + k μ_J_s
            ṽ_k = σ_s² Δt + k σ_J_s²

        Returns
        -------
        dens : np.ndarray, shape (len(log_returns),)
        """
        lam_dt = self.lam * dt
        kappa  = self.kappa

        # Poisson weights P(N=k) for k=0..K  —  shape (K+1,)
        log_pois = _K_VALS * np.log(max(lam_dt, 1e-300)) - lam_dt - _LOG_FACT_ARR
        log_pois -= log_pois.max()                    # numerical stability
        pois_w   = np.exp(log_pois)
        pois_w  /= pois_w.sum()

        # Component parameters  —  shape (K+1,)
        mu_k  = (self.mu - self.lam * kappa - 0.5 * self.sigma ** 2) * dt + _K_VALS * self.mu_j
        var_k = self.sigma ** 2 * dt + _K_VALS * self.sigma_j ** 2
        std_k = np.sqrt(np.maximum(var_k, 1e-14))

        # Component densities  —  shape (T, K+1)
        r_col = log_returns[:, np.newaxis]            # (T, 1)
        comp  = stats.norm.pdf(r_col, loc=mu_k[np.newaxis], scale=std_k[np.newaxis])

        # Mixture density  —  shape (T,)
        return (pois_w[np.newaxis] * comp).sum(axis=1)


@dataclass
class RSJMParams:
    """
    Full RSJM parameter set (paper §4.1).

    The RSJM SDE:
        dS_t = μ(X_t) S_t dt + σ(X_t) S_t dW_t + J(X_t) dN_t(X_t) S_t

    Transition matrix (paper §4.2, empirical MLE on EU ETS data):
        P = [[p11,      1 − p11],
             [1 − p22,  p22    ]]
    with p11 ≈ p22 ≈ 0.9868.
    """
    regime0: RegimeParams   # stable policy
    regime1: RegimeParams   # high policy uncertainty
    p11: float = 0.9868
    p22: float = 0.9868

    @property
    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic: P[i, j] = P(X_{t+1}=j | X_t=i)."""
        return np.array(
            [[self.p11,       1.0 - self.p11],
             [1.0 - self.p22, self.p22      ]],
            dtype=np.float64,
        )

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Analytic stationary probability vector π = [π₀, π₁]."""
        q01 = 1.0 - self.p11
        q10 = 1.0 - self.p22
        denom = q01 + q10
        return np.array([q10 / denom, q01 / denom], dtype=np.float64)

    def regime(self, state: int) -> RegimeParams:
        return self.regime0 if state == 0 else self.regime1


# ---------------------------------------------------------------------------
# CYO result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CYOOptionResult:
    """
    Carbon-Yield Outflow (CYO) Green Equity Option prices (paper §9.1).

    C_CYO = S·e^{−q_c T}·Φ(d₁) − K·e^{−rT}·Φ(d₂)
    P_CYO = K·e^{−rT}·Φ(−d₂) − S·e^{−q_c T}·Φ(−d₁)
    """
    call: float
    put: float
    d1: float
    d2: float
    carbon_yield: float   # q_c = E_annual × SCC / M_t


# ---------------------------------------------------------------------------
# RSJM engine
# ---------------------------------------------------------------------------

class RSJMEngine:
    """
    Regime-Switching Jump Diffusion Model engine (paper §4.1, §4.2, §9.1).

    Capabilities
    ------------
    1. Vectorised Monte Carlo simulation of the RSJM SDE.
    2. Hamilton (1989) filter for regime-tracking and log-likelihood.
    3. Risk-neutral Monte Carlo European option pricing.
    4. CYO Green Equity Option pricing via continuous-dividend BSM.
    """

    def __init__(
        self,
        params: RSJMParams,
        dt: float = 1.0 / 252.0,
        seed: int = 42,
    ) -> None:
        self.params = params
        self.dt     = dt
        self.rng    = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Vectorised Monte Carlo simulation of the RSJM SDE
    # (paper §4.1):  dS_t = μ(X_t)S_t dt + σ(X_t)S_t dW_t + J(X_t)dN_t(X_t)S_t
    #
    # Implementation strategy:
    #   - Outer loop over time steps (causally required by Markov chain).
    #   - Fully vectorised over all n_paths simultaneously — zero path loops.
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
    ) -> np.ndarray:
        """
        Vectorised RSJM path simulation (physical measure).

        Log-return decomposition per step:
            log(S_{t+1}/S_t) = (μ(X_t) − λ(X_t)κ(X_t) − σ(X_t)²/2)Δt
                                + σ(X_t)√Δt · Z_t
                                + Σ_{k=1}^{N_t} J_k(X_t)

        Parameters
        ----------
        S0      : float   Initial asset price.
        T       : float   Time horizon (years).
        n_steps : int     Number of discretisation steps.
        n_paths : int     Number of independent paths (vectorised axis).

        Returns
        -------
        paths : np.ndarray, shape (n_steps + 1, n_paths)
                Row 0 = S0, row -1 = terminal prices.
        """
        dt = T / n_steps
        p  = self.params
        P  = p.transition_matrix          # (2, 2)
        pi = p.stationary_distribution   # (2,)

        # --- Regime initialisation from stationary distribution ---
        regimes: np.ndarray = np.empty((n_steps, n_paths), dtype=np.int32)
        regimes[0] = (self.rng.uniform(size=n_paths) >= pi[0]).astype(np.int32)

        # Pre-generate all regime uniform draws — shape (n_steps-1, n_paths)
        u_reg = self.rng.uniform(size=(n_steps - 1, n_paths))

        # Markov chain: loop over time steps, vectorised over paths
        for t in range(1, n_steps):
            prev      = regimes[t - 1]
            threshold = np.where(prev == 0, P[0, 0], P[1, 1])
            regimes[t] = np.where(u_reg[t - 1] < threshold, prev, 1 - prev)

        # --- State-dependent parameter broadcasting ---
        r0, r1 = p.regime0, p.regime1

        mu_arr  = np.where(regimes == 0, r0.mu,      r1.mu)      # (n, n_paths)
        sig_arr = np.where(regimes == 0, r0.sigma,   r1.sigma)
        lam_arr = np.where(regimes == 0, r0.lam,     r1.lam)
        muj_arr = np.where(regimes == 0, r0.mu_j,    r1.mu_j)
        sj_arr  = np.where(regimes == 0, r0.sigma_j, r1.sigma_j)

        # Drift compensation κ(X_t) = exp(μ_J + σ_J²/2) − 1
        kappa_arr = np.exp(muj_arr + 0.5 * sj_arr ** 2) - 1.0

        # --- Brownian component (fully vectorised) ---
        drift     = (mu_arr - lam_arr * kappa_arr - 0.5 * sig_arr ** 2) * dt
        diffusion = sig_arr * np.sqrt(dt) * self.rng.standard_normal((n_steps, n_paths))

        # --- Compound Poisson component (fully vectorised) ---
        n_jumps   = self.rng.poisson(lam_arr * dt)          # integer (n, n_paths)
        jump_mean = muj_arr * n_jumps
        jump_std  = sj_arr * np.sqrt(np.maximum(n_jumps.astype(np.float64), 1e-12))
        jump_noise = self.rng.standard_normal((n_steps, n_paths))
        # Zero out jump component where no jump occurred
        jump_total = (jump_mean + jump_std * jump_noise) * (n_jumps > 0)

        log_incr = drift + diffusion + jump_total            # (n_steps, n_paths)
        log_path = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_incr, axis=0)])

        return S0 * np.exp(log_path)

    # ------------------------------------------------------------------
    # Hamilton (1989) filter — regime-filtered log-likelihood
    # ------------------------------------------------------------------

    def _hamilton_filter(
        self,
        log_returns: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Hamilton filter for the hidden two-state Markov chain.

        For each observation r_t the conditional density in regime s is the
        Merton (1976) Poisson mixture of Gaussians evaluated via
        RegimeParams.log_density_mixture.

        Algorithm
        ---------
        Initialise:  π = stationary distribution
        For t = 1..T:
            joint_t[s]  = π[s] · f(r_t | X_t = s)
            LL         += log Σ_s joint_t[s]
            π[s]        = joint_t[s] / Σ_s joint_t[s]      ← filtered probs
            π           = π @ P                              ← prediction step

        Returns
        -------
        log_lik    : float
        filtered_p : np.ndarray, shape (T, 2)   P(X_t=s | r_{1:t})
        """
        r  = np.asarray(log_returns, dtype=np.float64).ravel()
        T  = len(r)
        dt = self.dt
        P  = self.params.transition_matrix   # (2, 2)

        log_lik   = 0.0
        filtered  = np.empty((T, 2), dtype=np.float64)
        prob      = self.params.stationary_distribution.copy()   # (2,)

        for t in range(T):
            rt = r[t : t + 1]   # shape (1,) — reuse vectorised density method

            # Conditional densities in each regime  —  scalars
            d0 = float(self.params.regime0.log_density_mixture(rt, dt)[0])
            d1 = float(self.params.regime1.log_density_mixture(rt, dt)[0])
            cond_dens = np.array([d0, d1], dtype=np.float64)

            # Joint and marginal
            joint = prob * cond_dens
            marg  = joint.sum()
            log_lik += np.log(max(marg, 1e-300))

            # Filtered probability P(X_t | r_{1:t})
            prob          = joint / max(marg, 1e-300)
            filtered[t]   = prob

            # One-step-ahead prediction: P(X_{t+1}) = P(X_t) @ P
            prob = prob @ P

        return log_lik, filtered

    # ------------------------------------------------------------------
    # Public log-likelihood / regime-probability API
    # ------------------------------------------------------------------

    def log_likelihood(self, log_returns: np.ndarray) -> float:
        """Total Hamilton-filter log-likelihood for given log-returns."""
        ll, _ = self._hamilton_filter(log_returns)
        return ll

    def filtered_regime_probs(self, log_returns: np.ndarray) -> np.ndarray:
        """
        Return filtered regime probabilities P(X_t=1 | r_{1:t}).

        Returns
        -------
        filtered_p : np.ndarray, shape (T, 2)
        """
        _, fp = self._hamilton_filter(log_returns)
        return fp

    # ------------------------------------------------------------------
    # European option pricing via Monte Carlo (risk-neutral measure)
    # ------------------------------------------------------------------

    def price_option_mc(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ) -> dict[str, float]:
        """
        Risk-neutral Monte Carlo pricing for European call and put.

        Following the Esscher transform (paper §4.1), physical drift μ(X_t)
        is replaced by the risk-free rate r under the risk-neutral measure.

        Parameters
        ----------
        S0      : float   Spot price.
        K       : float   Strike.
        T       : float   Expiry (years).
        r       : float   Risk-free rate (continuous, annualised).
        n_paths : int     Monte Carlo paths.
        n_steps : int     Time discretisation steps.

        Returns
        -------
        dict with 'call', 'put', 'call_stderr'
        """
        p = self.params

        # Substitute risk-neutral drift r for physical drift μ(X_t)
        rn_params = RSJMParams(
            regime0=RegimeParams(
                mu=r,
                sigma=p.regime0.sigma,
                lam=p.regime0.lam,
                mu_j=p.regime0.mu_j,
                sigma_j=p.regime0.sigma_j,
            ),
            regime1=RegimeParams(
                mu=r,
                sigma=p.regime1.sigma,
                lam=p.regime1.lam,
                mu_j=p.regime1.mu_j,
                sigma_j=p.regime1.sigma_j,
            ),
            p11=p.p11,
            p22=p.p22,
        )
        engine_rn = RSJMEngine(rn_params, self.dt, seed=int(self.rng.integers(0, 2**31)))
        paths     = engine_rn.simulate_paths(S0, T, n_steps, n_paths)
        ST        = paths[-1]                       # (n_paths,)

        disc      = np.exp(-r * T)
        payoff_c  = np.maximum(ST - K, 0.0)
        payoff_p  = np.maximum(K - ST, 0.0)

        call      = float(disc * payoff_c.mean())
        put       = float(disc * payoff_p.mean())
        se_call   = float(disc * payoff_c.std(ddof=1) / np.sqrt(n_paths))

        return {"call": call, "put": put, "call_stderr": se_call}

    # ------------------------------------------------------------------
    # CYO Green Equity Option — paper §9.1
    # (static method: no RSJM parameters required for this closed-form)
    # ------------------------------------------------------------------

    @staticmethod
    def price_cyo_option(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        annual_emissions: float,
        scc: float,
        market_cap: float,
    ) -> CYOOptionResult:
        """
        Carbon-Yield Outflow (CYO) Green Equity Option pricing (paper §9.1).

        The continuous Carbon Yield (paper §9.1):
            q_c = E_annual × SCC / M_t

        CYO Call:
            C_CYO = S·e^{−q_c T}·Φ(d₁) − K·e^{−rT}·Φ(d₂)

        CYO Put:
            P_CYO = K·e^{−rT}·Φ(−d₂) − S·e^{−q_c T}·Φ(−d₁)

        d₁ = [ln(S/K) + (r − q_c + σ²/2)T] / (σ√T)
        d₂ = d₁ − σ√T

        The CYO model resolves the Bankruptcy Paradox of the CAU model
        (paper §7.3) because e^{−q_c T} > 0 always, preserving log-normality.

        Parameters
        ----------
        S                : float   Current stock price ($).
        K                : float   Strike price ($).
        T                : float   Time to expiry (years).
        r                : float   Risk-free rate (continuous, annualised).
        sigma            : float   Equity volatility (annualised).
        annual_emissions : float   Verified annual tCO₂e (Scope 1+2+3).
        scc              : float   Social Cost of Carbon ($/tCO₂e) — CONFIGURABLE,
                                   not hardcoded; pass EPA central estimate
                                   ($210 for 2025, $310 for 2050, paper §6.2).
        market_cap       : float   Total market capitalisation ($).

        Returns
        -------
        CYOOptionResult
        """
        if market_cap <= 0.0:
            raise ValueError("market_cap must be strictly positive.")
        if annual_emissions < 0.0:
            raise ValueError("annual_emissions cannot be negative.")
        if T <= 0.0:
            raise ValueError("T (time to expiry) must be positive.")

        # Carbon yield: q_c = E_annual × SCC / M_t  (paper §9.1)
        q_c = annual_emissions * scc / market_cap

        sqrtT   = np.sqrt(T)
        d1      = (np.log(S / K) + (r - q_c + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2      = d1 - sigma * sqrtT

        disc_q  = np.exp(-q_c * T)    # e^{−q_c T}  — always > 0, no bankruptcy paradox
        disc_r  = np.exp(-r   * T)    # e^{−rT}

        call = float(S * disc_q * stats.norm.cdf( d1) - K * disc_r * stats.norm.cdf( d2))
        put  = float(K * disc_r * stats.norm.cdf(-d2) - S * disc_q * stats.norm.cdf(-d1))

        return CYOOptionResult(
            call=call,
            put=put,
            d1=float(d1),
            d2=float(d2),
            carbon_yield=float(q_c),
        )

    # ------------------------------------------------------------------
    # Convenience: RMSE over option strip (for backtest)
    # ------------------------------------------------------------------

    def compute_rmse_mc(
        self,
        S0: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        observed_calls: np.ndarray,
        n_paths: int = 20_000,
    ) -> float:
        """RMSE of MC-priced RSJM calls against observed call prices."""
        predicted = np.array([
            self.price_option_mc(S0, K, T, r, n_paths=n_paths)["call"]
            for K in strikes
        ])
        return float(np.sqrt(np.mean((predicted - observed_calls) ** 2)))
