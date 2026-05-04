"""
src/data_generator.py
=====================
Synthesises realistic European Climate Exchange (ECX) carbon spot log-return
data under a two-regime jump-diffusion process (RSJM, paper §4.1).

Explicitly reproduces:
  Phase I  (2005–2007): over-allocation environment + forced 2006 crash.
  Phase II (2008–2012): post-reform, lower but persistent volatility.

Regime switching follows a discrete-time two-state Markov chain with
transition probabilities P11 = P22 ≈ 0.9868 (paper §4.2).

All Monte Carlo simulation is fully vectorised over paths; no Python loops
over the path dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseConfig:
    """Immutable configuration for one ECX market phase."""

    name: str
    n_days: int

    # Regime-level parameter overrides are kept in ECXDataGenerator
    regime_bias: float       # P(initial state = turbulent)
    shock_day: Optional[int] # index of forced structural shock
    shock_size: float        # log-return magnitude of shock


# Phase I: 2005-2007, dominated by over-allocation; ~70% crash mid-phase.
PHASE_I_CONFIG = PhaseConfig(
    name="Phase_I",
    n_days=520,
    regime_bias=0.60,
    shock_day=260,   # 2006 verified-emissions revelation
    shock_size=-1.20,
)

# Phase II: 2008-2012, post-reform; lower base vol, occasional policy shocks.
PHASE_II_CONFIG = PhaseConfig(
    name="Phase_II",
    n_days=1_040,
    regime_bias=0.35,
    shock_day=None,
    shock_size=0.0,
)

# ---------------------------------------------------------------------------
# Per-regime parameters for each phase (state 0 = stable, 1 = turbulent)
# ---------------------------------------------------------------------------

_PHASE_I_REGIME_PARAMS: dict[int, dict[str, float]] = {
    0: dict(mu=0.05,  sigma=0.30, lam=3.0,  mu_j=-0.02, sigma_j=0.08),
    1: dict(mu=0.15,  sigma=0.65, lam=14.0, mu_j=-0.08, sigma_j=0.18),
}

_PHASE_II_REGIME_PARAMS: dict[int, dict[str, float]] = {
    0: dict(mu=0.03,  sigma=0.20, lam=2.0,  mu_j=0.02,  sigma_j=0.06),
    1: dict(mu=0.08,  sigma=0.42, lam=7.0,  mu_j=0.05,  sigma_j=0.12),
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ECXDataGenerator:
    """
    Generates synthetic ECX Phase I + Phase II carbon spot log-returns.

    The underlying data-generating process is the RSJM SDE (paper §4.1):

        dS_t = μ(X_t) S_t dt + σ(X_t) S_t dW_t + J(X_t) dN_t(X_t) S_t

    X_t ∈ {0, 1} is governed by a two-state Markov chain with the empirical
    transition matrix estimated by MLE on EU ETS Phase I & II data (paper §4.2):

        P = [[p11,      1 - p11],
             [1 - p22,  p22    ]]

    with P11 = P22 ≈ 0.9868, reflecting near-absolute regime persistence.
    """

    # Empirical transition probabilities from paper §4.2
    DEFAULT_P11: float = 0.9868
    DEFAULT_P22: float = 0.9868

    def __init__(
        self,
        seed: int = 42,
        dt: float = 1.0 / 252.0,
        p11: float = DEFAULT_P11,
        p22: float = DEFAULT_P22,
    ) -> None:
        self.dt  = dt
        self.p11 = p11
        self.p22 = p22
        self.rng = np.random.default_rng(seed)

        self._transition_matrix: np.ndarray = np.array(
            [[p11,       1.0 - p11],
             [1.0 - p22, p22      ]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Transition matrix (read-only view)
    # ------------------------------------------------------------------

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix.copy()

    # ------------------------------------------------------------------
    # Markov chain regime simulation
    # (loop over time steps, fully vectorised over paths dimension)
    # ------------------------------------------------------------------

    def _simulate_regimes(
        self,
        n_steps: int,
        n_paths: int,
        initial_regime: int,
    ) -> np.ndarray:
        """
        Simulate discrete-time Markov chain regime sequences.

        Parameters
        ----------
        n_steps        : int
        n_paths        : int
        initial_regime : int   0 = stable, 1 = turbulent

        Returns
        -------
        regimes : np.ndarray, shape (n_steps, n_paths), dtype int32
        """
        regimes: np.ndarray = np.empty((n_steps, n_paths), dtype=np.int32)
        regimes[0, :] = initial_regime

        # Pre-generate all uniform draws at once — vectorised over paths
        u: np.ndarray = self.rng.uniform(size=(n_steps - 1, n_paths))

        for t in range(1, n_steps):
            prev      = regimes[t - 1]                                # (n_paths,)
            threshold = np.where(prev == 0, self.p11, self.p22)       # (n_paths,)
            # Stay in current regime if u < threshold, else switch
            regimes[t] = np.where(u[t - 1] < threshold, prev, 1 - prev)

        return regimes

    # ------------------------------------------------------------------
    # Single-phase log-return simulation (vectorised over paths)
    # ------------------------------------------------------------------

    def _simulate_phase_returns(
        self,
        cfg: PhaseConfig,
        regime_params: dict[int, dict[str, float]],
        n_paths: int,
    ) -> np.ndarray:
        """
        Simulate log-returns for one phase.

        Log-return decomposition (Itô's Lemma applied to RSJM SDE):
            r_t = (μ(X_t) − λ(X_t)κ(X_t) − ½σ(X_t)²)Δt
                   + σ(X_t)√Δt · Z_t
                   + Σ_{k=1}^{N_t} J_k(X_t)

        where κ(X_t) = E[e^{J(X_t)} − 1] is the drift-compensation term,
        N_t ~ Poisson(λ(X_t)Δt), J_k ~ N(μ_J(X_t), σ_J(X_t)²).

        Returns
        -------
        log_returns : np.ndarray, shape (cfg.n_days, n_paths)
        """
        n   = cfg.n_days
        dt  = self.dt

        # Initial regime biased toward turbulent regime per phase
        init = int(self.rng.uniform() < cfg.regime_bias)
        reg  = self._simulate_regimes(n, n_paths, init)   # (n, n_paths)

        # --- State-dependent parameter arrays (vectorised over time × paths) ---
        r0, r1 = regime_params[0], regime_params[1]

        mu_arr    = np.where(reg == 0, r0["mu"],      r1["mu"])       # (n, n_paths)
        sig_arr   = np.where(reg == 0, r0["sigma"],   r1["sigma"])
        lam_arr   = np.where(reg == 0, r0["lam"],     r1["lam"])
        mu_j_arr  = np.where(reg == 0, r0["mu_j"],    r1["mu_j"])
        sj_arr    = np.where(reg == 0, r0["sigma_j"], r1["sigma_j"])

        # --- Drift compensation κ(X_t) = E[e^{J} − 1] ---
        kappa_arr = np.exp(mu_j_arr + 0.5 * sj_arr ** 2) - 1.0

        # --- Continuous diffusion component ---
        drift     = (mu_arr - lam_arr * kappa_arr - 0.5 * sig_arr ** 2) * dt
        diffusion = sig_arr * np.sqrt(dt) * self.rng.standard_normal((n, n_paths))

        # --- Compound Poisson jump component ---
        n_jumps   = self.rng.poisson(lam_arr * dt)                    # (n, n_paths)
        # Aggregate jump: k i.i.d. N(μ_J, σ_J²) → N(k·μ_J, k·σ_J²)
        jump_mean = mu_j_arr * n_jumps
        jump_std  = sj_arr * np.sqrt(np.maximum(n_jumps.astype(np.float64), 1e-12))
        jump_noise = self.rng.standard_normal((n, n_paths))
        # Apply jump component only where at least one jump occurred
        jump_total = (jump_mean + jump_std * jump_noise) * (n_jumps > 0)

        log_returns = drift + diffusion + jump_total

        # --- Inject forced structural shock (Phase I: 2006 over-allocation crash) ---
        if cfg.shock_day is not None and 0 <= cfg.shock_day < n:
            log_returns[cfg.shock_day, :] += cfg.shock_size

        return log_returns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_paths: int = 1) -> dict[str, np.ndarray]:
        """
        Generate concatenated Phase I + Phase II log-return series.

        Parameters
        ----------
        n_paths : int
            Number of independent simulation paths. Default 1 (single series).

        Returns
        -------
        dict with keys:
          'log_returns'  : np.ndarray  shape (total_days,) if n_paths==1,
                                             else (total_days, n_paths)
          'phase_labels' : np.ndarray  shape (total_days,)  str array
          'n_days'       : int         total trading days
          'transition_matrix' : np.ndarray  shape (2, 2)
        """
        lr_i  = self._simulate_phase_returns(PHASE_I_CONFIG,  _PHASE_I_REGIME_PARAMS,  n_paths)
        lr_ii = self._simulate_phase_returns(PHASE_II_CONFIG, _PHASE_II_REGIME_PARAMS, n_paths)

        log_returns = np.concatenate([lr_i, lr_ii], axis=0)

        phase_labels: np.ndarray = np.array(
            ["Phase_I"]  * PHASE_I_CONFIG.n_days +
            ["Phase_II"] * PHASE_II_CONFIG.n_days,
            dtype="<U8",
        )

        if n_paths == 1:
            log_returns = log_returns[:, 0]   # squeeze for convenience

        return {
            "log_returns":       log_returns,
            "phase_labels":      phase_labels,
            "n_days":            PHASE_I_CONFIG.n_days + PHASE_II_CONFIG.n_days,
            "transition_matrix": self.transition_matrix,
        }
