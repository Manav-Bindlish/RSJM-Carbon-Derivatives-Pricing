# RSJM Carbon Derivatives Pricing

Production-grade implementation of the **Regime-Switching Jump Diffusion Model (RSJM)** and **Carbon-Yield Outflow (CYO) Green Equity Options** framework, as developed in:

> *Internalizing Carbon Externalities in Financial Derivatives: The Green Equity Options Framework and the Empirical Supremacy of Regime-Switching Models*
> Manav Bindlish, B.Sc (Honours) Statistics — *Ramjas Finvestment Review*

---

## Thesis: The Carbon-Yield Outflow Framework

Standard equity derivative pricing ignores the economic harm of corporate greenhouse gas emissions. Traditional models — including the Black-Scholes-Merton (BSM) framework and Geometric Brownian Motion (GBM) — treat a firm's stock price as a complete reflection of its assets and liabilities. In reality, unless a company operates within a fully binding cap-and-trade regime, the vast majority of its emissions represent an **unrecorded latent liability** subsidised by global society.

This framework corrects that market failure via two contributions:

### Part I — Pricing the Public Policy Asset (Carbon Allowances)
Policy-driven carbon markets exhibit **discontinuous price jumps** (triggered by regulatory shocks) and **volatility clustering** (persistent high-variance regimes). These stylised facts violate GBM's core assumptions of continuous sample paths and constant volatility. The **RSJM** resolves both deficiencies by embedding a hidden two-state Markov chain into the stochastic differential equation, making all parameters state-dependent.

### Part II — Pricing the Private Corporate Liability (Green Equity Options)
Using the **Social Cost of Carbon (SCC)** as an exogenous pricing vector, two models are derived to internalise corporate emissions into equity option valuation:

- **CAU (Carbon-Adjusted Underlying):** Subtracts a per-share carbon liability from the observed stock price. Intuitive but structurally flawed — produces negative underlying prices for carbon-intensive firms (the *Bankruptcy Paradox*) and distorts the implied volatility surface.
- **CYO (Carbon-Yield Outflow):** Treats the carbon penalty as a **continuous dividend-like yield** on the equity, anchored to the EPA's SCC. Mathematically sound, preserves log-normality, maps directly onto existing continuous-yield BSM engines, and creates direct financial incentives for corporate decarbonisation.

The CYO model integrates the **Scholes-Alankar Separation Principle** (*Carbon Emissions and Asset Management*, JOIM 2022, Harry M. Markowitz Special Distinction Award), which demonstrates that net-zero can be achieved via a predictable, deterministic yield outflow — approximately **7 bps/year** for broad US indices and **35 bps/year** for emerging markets — without destroying portfolio efficiency through exclusionary screening.

---

## Mathematical Framework

### RSJM Stochastic Differential Equation (Paper §4.1)

The RSJM SDE for the spot carbon allowance price $S_t$:

$$dS_t = \mu(X_t)\,S_t\,dt + \sigma(X_t)\,S_t\,dW_t + J(X_t)\,dN_t(X_t)\,S_t$$

where:

| Symbol | Description |
|---|---|
| $X_t \in \{0, 1\}$ | Hidden two-state Markov chain (0 = stable policy, 1 = high uncertainty) |
| $\mu(X_t)$ | State-dependent drift |
| $\sigma(X_t)$ | State-dependent Brownian (continuous) volatility |
| $dW_t$ | Standard Wiener process increment |
| $dN_t(X_t)$ | Poisson process, arrival rate $\lambda(X_t)$ |
| $J(X_t) \sim \mathcal{N}(\mu_J(X_t),\,\sigma_J^2(X_t))$ | Random log-jump magnitude |

The drift-compensation term enforces the martingale property:
$$\kappa(X_t) = \mathbb{E}\!\left[e^{J(X_t)} - 1\right] = \exp\!\left(\mu_J(X_t) + \tfrac{1}{2}\sigma_J^2(X_t)\right) - 1$$

The two-state Markov chain transition matrix (empirical MLE on EU ETS Phase I & II, §4.2):

$$P = \begin{pmatrix} P_{11} & 1-P_{11} \\ 1-P_{22} & P_{22} \end{pmatrix} \approx \begin{pmatrix} 0.9868 & 0.0132 \\ 0.0132 & 0.9868 \end{pmatrix}$$

$P_{11} \approx P_{22} \approx 0.9868$ reflects **near-absolute regime persistence** — once a regulatory shock drives the market into the high-variance regime, it remains there for an extended period, perfectly replicating observed volatility clustering.

### GBM null-hypothesis SDE (Paper §3)

$$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t$$

Log-returns are i.i.d. $\mathcal{N}\!\left((\mu - \tfrac{\sigma^2}{2})\Delta t,\;\sigma^2\Delta t\right)$.

### JDM intermediate SDE (Paper §3.2)

$$dS_t = (\mu - \lambda\kappa)\,S_t\,dt + \sigma\,S_t\,dW_t + (e^J - 1)\,S_t\,dN_t$$

Parameters $\lambda$ and $\sigma$ are **constants** — this is the structural deficit resolved by the RSJM.

### CYO Green Equity Option (Paper §9.1)

The continuous **Carbon Yield** $q_c$:

$$q_c = \frac{E_{\text{annual}} \times SCC_t}{M_t}$$

where $E_{\text{annual}}$ is total verified annual $\text{tCO}_2\text{e}$, $SCC_t$ is the EPA Social Cost of Carbon (configurable; central estimate $\$210/\text{t}$ for 2025, $\$310/\text{t}$ for 2050 at 2.0% Ramsey discount rate), and $M_t$ is total market capitalisation.

Green European Call and Put:

$$C_{\text{CYO}} = S\,e^{-q_c(T-t)}\,\Phi(d_1) - K\,e^{-r(T-t)}\,\Phi(d_2)$$

$$P_{\text{CYO}} = K\,e^{-r(T-t)}\,\Phi(-d_2) - S\,e^{-q_c(T-t)}\,\Phi(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q_c + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \qquad d_2 = d_1 - \sigma\sqrt{T-t}$$

Because $e^{-q_c(T-t)} > 0$ always, the **Bankruptcy Paradox** of the CAU model is eradicated — the effective asset price approaches zero asymptotically but can never become negative, preserving all log-normal distributional requirements.

---

## Likelihood Ratio Outperformance (Paper §3.2, §4.2)

The RSJM's superiority over both GBM and JDM is validated by $\chi^2$ Likelihood Ratio tests on EU ETS empirical data:

| Test (H₀ → Hₐ) | LR Statistic | df | Interpretation |
|---|---|---|---|
| GBM → JDM | **632.38** | 3 | Discontinuous jumps are a permanent structural feature of carbon pricing, not anomalies |
| JDM → RSJM | **134.21** | 7 | Carbon jumps are not independent Poisson events but manifestations of state-dependent policy regimes |

Both statistics are distributed $\chi^2(df)$ under the null. Both are conclusive at $p < 10^{-100}$, providing definitive rejection of GBM and JDM as carbon derivative pricing engines.

---

## Project Structure

```
RSJM-Derivatives-Pricing/
├── src/
│   ├── __init__.py
│   ├── data_generator.py   # ECX Phase I/II synthetic data (RSJM DGP)
│   ├── gbm_baseline.py     # GBM/BSM null-hypothesis model + MLE + option pricing
│   ├── rsjm_engine.py      # RSJM engine: simulation, Hamilton filter, MC pricing, CYO
│   └── likelihood.py       # MLE (GBM/JDM/RSJM) + bounded optimisation + LR tests
├── run_backtest.py         # Full pipeline: data → fit → LR tests → CYO pricing → report
├── requirements.txt
└── README.md
```

---

## Replication Instructions

```bash
# 1. Clone
git clone https://github.com/<your-handle>/RSJM-Derivatives-Pricing.git
cd RSJM-Derivatives-Pricing

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the full backtest
python run_backtest.py
```

Expected terminal output includes:

- ECX synthetic return statistics (skewness, excess kurtosis, jump magnitudes)
- GBM, JDM, and RSJM log-likelihoods and fitted parameters
- RSJM transition matrix with $P_{11}, P_{22} \approx 0.9868$
- LR test statistics benchmarked against paper values (632.38, 134.21)
- CYO Green Equity Option prices with configurable SCC and carbon yield decomposition

> **Note:** The RSJM MLE step (Hamilton filter + L-BFGS-B) typically runs in **60–180 seconds** on modern hardware due to the iterative filter evaluation at each gradient step. This is expected behaviour for a 12-parameter regime-switching model fitted on 1,560 observations.

---

## Key Design Decisions

| Constraint | Implementation |
|---|---|
| Zero `for` loops over Monte Carlo paths | All path simulation vectorised over the `n_paths` axis via NumPy broadcasting |
| No negative jump intensities | `scipy.optimize.Bounds(lb=[..., 0.0, ...])` enforced on all MLE optimisations |
| Configurable SCC | `scc` is a required argument to `RSJMEngine.price_cyo_option()`; never hardcoded |
| Regime persistence | Empirical $P_{11} = P_{22} = 0.9868$ used as warm-start; MLE recovers values consistent with paper §4.2 |
| Log-normal preservation | CYO uses exponential decay factor $e^{-q_c T}$; the CAU subtraction approach is deliberately excluded from the derivative engine |

---

## EPA SCC Reference Grid (Paper §6.2, Table)

Values in 2020 USD per $\text{tCO}_2\text{e}$:

| Year | 2.5% Discount Rate | **2.0% Central Estimate** | 1.5% Discount Rate |
|---|---|---|---|
| 2020 | $120 | **$190** | $340 |
| 2025 | $130 | **$210** | $360 |
| 2030 | $140 | **$230** | $380 |
| 2040 | $170 | **$270** | $430 |
| 2050 | $200 | **$310** | $480 |

The 2.0% Near-Term Ramsey Discount Rate central estimate represents a ~250% increase over the prior $51/tonne interim standard and is the default for all CYO model calculations.

---

## License

MIT
