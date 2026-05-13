# Narrative Contagion Model

An agent-based epidemic model of financial narrative spread, built on the theoretical framework developed in *When Beliefs Become Entrenched: Narrative Dominance and Mispricing in Equity Markets* (Wacker, 2026).

---

## Motivation

Standard asset pricing assumes that investors update beliefs rationally and continuously. Empirically, this fails: narratives — coherent, emotionally resonant stories about an asset or sector — spread through investor populations the way infectious diseases spread through biological ones. Adoption accelerates with social proof, peaks when the narrative is fully entrenched, and unwinds when contradicting evidence accumulates or the crowded trade reverses.

This project operationalises that mechanism. The central claim of Wacker (2026) is that **narrative dominance creates predictable mispricing windows**: overpricing during peak entrenchment, and corrective underpricing during the unwind. This model makes that claim empirically testable on any liquid equity.

---

## Model Architecture

### Epidemic Framework: SIRS

The model adapts the classic SIR epidemic system to financial belief dynamics:

| Compartment | Interpretation |
|---|---|
| **S** (Susceptible) | Investors not yet pricing in the narrative |
| **I** (Infected) | Investors actively driving price on the narrative |
| **R** (Recovered) | Investors who have fully priced it in or actively faded it |

The key extension over standard SIR is the **R → S feedback** (SIRS): recovered investors gradually re-enter the susceptible pool as beliefs reset. This operationalises the *narrative reset* mechanism from Wacker (2026) — entrenchment is not permanent, and new narrative waves can emerge from the same population.

```
dS/dt = −β(t)·S·I + δ·R
dI/dt =  β(t)·S·I − γ(t)·I
dR/dt =  γ(t)·I   − δ·R
```

`δ = 0.025` per day implies a ~40 trading day average belief-reset window.

### Time-Varying Parameters

**β(t) — Transmission rate** is calibrated from market observables that proxy narrative intensity:

```
β(t) = β_base × (0.4 + 1.2 · σ(0.5·momentum + 0.3·volume_z + 0.2·RSI_norm))
```

Strong price momentum, elevated volume, and overbought RSI all accelerate narrative spread. In a bear market, β falls below γ (R₀ < 1) and the narrative decays.

**γ(t) — Recovery rate** is calibrated from disruption signals:

```
γ(t) = γ_base × (0.7 + 1.8 · σ(0.6·vol_expansion + 0.4·mean_reversion))
```

Volatility spikes and mean-reverting return autocorrelation accelerate narrative abandonment — the quantitative signature of a crowding unwind.

### Heterogeneous Agent Population

Five investor archetypes drive the population-weighted β and γ:

| Agent | β | γ | Narrative role |
|---|---|---|---|
| Value | 0.08 | 0.12 | Slow to adopt; holds views for months |
| Momentum | 0.65 | 0.08 | Fast adopter; rides the narrative until forced out |
| Retail | 0.80 | 0.05 | Highest susceptibility; last to exit |
| Institutional | 0.35 | 0.10 | Moderate; subject to internal herding dynamics |
| Contrarian | 0.03 | 0.55 | Nearly immune; exits immediately when infected |

Population weights are adjustable in the dashboard — shifting weight toward Retail raises R₀ and accelerates spread; shifting toward Contrarian dampens it.

### Narrative Phase Classification

Each day is classified into one of five phases using the adaptive relative position of I within its 63-day rolling distribution:

| Phase | Condition | Signal |
|---|---|---|
| **Seeding** | I near rolling floor, rising | Narrative emerging |
| **Spreading** | I in middle range, rising | Active contagion — momentum building |
| **Peak Entrenchment** | I near rolling ceiling | Maximum crowding risk |
| **Unwinding** | I falling from recent high | Mean-reversion likely; fade signal |
| **Resolved** | I near rolling floor, falling | Belief fully priced in |

### Mispricing Score

A signed scalar in [−1, 1] measuring narrative-driven price distortion:

```
score(t) = I(t) · tanh(dI(t) · 20) · sign(momentum(t))
         normalized by rolling 252-day absolute max
```

Positive → crowded long, price overshooting fundamentals.  
Negative → narrative suppression or active unwind underway.

### Regime Detection

A Gaussian HMM on `[log_return, realized_vol, volume_zscore]` detects 2–3 latent market regimes. States are reordered by mean return (Bear / Neutral / Bull) and overlaid on the dashboard. The regime state modulates the economic interpretation of contagion phase — a Peak Entrenchment in a Bull regime carries different risk than the same phase in a Bear regime.

---

## Project Structure

```
narrative-contagion-model/
├── src/
│   ├── data/
│   │   └── loader.py           # yfinance download + feature engineering
│   ├── model/
│   │   ├── agents.py           # Agent type dataclasses (β, γ parameters)
│   │   ├── calibration.py      # Time-varying β(t), γ(t) from market data
│   │   ├── contagion.py        # SIRS Euler integration
│   │   └── regime.py           # Gaussian HMM regime detector
│   ├── signals/
│   │   └── mispricing.py       # Phase classification + mispricing score
│   └── viz/
│       └── charts.py           # Plotly dashboard builders
├── app.py                      # Streamlit interactive dashboard
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ciaramary13/narrative-contagion-model
cd narrative-contagion-model
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage

1. Enter any liquid US equity ticker (e.g. `NVDA`, `GME`, `TSLA`, `META`)
2. Set the date range
3. Adjust agent population weights to model different investor compositions
4. Click **Run Model**

The dashboard displays:
- Price chart with narrative phase bands overlaid
- S / I / R population dynamics over time
- Signed mispricing score with ±0.5 threshold lines
- HMM market regime classification

**Suggested case studies:**

| Ticker | Period | Narrative |
|---|---|---|
| `NVDA` | 2022–2024 | AI infrastructure supercycle |
| `GME` | 2020–2022 | Retail short-squeeze narrative |
| `TSLA` | 2019–2021 | EV disruption / retail cult |
| `META` | 2021–2023 | Metaverse rebrand and correction |

---

## Theoretical Grounding

The epidemic analogy for financial markets is not new — Shiller (2019) formalises it in *Narrative Economics* — but prior implementations treat β and γ as constants estimated from price data alone. This model differs in two ways:

1. **Heterogeneous agents:** β and γ are population-weighted across five investor archetypes with distinct behavioural parameters, rather than treated as market averages.

2. **Market-calibrated dynamics:** β(t) and γ(t) are driven by observable features (momentum, volume, volatility) that proxy the underlying narrative intensity and disruption mechanisms described in Wacker (2026).

The result is a model where the *timing* of phase transitions is determined by market data, while the *structure* of those transitions is governed by the epidemic mathematics — making the mispricing score an empirically grounded, forward-looking signal rather than a retrospective label.

---

## Dependencies

`yfinance` · `numpy` · `pandas` · `scipy` · `hmmlearn` · `scikit-learn` · `plotly` · `streamlit`

---

*Ciara Wacker — Fordham University*
