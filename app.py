import pandas as pd
import streamlit as st
from datetime import date, timedelta

from src.data.loader import fetch_data
from src.model.agents import (
    AgentType,
    CONTRARIAN,
    DEFAULT_POPULATION,
    INSTITUTIONAL,
    MOMENTUM,
    RETAIL,
    VALUE,
)
from src.model.contagion import run as run_contagion
from src.model.regime import REGIME_LABELS, fit_regime_hmm
from src.signals.mispricing import (
    PHASE_LABELS,
    classify_phase,
    compute_mispricing,
)
from src.viz.charts import make_dashboard

st.set_page_config(
    page_title="Narrative Contagion Model",
    page_icon="📡",
    layout="wide",
)


def _desc(name: str) -> str:
    return {
        "Value":         "Slow to adopt narratives; reverts to fundamentals quickly",
        "Momentum":      "Fast narrative adoption; slow to exit crowded positions",
        "Retail":        "Highest susceptibility; slowest to recover",
        "Institutional": "Moderate adoption; subject to internal herding",
        "Contrarian":    "Immune to narrative; actively fades crowded trades",
    }.get(name, "")

st.title("Narrative Contagion Model")
st.caption(
    "An agent-based epidemic model of financial narrative spread. "
    "Based on *When Beliefs Become Entrenched: Narrative Dominance and "
    "Mispricing in Equity Markets* (Wacker, 2026)."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Ticker", value="NVDA").upper().strip()

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Start", value=date.today() - timedelta(days=365 * 3))
    end_date   = col_d2.date_input("End",   value=date.today())

    st.subheader("Agent Population")
    st.caption("Weights are normalized automatically.")

    w_value         = st.slider("Value Investors",   0.0, 1.0, VALUE.weight,         0.05)
    w_momentum      = st.slider("Momentum Traders",  0.0, 1.0, MOMENTUM.weight,      0.05)
    w_retail        = st.slider("Retail Investors",  0.0, 1.0, RETAIL.weight,        0.05)
    w_institutional = st.slider("Institutional",     0.0, 1.0, INSTITUTIONAL.weight, 0.05)
    w_contrarian    = st.slider("Contrarians",       0.0, 1.0, CONTRARIAN.weight,    0.05)

    total_w = w_value + w_momentum + w_retail + w_institutional + w_contrarian
    if abs(total_w) < 1e-8:
        st.error("At least one agent must have weight > 0.")
        total_w = 1.0

    n_regimes = st.selectbox("HMM Regime States", [2, 3], index=1)

    run_btn = st.button("Run Model", type="primary", use_container_width=True)

    st.divider()
    st.caption("Suggested tickers for strong narrative dynamics:")
    st.caption("**NVDA** · AI bubble 2023–24")
    st.caption("**GME** · Meme stock 2021")
    st.caption("**META** · Rebrand narrative 2021–22")
    st.caption("**TSLA** · Retail narrative 2020–21")

# ── Run model ───────────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching data and running model for **{ticker}**…"):
        try:
            df = fetch_data(ticker, str(start_date), str(end_date))
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

        agents = [
            AgentType("Value",         VALUE.beta,         VALUE.gamma,         w_value         / total_w, VALUE.color),
            AgentType("Momentum",      MOMENTUM.beta,      MOMENTUM.gamma,      w_momentum      / total_w, MOMENTUM.color),
            AgentType("Retail",        RETAIL.beta,        RETAIL.gamma,        w_retail        / total_w, RETAIL.color),
            AgentType("Institutional", INSTITUTIONAL.beta, INSTITUTIONAL.gamma, w_institutional / total_w, INSTITUTIONAL.color),
            AgentType("Contrarian",    CONTRARIAN.beta,    CONTRARIAN.gamma,    w_contrarian    / total_w, CONTRARIAN.color),
        ]

        sir        = run_contagion(df, agents)
        phases     = classify_phase(sir["I"].values, sir["dI"].values)
        regimes, _ = fit_regime_hmm(df, n_states=n_regimes)
        mispricing = compute_mispricing(sir, df)

        st.session_state["results"] = (df, sir, phases, regimes, mispricing, ticker)

# ── Display results ──────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.info("Configure settings in the sidebar and click **Run Model** to begin.")
    st.stop()

df, sir, phases, regimes, mispricing, used_ticker = st.session_state["results"]

# Metrics
current_phase     = phases[-1]
current_i         = sir["I"].iloc[-1]
current_mispricing = mispricing.iloc[-1]
current_regime    = REGIME_LABELS.get(int(regimes[-1]), "Unknown") if len(regimes) else "—"
peak_idx          = sir["I"].idxmax()
days_since_peak   = (df.index[-1] - peak_idx).days

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker",            used_ticker)
m2.metric("Narrative Phase",   PHASE_LABELS.get(current_phase, current_phase))
m3.metric("Contagion (I)",     f"{current_i:.1%}")
m4.metric("Mispricing Score",  f"{current_mispricing:+.2f}")
m5.metric("Market Regime",     current_regime)

# Main chart
fig = make_dashboard(df, sir, phases, regimes, mispricing)
st.plotly_chart(fig, use_container_width=True)

# Reference tables
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("Agent Population Parameters"):
        agent_rows = [
            {
                "Agent":         a.name,
                "β (Spread)":    f"{a.beta:.2f}",
                "γ (Recovery)":  f"{a.gamma:.2f}",
                "Description":   _desc(a.name),
            }
            for a in DEFAULT_POPULATION
        ]
        st.dataframe(pd.DataFrame(agent_rows), hide_index=True, use_container_width=True)

with col_right:
    with st.expander("Phase Classification Guide"):
        phase_rows = [
            {"Phase": PHASE_LABELS["seeding"],   "Condition": "I < 12%",               "Signal": "Narrative emerging"},
            {"Phase": PHASE_LABELS["spreading"],  "Condition": "12% ≤ I < 45%, rising", "Signal": "Active contagion — momentum builds"},
            {"Phase": PHASE_LABELS["peak"],       "Condition": "I ≥ 45% or flat",       "Signal": "Max entrenchment — crowding risk"},
            {"Phase": PHASE_LABELS["unwinding"],  "Condition": "I falling",             "Signal": "Narrative decay — mean-reversion likely"},
            {"Phase": PHASE_LABELS["resolved"],   "Condition": "I < 12%, R dominant",   "Signal": "Belief fully priced in"},
        ]
        st.dataframe(pd.DataFrame(phase_rows), hide_index=True, use_container_width=True)
