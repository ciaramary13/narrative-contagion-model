import numpy as np
import pandas as pd

from .agents import AgentType, population_beta, population_gamma


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def calibrate_beta(
    df: pd.DataFrame,
    agents: list[AgentType],
) -> pd.Series:
    """
    Time-varying transmission rate driven by narrative intensity.

    In a strong momentum/volume environment (AI narrative, meme stock
    frenzy) β rises well above the population baseline, driving R0 >> 1.
    In a bear market β falls below baseline, letting γ dominate and
    causing the narrative to decay.

    Scale range: 0.40 × base  →  1.60 × base
    """
    base = population_beta(agents)

    mom = df["momentum_combined"].clip(-0.5, 0.5) / 0.5
    vol = df["volume_zscore"].clip(-3, 3) / 3
    rsi = df["rsi_normalized"]

    composite = (0.50 * mom + 0.30 * vol + 0.20 * rsi).clip(-3, 3)
    scale = 0.40 + 1.20 * _sigmoid(composite * 1.5)

    return pd.Series(base * scale, index=df.index, name="beta")


def calibrate_gamma(
    df: pd.DataFrame,
    agents: list[AgentType],
) -> pd.Series:
    """
    Time-varying recovery rate driven by narrative disruption signals.

    γ is kept intentionally low in calm conditions so narratives can
    accumulate. It rises sharply during volatility spikes and mean-
    reversion regimes — operationalising the 'crowding unwind' mechanism
    from Wacker (2026).

    Scale range: 0.70 × base  →  2.50 × base
    """
    base = population_gamma(agents)

    vol_exp   = df["vol_expansion"].clip(-2, 2) / 2
    reversion = df["mean_reversion_score"].clip(-1, 1)

    composite = (0.60 * vol_exp + 0.40 * reversion).clip(-2, 2)
    scale = 0.70 + 1.80 * _sigmoid(composite * 1.5)

    return pd.Series(base * scale, index=df.index, name="gamma")
