import numpy as np
import pandas as pd


PHASE_LABELS = {
    "seeding":   "Seeding",
    "spreading": "Spreading",
    "peak":      "Peak Entrenchment",
    "unwinding": "Unwinding",
    "resolved":  "Resolved",
}

PHASE_COLORS = {
    "seeding":   "#90CAF9",
    "spreading": "#FF9800",
    "peak":      "#F44336",
    "unwinding": "#CE93D8",
    "resolved":  "#A5D6A7",
}


def classify_phase(I: np.ndarray, dI: np.ndarray, window: int = 63) -> list[str]:
    """
    Adaptive phase classification using rolling relative position of I.

    Rather than fixed absolute thresholds, each day's I is measured relative
    to its recent history (rolling mean and max). This makes the classification
    robust across assets with very different narrative intensities.

    rel = 0 → I at its rolling mean (baseline)
    rel = 1 → I at its rolling max (peak narrative entrenchment)
    """
    I_s = pd.Series(I)
    i_mean = I_s.rolling(window, min_periods=10).mean().bfill()
    i_max  = I_s.rolling(window, min_periods=10).max().bfill()
    span   = (i_max - i_mean).clip(lower=1e-6)
    rel    = ((I_s - i_mean) / span).clip(0, 1)

    phases = []
    for r, di_val in zip(rel, dI):
        if r < 0.25:
            phase = "seeding" if di_val >= 0 else "resolved"
        elif r < 0.65 and di_val > 0:
            phase = "spreading"
        elif r >= 0.65:
            phase = "peak" if di_val >= -5e-4 else "unwinding"
        else:
            phase = "unwinding"
        phases.append(phase)
    return phases


def compute_mispricing(sir: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """
    Signed narrative mispricing score in [-1, 1].

    Intuition from Wacker (2026): when a narrative is actively spreading
    (high I, rising dI) in a momentum environment, price overshoots
    fundamentals. The score captures the signed distance from fair value
    attributable to narrative dynamics alone.

    Positive → crowded long / overpriced
    Negative → narrative suppressing price or unwinding
    """
    momentum_sign = np.sign(
        df["momentum_combined"].reindex(sir.index).fillna(0).values
    )
    raw = sir["I"].values * np.tanh(sir["dI"].values * 20) * momentum_sign

    raw_series = pd.Series(raw, index=sir.index)
    rolling_max = raw_series.abs().rolling(252, min_periods=21).max().clip(lower=1e-8)

    return (raw_series / rolling_max).clip(-1, 1).rename("mispricing_score")
