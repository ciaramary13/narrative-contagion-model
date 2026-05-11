import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


REGIME_LABELS = {0: "Bear", 1: "Neutral", 2: "Bull"}
REGIME_COLORS = {0: "#EF5350", 1: "#FFA726", 2: "#66BB6A"}


def fit_regime_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
) -> tuple[np.ndarray, hmm.GaussianHMM | None]:
    """
    Gaussian HMM on [log_return, realized_vol, volume_zscore].
    States are reordered by mean return so state 0 = bearish, n-1 = bullish.
    Falls back to a volatility-percentile classifier if HMM fails to converge.
    """
    features = np.column_stack([
        df["log_return"].fillna(0).values,
        df["realized_vol"].fillna(df["realized_vol"].median()).values,
        df["volume_zscore"].fillna(0).values,
    ])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=300,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
    except Exception:
        return _volatility_fallback(df, n_states), None

    # Reorder states by mean return: 0 = lowest (bear), n-1 = highest (bull)
    mean_returns = np.array([
        df["log_return"].values[states == s].mean() if (states == s).any() else 0.0
        for s in range(n_states)
    ])
    order = np.argsort(mean_returns)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    states = np.array([remap[s] for s in states])

    return states, model


def _volatility_fallback(df: pd.DataFrame, n_states: int) -> np.ndarray:
    vol = df["realized_vol"].fillna(df["realized_vol"].median()).values
    thresholds = np.percentile(vol, np.linspace(0, 100, n_states + 1)[1:-1])
    states = np.digitize(vol, thresholds)
    # Invert so low-vol = bull (high state index), high-vol = bear (low state index)
    return (n_states - 1) - states
