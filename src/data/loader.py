import numpy as np
import pandas as pd
import yfinance as yf


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = df.columns.str.lower()
    df = df[df["volume"] > 0].dropna()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_volume"] = np.log(df["volume"])

    # Momentum: weighted blend of 12-month and 1-month price change
    df["momentum_12m"] = df["close"].pct_change(252)
    df["momentum_1m"] = df["close"].pct_change(21)
    df["momentum_combined"] = 0.7 * df["momentum_12m"] + 0.3 * df["momentum_1m"]

    # Volume z-score over 63-day rolling window
    vol_roll = df["log_volume"].rolling(63)
    df["volume_zscore"] = (
        (df["log_volume"] - vol_roll.mean()) / vol_roll.std().clip(lower=1e-8)
    )

    df["rsi"] = _rsi(df["close"], 14)
    df["rsi_normalized"] = (df["rsi"] - 50) / 50  # maps to [-1, 1]

    df["realized_vol"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    vol_long = df["log_return"].rolling(126).std() * np.sqrt(252)
    df["vol_expansion"] = (df["realized_vol"] / vol_long.clip(lower=1e-8) - 1).clip(-2, 2)

    df["mean_reversion_score"] = -_rolling_autocorr(df["log_return"], window=63, lag=5)

    return df.dropna()


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-8)
    return 100 - (100 / (1 + rs))


def _rolling_autocorr(returns: pd.Series, window: int = 63, lag: int = 5) -> pd.Series:
    def _autocorr(x: np.ndarray) -> float:
        if len(x) < lag + 2:
            return 0.0
        corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return corr if np.isfinite(corr) else 0.0

    return returns.rolling(window).apply(_autocorr, raw=True)
