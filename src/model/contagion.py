import numpy as np
import pandas as pd

from .agents import AgentType, DEFAULT_POPULATION
from .calibration import calibrate_beta, calibrate_gamma


def run(
    df: pd.DataFrame,
    agents: list[AgentType] = DEFAULT_POPULATION,
    i0: float = 0.03,
    delta: float = 0.025,
) -> pd.DataFrame:
    """
    SIRS narrative contagion model with time-varying transmission parameters.

    S = fraction of market not yet adopting the narrative (susceptible)
    I = fraction actively pricing in the narrative (infected / crowded)
    R = fraction that has priced it in or faded it (recovered)

    The S→I→R→S cycle captures recurring narrative waves: δ is the daily
    re-susceptibility rate (1/δ ≈ 40 trading days average belief-reset window),
    operationalising the 'narrative reset' mechanism from Wacker (2026).

    β(t) and γ(t) are calibrated from market observables so the model tracks
    empirical dynamics rather than running free.
    """
    beta_series  = calibrate_beta(df, agents)
    gamma_series = calibrate_gamma(df, agents)

    n = len(df)
    S = np.empty(n)
    I = np.empty(n)
    R = np.empty(n)

    S[0] = 1.0 - i0
    I[0] = i0
    R[0] = 0.0

    for t in range(1, n):
        beta  = beta_series.iloc[t]
        gamma = gamma_series.iloc[t]

        new_infected    = beta  * S[t - 1] * I[t - 1]
        new_recovered   = gamma * I[t - 1]
        new_susceptible = delta * R[t - 1]  # belief reset: R → S

        S[t] = S[t - 1] - new_infected  + new_susceptible
        I[t] = I[t - 1] + new_infected  - new_recovered
        R[t] = R[t - 1] + new_recovered - new_susceptible

        S[t] = max(S[t], 0.0)
        I[t] = max(I[t], 0.0)
        R[t] = max(R[t], 0.0)
        total = S[t] + I[t] + R[t]
        if total > 0:
            S[t] /= total
            I[t] /= total
            R[t] /= total

    result = pd.DataFrame({"S": S, "I": I, "R": R}, index=df.index)
    result["beta"]  = beta_series.values
    result["gamma"] = gamma_series.values
    result["dI"]    = np.gradient(I)

    return result
