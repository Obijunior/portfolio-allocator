import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

import pickle

def rolling_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    window: int = 52
) -> dict:
    """
    Compute rolling Ledoit-Wolf covariance matrices.

    Returns: dict -> {timestamp: covariance_matrix (pd.DataFrame)}
    """
    covariances = {}

    for end_date in returns.index[window:]:
        window_returns = returns.loc[:end_date].tail(window)

        lw = LedoitWolf().fit(window_returns.values)
        cov = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )

        covariances[end_date] = cov

    return covariances

def save_covariances(covs, path="data/covariances.pkl"):
    with open(path, "wb") as f:
        pickle.dump(covs, f)