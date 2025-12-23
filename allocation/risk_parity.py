import numpy as np
import pandas as pd


def risk_parity_weights(cov, tol=1e-6, max_iter=1000):
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(max_iter):
        portfolio_var = w.T @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib

        target = portfolio_var / n

        diff = risk_contrib - target
        if np.linalg.norm(diff) < tol:
            break

        w -= 0.01 * diff
        w = np.maximum(w, 0)
        w /= w.sum()

    return pd.Series(w, index=cov.index)
