import cvxpy as cp
import pandas as pd


def mean_variance_weights(
    mu,
    cov,
    risk_aversion=10.0,
    max_weight=0.4
):
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Maximize(mu.values @ w - risk_aversion * cp.quad_form(w, cov.values))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(w.value, index=mu.index)
