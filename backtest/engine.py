import pandas as pd
import numpy as np

def run_backtest(
    returns,
    expected_returns,
    covs,
    weight_func,
    transaction_cost=0.001
):
    portfolio_returns = []
    dates = list(covs.keys())
    prev_weights = None
    for date, next_date in zip(dates, dates[1:]):
        mu = expected_returns.loc[date]
        cov = covs[date]

        weights = weight_func(mu, cov)

        realized = returns.loc[next_date]

        gross = (weights * realized).sum()
        cost = 0 if prev_weights is None else transaction_cost * (weights - prev_weights).abs().sum()

        portfolio_returns.append(gross - cost)
        prev_weights = weights

    return pd.Series(portfolio_returns, index=dates[1:])
