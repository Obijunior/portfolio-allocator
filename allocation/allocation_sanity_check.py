import pandas as pd
import numpy as np

from allocation.equal_weight import equal_weight
from allocation.risk_parity import risk_parity_weights
from allocation.mean_variance import mean_variance_weights


def main():
    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    mu = returns.rolling(52).mean().iloc[-1]

    from risk.covariance import rolling_ledoit_wolf_covariance
    covs = rolling_ledoit_wolf_covariance(returns)
    cov = list(covs.values())[-1]

    ew = equal_weight(mu.index)
    rp = risk_parity_weights(cov)
    mv = mean_variance_weights(mu, cov)

    print("\nEqual Weight:\n", ew.round(3))
    print("\nRisk Parity:\n", rp.round(3))
    print("\nMean-Variance:\n", mv.round(3))

    print("\nSum checks:")
    print(ew.sum(), rp.sum(), mv.sum())


if __name__ == "__main__":
    main()
