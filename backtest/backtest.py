import pandas as pd
import pickle

from backtest.engine import run_backtest
from backtest.metrics import performance_metrics
from backtest.plotting import plot_cumulative_returns

from allocation.equal_weight import equal_weight
from allocation.risk_parity import risk_parity_weights
from allocation.mean_variance import mean_variance_weights

USE_ML_PREDICTIONS = True

def main():

    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    if USE_ML_PREDICTIONS:
        expected_returns = pd.read_csv(
            "data/prices/predicted_returns_weekly.csv",
            index_col=0,
            parse_dates=True
        )
    else:
        expected_returns = pd.read_csv(
            "data/prices/expected_returns_weekly.csv",
            index_col=0,
            parse_dates=True
        )

    expected_returns = expected_returns.shift(1)

    with open("data/covariances.pkl", "rb") as f:
        covs = pickle.load(f)

    strategies = {
        "Equal Weight": lambda mu, cov: equal_weight(mu.index),
        "Risk Parity": lambda mu, cov: risk_parity_weights(cov),
        "Mean-Variance": lambda mu, cov: mean_variance_weights(mu, cov),
    }

    results = {}
    metrics = {}

    for name, strategy in strategies.items():
        strat_returns = run_backtest(
            returns=returns,
            expected_returns=expected_returns,
            covs=covs,
            weight_func=strategy
        )
        results[name] = strat_returns
        metrics[name] = performance_metrics(strat_returns)

    print("\nPerformance Summary:\n")
    for name, stats in metrics.items():
        print(name)
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}")
        print()

    plot_cumulative_returns(results)


if __name__ == "__main__":
    main()
