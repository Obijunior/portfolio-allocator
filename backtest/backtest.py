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

    expected_stat = pd.read_csv(
        "data/prices/expected_returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    expected_ml = pd.read_csv(
        "data/prices/predicted_returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    # prevent lookahead bias for BOTH
    expected_stat = expected_stat.shift(1)
    expected_ml = expected_ml.shift(1)

    def rank_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert per-date predictions into a cross-sectional ranked z-score signal.
        For each date: rank assets, then z-score ranks.
        """
        ranks = df.rank(axis=1, method="average", ascending=True)
        z = (ranks.sub(ranks.mean(axis=1), axis=0)).div(ranks.std(axis=1), axis=0)
        return z.fillna(0.0)

    expected_ml_ranked = rank_zscore(expected_ml)

    # Choose μ mode:
    MU_MODE = "blend"   # "stat", "ml_rank", "blend"
    ALPHA_STAT = 0.7    # used only if MU_MODE == "blend"

    if MU_MODE == "stat":
        expected_returns = expected_stat
    elif MU_MODE == "ml_rank":
        expected_returns = expected_ml_ranked
    elif MU_MODE == "blend":
        expected_returns = ALPHA_STAT * expected_stat + (1 - ALPHA_STAT) * expected_ml_ranked
    else:
        raise ValueError(f"Unknown MU_MODE: {MU_MODE}")

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
