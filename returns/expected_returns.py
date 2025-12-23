import pandas as pd
from pathlib import Path

from returns.momentum import (
    rolling_mean_returns,
    simple_momentum_returns
)

DATA_PATH = Path("data/prices/returns_weekly.csv")


def load_weekly_returns() -> pd.DataFrame:
    returns = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )
    returns = returns.sort_index()
    return returns


def compute_expected_returns(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    method: str = "momentum",
    lookback: int = 12
) -> pd.Series:
    """
    Compute expected returns for time t+1 using data up to time t.

    Parameters
    ----------
    returns : pd.DataFrame -> Weekly returns (entire dataset)
    date : pd.Timestamp -> Evaluation date t
    method : str -> 'mean' or 'momentum'
    lookback : int -> Lookback window in weeks

    Returns
    -------
    pd.Series -> Expected return per asset
    """
    if date not in returns.index:
        raise ValueError("Date not in returns index")

    historical = returns.loc[:date]

    if method == "mean":
        return rolling_mean_returns(historical, lookback)

    elif method == "momentum":
        return simple_momentum_returns(historical, lookback)

    else:
        raise ValueError(f"Unknown method: {method}")

def rolling_mean_expected_returns(returns, window=52):
    return returns.rolling(window).mean()
