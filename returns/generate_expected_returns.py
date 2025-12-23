import pandas as pd
from returns.expected_returns import rolling_mean_expected_returns


def main():
    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    print("Computing rolling expected returns...")
    expected_returns = rolling_mean_expected_returns(
        returns,
        window=52
    )

    expected_returns.to_csv("data/expected_returns_weekly.csv")
    print("Saved expected returns to data/expected_returns_weekly.csv")


if __name__ == "__main__":
    main()
