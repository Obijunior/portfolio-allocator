import pandas as pd
from risk.covariance import rolling_ledoit_wolf_covariance, save_covariances


def main():
    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    print("Computing rolling Ledoit–Wolf covariances...")
    covs = rolling_ledoit_wolf_covariance(returns, window=52)

    save_covariances(covs)
    print("Saved covariances to data/covariances.pkl")


if __name__ == "__main__":
    main()
