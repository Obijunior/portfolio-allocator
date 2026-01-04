import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import sys
from pathlib import Path

# path to project root folder, get modules from there like get_data
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# import assets from get_data.py, no longer locked
from get_data import ASSETS

LAGS = 4  # number of lag weeks for features
MIN_TRAIN = 52  # minimum training observations for expanding window


def create_features(df, lags=LAGS):
    """Generate lagged returns and rolling statistics as features."""
    X_list = []

    for lag in range(1, lags + 1):
        X_list.append(df.shift(lag).add_suffix(f"_lag{lag}"))

    X_list.append(df.rolling(window=lags).mean().add_suffix("_mean"))
    X_list.append(df.rolling(window=lags).std().add_suffix("_vol"))

    X = pd.concat(X_list, axis=1)
    return X


def main():
    returns = pd.read_csv(
        "data/prices/returns_weekly.csv",
        index_col=0,
        parse_dates=True
    )

    predicted_returns = pd.DataFrame(index=returns.index, columns=ASSETS, dtype=float)

    for asset in ASSETS:
        print(f"Training model for {asset}...")

        # Target: next week's return
        y = returns[asset].shift(-1)

        # Features
        X = create_features(returns[[asset]])

        # Combine features and target
        data = pd.concat([X, y.rename("target")], axis=1)

        preds = pd.Series(index=data.index, dtype=float)

        for date in data.index:
            row = data.loc[date]
            if row.isna().any():
                continue

            train = data.loc[:date].iloc[:-1].dropna()
            if len(train) < MIN_TRAIN:
                continue

            X_train = train.drop(columns="target")
            y_train = train["target"]

            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train, y_train)

            X_pred = row.drop("target").to_frame().T
            preds.loc[date] = model.predict(X_pred)[0]

        predicted_returns.loc[preds.index, asset] = preds

    predicted_returns.to_csv("data/prices/predicted_returns_weekly.csv")
    print("Saved ML predicted returns to data/prices/predicted_returns_weekly.csv")


if __name__ == "__main__":
    main()
