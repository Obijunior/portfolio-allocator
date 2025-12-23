import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ"]
LAGS = 4  # number of lag weeks for features

def create_features(df, lags=LAGS):
    """Generate lagged returns and rolling statistics as features."""
    X_list = []
    for lag in range(1, lags + 1):
        X_list.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    # Optional: rolling mean and volatility
    X_list.append(df.rolling(window=lags).mean().add_suffix("_mean"))
    X_list.append(df.rolling(window=lags).std().add_suffix("_vol"))
    X = pd.concat(X_list, axis=1)
    X = X.dropna()
    return X

def main():
    returns = pd.read_csv("data/prices/returns_weekly.csv", index_col=0, parse_dates=True)
    predicted_returns = pd.DataFrame(index=returns.index, columns=ASSETS)

    for asset in ASSETS:
        print(f"Training model for {asset}...")
        y = returns[asset].shift(-1)  # target: next week's return
        X = create_features(returns[[asset]])
        y = y.loc[X.index]  # align

        # Time series split for validation (optional)
        tscv = TimeSeriesSplit(n_splits=5)

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        predicted_returns.loc[y_pred.index, asset] = y_pred

    # Save predictions
    predicted_returns.to_csv("data/prices/predicted_returns_weekly.csv")
    print("Saved ML predicted returns to data/prices/predicted_returns_weekly.csv")

if __name__ == "__main__":
    main()
