import pandas as pd

returns = pd.read_csv(
    "data/prices/returns_weekly.csv",
    index_col=0,
    parse_dates=True
)

print("\nMean weekly returns:")
print(returns.mean())

print("\nWeekly volatility:")
print(returns.std())

print("\nCorrelation matrix:")
print(returns.corr())
