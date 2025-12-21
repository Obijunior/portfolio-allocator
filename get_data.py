import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# -----------------------
# Configuration (LOCKED)
# -----------------------
ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ"]
START_DATE = "2005-01-01"
DATA_DIR = Path("data/prices")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# STEP 1: Download data
# -----------------------
print("Downloading adjusted close prices...")

prices = yf.download(
    tickers=ASSETS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
    threads=False  # IMPORTANT: avoids Windows cache lock
)["Close"]

# Sanity check: ensure all tickers downloaded
missing = set(ASSETS) - set(prices.columns)

if missing:
    raise RuntimeError(f"Missing data for tickers: {missing}")

prices.to_csv(DATA_DIR / "raw_prices.csv")
print("Saved raw_prices.csv")

# -----------------------
# STEP 2: Clean prices
# -----------------------
print("Cleaning prices...")

# Drop rows with any missing asset
prices_clean = prices.dropna(how="any").sort_index()

if prices_clean.empty:
    raise RuntimeError("prices_clean is empty after dropna — check raw data.")

prices_clean.to_csv(DATA_DIR / "prices_clean.csv")
print("Saved prices_clean.csv")

# -----------------------
# STEP 3: Daily log returns
# -----------------------
print("Computing daily log returns...")

returns_daily = np.log(prices_clean / prices_clean.shift(1)).dropna()

returns_daily.to_csv(DATA_DIR / "returns_daily.csv")
print("Saved returns_daily.csv")

# -----------------------
# STEP 4: Weekly returns
# -----------------------
print("Computing weekly log returns...")

returns_weekly = (
    returns_daily
    .resample("W-FRI")
    .sum()
    .dropna()
)

returns_weekly.to_csv(DATA_DIR / "returns_weekly.csv")
print("Saved returns_weekly.csv")

print("[ SUCCESS ] :: Phase 2 complete.")
