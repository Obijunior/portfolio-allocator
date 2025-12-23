import numpy as np
import pandas as pd


def performance_metrics(returns):
    ann_return = (1 + returns).prod() ** (52 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(52)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }
