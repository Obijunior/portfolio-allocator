import pandas as pd

def equal_weight(assets):
    n = len(assets)
    return pd.Series(1.0 / n, index=assets)