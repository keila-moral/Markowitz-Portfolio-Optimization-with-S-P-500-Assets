# File: src/portfolio.py
"""Data loading and basic portfolio metrics"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

TICKERS = ["AAPL","MSFT","NVDA","XOM","JPM","JNJ","PG","UNH"]


def download_prices(tickers=TICKERS, start="2025-01-01", end="2025-11-10"):
    df = yf.download(tickers, start=start, end=end, progress=False)

    # MultiIndex DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        # Use 'Close' instead of 'Adj Close'
        df = df.xs('Close', axis=1, level=0)  # level 0 is Price type
        df.columns.name = None  # optional: flatten columns

    # Single ticker
    elif 'Close' in df.columns:
        df = df['Close'].to_frame(name=tickers[0])

    else:
        raise KeyError("No 'Close' column found in the downloaded data")

    return df


def load_cached(csv_path):
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def log_returns(price_df):
    return np.log(price_df).diff().dropna()


def expected_return(returns, annualize=252):
    return returns.mean() * annualize


def cov_matrix(returns, annualize=252):
    return returns.cov() * annualize


def ewma_cov(returns, span=60, annualize=252):
    # Exponentially weighted covariance: approximate by computing EWMA of demeaned returns outer products
    rets = returns - returns.mean()
    weights = np.exp(-np.arange(len(rets))[::-1] / span)
    weights = weights / weights.sum()
    cov = np.zeros((returns.shape[1], returns.shape[1]))
    cols = returns.columns
    for t in range(len(rets)):
        r = rets.iloc[t].values.reshape(-1,1)
        cov += weights[t] * (r @ r.T)
    return pd.DataFrame(cov * annualize, index=cols, columns=cols)

