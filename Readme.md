# Markowitz Portfolio Optimization with S&P 500 Assets

## Project overview

This repository implements a professional-grade Mean Variance (Markowitz) portfolio optimization pipeline using a small, diversified set of S&P 500 tickers. The objective is to find efficient frontiers, construct the tangency (maximum Sharpe) portfolio, compute the Capital Market Line (CML), and provide model-validation style analysis (stress testing, backtesting, risk decomposition).

## What you'll find in this repo (high level)

* `data/`: sample CSVs and small cached datasets (optional)
* `notebooks/`: Jupyter notebooks demonstrating analysis and plots (`efficient_frontier.ipynb`, `backtest.ipynb`)
* `src/`: production-ready Python modules

  * `portfolio.py`: data loading, returns, covariances
  * `optimization.py`: mean-variance optimization, efficient frontier, tangency portfolio
  * `risk_metrics.py`: Sharpe, drawdown, MCTR / PCTR
  * `backtest.py`: rebalancing backtest harness with transaction costs
  * `plotting.py`: static + interactive plotting helpers (matplotlib + plotly wrappers)
* `app/`: optional Streamlit or Dash app for interactive exploration
* `docs/Model_Validation_Report.md`: short model validation write-up (assumptions, limitations, sensitivity analysis)
* `requirements.txt`: pinned environment

---

## Quick start (local)

1. Clone repo

```bash
git clone https://github.com/your-username/markowitz-sp500.git
cd markowitz-sp500
python -m venv .venv
source .venv/bin/activate  # mac / linux
pip install -r requirements.txt
```

2. Run the notebook(s)

```bash
jupyter lab  # or jupyter notebook
```

3. (Optional) Run streamlit app

```bash
streamlit run app/dashboard.py
```

---

## Chosen asset universe 

We use 8 tickers to keep the model compact and explainable:

* `AAPL`, `MSFT`, `NVDA`, `XOM`, `JPM`, `JNJ`, `PG`, `UNH`

---

## Features & deliverables

1. **Data ingestion**: `yfinance` downloader with caching to CSV
2. **Return estimation**: daily log-returns, configurable lookback window (1Y / 3Y / 5Y)
3. **Covariance estimation**: sample covariance + exponentially weighted covariance (EWMA)
4. **Optimization**:

   * Efficient frontier for N assets
   * Tangency portfolio (max Sharpe) under different constraints
   * Solver variants: `scipy.optimize.minimize` (SLSQP) and `cvxpy` (if available)
5. **Backtest**: monthly rebalancing with transaction costs, turnover constraints
6. **Stress tests**: compare behavior during 2008, 2020 and custom shocks
7. **Risk decomposition**: MCTR, PCTR
8. **Reporting**: automated PDF/Markdown report summarizing the model, performance, and validation notes
9. **Interactive dashboard (optional)**: sliders to change lookback, risk-free rate, allow/disallow shorting

---

## Code: starter snippets (in `src/`)

### `portfolio.py` (data + basic metrics)

```python
import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["AAPL","MSFT","NVDA","XOM","JPM","JNJ","PG","UNH"]

def download_prices(tickers=TICKERS, start="2018-01-01", end=None, cache_path=None):
    df = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    if cache_path:
        df.to_csv(cache_path)
    return df

def log_returns(price_df):
    return np.log(price_df).diff().dropna()

def expected_return(returns, annualize=252):
    return returns.mean() * annualize

def cov_matrix(returns, annualize=252):
    return returns.cov() * annualize
```

### `optimization.py` (efficient frontier + tangency)

```python
import numpy as np
from scipy.optimize import minimize


def portfolio_metrics(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return ret, vol


def min_variance(mean_returns, cov_matrix, bounds=None, constraints=()):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    if constraints:
        cons = cons + constraints
    res = minimize(lambda w: w.T @ cov_matrix @ w, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x


def max_sharpe(mean_returns, cov_matrix, risk_free=0.0, bounds=None):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    def neg_sharpe(w):
        r, v = portfolio_metrics(w, mean_returns, cov_matrix)
        return -(r - risk_free) / v
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x
```

---

## Model validation write-up (docs/Model_Validation_Report.md)

This file will describe limitations and sanity checks, including:

* Estimation error in mean returns and covariance (small-sample noise)
* Sensitivity to lookback window
* Robustness to outliers (Winsorization, trimming)
* Transaction costs and turnover impact
* Backtest caveats (forward-looking bias, data snooping)

---

## Suggested roadmap 

1. Complete `src/` module files with tests
2. Build `notebooks/efficient_frontier.ipynb` based on my Coursera visuals (with CML and tangent point highlighted)
3. Build `backtest.py` and `notebooks/backtest.ipynb`
4. Create `docs/Model_Validation_Report.md` and a short `slides/` presentation
5. (Optional) Streamlit dashboard in `app/`

---

## Requirements (starter)

```
python>=3.9
pandas
numpy
scipy
matplotlib
seaborn
yfinance
plotly
jupyter
streamlit
```

---


