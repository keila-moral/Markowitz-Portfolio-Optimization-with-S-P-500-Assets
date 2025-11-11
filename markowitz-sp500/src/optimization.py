# File: src/optimization.py
"""Optimization helpers: efficient frontier, min-variance, max-Sharpe"""
import numpy as np
from scipy.optimize import minimize


def portfolio_metrics(weights, mean_returns, cov_matrix):
    weights = np.asarray(weights)
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    return ret, vol


def _check_bounds(n, allow_short=False, max_w=1.0):
    if allow_short:
        return [(-1.0, 1.0) for _ in range(n)]
    return [(0.0, max_w) for _ in range(n)]


def min_variance(mean_returns, cov_matrix, bounds=None, allow_short=False):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    if bounds is None:
        bounds = _check_bounds(n, allow_short=allow_short)
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    fun = lambda w: float(w.T @ cov_matrix @ w)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x


def max_sharpe(mean_returns, cov_matrix, risk_free=0.0411, bounds=None, allow_short=False):
    # We are using 4.11% as the risk-free rate for the Sharpe ratio which is the 10-year U.S. Treasury yield value in 2025
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    if bounds is None:
        bounds = _check_bounds(n, allow_short=allow_short)
    def neg_sharpe(w):
        r, v = portfolio_metrics(w, mean_returns, cov_matrix)
        # penalize invalid vols
        if v <= 0:
            return 1e6
        return - (r - risk_free) / v
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x


def efficient_frontier(mean_returns, cov_matrix, n_points=50, bounds=None, allow_short=False):
    # generate target returns between min and max achievable returns
    n = len(mean_returns)
    if bounds is None:
        bounds = _check_bounds(n, allow_short=allow_short)
    ret_min = float(np.min(mean_returns))
    ret_max = float(np.max(mean_returns))
    target_returns = np.linspace(ret_min, ret_max, n_points)
    frontier = []
    cons_base = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
    for rt in target_returns:
        cons = cons_base + ({'type':'eq','fun': lambda x, rt=rt: float(np.dot(x, mean_returns)) - rt},)
        x0 = np.repeat(1/n, n)
        res = minimize(lambda w: float(w.T @ cov_matrix @ w), x0, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            r, v = portfolio_metrics(res.x, mean_returns, cov_matrix)
            frontier.append({'ret': r, 'vol': v, 'weights': res.x})
    return frontier





