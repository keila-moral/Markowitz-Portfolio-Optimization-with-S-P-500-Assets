"""Sharpe, drawdown, MCTR/PCTR, and helper functions"""
import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free=0.0411, annualize=252):
    excess = returns - risk_free/annualize
    return (excess.mean() / excess.std()) * np.sqrt(annualize)


def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def mctr_pctr(weights, cov_matrix):
    # Marginal contribution to risk and percent contribution
    w = np.asarray(weights)
    sigma = np.sqrt(w.T @ cov_matrix @ w)
    mctr = (cov_matrix @ w) / sigma
    pctr = (w * mctr) / sigma
    return mctr, pctr




