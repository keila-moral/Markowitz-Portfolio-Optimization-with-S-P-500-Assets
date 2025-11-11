"""Simple monthly rebalancing backtest with transaction costs"""
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def rebalancing_backtest(prices, weights_series, start=None, end=None, tc=0.0):
    """prices: DataFrame of adj close indexed by date
       weights_series: DataFrame indexed by date with target weights (rebal dates)
       tc: proportional transaction cost (e.g., 0.001 = 0.1%)
    Returns: portfolio returns series and turnover stats
    """
    prices = prices.sort_index()
    rebal_dates = weights_series.index
    # forward fill prices to business days
    returns = prices.pct_change().fillna(0)
    portfolio_value = 1.0
    positions = None
    pv_list = []
    turnover = []
    for i in range(len(returns)):
        date = returns.index[i]
        if date in rebal_dates:
            target = weights_series.loc[date].values
            if positions is None:
                # initial buy
                cost = tc * np.sum(np.abs(target))
                portfolio_value *= (1 - cost)
                positions = target * portfolio_value
                turnover.append(np.sum(np.abs(target)))
            else:
                # compute turnover from current weights
                current_weights = positions * prices.loc[date] / portfolio_value
                trade = target - current_weights
                cost = tc * np.sum(np.abs(trade))
                portfolio_value *= (1 - cost)
                positions = target * portfolio_value
                turnover.append(np.sum(np.abs(trade)))
        # apply daily P&L
        if positions is None:
            pv_list.append(portfolio_value)
            continue
        daily_ret = np.dot((positions * returns.iloc[i]).values / portfolio_value, np.ones(len(positions)))
        portfolio_value *= (1 + daily_ret)
        pv_list.append(portfolio_value)
    pv = pd.Series(pv_list, index=returns.index)
    return pv.pct_change().fillna(0), pd.Series(turnover, index=rebal_dates)
