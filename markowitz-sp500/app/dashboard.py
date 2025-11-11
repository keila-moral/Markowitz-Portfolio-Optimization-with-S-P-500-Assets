"""Streamlit dashboard for interactive exploration"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from src import portfolio as pf
from src import optimization as opt
from src import plotting as plotting
from src import risk_metrics as rm

st.set_page_config(layout='wide', page_title='Markowitz Explorer')

st.title('Markowitz Portfolio Explorer — S&P 500 (Sector Leaders)')

# Sidebar controls
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2025-01-01'))
end_date = st.sidebar.date_input('End date', value=pd.to_datetime(date.today()))
lookback_years = st.sidebar.selectbox('Return lookback', [1,3,5], index=1)
risk_free = st.sidebar.number_input('Risk-free rate (annual, decimal)', value=0.0411, step=0.001)
allow_short = st.sidebar.checkbox('Allow shorting', value=False)
max_w = st.sidebar.slider('Max weight (if no shorting)', 0.1, 1.0, 0.6)
rebal_freq = st.sidebar.selectbox('Rebalancing frequency', ['Monthly','Quarterly'], index=0)
show_backtest = st.sidebar.checkbox('Run backtest', value=True)

# Load data
with st.spinner('Downloading price data...'):
    prices = pf.download_prices(pf.TICKERS, start=str(start_date), end=str(end_date))

returns = pf.log_returns(prices)
mean_returns = pf.expected_return(returns)
cov = pf.cov_matrix(returns)

# Compute efficient frontier
n_points = 40
bounds = None if allow_short else None
frontier = opt.efficient_frontier(mean_returns.values, cov.values, n_points=n_points, bounds=None, allow_short=allow_short)

# Find tangency via optimizer
weights_tan = opt.max_sharpe(mean_returns.values, cov.values, risk_free=risk_free, bounds=None, allow_short=allow_short)
ret_tan, vol_tan = opt.portfolio_metrics(weights_tan, mean_returns.values, cov.values)

# Build CML line
max_vol = max([p['vol'] for p in frontier]) * 1.2
cml_x = np.linspace(0, max_vol, 100)
cml_y = risk_free + (ret_tan - risk_free) / vol_tan * cml_x

# Display results
st.subheader('Tangency portfolio (max Sharpe)')
col1, col2 = st.columns([2,1])
with col1:
    #fig, ax = plt.subplots() 
    plotting.plot_efficient_frontier(frontier, tangency_point={'ret':ret_tan,'vol':vol_tan}, cml={'x':cml_x,'y':cml_y})
    st.pyplot()
with col2:
    tickers = prices.columns.tolist()
    df_w = pd.DataFrame({'ticker': tickers, 'weight': weights_tan})
    st.table(df_w.set_index('ticker'))
    st.write(f'Expected annual return: {ret_tan:.4f}')
    st.write(f'Annual vol: {vol_tan:.4f}')
    st.write(f'Sharpe (excess): {(ret_tan - risk_free)/vol_tan:.4f}')

# Backtest
if show_backtest:
    st.subheader('Backtest (simple monthly rebalance)')
    # create monthly rebalancing dates and weights (constant tangency)
    rebal_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='M')
    weights_df = pd.DataFrame([weights_tan]*len(rebal_dates), index=rebal_dates, columns=prices.columns)
    pv_returns, turnover = pf.rebalancing_backtest(prices, weights_df, tc=0.001) if hasattr(pf, 'rebalancing_backtest') else (None, None)
    # fallback to src.backtest
    from src.backtest import rebalancing_backtest
    port_rets, turnover = rebalancing_backtest(prices, weights_df, tc=0.001)
    st.line_chart((1+port_rets).cumprod())
    st.write('Average turnover per rebalance:', float(turnover.mean()))


# Footer
st.markdown('---')
st.markdown('Repo: Markowitz Portfolio Optimization — includes model validation notes and notebooks.')
