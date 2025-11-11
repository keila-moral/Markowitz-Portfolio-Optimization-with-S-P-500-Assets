"""Plotting helpers (matplotlib + plotly optional)"""
import matplotlib.pyplot as plt
import numpy as np


def plot_efficient_frontier(frontier, tangency_point=None, cml=None):
    vols = [p['vol'] for p in frontier]
    rets = [p['ret'] for p in frontier]
    plt.figure(figsize=(8,5))
    plt.plot(vols, rets, label='Efficient frontier')
    if tangency_point is not None:
        plt.scatter([tangency_point['vol']], [tangency_point['ret']], c='red', label='Tangency')
    if cml is not None:
        # cml: dict with 'x' and 'y' arrays
        plt.plot(cml['x'], cml['y'], '--', label='CML')
    plt.xlabel('Risk (vol)')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt
