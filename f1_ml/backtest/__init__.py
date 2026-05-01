"""Backtest harness — walk-forward training, in-race replay, look-ahead audit.

The backtest is the source of truth for "is this model worth deploying."
Calibration metrics + simulated P&L vs. historical market quotes.
"""
