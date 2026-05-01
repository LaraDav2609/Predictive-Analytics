"""Core probability engine — gradient-boosted heads with calibration wrappers.

These are the production workhorses for v1: LightGBM on engineered features,
wrapped in conformal / Platt / deep ensembles for calibrated uncertainty.
"""
