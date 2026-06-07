"""Stochastic event models — DNF, safety car, pit stop, overtake.

These get sampled inside the Monte Carlo simulator each lap. Failures here
(badly calibrated DNF rates, etc.) directly degrade winner probability calibration.
"""
