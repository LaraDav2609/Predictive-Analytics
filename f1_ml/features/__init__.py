"""Feature extraction layer — turns raw telemetry into model-ready signals.

All features carry a `knowable_as_of` timestamp (see knowable_as_of.py) so the
backtest harness can reject look-ahead violations.
"""
