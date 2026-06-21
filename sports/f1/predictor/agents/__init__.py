"""F1 scoped analyst agents.

These agents convert existing predictor feature payloads into human-readable,
source-labeled diagnostics. They do not mutate model weights or prediction
outputs.
"""

from .car_performance import build_car_performance_agent

__all__ = ["build_car_performance_agent"]
