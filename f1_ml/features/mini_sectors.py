"""Mini-sector decomposition — project GPS samples onto the reference racing line.

Splits each lap into ~20-30 sub-sectors, computes per-sub-sector time. Reveals
where each driver is fast/slow (low-speed corners vs. straights vs. high-speed
corners), which exposes setup choices (downforce vs. drag) and driver skill
that whole-lap times mask.

Inputs: stream of TelemetrySample (with s_coord pre-projected).
Outputs: pandas DataFrame [driver, lap, mini_sector_idx, time_s, avg_speed].
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from f1_ml.common.types import TelemetrySample


def build_reference_line(samples: Iterable[TelemetrySample], n_segments: int = 25) -> pd.DataFrame:
    """Build per-track reference racing line from the fastest observed lap.
    Returns segment boundaries in s_coord."""
    raise NotImplementedError("group by lap, find min lap_time lap, evenly slice s_coord by quantile")


def decompose(
    samples: Iterable[TelemetrySample],
    reference_line: pd.DataFrame,
) -> pd.DataFrame:
    """Project each tick to its mini-sector index, aggregate per-sector duration."""
    raise NotImplementedError
