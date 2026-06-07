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

import numpy as np
import pandas as pd

from sports.f1.ml.common.types import TelemetrySample


def _samples_to_frame(samples: Iterable[TelemetrySample]) -> pd.DataFrame:
    """Materialize the iterable of TelemetrySample into a single DataFrame."""
    rows = []
    for s in samples:
        rows.append({
            "driver_code": s.driver_code,
            "timestamp": s.timestamp,
            "lap": s.lap,
            "s_coord": s.s_coord,
            "speed_kph": s.speed_kph,
        })
    if not rows:
        return pd.DataFrame(
            columns=["driver_code", "timestamp", "lap", "s_coord", "speed_kph"]
        )
    return pd.DataFrame(rows)


def build_reference_line(
    samples: Iterable[TelemetrySample],
    n_segments: int = 25,
) -> pd.DataFrame:
    """Build per-track reference racing line from the fastest observed lap.

    Returns a DataFrame with columns [segment_idx, s_start, s_end] partitioning
    [s_min, s_max] into `n_segments` equal-width windows along the projected
    arc-length coordinate.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    df = _samples_to_frame(samples)
    if df.empty:
        return pd.DataFrame(columns=["segment_idx", "s_start", "s_end"])

    # Identify the fastest lap by elapsed wall time across its samples; use it
    # as the reference (it has the cleanest racing line).
    elapsed = (
        df.groupby(["driver_code", "lap"])["timestamp"]
        .agg(lambda t: (t.max() - t.min()).total_seconds())
        .reset_index(name="lap_time")
    )
    elapsed = elapsed[elapsed["lap_time"] > 0]
    if elapsed.empty:
        # Single-tick laps; fall back to s-range from all samples.
        ref = df
    else:
        best = elapsed.sort_values("lap_time").iloc[0]
        ref = df[(df["driver_code"] == best["driver_code"]) & (df["lap"] == best["lap"])]

    s_min = float(ref["s_coord"].min())
    s_max = float(ref["s_coord"].max())
    if s_max <= s_min:
        s_max = s_min + 1.0  # degenerate: still produce a single segment

    edges = np.linspace(s_min, s_max, n_segments + 1)
    return pd.DataFrame({
        "segment_idx": np.arange(n_segments),
        "s_start": edges[:-1],
        "s_end": edges[1:],
    })


def decompose(
    samples: Iterable[TelemetrySample],
    reference_line: pd.DataFrame,
) -> pd.DataFrame:
    """Project each tick to its mini-sector index, aggregate per-sector duration.

    Output columns: driver_code, lap, mini_sector_idx, time_s, avg_speed_kph.
    `time_s` is the wall-clock time spent in the sector on that lap; `avg_speed_kph`
    is the simple mean of speed samples in the sector.
    """
    df = _samples_to_frame(samples)
    if df.empty or reference_line.empty:
        return pd.DataFrame(
            columns=["driver_code", "lap", "mini_sector_idx", "time_s", "avg_speed_kph"]
        )

    edges = reference_line["s_end"].to_numpy()
    # np.searchsorted maps each s_coord to its segment index.
    idx = np.searchsorted(edges, df["s_coord"].to_numpy(), side="right")
    df = df.assign(mini_sector_idx=np.clip(idx, 0, len(edges) - 1))

    grouped = df.groupby(["driver_code", "lap", "mini_sector_idx"], sort=True)
    out = grouped.agg(
        time_s=("timestamp", lambda t: (t.max() - t.min()).total_seconds()),
        avg_speed_kph=("speed_kph", "mean"),
    ).reset_index()
    return out
