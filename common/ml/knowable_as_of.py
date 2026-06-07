"""Look-ahead defense — every feature carries a `knowable_as_of` timestamp.

The backtest harness rejects any prediction that consumes a feature whose
knowable_as_of is later than the decision time. This is the single most important
defense against subtle look-ahead bias in time-series ML.

Pattern: every feature-extraction function returns a (DataFrame, timestamp) pair
or annotates each row with a knowable_as_of column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class TimestampedFeatures:
    data: pd.DataFrame
    knowable_as_of: datetime  # earliest moment all rows became available


def annotate(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """Attach a uniform knowable_as_of to a feature frame."""
    df = df.copy()
    df["knowable_as_of"] = ts
    return df


def assert_no_lookahead(features: pd.DataFrame, decision_time: datetime) -> None:
    """Raise if any feature row claims to be knowable after the decision time."""
    if "knowable_as_of" not in features.columns:
        raise ValueError("feature frame missing knowable_as_of column")
    bad = features[features["knowable_as_of"] > decision_time]
    if not bad.empty:
        raise LookaheadError(
            f"{len(bad)} feature rows leak future info "
            f"(latest={bad['knowable_as_of'].max()}, decision={decision_time})"
        )


class LookaheadError(RuntimeError):
    pass
