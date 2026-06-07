"""Look-ahead audit — the safety net that catches accidental future leakage.

Single most-important defense against subtle backtest bugs. For every prediction
the backtester emits, it inspects the feature frame's `knowable_as_of` column
(annotated by features/knowable_as_of.py); any row with a timestamp later than
the decision time aborts the backtest with a LookaheadError.

A passing backtest with this audit on means: no feature used to make this
prediction was knowable after the prediction time. Without it, your beautiful
PnL chart is probably leaking.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from sports.f1.ml.features.knowable_as_of import LookaheadError, assert_no_lookahead


def audit_features(features: pd.DataFrame, decision_time: datetime) -> None:
    """Wrapper around assert_no_lookahead — re-exported here so the backtest
    code path imports from one obvious location."""
    assert_no_lookahead(features, decision_time)


def audit_artifact_manifest(manifest_training_cutoff: datetime, decision_time: datetime) -> None:
    """A model's training cutoff must precede the decision time."""
    if manifest_training_cutoff > decision_time:
        raise LookaheadError(
            f"model trained on data up to {manifest_training_cutoff}, "
            f"used to predict at {decision_time}"
        )
