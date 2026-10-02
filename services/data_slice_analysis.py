"""Module duty: Data slice analysis.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score


def _metrics(y, pred, task):
    """Perform the metrics operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if task == "classification":
        return {"accuracy": float(accuracy_score(y, pred)), "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0))}
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}


class DataSliceAnalyzer:
    """Find materially different performance slices with sample-size safeguards."""

    @staticmethod
    def run(X: pd.DataFrame, y, predictions, task: str, min_rows: int = 25, max_slices: int = 12) -> dict[str, Any]:
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if X is None or len(X) == 0:
            return {"status": "insufficient_data", "reason": "No evaluation rows."}
        base = _metrics(y, predictions, task)
        primary = "f1_weighted" if task == "classification" else "rmse"
        higher_is_better = task == "classification"
        candidates = []
        X_local = X.reset_index(drop=True)
        y_local = np.asarray(y)
        pred_local = np.asarray(predictions)
        for col in X_local.columns:
            s = X_local[col]
            if s.nunique(dropna=True) < 2 or s.nunique(dropna=True) > 30:
                continue
            groups = s.fillna("<missing>")
            for value, idx in groups.groupby(groups).groups.items():
                if len(idx) < min_rows or len(idx) > .9 * len(X):
                    continue
                pos = list(idx)
                m = _metrics(y_local[pos], pred_local[pos], task)
                delta = m[primary] - base[primary]
                if not higher_is_better:
                    delta = -delta
                candidates.append({"feature": str(col), "value": str(value), "n": int(len(idx)), "metrics": m, "primary_delta_vs_overall": float(delta)})
        candidates.sort(key=lambda r: abs(r["primary_delta_vs_overall"]), reverse=True)
        return {"status": "ok", "overall": base, "primary_metric": primary, "slices": candidates[:max_slices], "interpretation": "Slice findings identify subpopulations for investigation; they do not establish causality or group-level root cause."}
