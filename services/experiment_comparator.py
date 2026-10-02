"""Module duty: Experiment comparator.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any
import numpy as np


class ExperimentComparator:
    """Compare experiments only on explicitly declared compatible metrics/protocols."""
    @staticmethod
    def compare(records: list[dict[str, Any]], metric: str | None = None) -> dict[str, Any]:
        """Perform the compare operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not records:
            return {"status": "insufficient_data", "rows": []}
        rows = []
        for r in records:
            metrics = r.get("metrics") or r.get("best_metrics") or {}
            value = metrics.get(metric) if metric else None
            rows.append({"experiment": r.get("experiment") or r.get("model") or r.get("agent"), "agent": r.get("agent"), "model": r.get("model") or r.get("best_model"), "task": r.get("task"), "metric": metric, "value": value, "dataset_fingerprint": r.get("dataset_fingerprint"), "validation_strategy": r.get("validation_strategy")})
        compatible = len({(x["task"], x["dataset_fingerprint"], x["validation_strategy"]) for x in rows}) <= 1
        return {"status": "ok", "comparable": compatible, "rows": rows, "interpretation": "Experiments with different datasets, targets, or validation protocols should not be ranked as if they were equivalent."}
