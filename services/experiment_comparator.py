from __future__ import annotations

from typing import Any
import numpy as np


class ExperimentComparator:
    """Compare experiments only on explicitly declared compatible metrics/protocols."""
    @staticmethod
    def compare(records: list[dict[str, Any]], metric: str | None = None) -> dict[str, Any]:
        if not records:
            return {"status": "insufficient_data", "rows": []}
        rows = []
        for r in records:
            metrics = r.get("metrics") or r.get("best_metrics") or {}
            value = metrics.get(metric) if metric else None
            rows.append({"experiment": r.get("experiment") or r.get("model") or r.get("agent"), "agent": r.get("agent"), "model": r.get("model") or r.get("best_model"), "task": r.get("task"), "metric": metric, "value": value, "dataset_fingerprint": r.get("dataset_fingerprint"), "validation_strategy": r.get("validation_strategy")})
        compatible = len({(x["task"], x["dataset_fingerprint"], x["validation_strategy"]) for x in rows}) <= 1
        return {"status": "ok", "comparable": compatible, "rows": rows, "interpretation": "Experiments with different datasets, targets, or validation protocols should not be ranked as if they were equivalent."}
