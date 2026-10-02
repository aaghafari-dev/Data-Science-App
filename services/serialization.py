"""Module duty: Serialization.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

"""Bounded JSON/msgpack-safe serialization for agent state and evidence."""

import json
from typing import Any


def json_safe(value: Any, *, max_rows: int = 20, max_items: int = 200) -> Any:
    """Convert scientific Python objects into bounded primitive structures.

    DataFrames are summarized rather than serialized in full.  Model/figure objects
    are intentionally omitted because LangGraph checkpointing and msgpack do not
    support them reliably and they are not required as narrative evidence.
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        pd = None
        np = None

    if pd is not None and isinstance(value, pd.DataFrame):
        preview = value.head(max_rows).copy()
        return {
            "__type__": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(c) for c in value.columns],
            "dtypes": {str(c): str(value[c].dtype) for c in value.columns},
            "preview": json_safe(preview.to_dict(orient="records"), max_rows=max_rows, max_items=max_items),
        }
    if pd is not None and isinstance(value, pd.Series):
        return json_safe(value.head(max_rows).tolist(), max_rows=max_rows, max_items=max_items)
    if np is not None and isinstance(value, np.ndarray):
        return json_safe(value.tolist(), max_rows=max_rows, max_items=max_items)
    if np is not None and isinstance(value, (np.integer, np.floating, np.bool_)):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                out["__truncated__"] = True
                break
            if str(k) in {"model_object", "estimator", "figure", "pipeline", "_estimators"}:
                continue
            out[str(k)] = json_safe(v, max_rows=max_rows, max_items=max_items)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        result = [json_safe(v, max_rows=max_rows, max_items=max_items) for v in seq[:max_items]]
        if len(seq) > max_items:
            result.append({"__truncated__": True, "original_length": len(seq)})
        return result
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
