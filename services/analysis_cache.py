"""Module duty: Analysis cache.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any, Callable
import hashlib, json
import pandas as pd


class AnalysisCache:
    """Small process-local deterministic cache for repeated pure analytical operations."""
    def __init__(self, max_items: int = 32):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.max_items = max_items
        self._data: dict[str, Any] = {}

    @staticmethod
    def key(name: str, df: pd.DataFrame | None = None, params: dict[str, Any] | None = None) -> str:
        """Perform the key operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        payload = json.dumps(params or {}, sort_keys=True, default=str).encode()
        if df is not None:
            payload += pd.util.hash_pandas_object(df, index=True).values.tobytes()
            payload += "|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode()
        return hashlib.sha256(name.encode() + payload).hexdigest()

    def get_or_compute(self, name: str, fn: Callable[[], Any], df: pd.DataFrame | None = None, params: dict[str, Any] | None = None):
        """Perform the get or compute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        k = self.key(name, df, params)
        if k in self._data:
            return self._data[k], True
        value = fn()
        if len(self._data) >= self.max_items:
            self._data.pop(next(iter(self._data)))
        self._data[k] = value
        return value, False
