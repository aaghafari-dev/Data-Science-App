"""Module duty: Lineage.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Perform the dataset fingerprint operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


class DatasetLineage:
    """DVC-style content-addressed lineage without requiring a DVC server."""
    def __init__(self, root: str = "lineage"):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lineage.jsonl"

    def record(self, dataset_hash: str, parent_hash: str | None, operation: str, metadata: dict[str, Any] | None = None):
        """Perform the record operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        rec = {"dataset_hash": dataset_hash, "parent_hash": parent_hash, "operation": operation,
               "metadata": metadata or {}}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def snapshot(self, df: pd.DataFrame, parent_hash: str | None = None, operation: str = "snapshot", metadata=None):
        """Perform the snapshot operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        h = dataset_fingerprint(df)
        return self.record(h, parent_hash, operation, {"rows": len(df), "columns": list(df.columns), **(metadata or {})})
