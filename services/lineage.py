from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def dataset_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


class DatasetLineage:
    """DVC-style content-addressed lineage without requiring a DVC server."""
    def __init__(self, root: str = "lineage"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lineage.jsonl"

    def record(self, dataset_hash: str, parent_hash: str | None, operation: str, metadata: dict[str, Any] | None = None):
        rec = {"dataset_hash": dataset_hash, "parent_hash": parent_hash, "operation": operation,
               "metadata": metadata or {}}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def snapshot(self, df: pd.DataFrame, parent_hash: str | None = None, operation: str = "snapshot", metadata=None):
        h = dataset_fingerprint(df)
        return self.record(h, parent_hash, operation, {"rows": len(df), "columns": list(df.columns), **(metadata or {})})
