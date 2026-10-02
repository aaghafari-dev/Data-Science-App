"""Module duty: Data contracts.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import json
import pandas as pd
from .semantic_types import classify_dtype


def _now():
    """Perform the now operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return datetime.now(timezone.utc).isoformat()


class DataContractEngine:
    """Create, validate and compare explicit dataset contracts."""

    @staticmethod
    def build(df: pd.DataFrame, name: str = "Dataset Contract") -> dict[str, Any]:
        """Perform the build operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {
            "name": name,
            "created_at": _now(),
            "columns": {
                str(c): {
                    "dtype": str(df[c].dtype),
                    "core_type": classify_dtype(df[c]),
                    "nullable": bool(df[c].isna().any()),
                    "min": float(df[c].min()) if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any() else None,
                    "max": float(df[c].max()) if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any() else None,
                    "unique_max": int(df[c].nunique(dropna=True)),
                } for c in df.columns
            },
        }

    @staticmethod
    def validate(df: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
        """Perform the validate operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        expected = contract.get("columns", {})
        actual = {str(c): c for c in df.columns}
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        dtype_changes = []
        range_violations = []
        for name, spec in expected.items():
            if name not in actual:
                continue
            c = actual[name]
            observed = str(df[c].dtype)
            if observed != spec.get("dtype"):
                dtype_changes.append({"column": name, "expected": spec.get("dtype"), "observed": observed})
            if pd.api.types.is_numeric_dtype(df[c]):
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None and (df[c] < lo).any(): range_violations.append({"column": name, "type": "below_min", "expected": lo})
                if hi is not None and (df[c] > hi).any(): range_violations.append({"column": name, "type": "above_max", "expected": hi})
        drift = bool(missing or added or dtype_changes or range_violations)
        return {
            "status": "drift" if drift else "pass",
            "missing_columns": missing,
            "added_columns": added,
            "dtype_changes": dtype_changes,
            "range_violations": range_violations,
            "checked_at": _now(),
        }

    @staticmethod
    def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
        """Perform the drift report operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        contract = DataContractEngine.build(reference, "Reference Contract")
        result = DataContractEngine.validate(current, contract)
        result["reference_rows"] = int(len(reference)); result["current_rows"] = int(len(current))
        result["row_delta"] = int(len(current) - len(reference))
        return result

    @staticmethod
    def save(path, contract):
        """Perform the save operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        from pathlib import Path
        Path(path).write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path):
        """Perform the load operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        from pathlib import Path
        return json.loads(Path(path).read_text(encoding="utf-8"))
