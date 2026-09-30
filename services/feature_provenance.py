from __future__ import annotations

from typing import Any
import re
import pandas as pd


class FeatureProvenanceEngine:
    """Create a lightweight auditable feature lineage and availability record."""
    @staticmethod
    def build(df: pd.DataFrame, target: str | None = None, approved_features: list[str] | None = None) -> dict[str, Any]:
        if df is None:
            return {"status": "insufficient_data"}
        rows = []
        approved = set(approved_features or [c for c in df.columns if c != target])
        for c in df.columns:
            if c == target:
                role = "target"
            elif c in approved:
                role = "feature"
            else:
                role = "excluded"
            rows.append({"feature": str(c), "role": role, "dtype": str(df[c].dtype), "missing_pct": float(df[c].isna().mean()*100), "nunique": int(df[c].nunique(dropna=True)), "name_flags": [x for x in ("target", "future", "result", "prediction", "post") if re.search(x, str(c), re.I)]})
        return {"status": "ok", "target": target, "features": rows, "policy": "Names are screening signals only; provenance and prediction-time availability require user/domain validation."}
