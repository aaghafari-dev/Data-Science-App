"""Statistical-insight specialist for descriptive, association, and data-structure evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.agent_memory import AgentMemory
from services.statistical_analysis import StatisticalAnalysisEngine


def run_statistics_step(df: pd.DataFrame, question: str = "") -> dict[str, Any]:
    """Create descriptive and association evidence without performing unsupported inference."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available for statistical analysis."}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    corr = df[numeric].corr(numeric_only=True) if len(numeric) >= 2 else pd.DataFrame()
    strongest = []
    if not corr.empty:
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                value = corr.loc[a, b]
                if pd.notna(value):
                    pairs.append((abs(float(value)), a, b, float(value)))
        strongest = [{"x": a, "y": b, "correlation": r} for _, a, b, r in sorted(pairs, reverse=True)[:10]]
    try:
        descriptive = StatisticalAnalysisEngine.describe(df)
    except Exception as exc:
        descriptive = {"status": "unavailable", "reason": str(exc)}
    memory = AgentMemory("Statistics", 20)
    memory.remember("statistical_summary", {"numeric": len(numeric), "categorical": len(categorical)})
    return {
        "status": "ok",
        "agent": "Statistical Insight",
        "question": question,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "strongest_numeric_associations": strongest,
        "descriptive": descriptive,
        "interpretation_note": "Associations are descriptive evidence and do not establish causal direction.",
        "memory": memory.to_dict(),
    }
