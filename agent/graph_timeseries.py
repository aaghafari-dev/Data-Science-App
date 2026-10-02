"""Time-series specialist for temporal structure, trend, and bounded forecasting evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.agent_memory import AgentMemory
from services.timeseries import analyze_time_series


def run_time_series_step(df: pd.DataFrame, date_col: str | None = None, value_col: str | None = None) -> dict[str, Any]:
    """Select a date/value pair and run the professional time-series baseline service."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available for time-series analysis."}
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dates:
        dates = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().mean() >= 0.8]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not dates or not numeric:
        return {"status": "error", "message": "A date-like column and numeric value column are required."}
    date_col = date_col or dates[0]
    value_col = value_col or numeric[0]
    result = analyze_time_series(df, date_col, value_col)
    memory = AgentMemory("Time-Series", 20)
    memory.remember("time_series_pair", {"date": date_col, "value": value_col})
    return {
        "status": "ok",
        "agent": "Time-Series",
        "date_column": date_col,
        "value_column": value_col,
        "analysis": result,
        "memory": memory.to_dict(),
    }
