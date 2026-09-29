from __future__ import annotations

import pandas as pd


def classify_dtype(series: pd.Series) -> str:
    """Stage-1 schema typing: keep the four core types explicit and deterministic."""
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    return "object"


def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append({
            "column": str(c),
            "dtype": str(df[c].dtype),
            "core_type": classify_dtype(df[c]),
            "nullable": bool(df[c].isna().any()),
            "unique": int(df[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)
