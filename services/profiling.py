from __future__ import annotations

import pandas as pd


def auto_profile(df: pd.DataFrame, title: str = "Data Science Studio Pro Profile") -> dict:
    """ydata-profiling integration with a dependency-free statistical fallback."""
    try:
        from ydata_profiling import ProfileReport
        report = ProfileReport(df, title=title, minimal=True)
        return {"engine": "ydata-profiling", "report": report, "html": report.to_html()}
    except Exception as exc:
        return {"engine": "pandas-fallback", "error": str(exc), "summary": {
            "shape": df.shape, "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isna().sum().to_dict(), "duplicates": int(df.duplicated().sum())}}
