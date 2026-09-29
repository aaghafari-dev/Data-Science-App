from __future__ import annotations

"""Practical, dependency-light data-quality diagnostics for desktop analysis."""

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class QualityIssue:
    severity: str
    check: str
    column: str | None
    message: str
    count: int = 0
    recommendation: str = ""


class DataQualityEngine:
    """Runs fast diagnostics before exploratory analysis or modelling.

    The score is a diagnostic indicator, not a statistical guarantee. It is designed
    to tell the analyst where attention is needed before downstream modelling.
    """

    def assess(self, df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
        if df is None or df.empty:
            return {"status": "blocked", "score": 0.0, "issues": [], "summary": "No data available."}

        issues: list[QualityIssue] = []
        n = len(df)
        p = len(df.columns)

        # Dataset-level checks.
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            issues.append(QualityIssue("warning", "duplicate_rows", None,
                                       f"{duplicate_count:,} duplicate rows detected.", duplicate_count,
                                       "Review whether duplicates are valid repeated observations."))

        constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
        for c in constant_cols:
            issues.append(QualityIssue("warning", "constant_column", c,
                                       "Column contains one unique value.", int(n),
                                       "Remove it unless it has a documented business meaning."))

        # Per-column checks.
        for c in df.columns:
            s = df[c]
            missing = int(s.isna().sum())
            if missing:
                pct = 100.0 * missing / max(n, 1)
                severity = "high" if pct >= 40 else "warning"
                issues.append(QualityIssue(severity, "missing_values", str(c),
                                           f"{missing:,} missing values ({pct:.1f}%).", missing,
                                           "Choose deletion, imputation, or an explicit missing category."))

            if pd.api.types.is_numeric_dtype(s):
                inf = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum()) if len(s) else 0
                if inf:
                    issues.append(QualityIssue("high", "infinite_values", str(c),
                                               f"{inf:,} infinite numeric values detected.", inf,
                                               "Replace or remove infinities before scaling/model fitting."))
                non_null = s.dropna()
                if len(non_null) >= 8 and non_null.nunique() > 1:
                    q1, q3 = non_null.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if pd.notna(iqr) and iqr > 0:
                        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        outliers = int(((non_null < low) | (non_null > high)).sum())
                        if outliers:
                            issues.append(QualityIssue("info", "iqr_outliers", str(c),
                                                       f"{outliers:,} observations are outside the 1.5×IQR range.", outliers,
                                                       "Inspect before clipping/removing; an outlier can be scientifically meaningful."))

            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                text = s.dropna().astype(str)
                if len(text):
                    empty = int((text.str.strip() == "").sum())
                    if empty:
                        issues.append(QualityIssue("warning", "empty_strings", str(c),
                                                   f"{empty:,} blank strings detected.", empty,
                                                   "Normalize blanks to missing values before analysis."))
                    unique_ratio = text.nunique() / max(len(text), 1)
                    if unique_ratio > 0.98 and len(text) > 20:
                        issues.append(QualityIssue("info", "high_cardinality", str(c),
                                                   "Very high categorical/text cardinality detected.", int(text.nunique()),
                                                   "Review whether this is an identifier, free text, or a useful feature."))

        if target and target in df.columns:
            ts = df[target]
            if ts.nunique(dropna=True) <= 1:
                issues.append(QualityIssue("high", "invalid_target", target,
                                           "Target has one or fewer non-null unique values.", int(ts.nunique(dropna=True)),
                                           "Choose a target with sufficient variation for the intended task."))
            if pd.api.types.is_numeric_dtype(ts) and ts.nunique(dropna=True) >= 2:
                # A very simple imbalance warning for binary numeric targets.
                vals = ts.dropna().value_counts(normalize=True)
                if len(vals) == 2 and float(vals.min()) < 0.10:
                    issues.append(QualityIssue("warning", "class_imbalance", target,
                                               f"Minority class proportion is only {100*float(vals.min()):.1f}%.", int(ts.notna().sum()),
                                               "Use stratified splitting and consider imbalance-aware metrics."))

        weights = {"high": 4.0, "warning": 1.5, "info": 0.25}
        penalty = sum(weights.get(i.severity, 0.0) for i in issues)
        scale = max(10.0, 0.35 * p + 0.02 * n**0.5)
        score = max(0.0, min(100.0, 100.0 - 100.0 * penalty / (penalty + scale)))
        high = sum(i.severity == "high" for i in issues)
        warnings = sum(i.severity == "warning" for i in issues)
        return {
            "status": "review" if high or warnings else "pass",
            "score": round(float(score), 1),
            "rows": int(n),
            "columns": int(p),
            "duplicate_rows": duplicate_count,
            "constant_columns": constant_cols,
            "high_severity": high,
            "warnings": warnings,
            "issues": [asdict(i) for i in issues],
        }

    @staticmethod
    def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in df.columns:
            s = df[c]
            example = ""
            non_null = s.dropna()
            if not non_null.empty:
                example = str(non_null.iloc[0])[:120]
            rows.append({
                "column": str(c),
                "dtype": str(s.dtype),
                "rows": int(len(s)),
                "non_null": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(100*s.isna().mean()), 2),
                "unique": int(s.nunique(dropna=True)),
                "example": example,
            })
        return pd.DataFrame(rows)
