from __future__ import annotations
from typing import Any
import re
import numpy as np
import pandas as pd

_TARGET_PATTERNS = re.compile(r"(^|_)(target|label|class|y|outcome|response|result|score|price|sales|revenue|churn|survived|default|bandgap|conductivity)($|_)", re.I)
_ID_PATTERNS = re.compile(r"(^|_)(id|uuid|key|index|identifier)($|_)", re.I)
_FUTURE_PATTERNS = re.compile(r"future|next|lead|after|outcome|post_|future_|timestamp_end|prediction", re.I)


def _target_candidate_score(df: pd.DataFrame, c: str) -> tuple[float, list[str]]:
    s = df[c]; reasons: list[str] = []; score = 0.0
    if _TARGET_PATTERNS.search(str(c)):
        score += 3; reasons.append("target-like name")
    if pd.api.types.is_numeric_dtype(s):
        score += 1.0; reasons.append("numeric")
    elif pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        nun = s.nunique(dropna=True)
        if 2 <= nun <= min(20, max(2, int(len(df) * .05))):
            score += 2; reasons.append("classification-like cardinality")
    miss = float(s.isna().mean()) if len(s) else 1.0
    score -= 2 * miss
    if s.nunique(dropna=True) <= 1:
        score -= 5; reasons.append("constant/invalid")
    if _ID_PATTERNS.search(str(c)):
        score -= 4; reasons.append("identifier-like")
    return score, reasons


def recommend_targets_and_features(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"target_candidates": [], "feature_candidates": [], "warnings": ["No data available."]}
    targets = []
    for c in df.columns:
        score, reasons = _target_candidate_score(df, str(c))
        targets.append({"column": str(c), "score": round(score, 3), "reasons": reasons,
                        "missing_pct": round(float(df[c].isna().mean()*100), 2),
                        "unique": int(df[c].nunique(dropna=True)), "dtype": str(df[c].dtype)})
    targets.sort(key=lambda x: x["score"], reverse=True)
    chosen = targets[0]["column"] if targets else None
    features = []
    warnings = []
    for c in df.columns:
        if c == chosen: continue
        s = df[c]
        reasons = []
        score = 0.0
        if _ID_PATTERNS.search(str(c)):
            score -= 4; reasons.append("identifier-like")
        if _FUTURE_PATTERNS.search(str(c)):
            score -= 3; reasons.append("possible future/post-outcome information")
        if s.isna().all():
            score -= 6; reasons.append("all values missing")
        if s.nunique(dropna=True) <= 1:
            score -= 5; reasons.append("constant")
        if pd.api.types.is_numeric_dtype(s): score += 1; reasons.append("numeric")
        else: score += 0.5; reasons.append("usable categorical/object")
        if chosen is not None and pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(df[chosen]):
            a = pd.to_numeric(s, errors="coerce"); b = pd.to_numeric(df[chosen], errors="coerce")
            valid = a.notna() & b.notna()
            if valid.sum() >= 5:
                corr = float(a[valid].corr(b[valid]))
                if np.isfinite(corr) and abs(corr) >= .995:
                    score -= 3; warnings.append(f"{c}: near-perfect target correlation may indicate leakage or duplication.")
        features.append({"column": str(c), "score": round(score,3), "reasons": reasons, "dtype": str(s.dtype)})
    features.sort(key=lambda x: x["score"], reverse=True)
    return {"target_candidates": targets, "feature_candidates": features, "suggested_target": chosen,
            "suggested_features": [x["column"] for x in features if x["score"] > -1], "warnings": sorted(set(warnings))}
