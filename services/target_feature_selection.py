"""Module duty: Target feature selection.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations
from typing import Any
import re
import numpy as np
import pandas as pd

_TARGET_PATTERNS = re.compile(r"(^|_)(target|label|class|y|outcome|response|result|score|price|sales|revenue|salary|wage|income|compensation|earnings|amount|churn|survived|default|bandgap|conductivity)($|_)", re.I)
_ID_PATTERNS = re.compile(r"(^|_)(id|uuid|key|index|identifier)($|_)", re.I)
_FUTURE_PATTERNS = re.compile(r"future|next|lead|after|outcome|post_|future_|timestamp_end|prediction", re.I)


def _target_candidate_score(df: pd.DataFrame, c: str) -> tuple[float, list[str]]:
    """Perform the target candidate score operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
    if s.isna().all():
        score -= 8; reasons.append("all values missing")
    return score, reasons


def _pairwise_redundancy(df: pd.DataFrame, max_cols: int = 40) -> list[dict[str, Any]]:
    """Perform the pairwise redundancy operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 5][:max_cols]
    out: list[dict[str, Any]] = []
    if len(nums) < 2: return out
    sample = df[nums].dropna(how="all").head(10000)
    for i, a in enumerate(nums):
        av = pd.to_numeric(sample[a], errors="coerce")
        for b in nums[i+1:]:
            bv = pd.to_numeric(sample[b], errors="coerce")
            valid = av.notna() & bv.notna()
            if valid.sum() < 5: continue
            x, y = av[valid].to_numpy(float), bv[valid].to_numpy(float)
            corr = float(np.corrcoef(x, y)[0,1]) if np.std(x) > 0 and np.std(y) > 0 else 1.0
            if abs(corr) >= .995:
                out.append({"columns":[str(a),str(b)],"type":"near_duplicate_or_linear_transform","correlation":round(corr,6),
                            "recommendation":"Keep one representative unless both variables have distinct documented meaning."})
                continue
            # Detect simple square/cubic-style deterministic transforms without assuming causality.
            for power in (2, 3):
                xp = np.sign(x) * (np.abs(x) ** power)
                if np.std(xp) > 0 and np.std(y) > 0:
                    r = float(np.corrcoef(xp, y)[0,1])
                    if abs(r) >= .995:
                        out.append({"columns":[str(a),str(b)],"type":f"possible_power_{power}_transform","correlation":round(r,6),
                                    "recommendation":"Flag as mathematically redundant; do not automatically remove because temporal/causal direction is unknown."})
                        break
    return out


def recommend_targets_and_features(df: pd.DataFrame) -> dict[str, Any]:
    """Perform the recommend targets and features operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if df is None or df.empty:
        return {"target_candidates": [], "feature_candidates": [], "warnings": ["No data available."]}
    targets = []
    for c in df.columns:
        score, reasons = _target_candidate_score(df, str(c))
        targets.append({"column": str(c), "score": round(score, 3), "reasons": reasons,
                        "missing_pct": round(float(df[c].isna().mean()*100), 2),
                        "unique": int(df[c].nunique(dropna=True)), "dtype": str(df[c].dtype)})
    targets.sort(key=lambda x: x["score"], reverse=True)
    suggested = targets[0]["column"] if targets else None
    redundancy = _pairwise_redundancy(df)
    warnings = []
    for pair in redundancy:
        warnings.append(f"{pair['columns']}: {pair['type']} detected; target/feature direction is not inferable from correlation alone.")

    features = []
    for c in df.columns:
        if c == suggested: continue
        s = df[c]; reasons = []; score = 0.0
        if _ID_PATTERNS.search(str(c)): score -= 4; reasons.append("identifier-like")
        if _FUTURE_PATTERNS.search(str(c)): score -= 3; reasons.append("possible future/post-outcome information")
        if s.isna().all(): score -= 8; reasons.append("all values missing")
        if s.nunique(dropna=True) <= 1: score -= 5; reasons.append("constant")
        if pd.api.types.is_numeric_dtype(s): score += 1; reasons.append("numeric")
        else: score += .5; reasons.append("usable categorical/object")
        # Target leakage/redundancy check.
        if suggested and pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(df[suggested]):
            a = pd.to_numeric(s, errors="coerce"); b = pd.to_numeric(df[suggested], errors="coerce"); valid = a.notna() & b.notna()
            if valid.sum() >= 5:
                corr = float(a[valid].corr(b[valid]))
                if np.isfinite(corr) and abs(corr) >= .995:
                    score -= 5; reasons.append("near-perfect association with proposed target; possible leakage/duplicate")
                for power in (2,3):
                    ap = np.sign(a[valid].to_numpy(float)) * (np.abs(a[valid].to_numpy(float)) ** power)
                    y = b[valid].to_numpy(float)
                    if np.std(ap) > 0 and np.std(y) > 0 and abs(float(np.corrcoef(ap,y)[0,1])) >= .995:
                        score -= 5; reasons.append(f"possible power-{power} transform of target; direction ambiguous")
                        break
        features.append({"column": str(c), "score": round(score,3), "reasons": reasons, "dtype": str(s.dtype)})
    features.sort(key=lambda x: x["score"], reverse=True)
    return {"target_candidates": targets, "feature_candidates": features,
            "suggested_target": suggested,
            "suggested_features": [x["column"] for x in features if x["score"] > -1],
            "redundancy_pairs": redundancy,
            "warnings": sorted(set(warnings))}
