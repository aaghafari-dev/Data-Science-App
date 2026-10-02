"""Module duty: Model preprocessing.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def _to_string(x):
    """Perform the to string operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return pd.DataFrame(x).astype(str).to_numpy()


def _datetime_to_numeric(x):
    """Perform the datetime to numeric operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    frame = pd.DataFrame(x).copy()
    out = pd.DataFrame(index=frame.index)
    for c in frame.columns:
        dt = pd.to_datetime(frame[c], errors="coerce")
        # Use UTC nanoseconds converted to seconds. NaT becomes NaN and is imputed downstream.
        vals = dt.astype("int64", copy=False).astype("float64")
        vals[dt.isna()] = np.nan
        out[c] = vals / 1e9
    return out.to_numpy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Robust preprocessing for the four core dtypes.

    Datetime is handled explicitly so sklearn never receives a mixed
    ['datetime', 'str'] categorical matrix.
    """
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    # Columns with zero observed values cannot be imputed and must not enter sklearn.
    all_missing = [c for c in X.columns if X[c].isna().all()]
    numeric = [c for c in numeric if c not in all_missing]
    datetime_cols = [c for c in datetime_cols if c not in all_missing]
    categorical = [c for c in X.columns if c not in numeric and c not in datetime_cols and c not in all_missing]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if datetime_cols:
        transformers.append(("date", Pipeline([("to_numeric", FunctionTransformer(_datetime_to_numeric, validate=False)), ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), datetime_cols))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("to_string", FunctionTransformer(_to_string, validate=False)), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")
