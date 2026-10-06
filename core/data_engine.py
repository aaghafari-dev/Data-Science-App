"""Module duty: Data engine.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from services.semantic_types import classify_dtype


class DataEngine:
    """Data source, filtering, schema and transformation service with reproducible lineage hooks."""
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.filter_specs: dict[str, dict[str, Any]] = {}
        self.source_path: str | None = None
        self.dimensions: list[str] = []
        self.measures: list[str] = []
        self.core_types: dict[str, str] = {}
        self._refresh_schema()

    def _refresh_schema(self):
        """Perform the refresh schema operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.df is None:
            self.dimensions, self.measures = [], []
            return
        self.measures = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.dimensions = [c for c in self.df.columns if c not in self.measures]
        self.core_types = {str(c): classify_dtype(self.df[c]) for c in self.df.columns}

    def _reapply_filters(self):
        """Perform the reapply filters operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.original_df is None:
            return
        filtered = self.original_df.copy()
        for col, spec in self.filter_specs.items():
            if col not in filtered.columns:
                continue
            if spec.get("kind") == "numeric":
                filtered = filtered[filtered[col].between(spec["min"], spec["max"], inclusive="both")]
            elif spec.get("kind") == "categorical":
                filtered = filtered[filtered[col].isin(spec.get("values", []))]
            elif spec.get("kind") == "date":
                series = pd.to_datetime(filtered[col], errors="coerce")
                if spec.get("start"):
                    filtered = filtered[series >= pd.Timestamp(spec["start"])]
                if spec.get("end"):
                    filtered = filtered[series <= pd.Timestamp(spec["end"])]
        self.df = filtered.reset_index(drop=True)
        self._refresh_schema()

    def read_file(self, filepath: str) -> pd.DataFrame:
        """Perform the read file operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        raise ValueError(f"Unsupported file type: {suffix}")

    def load_data(self, filepath: str):
        """Perform the load data operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.source_path = str(filepath)
        self.original_df = self.read_file(filepath).copy()
        self.filter_specs.clear()
        self.df = self.original_df.copy()
        self._refresh_schema()
        return self.df

    def set_active_dataframe(self, df: pd.DataFrame, source_path: str | None = None):
        """Perform the set active dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.original_df = df.copy()
        self.df = df.copy()
        self.filter_specs.clear()
        self.source_path = source_path
        self._refresh_schema()
        return self.df

    def commit_dataframe(self, df: pd.DataFrame, keep_as_source: bool = False):
        """Perform the commit dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.df = df.copy()
        if keep_as_source or self.original_df is None:
            self.original_df = self.df.copy()
        self._refresh_schema()
        return self.df

    def apply_filters(self, filters: dict):
        """Perform the apply filters operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        for col, values in filters.items():
            if isinstance(values, dict):
                self.filter_specs[col] = values
            elif isinstance(values, tuple) and len(values) == 2:
                self.filter_specs[col] = {"kind": "numeric", "min": values[0], "max": values[1]}
            else:
                self.filter_specs[col] = {"kind": "categorical", "values": list(values) if isinstance(values, (list, tuple, set)) else [values]}
        self._reapply_filters()
        return self.df

    def set_filter(self, col: str, spec: dict[str, Any]):
        """Perform the set filter operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.filter_specs[col] = spec
        self._reapply_filters()
        return self.df

    def remove_filter(self, col: str):
        """Perform the remove filter operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.filter_specs.pop(col, None)
        self._reapply_filters()
        return self.df

    def clear_filters(self):
        """Perform the clear filters operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.filter_specs.clear()
        self._reapply_filters()
        return self.df

    def get_data_info(self) -> str:
        """Perform the get data info operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return "No data loaded." if self.df is None else f"Shape: {self.df.shape}\n\nDtypes:\n{self.df.dtypes}"

    def get_missing_values(self):
        """Perform the get missing values operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {} if self.df is None else self.df.isnull().sum().to_dict()

    def describe_data(self) -> str:
        """Perform the describe data operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return "No data." if self.df is None else self.df.describe(include="all").to_string()

    def remove_duplicates(self):
        """Perform the remove duplicates operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.original_df = self.df.copy()
        self._refresh_schema()
        return f"Removed {before - len(self.df)} duplicates."

    def fill_missing_mean(self):
        """Perform the fill missing mean operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())
        self.original_df = self.df.copy(); self._refresh_schema()
        return "Filled numerical missing values with mean."

    def drop_missing_rows(self):
        """Perform the drop missing rows operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        before = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        self.original_df = self.df.copy(); self._refresh_schema()
        return f"Dropped {before - len(self.df)} rows."

    def cap_outliers(self):
        """Perform the cap outliers operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        for c in self.df.select_dtypes(include=[np.number]).columns:
            q1, q3 = self.df[c].quantile([0.25, 0.75]); iqr = q3 - q1
            self.df[c] = self.df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        self.original_df = self.df.copy(); self._refresh_schema()
        return "Capped outliers using IQR."

    def label_encode(self, column):
        """Perform the label encode operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        le = LabelEncoder(); self.df[column] = le.fit_transform(self.df[column].astype(str))
        self.original_df = self.df.copy(); self._refresh_schema()
        return f"Label Encoded {column}."

    def scale_data(self, column):
        """Perform the scale data operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        scaler = StandardScaler(); self.df[[column]] = scaler.fit_transform(self.df[[column]])
        self.original_df = self.df.copy(); self._refresh_schema()
        return f"Scaled {column}."

    def get_extra_insight(self, column, top_n=10, ascending=False):
        """Perform the get extra insight operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.df is None or column not in self.df.columns:
            return None
        if pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df[[column]].dropna().sort_values(column, ascending=ascending).head(top_n).copy()
        counts=self.df[column].value_counts(dropna=False).sort_values(ascending=ascending).head(top_n)
        out = counts.reset_index()
        out.columns = [column, "Count"]
        return out

    def dataset_hash(self) -> str | None:
        """Perform the dataset hash operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.original_df is None:
            return None
        payload = pd.util.hash_pandas_object(self.original_df, index=True).values.tobytes()
        return hashlib.sha256(payload).hexdigest()
