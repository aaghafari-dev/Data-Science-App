"""Module duty: Evaluation protocol.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

"""Validation-strategy selection and integrity checks for professional analysis.

The advisor never claims that one split is universally correct.  It inspects the
observed data and the intended prediction setting, then returns an explicit,
auditable recommendation and alternatives.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
)


@dataclass
class ValidationPlan:
    strategy: str
    rationale: list[str]
    warnings: list[str]
    time_column: str | None = None
    group_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Perform the to dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {
            "strategy": self.strategy,
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "time_column": self.time_column,
            "group_column": self.group_column,
        }


class ValidationProtocolAdvisor:
    TIME_NAMES = ("date", "time", "timestamp", "datetime", "period")
    GROUP_NAMES = ("id", "customer", "patient", "subject", "sample", "experiment", "batch", "group")

    @classmethod
    def infer(cls, df: pd.DataFrame, target: str | None = None) -> ValidationPlan:
        """Perform the infer operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if df is None or df.empty:
            return ValidationPlan("random", ["No data were supplied."], ["Validation strategy cannot be inferred."])
        time_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if not time_cols:
            time_cols = [c for c in df.columns if any(n in str(c).lower() for n in cls.TIME_NAMES)]
        group_cols = []
        for c in df.columns:
            if c == target:
                continue
            name = str(c).lower()
            nunique = int(df[c].nunique(dropna=True))
            if any(n in name for n in cls.GROUP_NAMES) and 2 <= nunique < len(df) * .9:
                group_cols.append(c)
        rationale, warnings = [], []
        if time_cols:
            col = str(time_cols[0])
            rationale.append(f"Datetime-like field '{col}' was detected; future prediction may require chronological validation.")
            if group_cols:
                warnings.append(f"A group-like field '{group_cols[0]}' is also present; repeated entities may require group-aware temporal validation.")
            return ValidationPlan("time", rationale, warnings, time_column=col, group_column=str(group_cols[0]) if group_cols else None)
        if group_cols:
            col = str(group_cols[0])
            rationale.append(f"Group-like field '{col}' may identify repeated entities; random row splitting can put the same entity in train and test.")
            return ValidationPlan("group", rationale, warnings, group_column=col)
        if target is not None and target in df.columns:
            rationale.append("No strong temporal or group structure was detected; stratification can preserve class proportions when appropriate.")
        else:
            rationale.append("No target was supplied; a standard random split is only a provisional default.")
        return ValidationPlan("stratified" if target is not None else "random", rationale, warnings)

    @staticmethod
    def split(df: pd.DataFrame, target: str, plan: ValidationPlan, test_size: float = .2, seed: int = 42):
        """Perform the split operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if plan.strategy == "time" and plan.time_column and plan.time_column in df.columns:
            order = pd.to_datetime(df[plan.time_column], errors="coerce").sort_values().index
            ordered = df.loc[order]
            cut = max(1, min(len(ordered)-1, int(round(len(ordered) * (1-test_size)))))
            return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
        if plan.strategy == "group" and plan.group_column and plan.group_column in df.columns:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            groups = df[plan.group_column]
            tr_idx, te_idx = next(splitter.split(df, df[target], groups=groups))
            return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        y = df[target]
        strat = y if plan.strategy == "stratified" and y.dtype == object and y.value_counts().min() >= 2 else None
        tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
        return tr.copy(), te.copy()

    @staticmethod
    def audit(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str | None = None, group_column: str | None = None) -> dict[str, Any]:
        """Perform the audit operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        checks = []
        if train_df is None or test_df is None:
            return {"status": "blocked", "checks": [{"check": "datasets_present", "status": "fail"}]}
        common = [c for c in train_df.columns if c in test_df.columns]
        try:
            train_hash = pd.util.hash_pandas_object(train_df[common].astype(str), index=False)
            test_hash = pd.util.hash_pandas_object(test_df[common].astype(str), index=False)
            overlap = len(set(train_hash.tolist()).intersection(set(test_hash.tolist())))
        except Exception:
            overlap = None
        checks.append({"check": "exact_row_overlap", "status": "pass" if overlap == 0 else "fail", "overlap": overlap})
        if group_column and group_column in train_df.columns and group_column in test_df.columns:
            overlap_groups = len(set(train_df[group_column].dropna()).intersection(set(test_df[group_column].dropna())))
            checks.append({"check": "group_overlap", "status": "pass" if overlap_groups == 0 else "fail", "overlap_groups": overlap_groups})
        if target and target in train_df.columns and target in test_df.columns:
            checks.append({"check": "target_present", "status": "pass"})
        else:
            checks.append({"check": "target_present", "status": "review"})
        status = "blocked" if any(x["status"] == "fail" for x in checks) else ("review" if any(x["status"] == "review" for x in checks) else "pass")
        return {"status": status, "checks": checks, "train_rows": len(train_df), "test_rows": len(test_df)}
