from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


class SplitWizard:
    """Leakage-aware train/validation/test planning and execution."""
    MODES = ["Random", "Stratified", "Group", "Time-aware"]

    @staticmethod
    def plan(df: pd.DataFrame, target: str | None, mode: str = "Random", group_col: str | None = None,
             time_col: str | None = None, test_size: float = .2, validation_size: float = .1) -> dict[str, Any]:
        warnings = []
        if mode == "Stratified" and target and target in df.columns and df[target].value_counts(dropna=True).min() < 2:
            warnings.append("Some target classes have fewer than two observations; stratification may fail.")
        if mode == "Group" and group_col not in df.columns:
            warnings.append("A valid group column is required for a group-aware split.")
        if mode == "Time-aware" and time_col not in df.columns:
            warnings.append("A valid datetime column is required for a time-aware split.")
        return {"mode": mode, "target": target, "group_col": group_col, "time_col": time_col,
                "test_size": test_size, "validation_size": validation_size, "warnings": warnings}

    @staticmethod
    def execute(df: pd.DataFrame, target: str | None, mode: str = "Random", group_col: str | None = None,
                time_col: str | None = None, test_size: float = .2, validation_size: float = .1, seed: int = 42):
        work = df.copy()
        if mode == "Time-aware":
            if not time_col or time_col not in work.columns: raise ValueError("A valid time column is required.")
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work.sort_values(time_col).reset_index(drop=True)
            n_test = max(1, int(len(work) * test_size)); n_val = max(1, int(len(work) * validation_size))
            train = work.iloc[:len(work)-n_val-n_test].copy(); val = work.iloc[len(work)-n_val-n_test:len(work)-n_test].copy(); test = work.iloc[len(work)-n_test:].copy()
            return train, val, test
        stratify = None
        if mode == "Stratified" and target in work.columns:
            counts = work[target].value_counts(dropna=True)
            if len(counts) > 1 and counts.min() >= 2: stratify = work[target]
        if mode == "Group":
            if not group_col or group_col not in work.columns: raise ValueError("A valid group column is required.")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(gss.split(work, groups=work[group_col]))
            train_all, test = work.iloc[train_idx].copy(), work.iloc[test_idx].copy()
            val_frac = validation_size / max(1e-9, 1-test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_idx, val_idx = next(gss2.split(train_all, groups=train_all[group_col]))
            return train_all.iloc[tr_idx].copy(), train_all.iloc[val_idx].copy(), test
        train_all, test = train_test_split(work, test_size=test_size, random_state=seed, stratify=stratify)
        val_frac = validation_size / max(1e-9, 1-test_size)
        strat2 = train_all[target] if mode == "Stratified" and target in train_all.columns else None
        if strat2 is not None and strat2.value_counts().min() < 2: strat2 = None
        train, val = train_test_split(train_all, test_size=val_frac, random_state=seed, stratify=strat2)
        return train.copy(), val.copy(), test.copy()
