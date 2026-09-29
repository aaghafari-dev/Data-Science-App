from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np


class StatisticalAnalysisEngine:
    @staticmethod
    def describe(df: pd.DataFrame) -> dict[str, Any]:
        num = df.select_dtypes(include=[np.number])
        return {"rows": len(df), "columns": len(df.columns), "numeric_summary": num.describe().to_dict(),
                "missing_pct": (100*df.isna().mean()).round(2).to_dict()}

    @staticmethod
    def correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        return df.select_dtypes(include=[np.number]).corr(method=method)

    @staticmethod
    def hypothesis(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
        from scipy import stats
        groups = [g[value_col].dropna().to_numpy(float) for _, g in df.groupby(group_col)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) == 2:
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            return {"test": "Welch t-test", "statistic": float(stat), "p_value": float(p), "groups": len(groups)}
        if len(groups) >= 3:
            stat, p = stats.f_oneway(*groups)
            return {"test": "One-way ANOVA", "statistic": float(stat), "p_value": float(p), "groups": len(groups)}
        raise ValueError("Need two or more groups with at least two observations each.")

    @staticmethod
    def regression(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
        from scipy.stats import linregress
        work = df[[x,y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 3: raise ValueError("Need at least three paired observations.")
        result = linregress(work[x], work[y])
        return {"slope": float(result.slope), "intercept": float(result.intercept), "rvalue": float(result.rvalue),
                "r_squared": float(result.rvalue**2), "p_value": float(result.pvalue), "stderr": float(result.stderr), "n": len(work)}
