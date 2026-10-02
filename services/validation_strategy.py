"""Evidence-based validation strategy selection for the Master Agent."""
from __future__ import annotations
from typing import Any
import pandas as pd

class ValidationStrategyAgent:
    """Propose validation designs independently from model selection."""
    @staticmethod
    def propose(df: pd.DataFrame, task: str, target: str | None = None) -> dict[str, Any]:
        """Return a bounded validation proposal with explicit evidence and risks."""
        n = len(df)
        if task == "REINFORCEMENT_LEARNING":
            return {"task": task, "protocol": "multi-seed policy evaluation with held-out episodes/environment conditions", "rationale": "RL performance is sequential and depends on environment dynamics, policy stochasticity and reward definition; ordinary K-fold validation is not appropriate.", "sample_size": n, "checks": ["multiple random seeds", "learning-curve stability", "held-out episodes or environment conditions", "reward and safety diagnostics"], "risks": ["reward misspecification", "policy instability", "environment shift", "off-policy evaluation bias"]}
        if task == "UNSUPERVISED":
            return {"task": task, "protocol": "repeated stability analysis plus internal validity and domain validation", "rationale": "There is no target for ordinary supervised cross-validation; stability of discovered structure is more appropriate.", "sample_size": n, "checks": ["multiple seeds/resamples", "cluster validity", "feature-scaling sensitivity", "domain validation"], "risks": ["algorithm dependence", "scaling sensitivity", "unstable clusters", "spurious structure"]}
        has_time = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
        has_group = any(str(c).lower() in {"group","subject","patient","id","entity","batch"} for c in df.columns)
        imbalance = False
        if target and target in df.columns and task == "CLASSIFICATION":
            p = df[target].value_counts(normalize=True, dropna=True)
            imbalance = bool(len(p) > 1 and p.min() < 0.20)
        if task == "TIME_SERIES" or has_time:
            protocol = "temporal / walk-forward validation"
            rationale = "Temporal ordering is available or the task is explicitly temporal; random shuffling may leak future information."
        elif has_group:
            protocol = "group-aware cross-validation"
            rationale = "Potential entity/group identifiers suggest that observations may not be independent across groups."
        elif task == "CLASSIFICATION" and imbalance:
            protocol = "stratified cross-validation with imbalance-aware metrics"
            rationale = "Class proportions indicate potential imbalance; accuracy alone is insufficient."
        elif n < 1000:
            protocol = "repeated K-fold cross-validation with locked holdout where feasible"
            rationale = "Moderate/small sample size makes a single split potentially unstable."
        else:
            protocol = "K-fold cross-validation with locked test evaluation"
            rationale = "A conventional IID validation design is a reasonable starting point when no grouping or temporal constraint is evident."
        return {"task": task, "protocol": protocol, "rationale": rationale, "sample_size": n, "checks": ["fit preprocessing inside training folds", "keep final test partition untouched", "report uncertainty or variability", "check subgroup performance"], "risks": ["sampling dependence", "distribution shift", "leakage", "small subgroup counts"]}
