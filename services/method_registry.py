"""Capability registry for analytical methods considered by the Master Agent."""

from __future__ import annotations

from typing import Any


class AnalyticalMethodRegistry:
    """Provide machine-readable capability cards for governed analytical method families."""

    _CARDS: dict[str, dict[str, Any]] = {
        "classical_ml": {"family": "supervised", "data": ["tabular", "mixed_type"], "complexity": "moderate", "interpretability": "medium", "validation": ["cross_validation", "locked_test"], "challenger": True},
        "deep_learning": {"family": "supervised", "data": ["tabular"], "complexity": "high", "interpretability": "low", "validation": ["cross_validation", "locked_test", "early_stopping"], "challenger": True},
        "cnn_transfer_learning": {"family": "computer_vision", "data": ["images"], "complexity": "high", "interpretability": "medium", "validation": ["class_stratified_split", "locked_test", "classwise_metrics", "explainability_review"], "challenger": False},
        "clustering_analysis": {"family": "clustering", "data": ["numeric"], "complexity": "moderate", "interpretability": "medium", "validation": ["silhouette", "calinski_harabasz", "davies_bouldin", "stability", "domain_review"], "challenger": False},
        "isolation_forest": {"family": "anomaly", "data": ["numeric"], "complexity": "moderate", "interpretability": "medium", "validation": ["contamination_sensitivity", "domain_review"], "challenger": False},
        "temporal_baseline": {"family": "time_series", "data": ["datetime_numeric"], "complexity": "low", "interpretability": "high", "validation": ["time_aware_split", "forecast_diagnostics"], "challenger": False},
        "descriptive_association": {"family": "statistics", "data": ["mixed_type"], "complexity": "low", "interpretability": "high", "validation": ["assumption_review", "effect_context"], "challenger": False},
        "evidence_profile": {"family": "exploration", "data": ["mixed_type"], "complexity": "low", "interpretability": "high", "validation": ["schema", "missingness", "distribution_review"], "challenger": False},
    }

    @classmethod
    def get(cls, method_id: str) -> dict[str, Any]:
        """Return a copy of one method capability card."""
        return dict(cls._CARDS.get(method_id, {}))

    @classmethod
    def annotate(cls, candidate: dict[str, Any]) -> dict[str, Any]:
        """Attach the capability card to a candidate without changing its core decision fields."""
        result = dict(candidate)
        result["capabilities"] = cls.get(str(candidate.get("method_id", "")))
        return result

    @classmethod
    def describe(cls) -> list[dict[str, Any]]:
        """Return all registered method capability cards for inspection and testing."""
        return [{"method_id": key, **value} for key, value in cls._CARDS.items()]
