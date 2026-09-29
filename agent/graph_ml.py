from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from services.error_analysis import ErrorAnalysisEngine
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory
from services.agent_quality import (
    ProfessionalEvaluationEngine, FeatureStabilityAnalyzer, TemporalAvailabilityAnalyzer,
    CounterfactualLeakageAnalyzer, ModelDiagnosisEngine,
)


def _prepare(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target]); y = work[target]
    task = "classification" if (y.dtype == object or y.nunique() <= min(20, max(2, int(len(y) * .05)))) else "regression"
    if task == "classification": y = y.astype(str)
    from services.model_preprocessing import build_preprocessor
    return X, y, task, build_preprocessor(X)


def _pipeline_for(task: str, estimator, X: pd.DataFrame) -> Pipeline:
    from services.model_preprocessing import build_preprocessor
    return Pipeline([("pre", build_preprocessor(X)), ("model", clone(estimator))])


def _suspicious_features(df: pd.DataFrame, target: str, rec: dict[str, Any]) -> list[str]:
    rows = rec.get("feature_candidates", [])
    selected = []
    for item in rows:
        reasons = " ".join(item.get("reasons", [])) if isinstance(item, dict) else ""
        name = item.get("column") if isinstance(item, dict) else str(item)
        if name != target and any(k in reasons.lower() for k in ("future", "near-perfect", "leakage", "power-")):
            selected.append(str(name))
    return selected[:10]


def run_ml_step(df: pd.DataFrame, target: str | None = None, seed: int = 42) -> dict[str, Any]:
    """Master-agent ML internal agent with professional evaluation and diagnostics."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available."}
    rec = recommend_targets_and_features(df)
    target = target or rec.get("suggested_target") or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    X, y, task, pre = _prepare(df, target)
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after target/leakage-safe preprocessing.")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=stratify)
    if task == "classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
            "Logistic Regression": LogisticRegression(max_iter=2000),
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
            "Ridge Regression": Ridge(),
        }
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in models.items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        t0 = time.time(); pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
        if task == "classification":
            metrics = {"accuracy": float(accuracy_score(yte, pred)), "f1_weighted": float(f1_score(yte, pred, average="weighted"))}
        else:
            metrics = {"r2": float(r2_score(yte, pred)), "mae": float(mean_absolute_error(yte, pred)),
                       "rmse": float(mean_squared_error(yte, pred) ** .5)}
        evaluation = ErrorAnalysisEngine.classification(yte, pred) if task == "classification" else ErrorAnalysisEngine.regression(yte, pred)
        professional = ProfessionalEvaluationEngine.evaluate_bundle(pipe, Xtr, Xte, ytr, yte, task, seed)
        results[name] = {"metrics": metrics, "seconds": round(time.time() - t0, 3), "error_analysis": evaluation, "professional_evaluation": professional}
        fitted[name] = pipe
    primary = "f1_weighted" if task == "classification" else "r2"
    best = max(results, key=lambda k: results[k]["metrics"].get(primary, -np.inf))
    best_pipe = fitted[best]

    def factory() -> Pipeline:
        estimator = models[best]
        return _pipeline_for(task, estimator, X)

    stability = FeatureStabilityAnalyzer.run(factory, X, y, task, repeats=3, max_features=25, seed=seed)
    temporal = TemporalAvailabilityAnalyzer.run(df, target=target)
    suspicious = _suspicious_features(df, target, rec)
    leakage = CounterfactualLeakageAnalyzer.run(
        lambda cols: _pipeline_for(task, models[best], X[cols]), X, y, task, suspicious, seed
    ) if suspicious else {"status": "not_available", "reason": "No high-priority suspicious features were identified for the counterfactual test."}
    diagnosis = ModelDiagnosisEngine.diagnose(task, results[best]["professional_evaluation"], stability, temporal, leakage)

    return {
        "status": "ok", "agent": "ML", "target": target, "task": task,
        "models": results, "best_model": best, "best_metrics": results[best]["metrics"],
        "model_object": best_pipe,
        "evaluation": {"level": "professional", "best_model": results[best]["professional_evaluation"], "baseline": results[best]["professional_evaluation"].get("baseline"), "cross_validation": results[best]["professional_evaluation"].get("cross_validation"), "prediction_uncertainty": results[best]["professional_evaluation"].get("prediction_uncertainty"), "robustness_perturbation": results[best]["professional_evaluation"].get("robustness_perturbation")},
        "feature_stability": stability,
        "temporal_availability": temporal,
        "counterfactual_leakage": leakage,
        "model_diagnosis": diagnosis,
        "summary": f"ML compared {len(results)} tabular models using locked holdout metrics, repeated cross-validation, baseline comparison, uncertainty, perturbation robustness and diagnostic evidence; selected {best} by {primary} on the held-out set.",
        "target_analysis": rec, "memory": AgentMemory("ML", 20).to_dict(),
    }
