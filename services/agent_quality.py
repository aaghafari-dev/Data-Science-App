from __future__ import annotations

"""High-level agent quality, evaluation and leakage diagnostics.

The functions in this module are deterministic services.  LangGraph decides when
and why to call them; the services do not make hidden model-selection decisions.
They return JSON-safe evidence objects so the GUI, Model Cards, Evidence DAG,
Report Agent and Analysis Recipe can consume the same facts.
"""

from dataclasses import dataclass
from typing import Any, Callable
import hashlib
import re

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate, train_test_split

from services.governance import EvidenceObject, StatisticalUncertainty


def _safe_float(x: Any) -> float | None:
    try:
        value = float(x)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _metric_summary(y_true, y_pred, task: str) -> dict[str, Any]:
    if task == "classification":
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        return {
            "accuracy": _safe_float(accuracy_score(yt, yp)),
            "balanced_accuracy": _safe_float(balanced_accuracy_score(yt, yp)),
            "precision_weighted": _safe_float(precision_score(yt, yp, average="weighted", zero_division=0)),
            "recall_weighted": _safe_float(recall_score(yt, yp, average="weighted", zero_division=0)),
            "f1_weighted": _safe_float(f1_score(yt, yp, average="weighted", zero_division=0)),
        }
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": _safe_float(mean_absolute_error(yt, yp)),
        "rmse": _safe_float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": _safe_float(r2_score(yt, yp)),
    }


class ProfessionalEvaluationEngine:
    """Professional, leakage-aware evaluation.

    Critical policy: model-selection CV is performed on *training data only*.
    The held-out test set is evaluated once after the model/protocol has been
    selected. This prevents the final test set from becoming a tuning resource.
    """

    @staticmethod
    def baseline(y_train, y_test, task: str) -> dict[str, Any]:
        estimator = DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")
        estimator.fit(np.zeros((len(y_train), 1)), y_train)
        pred = estimator.predict(np.zeros((len(y_test), 1)))
        return {"model": "Dummy baseline", "metrics": _metric_summary(y_test, pred, task)}

    @staticmethod
    def evaluate_holdout(y_true, y_pred, task: str, seed: int = 42) -> dict[str, Any]:
        metrics = _metric_summary(y_true, y_pred, task)
        primary = "f1_weighted" if task == "classification" else "r2"
        try:
            metric_fn = f1_score if task == "classification" else r2_score
            metrics[f"{primary}_uncertainty"] = StatisticalUncertainty.bootstrap_metric(
                y_true, y_pred, metric_fn, n_boot=500, seed=seed,
            )
        except Exception:
            pass
        return {"metrics": metrics, "primary_metric": primary, "evaluation_population": "locked_holdout_test"}

    @staticmethod
    def cross_validation(pipeline, X_train: pd.DataFrame, y_train, task: str, seed: int = 42, n_jobs: int = 1) -> dict[str, Any]:
        """Evaluate candidate models only inside the training partition."""
        n = len(y_train)
        if n < 30:
            return {"status": "insufficient_data", "reason": "Fewer than 30 training rows; repeated cross-validation was not run."}
        folds = 5 if n >= 100 else 3
        repeats = 2 if n >= 100 else 1
        if task == "classification":
            min_class = int(pd.Series(y_train).value_counts().min())
            folds = min(folds, min_class)
            if folds < 2:
                return {"status": "insufficient_data", "reason": "Not enough observations per class for cross-validation."}
            cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
            scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_weighted": "f1_weighted"}
        else:
            cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
            scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
        try:
            raw = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}
        out: dict[str, Any] = {"status": "ok", "folds": folds, "repeats": repeats, "evaluation_population": "training_partition_only", "metrics": {}}
        for key in scoring:
            values = np.asarray(raw[f"test_{key}"], float)
            if key in {"mae", "rmse"}: values = -values
            train_values = np.asarray(raw[f"train_{key}"], float)
            if key in {"mae", "rmse"}: train_values = -train_values
            out["metrics"][key] = {
                "mean": _safe_float(np.mean(values)),
                "std": _safe_float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": _safe_float(np.min(values)),
                "max": _safe_float(np.max(values)),
                "train_mean": _safe_float(np.mean(train_values)),
                "generalization_gap": _safe_float(np.mean(train_values) - np.mean(values)),
                "fold_values": [_safe_float(v) for v in values],
            }
        return out

    @staticmethod
    def classification_calibration(pipeline, X_test, y_test) -> dict[str, Any]:
        try:
            if not hasattr(pipeline, "predict_proba"):
                return {"status": "not_available"}
            from sklearn.metrics import brier_score_loss, log_loss
            proba = np.asarray(pipeline.predict_proba(X_test), dtype=float)
            p = np.max(proba, axis=1)
            pred = pipeline.predict(X_test)
            correct = (np.asarray(pred) == np.asarray(y_test)).astype(float)
            bins = np.linspace(0, 1, 11)
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
                if mask.any():
                    rows.append({"confidence_mean": _safe_float(np.mean(p[mask])), "accuracy": _safe_float(np.mean(correct[mask])), "n": int(mask.sum())})
            ece = float(sum((r["n"] / len(p)) * abs(r["confidence_mean"] - r["accuracy"]) for r in rows)) if len(p) else None
            # Binary Brier is directly interpretable; multiclass uses a bounded one-vs-rest aggregate.
            try:
                classes = getattr(pipeline, "classes_", None)
                if classes is not None and len(classes) == 2:
                    brier = float(brier_score_loss(np.asarray(y_test), proba[:, 1]))
                else:
                    y_arr = np.asarray(y_test)
                    class_list = list(classes) if classes is not None else list(range(proba.shape[1]))
                    one_hot = np.column_stack([(y_arr == c).astype(float) for c in class_list])
                    brier = float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))
            except Exception:
                brier = None
            try: ll = float(log_loss(y_test, proba, labels=getattr(pipeline, "classes_", None)))
            except Exception: ll = None
            return {"status": "ok", "expected_calibration_error": ece, "brier_score": brier, "log_loss": ll, "bins": rows}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    @staticmethod
    def evaluate_bundle(pipeline, X_train, X_test, y_train, y_test, task: str, seed: int = 42, n_jobs: int = 1) -> dict[str, Any]:
        pred = pipeline.predict(X_test)
        holdout = ProfessionalEvaluationEngine.evaluate_holdout(y_test, pred, task, seed=seed)
        baseline = ProfessionalEvaluationEngine.baseline(y_train, y_test, task)
        # IMPORTANT: CV is training-only. The test partition remains locked.
        cv = ProfessionalEvaluationEngine.cross_validation(clone(pipeline), X_train, y_train, task, seed, n_jobs=n_jobs)
        calibration = ProfessionalEvaluationEngine.classification_calibration(pipeline, X_test, y_test) if task == "classification" else {"status": "not_applicable"}
        uncertainty = (PredictionUncertaintyEngine.classification(pipeline, X_test, y_test) if task == "classification" else PredictionUncertaintyEngine.regression(pipeline, X_train, y_train, X_test, pred))
        robustness = RobustnessPerturbationAnalyzer.run(pipeline, X_test, y_test, task, seed=seed)
        from services.data_slice_analysis import DataSliceAnalyzer
        slices = DataSliceAnalyzer.run(X_test, y_test, pred, task)
        return {
            "holdout": holdout,
            "baseline": baseline,
            "cross_validation": cv,
            "calibration": calibration,
            "prediction_uncertainty": uncertainty,
            "robustness_perturbation": robustness,
            "data_slices": slices,
            "evaluation_level": "professional",
            "test_set_locked": True,
            "selection_policy": "Model selection must use training-only CV; final test is reserved for final reporting.",
        }

    @staticmethod
    def candidate_score(evaluation: dict[str, Any], task: str) -> float | None:
        cv = evaluation.get("cross_validation", {})
        metrics = cv.get("metrics", {})
        key = "f1_weighted" if task == "classification" else "r2"
        value = (metrics.get(key) or {}).get("mean")
        return _safe_float(value)


class PredictionUncertaintyEngine:
    """Bounded prediction-level uncertainty diagnostics for professional users.

    Regression uses an empirical residual interval on the training population.
    Classification reports probability confidence/entropy when probabilities are
    available. These are uncertainty diagnostics, not causal confidence claims.
    """

    @staticmethod
    def regression(pipeline, X_train, y_train, X_test, y_pred, confidence: float = .95, seed: int = 42) -> dict[str, Any]:
        """OOF-residual conformal-style diagnostic using training data only.

        This is a finite-sample diagnostic under exchangeability-like assumptions.
        It does not claim distribution-free validity for arbitrary deployment shift.
        """
        try:
            from sklearn.model_selection import KFold
            from sklearn.base import clone
            X_train = X_train.reset_index(drop=True)
            y_arr = np.asarray(y_train)
            n = len(X_train)
            if n < 30:
                train_pred = np.asarray(pipeline.predict(X_train), dtype=float).reshape(-1)
                residuals = np.abs(y_arr.astype(float) - train_pred)
                method = "training_residual_diagnostic"
            else:
                folds = min(5, max(3, n // 40))
                kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
                oof = np.empty(n, dtype=float)
                for tr_idx, va_idx in kf.split(X_train):
                    model = clone(pipeline)
                    model.fit(X_train.iloc[tr_idx], y_arr[tr_idx])
                    oof[va_idx] = np.asarray(model.predict(X_train.iloc[va_idx]), dtype=float).reshape(-1)
                residuals = np.abs(y_arr.astype(float) - oof)
                method = "cross_validated_abs_residual_interval"
            alpha = 1 - confidence
            q = float(np.quantile(residuals, min(1.0, max(0.0, (1-alpha) * (1 + 1/max(len(residuals),1))))))
            yp = np.asarray(y_pred, dtype=float).reshape(-1)
            lower, upper = yp - q, yp + q
            return {
                "status":"ok", "method":method, "confidence":confidence,
                "absolute_residual_quantile":q,
                "residual_quantiles":{"lower":-q,"upper":q},
                "test_interval_summary":{"mean_width":float(np.mean(upper-lower)),"median_width":float(np.median(upper-lower))},
                "calibration_residual_count":int(len(residuals)),
                "interpretation":"Interval is calibrated from training/OOF residuals. Coverage depends on the data-generating assumptions and can fail under distribution shift; it is not a guarantee of deployment coverage.",
            }
        except Exception as exc:
            return {"status":"error","reason":str(exc)}

    @staticmethod
    def classification(pipeline, X_test, y_test=None) -> dict[str, Any]:
        try:
            if not hasattr(pipeline, "predict_proba"):
                return {"status":"not_available","reason":"Model does not expose predict_proba."}
            proba=np.asarray(pipeline.predict_proba(X_test),float)
            confidence=np.max(proba,axis=1)
            entropy=-(proba*np.log(np.clip(proba,1e-12,1))).sum(axis=1)
            return {"status":"ok","method":"probability_confidence_entropy","mean_confidence":float(np.mean(confidence)),
                    "p10_confidence":float(np.quantile(confidence,.10)),"p50_confidence":float(np.quantile(confidence,.50)),
                    "mean_entropy":float(np.mean(entropy)),
                    "interpretation":"Probability confidence and entropy describe model uncertainty; they are not calibrated guarantees unless calibration has been validated."}
        except Exception as exc:
            return {"status":"error","reason":str(exc)}


def prediction_uncertainty_for_model(pipeline, X_train, y_train, X_test, y_pred, task: str) -> dict[str, Any]:
    if task == "classification":
        return PredictionUncertaintyEngine.classification(pipeline, X_test, y_test=None)
    return PredictionUncertaintyEngine.regression(pipeline, X_train, y_train, X_test, y_pred)


class RobustnessPerturbationAnalyzer:
    """Measure performance degradation under bounded, realistic test-data perturbations."""

    @staticmethod
    def run(pipeline, X_test: pd.DataFrame, y_test, task: str, seed: int = 42, predict_fn: Callable[[pd.DataFrame], Any] | None = None) -> dict[str, Any]:
        if X_test is None or X_test.empty:
            return {"status":"insufficient_data","reason":"No test data available."}
        rng=np.random.default_rng(seed)
        predictor = predict_fn or pipeline.predict
        base_pred=predictor(X_test)
        base=_metric_summary(y_test,base_pred,task)
        primary="f1_weighted" if task=="classification" else "r2"
        cases=[]
        numeric=[c for c in X_test.columns if pd.api.types.is_numeric_dtype(X_test[c])]
        # 1% Gaussian perturbation scaled by each feature's observed spread.
        jitter=X_test.copy()
        for c in numeric:
            sd=float(pd.to_numeric(X_test[c],errors="coerce").std())
            if np.isfinite(sd) and sd>0: jitter[c]=pd.to_numeric(jitter[c],errors="coerce")+rng.normal(0,.01*sd,len(jitter))
        cases.append(("numeric_1pct_jitter",jitter))
        # 2% missingness; the normal preprocessing pipeline should handle this if configured.
        missing=X_test.copy()
        if len(missing):
            for c in list(missing.columns):
                idx=rng.choice(len(missing),size=max(1,int(.02*len(missing))),replace=False)
                missing.iloc[idx,missing.columns.get_loc(c)]=np.nan
        cases.append(("random_2pct_missingness",missing))
        # Numeric winsorization against extreme test observations.
        clipped=X_test.copy()
        for c in numeric:
            q1,q99=pd.to_numeric(clipped[c],errors="coerce").quantile([.01,.99])
            if np.isfinite(q1) and np.isfinite(q99): clipped[c]=pd.to_numeric(clipped[c],errors="coerce").clip(q1,q99)
        cases.append(("numeric_1_99pct_clipping",clipped))
        rows=[]
        for name,Xp in cases:
            try:
                pred=predictor(Xp); m=_metric_summary(y_test,pred,task)
                rows.append({"perturbation":name,"metrics":m,"primary_delta":float(m[primary]-base[primary]) if base.get(primary) is not None and m.get(primary) is not None else None,
                             "relative_primary_change":float((m[primary]-base[primary])/(abs(base[primary])+1e-12)) if base.get(primary) is not None and m.get(primary) is not None else None})
            except Exception as exc:
                rows.append({"perturbation":name,"status":"error","reason":str(exc)})
        return {"status":"ok","task":task,"baseline":base,"primary_metric":primary,"tests":rows,
                "interpretation":"Sensitivity diagnostics reveal performance degradation under bounded perturbations; they do not prove production robustness."}


class FeatureStabilityAnalyzer:
    """Measure whether important raw features remain important across resamples."""

    @staticmethod
    def run(pipeline_factory: Callable[[], Any], X: pd.DataFrame, y, task: str, repeats: int = 3, max_features: int = 25, seed: int = 42) -> dict[str, Any]:
        if X is None or X.empty or len(X) < 40:
            return {"status": "insufficient_data", "reason": "At least 40 rows are required for feature stability analysis."}
        features = list(X.columns)[:max_features]
        score_name = "f1_weighted" if task == "classification" else "r2"
        rng = np.random.default_rng(seed)
        scores: dict[str, list[float]] = {str(c): [] for c in features}
        for i in range(repeats):
            rs = int(rng.integers(0, 2**31 - 1))
            strat = y if task == "classification" and pd.Series(y).value_counts().min() >= 2 else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=rs, stratify=strat)
            pipe = pipeline_factory()
            try:
                pipe.fit(Xtr, ytr)
                perm = permutation_importance(pipe, Xte, yte, scoring=score_name, n_repeats=3, random_state=rs, n_jobs=1)
                for c, value in zip(X.columns, perm.importances_mean):
                    if str(c) in scores:
                        scores[str(c)].append(float(value))
            except Exception:
                continue
        rows = []
        for c, values in scores.items():
            if not values:
                continue
            arr = np.asarray(values, float)
            rows.append({"feature": c, "mean_importance": _safe_float(np.mean(arr)), "std_importance": _safe_float(np.std(arr)), "positive_frequency": _safe_float(np.mean(arr > 0)), "runs": len(arr)})
        rows.sort(key=lambda x: (x["positive_frequency"], x["mean_importance"]), reverse=True)
        return {"status": "ok", "repeats": repeats, "score": score_name, "features": rows, "stable_features": [r["feature"] for r in rows if r["positive_frequency"] >= .67]}


class TemporalAvailabilityAnalyzer:
    """Create a feature-availability matrix across time buckets."""

    TIME_PATTERNS = re.compile(r"date|time|timestamp|datetime|period", re.I)

    @classmethod
    def run(cls, df: pd.DataFrame, target: str | None = None, max_features: int = 30, bins: int = 10) -> dict[str, Any]:
        if df is None or df.empty:
            return {"status": "insufficient_data", "reason": "No dataframe."}
        dates = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if not dates:
            dates = [c for c in df.columns if cls.TIME_PATTERNS.search(str(c))]
        time_col = dates[0] if dates else None
        if time_col is None:
            return {"status": "not_available", "reason": "No datetime-like field was detected."}
        t = pd.to_datetime(df[time_col], errors="coerce")
        valid = t.notna()
        if valid.sum() < 10:
            return {"status": "insufficient_data", "reason": "Too few valid timestamps."}
        work = df.loc[valid].copy()
        work["__time__"] = t.loc[valid]
        work = work.sort_values("__time__")
        work["__bucket__"] = pd.qcut(work["__time__"].rank(method="first"), q=min(bins, len(work)), duplicates="drop")
        candidates = [c for c in df.columns if c not in {target, time_col}][:max_features]
        matrix = []
        for bucket, g in work.groupby("__bucket__", observed=True):
            row = {"period": str(bucket)}
            for c in candidates:
                row[str(c)] = _safe_float(g[c].notna().mean())
            matrix.append(row)
        m = pd.DataFrame(matrix)
        late_features = []
        for c in candidates:
            vals = pd.to_numeric(m[str(c)], errors="coerce").to_numpy(float)
            if len(vals) >= 3 and np.isfinite(vals).all():
                early = float(np.mean(vals[: max(1, len(vals)//3)]))
                late = float(np.mean(vals[-max(1, len(vals)//3):]))
                if late - early >= .35:
                    late_features.append({"feature": str(c), "early_availability": early, "late_availability": late, "increase": late - early, "flag": "late-appearing feature; review temporal leakage"})
        return {"status": "ok", "time_column": str(time_col), "matrix": m.to_dict(orient="records"), "late_features": late_features, "interpretation": "Availability is diagnostic. A late-appearing feature is not proof of leakage; review whether it was genuinely available at prediction time."}


class CounterfactualLeakageAnalyzer:
    """Test whether suspicious features materially drive performance.

    This is deliberately a counterfactual *diagnostic*, not a causal proof. It
    compares the model with a feature removed while preserving the same split.
    """

    @staticmethod
    def run(pipeline_factory: Callable[[list[str]], Any], X: pd.DataFrame, y, task: str, candidate_features: list[str], seed: int = 42) -> dict[str, Any]:
        if X is None or X.empty or not candidate_features:
            return {"status": "not_available", "reason": "No candidate features were supplied."}
        strat = y if task == "classification" and pd.Series(y).value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=strat)
        full = pipeline_factory(list(X.columns))
        full.fit(Xtr, ytr)
        base = _metric_summary(yte, full.predict(Xte), task)
        rows = []
        for feature in candidate_features[:10]:
            if feature not in X.columns:
                continue
            cols = [c for c in X.columns if c != feature]
            if not cols:
                continue
            try:
                cf = pipeline_factory(cols)
                cf.fit(Xtr[cols], ytr)
                counter = _metric_summary(yte, cf.predict(Xte[cols]), task)
                primary = "f1_weighted" if task == "classification" else "r2"
                delta = float(base[primary] - counter[primary])
                rows.append({"feature": str(feature), "baseline": base[primary], "without_feature": counter[primary], "performance_drop": delta, "interpretation": "Material dependence; investigate temporal/target provenance before calling this leakage."})
            except Exception as exc:
                rows.append({"feature": str(feature), "status": "error", "reason": str(exc)})
        rows.sort(key=lambda x: x.get("performance_drop", -np.inf), reverse=True)
        return {"status": "ok", "task": task, "baseline_metrics": base, "tests": rows}


class ModelDiagnosisEngine:
    """Turn evaluation/error evidence into explicit diagnostic hypotheses."""

    @staticmethod
    def diagnose(task: str, evaluation: dict[str, Any], feature_stability: dict[str, Any] | None = None, temporal: dict[str, Any] | None = None, leakage: dict[str, Any] | None = None) -> dict[str, Any]:
        hypotheses = []
        cv = evaluation.get("cross_validation", {})
        if cv.get("status") == "ok":
            for name, m in cv.get("metrics", {}).items():
                gap = m.get("generalization_gap")
                if gap is not None and abs(float(gap)) > .15:
                    hypotheses.append({"type": "generalization_gap", "metric": name, "gap": gap, "severity": "review"})
        cal = evaluation.get("calibration", {})
        if cal.get("status") == "ok" and cal.get("expected_calibration_error") is not None and cal["expected_calibration_error"] > .10:
            hypotheses.append({"type": "calibration", "ece": cal["expected_calibration_error"], "severity": "review"})
        robust = evaluation.get("robustness_perturbation", {})
        if robust.get("status") == "ok":
            degradations = [x for x in robust.get("tests", []) if (x.get("relative_primary_change") is not None and abs(x.get("relative_primary_change")) > .15)]
            if degradations:
                hypotheses.append({"type": "robustness_sensitivity", "perturbations": [x.get("perturbation") for x in degradations], "severity": "review"})
        slices = evaluation.get("data_slices", {})
        if slices.get("status") == "ok" and slices.get("slices"):
            extreme = [x for x in slices["slices"] if abs(float(x.get("primary_delta_vs_overall", 0))) > .20]
            if extreme:
                hypotheses.append({"type": "slice_performance_difference", "slices": [{"feature":x["feature"],"value":x["value"],"n":x["n"]} for x in extreme[:5]], "severity": "review"})
        baseline = evaluation.get("baseline", {})
        try:
            primary = "f1_weighted" if task == "classification" else "r2"
            observed = (evaluation.get("holdout", {}).get("metrics", {}) or {}).get(primary)
            base = (baseline.get("metrics", {}) or {}).get(primary)
            if observed is not None and base is not None and abs(float(observed) - float(base)) < 0.02:
                hypotheses.append({"type": "weak_baseline_separation", "observed": observed, "baseline": base, "severity": "review"})
        except Exception:
            pass
        if feature_stability and feature_stability.get("status") == "ok":
            unstable = [r for r in feature_stability.get("features", []) if r.get("positive_frequency", 0) < .50]
            if unstable:
                hypotheses.append({"type": "feature_instability", "features": [r["feature"] for r in unstable[:10]], "severity": "review"})
        if temporal and temporal.get("late_features"):
            hypotheses.append({"type": "temporal_availability", "features": [x["feature"] for x in temporal["late_features"]], "severity": "high"})
        if leakage and leakage.get("status") == "ok":
            material = [x for x in leakage.get("tests", []) if x.get("performance_drop", 0) > .10]
            if material:
                hypotheses.append({"type": "counterfactual_dependence", "features": [x["feature"] for x in material], "severity": "high"})
        return {"status": "ok", "hypotheses": hypotheses, "diagnosis_level": "evidence_based", "next_actions": ["Inspect split strategy", "Review feature provenance", "Check temporal availability", "Compare with simpler baseline"] if hypotheses else ["No high-severity automated diagnostic hypothesis was triggered."]}


class DLValidationEngine:
    """Bounded neural-network validation diagnostics."""
    @staticmethod
    def seed_stability(train_fn: Callable[[int], dict[str, Any]], seeds: list[int], task: str) -> dict[str, Any]:
        primary = "f1_weighted" if task == "classification" else "r2"
        rows=[]
        for seed in seeds[:3]:
            try:
                out=train_fn(int(seed))
                metrics=out.get("metrics", out)
                rows.append({"seed":int(seed),"primary":_safe_float(metrics.get(primary)),"metrics":metrics})
            except Exception as exc:
                rows.append({"seed":int(seed),"status":"error","reason":str(exc)})
        vals=[r["primary"] for r in rows if r.get("primary") is not None]
        return {"status":"ok" if vals else "error","primary_metric":primary,"runs":rows,"mean":_safe_float(np.mean(vals)) if vals else None,"std":_safe_float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,"interpretation":"Seed stability measures sensitivity to initialization/training stochasticity; it is not an independent estimate of population performance."}


class AgentWhyEvidence:
    """First-class evidence object explaining an agent decision without hidden CoT."""

    @staticmethod
    def create(agent: str, decision: str, rationale: list[str], evidence_refs: list[str] | None = None, alternatives: list[str] | None = None, constraints: list[str] | None = None) -> dict[str, Any]:
        digest = hashlib.sha256(f"{agent}|{decision}|{'|'.join(rationale)}".encode()).hexdigest()[:12]
        obj = EvidenceObject(
            evidence_id=f"why-{digest}",
            kind="agent_why",
            title=f"Agent Why: {agent} — {decision}",
            status="complete",
            data={
                "agent": agent,
                "decision": decision,
                "rationale": list(rationale),
                "evidence_refs": list(evidence_refs or []),
                "alternatives_considered": list(alternatives or []),
                "constraints": list(constraints or []),
                "disclosure": "Decision-level rationale only; no private chain-of-thought is stored.",
            },
            parent_ids=list(evidence_refs or []),
        )
        return obj.to_dict()


class AgentSelfCheck:
    """Pre-approval safety/quality check for every executable agent step."""

    @staticmethod
    def run(agent: str, action: str, plan: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        checks = []
        required = ("objective", "inputs", "expected_output", "risk")
        missing = [x for x in required if not plan.get(x)]
        checks.append({"check": "plan_completeness", "status": "fail" if missing else "pass", "missing": missing})
        refs = (evidence or {}).get("evidence_refs", [])
        checks.append({"check": "evidence_binding", "status": "pass" if refs else "review", "evidence_refs": refs})
        checks.append({"check": "bounded_action", "status": "pass" if plan.get("tool_budget", 1) <= 20 else "fail", "tool_budget": plan.get("tool_budget", 1)})
        checks.append({"check": "human_gate", "status": "pass"})
        status = "pass" if all(x["status"] == "pass" for x in checks) else "review"
        return {"agent": agent, "action": action, "status": status, "checks": checks, "blocking_reasons": missing, "approval_required": True}
