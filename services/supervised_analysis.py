"""Module duty: Supervised analysis.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

"""Professional end-to-end supervised learning workspaces.

All preprocessing lives inside sklearn Pipelines, so imputation, encoding and
scaling are fitted inside each training fold.  The final test partition is
never used for hyperparameter selection.
"""

from dataclasses import dataclass
from typing import Any
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV,
    cross_validate, cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from services.mlp_training import make_mlp, fit_search_with_convergence_audit, MLPPolicy


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str


def _ohe():
    """Perform the ohe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prepare_features(X: pd.DataFrame, scaling: str = "standard") -> ColumnTransformer:
    """Perform the prepare features operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    numeric = list(X.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in X.columns if c not in numeric]
    scale = StandardScaler() if scaling == "standard" else MinMaxScaler() if scaling == "minmax" else "passthrough"
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale != "passthrough":
        num_steps.append(("scaler", scale))
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _ohe())]
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline(cat_steps), categorical),
    ], remainder="drop")


def _infer_task(y: pd.Series) -> str:
    """Perform the infer task operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if y.dtype == object or str(y.dtype).startswith("category") or y.nunique(dropna=True) <= max(10, min(20, int(len(y) * .05))):
        return "classification"
    return "regression"


def _split(df: pd.DataFrame, target: str, test_size: float = .2, seed: int = 42) -> SplitResult:
    """Perform the split operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    work = df.dropna(subset=[target]).copy()
    y = work[target]
    task = _infer_task(y)
    strat = y.astype(str) if task == "classification" and y.value_counts().min() >= 2 else None
    tr, te = train_test_split(work, test_size=test_size, random_state=seed, stratify=strat)
    return SplitResult(tr, te, "stratified" if strat is not None else "random")


def _cv(task: str, y, folds: int, seed: int):
    """Perform the cv operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if task == "classification":
        counts = pd.Series(y).value_counts()
        folds = min(folds, int(counts.min()))
        if folds < 2:
            raise ValueError("At least two observations per class are required for cross-validation.")
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _classification_metrics(y, pred, proba=None):
    """Perform the classification metrics operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classes": [str(x) for x in sorted(pd.Series(y).astype(str).unique())],
    }
    if proba is not None and np.asarray(proba).ndim == 2:
        try:
            classes = np.asarray(sorted(pd.Series(y).astype(str).unique()))
            if proba.shape[1] == 2:
                y_binary=(pd.Series(y).astype(str)==classes[1]).astype(int)
                out["roc_auc"] = float(roc_auc_score(y_binary, proba[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_binary, proba[:, 1]))
                out["brier_score"] = float(brier_score_loss(y_binary, proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(pd.get_dummies(pd.Series(y).astype(str)), proba, multi_class="ovr", average="macro"))
            out["log_loss"] = float(log_loss(y, proba))
        except Exception:
            pass
    return out


def _regression_metrics(y, pred):
    """Perform the regression metrics operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    yv, pv = np.asarray(y, float), np.asarray(pred, float)
    return {
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": float(np.sqrt(mean_squared_error(yv, pv))),
        "r2": float(r2_score(yv, pv)),
        "median_absolute_error": float(median_absolute_error(yv, pv)),
        "mean_bias_error": float(np.mean(pv - yv)),
        "mape_percent": float(np.mean(np.abs((yv - pv) / np.where(np.abs(yv) < 1e-12, np.nan, yv))) * 100) if np.any(np.abs(yv) >= 1e-12) else None,
    }


def _feature_screening(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Non-destructive leakage/feature screening for the professional workspaces."""
    try:
        from services.target_feature_selection import recommend_targets_and_features
        rec=recommend_targets_and_features(df)
        candidates=rec.get("feature_candidates",[])
        flagged=[]
        for item in candidates:
            if isinstance(item,dict):
                reasons=" ".join(item.get("reasons",[])).lower()
                if any(k in reasons for k in ("leakage","future","near-perfect","power-")):
                    flagged.append({"column":item.get("column"),"reasons":item.get("reasons",[])})
        return {"status":"ok","target":target,"flagged_features":flagged[:20],"policy":"flag for human review; never silently drop a feature without provenance/availability review."}
    except Exception as exc:
        return {"status":"review","reason":str(exc)}


class ClassificationAnalysisEngine:
    METHODS = ["Logistic Regression", "SVM", "Random Forest", "Decision Tree", "KNN", "Naive Bayes", "Neural Network"]

    @staticmethod
    def run(df: pd.DataFrame, target: str, test_size=.2, folds=5, seed=42, search="grid", scaling="standard", pca=False) -> dict[str, Any]:
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")
        split = _split(df, target, test_size, seed)
        Xtr, ytr = split.train.drop(columns=[target]), split.train[target].astype(str)
        Xte, yte = split.test.drop(columns=[target]), split.test[target].astype(str)
        label_encoder = LabelEncoder(); label_encoder.fit(ytr.astype(str))
        unseen = sorted(set(yte.astype(str)) - set(label_encoder.classes_))
        if unseen: raise ValueError(f"Test partition contains classes absent from training data: {unseen}. Use a stratified split with sufficient samples per class.")
        ytr_encoded, yte_encoded = label_encoder.transform(ytr), label_encoder.transform(yte)
        pre = _prepare_features(Xtr, scaling=scaling)
        if pca:
            pre_steps = [("pre", pre), ("pca", PCA(n_components=.95, svd_solver="full"))]
        else:
            pre_steps = [("pre", pre)]
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=5000), {"model__C": [.1, 1, 10]}),
            "SVM": (SVC(probability=True), {"model__C": [.5, 1, 10], "model__kernel": ["rbf", "linear"]}),
            "Random Forest": (RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=1), {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=seed), {"model__max_depth": [None, 4, 8, 16]}),
            "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 9], "model__weights": ["uniform", "distance"]}),
            "Naive Bayes": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
            "Neural Network": (make_mlp("classification", seed), {"model__alpha": [1e-5, 1e-4, 1e-3]}),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = (XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=seed, n_jobs=1), {"model__max_depth":[3,6],"model__learning_rate":[.03,.1]})
        except Exception: pass
        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"] = (LGBMClassifier(n_estimators=200, random_state=seed, verbosity=-1), {"model__num_leaves":[15,31],"model__learning_rate":[.03,.1]})
        except Exception: pass
        cv = _cv("classification", ytr_encoded, folds, seed)
        results = {}
        for name, (est, params) in models.items():
            pipe = Pipeline(pre_steps + [("model", est)])
            t0 = time.time()
            searcher = GridSearchCV(pipe, params, scoring="f1_weighted", cv=cv, n_jobs=1, refit=True, return_train_score=True)
            convergence = fit_search_with_convergence_audit(searcher, Xtr, ytr_encoded)
            pred_encoded = searcher.predict(Xte)
            pred = label_encoder.inverse_transform(np.asarray(pred_encoded, dtype=int))
            proba = searcher.predict_proba(Xte) if hasattr(searcher, "predict_proba") else None
            results[name] = {
                "metrics": _classification_metrics(yte, pred, proba),
                "cv_best_score": float(searcher.best_score_),
                "best_params": searcher.best_params_,
                "fit_seconds": round(time.time() - t0, 3),
                "estimator": searcher.best_estimator_,
                "convergence": convergence,
            }
        best = max(results, key=lambda k: results[k]["cv_best_score"])
        screening = _feature_screening(df, target)
        return {
            "status": "ok", "analysis": "Classification Analysis", "target": target, "task": "classification",
            "validation": {"strategy": split.strategy, "test_size": test_size, "folds": folds, "selection_population": "training_partition_only", "test_locked": True},
            "preprocessing": {"imputation": "median numeric / most_frequent categorical", "encoding": "OneHotEncoder(handle_unknown='ignore')", "scaling": scaling, "pca": pca, "fit_scope": "inside CV training folds"},
            "methods": {k: {kk: vv for kk, vv in v.items() if kk != "estimator"} for k, v in results.items()},
            "_estimators": {k: v["estimator"] for k,v in results.items()},
            "best_model": best, "best_test_metrics": results[best]["metrics"],
            "selection_metric": "weighted F1 on training-only cross-validation",
            "deployment": {"artifact_versioning": "Model Registry / Experiment Registry", "monitoring": "Model Monitoring / Drift"},
            "class_distribution": ytr.value_counts().to_dict(),
            "feature_screening": screening,
            "label_encoding": {"classes": [str(x) for x in label_encoder.classes_], "fit_scope": "training partition labels only"},
        }


class RegressionAnalysisEngine:
    METHODS = ["OLS", "Ridge", "Lasso", "Elastic Net", "Polynomial Regression", "SVR", "Decision Tree", "Random Forest", "Gradient Boosting", "Neural Network"]

    @staticmethod
    def run(df: pd.DataFrame, target: str, test_size=.2, folds=5, seed=42, scaling="standard", pca=False) -> dict[str, Any]:
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")
        split = _split(df, target, test_size, seed)
        Xtr, ytr = split.train.drop(columns=[target]), pd.to_numeric(split.train[target], errors="coerce")
        Xte, yte = split.test.drop(columns=[target]), pd.to_numeric(split.test[target], errors="coerce")
        valid_tr, valid_te = ytr.notna(), yte.notna(); Xtr, ytr = Xtr.loc[valid_tr], ytr.loc[valid_tr]; Xte, yte = Xte.loc[valid_te], yte.loc[valid_te]
        pre = _prepare_features(Xtr, scaling=scaling)
        models = {
            "OLS": (LinearRegression(), {}),
            "Ridge": (Ridge(), {"model__alpha": [.01, .1, 1, 10]}),
            "Lasso": (Lasso(max_iter=20000,tol=1e-4), {"model__alpha": [.0001, .001, .01, .1]}),
            "Elastic Net": (ElasticNet(max_iter=20000,tol=1e-4), {"model__alpha": [.001, .01, .1], "model__l1_ratio": [.2, .5, .8]}),
            "Polynomial Regression": (Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("linear", Ridge())]), {"model__linear__alpha": [.1, 1, 10]}),
            "SVR": (SVR(), {"model__C": [.5, 1, 10], "model__epsilon": [.05, .1, .2]}),
            "Decision Tree": (DecisionTreeRegressor(random_state=seed), {"model__max_depth": [None, 5, 10, 20]}),
            "Random Forest": (RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=1), {"model__max_depth": [None, 10, 20]}),
            "Gradient Boosting": (GradientBoostingRegressor(random_state=seed), {"model__n_estimators": [100, 200], "model__learning_rate": [.03, .1], "model__max_depth": [2, 3]}),
            "Neural Network": (make_mlp("regression", seed), {"model__alpha": [1e-5, 1e-4, 1e-3]}),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = (XGBRegressor(n_estimators=200, random_state=seed, n_jobs=1, objective="reg:squarederror"), {"model__max_depth":[3,6],"model__learning_rate":[.03,.1]})
        except Exception: pass
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = (LGBMRegressor(n_estimators=200, random_state=seed, verbosity=-1), {"model__num_leaves":[15,31],"model__learning_rate":[.03,.1]})
        except Exception: pass
        cv = _cv("regression", ytr, folds, seed)
        results = {}
        for name, (est, params) in models.items():
            # Polynomial model already owns an inner pipeline; prepend preprocessing.
            pipe = Pipeline([("pre", pre), ("model", est)])
            t0 = time.time()
            searcher = GridSearchCV(pipe, params, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1, refit=True, return_train_score=True)
            convergence = fit_search_with_convergence_audit(searcher, Xtr, ytr)
            pred = searcher.predict(Xte)
            results[name] = {"metrics": _regression_metrics(yte, pred), "cv_best_score": float(searcher.best_score_), "best_params": searcher.best_params_, "fit_seconds": round(time.time()-t0,3), "estimator": searcher.best_estimator_, "convergence": convergence}
        best = min(results, key=lambda k: results[k]["metrics"]["rmse"])
        screening = _feature_screening(df, target)
        return {
            "status":"ok", "analysis":"Regression Analysis", "target":target, "task":"regression",
            "validation":{"strategy":split.strategy,"test_size":test_size,"folds":folds,"selection_population":"training_partition_only","test_locked":True},
            "preprocessing":{"imputation":"median numeric / most_frequent categorical","encoding":"OneHotEncoder(handle_unknown='ignore')","scaling":scaling,"pca":pca,"fit_scope":"inside CV training folds"},
            "methods":{k:{kk:vv for kk,vv in v.items() if kk!="estimator"} for k,v in results.items()},
            "_estimators": {k:v["estimator"] for k,v in results.items()},
            "best_model":best,"best_test_metrics":results[best]["metrics"],
            "feature_screening":screening,
            "selection_metric":"negative RMSE on training-only cross-validation","deployment":{"artifact_versioning":"Model Registry / Experiment Registry","monitoring":"Model Monitoring / Drift"},
        }
