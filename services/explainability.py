from __future__ import annotations

import numpy as np
import pandas as pd


def explain_model(model, X: pd.DataFrame, y=None, max_features: int = 20) -> dict:
    """SHAP when installed; otherwise permutation importance. Always returns explicit method metadata."""
    try:
        import shap
        explainer = shap.Explainer(model, X)
        values = explainer(X.iloc[: min(200, len(X))])
        arr = np.asarray(values.values)
        importance = np.abs(arr).mean(axis=0)
        if importance.ndim > 1: importance = importance.mean(axis=-1)
        names = list(X.columns)
        ranking = sorted(zip(names, importance.tolist()), key=lambda x: x[1], reverse=True)[:max_features]
        return {"method": "SHAP", "features": [{"feature": n, "importance": float(v)} for n, v in ranking]}
    except Exception as shap_error:
        try:
            from sklearn.inspection import permutation_importance
            if y is None: raise ValueError("y is required for permutation importance")
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            ranking = sorted(zip(X.columns, result.importances_mean), key=lambda x: x[1], reverse=True)[:max_features]
            return {"method": "Permutation Importance", "features": [{"feature": n, "importance": float(v)} for n, v in ranking],
                    "fallback_reason": str(shap_error)}
        except Exception as exc:
            return {"method": "Unavailable", "features": [], "error": str(exc), "shap_error": str(shap_error)}
