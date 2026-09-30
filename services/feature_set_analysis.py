from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.base import clone


def _primary(y, pred, task):
    from sklearn.metrics import f1_score, r2_score
    if task == "classification": return float(f1_score(y,pred,average="weighted",zero_division=0))
    return float(r2_score(y,pred))


class FeatureSetChallenge:
    """Compare scientifically defensible feature sets under the same locked split."""
    @staticmethod
    def run(pipeline_factory: Callable[[list[str]], Any], X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, task: str, stable_features: list[str] | None = None, suspicious_features: list[str] | None = None) -> dict[str, Any]:
        if X_train is None or X_train.empty:
            return {"status":"insufficient_data"}
        all_features=list(X_train.columns)
        candidates={"all_eligible":all_features}
        if stable_features:
            stable=[c for c in stable_features if c in all_features]
            if len(stable)>=2: candidates["stable_features_only"]=stable
        if suspicious_features:
            restricted=[c for c in all_features if c not in set(suspicious_features)]
            if len(restricted)>=2: candidates["suspicious_features_excluded"]=restricted
        rows=[]
        for name,cols in candidates.items():
            try:
                model=pipeline_factory(cols); model.fit(X_train[cols],y_train); pred=model.predict(X_test[cols])
                rows.append({"feature_set":name,"n_features":len(cols),"primary_metric": "f1_weighted" if task=="classification" else "r2","primary_value":_primary(y_test,pred,task),"features":cols})
            except Exception as exc:
                rows.append({"feature_set":name,"status":"error","reason":str(exc),"n_features":len(cols)})
        return {"status":"ok","task":task,"sets":rows,"interpretation":"Feature-set challenge compares alternatives on the same locked test partition; it does not establish causal feature importance. Feature selection should be nested inside validation when used for model tuning."}
