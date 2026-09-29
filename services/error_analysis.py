from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


class ErrorAnalysisEngine:
    @staticmethod
    def classification(y_true, y_pred) -> dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred)
        errors = np.asarray(y_true) != np.asarray(y_pred)
        return {"type": "classification", "n": int(len(y_true)), "errors": int(errors.sum()),
                "error_rate": float(errors.mean()) if len(errors) else float("nan"),
                "confusion_matrix": cm.tolist(), "classes": [str(x) for x in np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))]}

    @staticmethod
    def regression(y_true, y_pred) -> dict[str, Any]:
        yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
        residual = yt - yp
        return {"type": "regression", "n": int(len(yt)), "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(mean_squared_error(yt, yp) ** .5), "r2": float(r2_score(yt, yp)),
                "residual_mean": float(np.mean(residual)), "residual_std": float(np.std(residual))}

    @staticmethod
    def slice_metrics(df: pd.DataFrame, y_true, y_pred, column: str, classification: bool = True) -> pd.DataFrame:
        work = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)})
        work[column] = df[column].to_numpy()[:len(work)]
        rows = []
        for value, group in work.groupby(column, dropna=False):
            if classification:
                rows.append({"slice": str(value), "n": len(group), "accuracy": float((group.y_true == group.y_pred).mean())})
            else:
                err = group.y_true.astype(float) - group.y_pred.astype(float)
                rows.append({"slice": str(value), "n": len(group), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(err**2)))})
        return pd.DataFrame(rows).sort_values("n", ascending=False)
