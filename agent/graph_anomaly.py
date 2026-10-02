"""Anomaly-detection specialist sub-agent for bounded unsupervised diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from services.agent_memory import AgentMemory


def run_anomaly_step(df: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Run a bounded Isolation Forest analysis and return anomaly evidence."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available for anomaly detection."}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        return {"status": "error", "message": "Anomaly detection requires at least two numerical variables."}
    X = df[numeric].replace([np.inf, -np.inf], np.nan)
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    contamination = min(0.10, max(0.01, 10 / max(len(df), 100)))
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=seed)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)
    anomaly_count = int((labels == -1).sum())
    memory = AgentMemory("Anomaly", 20)
    memory.remember("anomaly_summary", {"count": anomaly_count, "fraction": anomaly_count / len(df)})
    return {
        "status": "ok",
        "agent": "Anomaly Detection",
        "method": "IsolationForest",
        "features": numeric,
        "contamination_assumption": contamination,
        "anomaly_count": anomaly_count,
        "anomaly_fraction": float(anomaly_count / len(df)),
        "score_summary": {"min": float(scores.min()), "median": float(np.median(scores)), "max": float(scores.max())},
        "interpretation_note": "Anomaly flags are model-dependent diagnostics and require domain review before being treated as true anomalies.",
        "memory": memory.to_dict(),
    }
