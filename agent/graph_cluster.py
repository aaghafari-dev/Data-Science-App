"""LangGraph specialist implementation for governed clustering analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.agent_memory import AgentMemory
from services.clustering_analysis import ClusteringAnalysisEngine


def run_clustering_step(df: pd.DataFrame, seed: int = 42, max_k: int = 8) -> dict[str, Any]:
    """Run the professional clustering service and return bounded evidence for the Master Agent."""
    try:
        result = ClusteringAnalysisEngine.run(df, k_min=2, k_max=max_k, seed=seed)
        if result.get("status") != "ok":
            return result
        selected = result.get("selected_method") or {}
        memory = AgentMemory("Clustering", 20)
        memory.remember("cluster_selection", {"algorithm": selected.get("algorithm"), "parameters": selected.get("parameters"), "metrics": selected.get("metrics", {})})
        result.update({
            "agent": "Clustering",
            "method": selected.get("algorithm", "Clustering Analysis"),
            "best_k": (selected.get("parameters") or {}).get("k"),
            "best_metrics": selected.get("metrics", {}),
            "memory": memory.to_dict(),
        })
        return result
    except Exception as exc:
        return {"status": "error", "agent": "Clustering", "message": str(exc)}
