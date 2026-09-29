from __future__ import annotations

from typing import Any
import pandas as pd

from core.viz_engine import VizEngine
from services.agent_memory import AgentMemory


def run_plot_agent(df: pd.DataFrame, ml_results: dict | None = None, dl_results: dict | None = None,
                   target: str | None = None, chart_type: str = "auto") -> dict[str, Any]:
    """Agent Plot turns ML/DL results into an auditable results table and Plotly figure."""
    rows = []
    for bundle in (ml_results, dl_results):
        if not bundle or bundle.get("status") != "ok":
            continue
        agent = bundle.get("agent", "Model")
        for name, result in bundle.get("models", {}).items():
            for metric, value in result.get("metrics", {}).items():
                rows.append({"Agent": agent, "Model": name, "Metric": metric, "Value": value})
    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return {"status": "error", "message": "No ML/DL results are available for plotting."}
    metric = result_df["Metric"].iloc[0]
    fig = None
    try:
        import plotly.express as px
        plot_df = result_df[result_df["Metric"] == metric].copy()
        fig = px.bar(plot_df, x="Model", y="Value", color="Agent", text="Value",
                     title=f"Model comparison — {metric}", barmode="group")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(template="plotly_white", yaxis_title=metric)
    except Exception as exc:
        return {"status": "ok", "agent": "Plot", "table": result_df, "figure": None,
                "summary": f"Plot data prepared; interactive renderer unavailable: {exc}"}
    return {"status": "ok", "agent": "Plot", "table": result_df, "figure": fig,
            "summary": f"Agent Plot compared ML/DL model metrics using {metric}.", "memory": AgentMemory("Plot",20).to_dict()}
