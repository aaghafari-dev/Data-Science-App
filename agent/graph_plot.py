from __future__ import annotations

from typing import Any, TypedDict
import json
import numpy as np
import pandas as pd

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from services.agent_memory import AgentMemory


class PlotAgentState(TypedDict, total=False):
    dataframe: pd.DataFrame
    target: str | None
    ml_results: dict[str, Any] | None
    dl_results: dict[str, Any] | None
    memory: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    plots: list[dict[str, Any]]
    status: str
    summary: str
    messages: list[BaseMessage]


def _numeric(df):
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical(df):
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            nun = int(s.nunique(dropna=True))
            if 2 <= nun <= 50:
                out.append(str(c))
    return out


def _datetime(df):
    return [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _model_comparison(ml_results, dl_results):
    rows = []
    for bundle in (ml_results, dl_results):
        if not bundle or bundle.get("status") != "ok":
            continue
        agent = bundle.get("agent", "Model")
        for name, result in bundle.get("models", {}).items():
            for metric, value in result.get("metrics", {}).items():
                try:
                    rows.append({"Agent": agent, "Model": name, "Metric": metric, "Value": float(value)})
                except Exception:
                    pass
    return pd.DataFrame(rows)


def _plan_node(state: PlotAgentState):
    df = state.get("dataframe")
    if df is None or df.empty:
        return {"status": "error", "summary": "No active dataset is available for Agent Plot."}
    nums, cats, dates = _numeric(df), _categorical(df), _datetime(df)
    target = state.get("target") if state.get("target") in df.columns else None
    plan: list[dict[str, Any]] = []

    # 1) Correlation heatmap is a high-value global diagnostic when enough numeric data exist.
    if len(nums) >= 2:
        plan.append({"kind": "correlation_heatmap", "fields": nums[:20],
                     "title": "Correlation Heatmap", "rationale": "Assess linear relationships, redundancy and potential leakage signals among numeric variables.", "priority": 100})

    # 2) Strongest numeric pair for an actual relationship plot.
    if len(nums) >= 2:
        corr = df[nums].corr(numeric_only=True).abs().where(lambda x: ~np.eye(len(x), dtype=bool))
        if not corr.empty and np.isfinite(corr.to_numpy()).any():
            stacked = corr.stack().dropna().sort_values(ascending=False)
            if not stacked.empty:
                a, b = stacked.index[0]
                if a != b:
                    plan.append({"kind": "scatter", "fields": [a, b], "title": f"{b} vs. {a}",
                                 "rationale": f"Strongest observed absolute Pearson correlation ({corr.loc[a,b]:.3f}) among numeric variables.", "priority": 95})

    # 3) Target distribution / target-vs-feature.
    if target:
        if pd.api.types.is_numeric_dtype(df[target]):
            plan.append({"kind": "distribution", "fields": [target], "title": f"Distribution of {target}",
                         "rationale": "Inspect target range, concentration and potential outliers before modelling.", "priority": 90})
            if nums:
                candidates = [c for c in nums if c != target]
                if candidates:
                    c = max(candidates, key=lambda x: abs(df[[x, target]].corr().iloc[0,1]) if df[[x,target]].notna().all(axis=1).sum() > 2 else 0)
                    plan.append({"kind": "target_scatter", "fields": [c, target], "title": f"{target} vs. {c}",
                                 "rationale": "Inspect a high-information feature/target relationship without claiming causality.", "priority": 88})
        else:
            plan.append({"kind": "count", "fields": [target], "title": f"Target Class Distribution — {target}",
                         "rationale": "Inspect class balance and rare target categories.", "priority": 90})

    # 4) Time trend.
    if dates and nums:
        plan.append({"kind": "time_series", "fields": [dates[0], nums[0]], "title": f"{nums[0]} over {dates[0]}",
                     "rationale": "Inspect temporal trend, seasonality-like structure and abrupt changes.", "priority": 85})

    # 5) Data scientist-style category/measure comparison. Prefer City when available.
    cat_priority = sorted(cats, key=lambda c: (0 if c.lower() in {"city", "region", "country", "category", "segment"} else 1, df[c].nunique()))
    if cat_priority and nums:
        c = cat_priority[0]
        m = target if target and pd.api.types.is_numeric_dtype(df[target]) else nums[0]
        plan.append({"kind": "category_bar", "fields": [c, m], "title": f"{m} by {c}",
                     "rationale": "Compare an important numeric measure across a useful categorical dimension.", "priority": 80})

    # 6) General categorical distribution if there is room.
    if cats:
        c = cat_priority[0]
        plan.append({"kind": "count", "fields": [c], "title": f"Record Distribution — {c}",
                     "rationale": "Check category prevalence and possible imbalance.", "priority": 70})

    # Deduplicate by kind/fields and keep a compact, useful plan.
    seen = set(); unique = []
    for p in sorted(plan, key=lambda x: -x.get("priority", 0)):
        key = (p["kind"], tuple(p["fields"]))
        if key not in seen:
            seen.add(key); unique.append(p)
    unique = unique[:6]
    mem = AgentMemory("Plot", 30)
    mem.remember("plot_plan", [{"kind": p["kind"], "fields": p["fields"]} for p in unique])
    return {"plan": unique, "memory": mem.to_dict(), "status": "planned",
            "messages": [AIMessage(content=f"Agent Plot selected {len(unique)} data-driven analytical plots.")]}


def _render_node(state: PlotAgentState):
    df = state.get("dataframe")
    plots = []
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as exc:
        return {"status": "ok", "plots": [{"kind": "plan_only", "title": p["title"], "fields": p["fields"], "rationale": p["rationale"]} for p in state.get("plan", [])],
                "summary": f"Analytical plot plan created; Plotly renderer unavailable: {exc}",
                "messages": [AIMessage(content="Plot plan created without interactive rendering.")]}

    for p in state.get("plan", []):
        try:
            kind, fields = p["kind"], p["fields"]
            fig = None
            if kind == "correlation_heatmap":
                corr = df[fields].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", title=p["title"], color_continuous_scale="RdBu_r")
            elif kind in {"scatter", "target_scatter"}:
                work = df[[fields[0], fields[1]]].dropna()
                fig = px.scatter(work, x=fields[0], y=fields[1], trendline="ols" if len(work) >= 5 else None, title=p["title"])
            elif kind == "distribution":
                fig = px.histogram(df, x=fields[0], marginal="box", title=p["title"])
            elif kind == "time_series":
                work = df[[fields[0], fields[1]]].dropna().sort_values(fields[0])
                fig = px.line(work, x=fields[0], y=fields[1], title=p["title"])
            elif kind == "category_bar":
                work = df[[fields[0], fields[1]]].dropna().groupby(fields[0], as_index=False)[fields[1]].sum().sort_values(fields[1], ascending=False).head(30)
                fig = px.bar(work, x=fields[0], y=fields[1], title=p["title"], text_auto=".3s")
            elif kind == "count":
                work = df[fields[0]].astype(str).value_counts().head(30).rename_axis(fields[0]).reset_index(name="Count")
                fig = px.bar(work, x=fields[0], y="Count", title=p["title"], text_auto=True)
            if fig is None:
                continue
            fig.update_layout(template="plotly_white", margin=dict(l=60, r=30, t=70, b=70))
            insights = []
            if kind == "category_bar" and len(df):
                top = df.groupby(fields[0], dropna=False)[fields[1]].sum().sort_values(ascending=False).head(3)
                insights.append("Top categories by aggregated measure: " + ", ".join(f"{k} ({v:.3g})" for k,v in top.items()))
            if kind in {"scatter", "target_scatter"}:
                c = df[[fields[0], fields[1]]].corr().iloc[0,1]
                if np.isfinite(c): insights.append(f"Pearson correlation = {c:.3f}; this is an association, not a causal conclusion.")
            plots.append({"kind": kind, "title": p["title"], "fields": fields, "rationale": p["rationale"],
                          "insights": insights, "figure_json": fig.to_json()})
        except Exception as exc:
            plots.append({"kind": p["kind"], "title": p["title"], "fields": p["fields"], "rationale": p["rationale"], "error": str(exc)})

    # Model comparison is supplemental, not the main purpose of Agent Plot.
    model_df = _model_comparison(state.get("ml_results"), state.get("dl_results"))
    if not model_df.empty:
        metric = "R2" if "R2" in set(model_df["Metric"]) else model_df["Metric"].iloc[0]
        md = model_df[model_df["Metric"] == metric]
        fig = px.bar(md, x="Model", y="Value", color="Agent", text="Value", title=f"Model Comparison — {metric}", barmode="group")
        fig.update_layout(template="plotly_white")
        plots.append({"kind":"model_comparison", "title":f"Model Comparison — {metric}", "fields":["Model",metric],
                      "rationale":"Supplemental comparison of recorded model evaluation metrics.", "insights":[], "figure_json":fig.to_json()})

    summary = f"Agent Plot generated {len(plots)} evidence-linked analytical visualizations from the active Data Management dataset, plus model comparison when model results were available."
    return {"status":"ok", "agent":"Plot", "plots":plots, "summary":summary,
            "memory":state.get("memory",[]), "messages":[AIMessage(content=summary)]}


def _finish(state):
    return state


workflow = StateGraph(PlotAgentState)
workflow.add_node("plan", _plan_node)
workflow.add_node("render", _render_node)
workflow.set_entry_point("plan")
workflow.add_edge("plan", "render")
workflow.add_edge("render", END)
agent_plot_app = workflow.compile()


def run_plot_agent(df: pd.DataFrame, ml_results: dict | None = None, dl_results: dict | None = None,
                   target: str | None = None, chart_type: str = "auto", memory=None) -> dict[str, Any]:
    """Compatibility facade. LangGraph performs the actual plot planning and rendering."""
    state = {"dataframe": df, "ml_results": ml_results, "dl_results": dl_results, "target": target,
             "memory": memory or []}
    return agent_plot_app.invoke(state, config={"configurable": {"thread_id": "agent-plot"}})
