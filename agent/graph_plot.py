"""Module duty: Graph plot.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any, TypedDict
import json
import numpy as np
import pandas as pd

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from services.agent_memory import AgentMemory
from services.llm_config import provider_from_state, get_llm
from services.serialization import json_safe


class PlotAgentState(TypedDict, total=False):
    dataframe: pd.DataFrame
    target: str | None
    ml_results: dict[str, Any] | None
    dl_results: dict[str, Any] | None
    memory: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    plots: list[dict[str, Any]]
    plot_quantities: dict[str, Any]
    llm_reasoning: str
    status: str
    summary: str
    messages: list[BaseMessage]
    llm_provider: str
    model_name: str
    api_key: str
    local_path: str


def _numeric(df):
    """Perform the numeric operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _categorical(df):
    """Perform the categorical operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    out=[]
    for c in df.columns:
        s=df[c]
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype,pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            nun=int(s.nunique(dropna=True))
            if 2 <= nun <= 50: out.append(str(c))
    return out

def _datetime(df):
    """Perform the datetime operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _model_comparison(ml_results, dl_results):
    """Perform the model comparison operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    rows=[]
    for bundle in (ml_results,dl_results):
        if not bundle or bundle.get("status")!="ok": continue
        agent=bundle.get("agent","Model")
        for name,result in bundle.get("models",{}).items():
            for metric,value in result.get("metrics",{}).items():
                try: rows.append({"Agent":agent,"Model":name,"Metric":metric,"Value":float(value)})
                except Exception: pass
    return pd.DataFrame(rows)


def _analytical_quantities(df, target, ml, dl):
    """Perform the analytical quantities operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    nums=_numeric(df); cats=_categorical(df); dates=_datetime(df)
    corr_pairs=[]
    if len(nums)>=2:
        c=df[nums].corr(numeric_only=True).abs()
        pairs=[]
        for i,a in enumerate(nums):
            for b in nums[i+1:]:
                v=c.loc[a,b]
                if np.isfinite(v): pairs.append((float(v),a,b))
        for v,a,b in sorted(pairs,reverse=True)[:8]: corr_pairs.append({"x":a,"y":b,"abs_correlation":v})
    category_measures=[]
    for c in cats[:10]:
        if nums:
            m=target if target in nums else nums[0]
            category_measures.append({"dimension":c,"measure":m,"cardinality":int(df[c].nunique(dropna=True))})
    target_stats={}
    if target in df.columns:
        s=df[target]
        if pd.api.types.is_numeric_dtype(s):
            target_stats={"type":"numeric","mean":float(s.mean()),"median":float(s.median()),"std":float(s.std()),"missing_pct":float(s.isna().mean()*100)}
        else:
            target_stats={"type":"categorical","classes":int(s.nunique(dropna=True)),"missing_pct":float(s.isna().mean()*100)}
    return {"numeric_columns":nums[:30],"categorical_columns":cats[:20],"datetime_columns":dates[:10],"strong_numeric_pairs":corr_pairs,"category_measure_candidates":category_measures,"target":target,"target_summary":target_stats,
            "model_evidence":json_safe({"ML":ml,"DL":dl})}


def _deterministic_plan(df, target, ml, dl):
    """Perform the deterministic plan operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    nums,cats,dates=_numeric(df),_categorical(df),_datetime(df); plan=[]
    if len(nums)>=2: plan.append({"kind":"correlation_heatmap","fields":nums[:20],"title":"Correlation Heatmap","quantity":"pairwise Pearson correlation","rationale":"Assess global numeric relationships, redundancy and possible leakage signals.","priority":100})
    if len(nums)>=2:
        c=df[nums].corr(numeric_only=True).abs().where(lambda x:~np.eye(len(x),dtype=bool)); stacked=c.stack().dropna().sort_values(ascending=False)
        if not stacked.empty:
            a,b=stacked.index[0]; plan.append({"kind":"scatter","fields":[a,b],"title":f"{b} vs. {a}","quantity":"paired observations + Pearson correlation","rationale":f"Inspect the strongest observed numeric association (|r|={c.loc[a,b]:.3f}).","priority":95})
    if target:
        if target in nums:
            plan.append({"kind":"distribution","fields":[target],"title":f"Distribution of {target}","quantity":"count / density / outlier range","rationale":"Inspect target distribution and outlier structure before interpreting model performance.","priority":92})
            cand=[c for c in nums if c!=target]
            if cand:
                c=max(cand,key=lambda x: abs(df[[x,target]].corr().iloc[0,1]) if df[[x,target]].dropna().shape[0]>2 else 0)
                plan.append({"kind":"target_scatter","fields":[c,target],"title":f"{target} vs. {c}","quantity":"target-feature association","rationale":"Show the most informative numeric feature/target relationship without implying causality.","priority":90})
        elif target in df.columns:
            plan.append({"kind":"count","fields":[target],"title":f"Target Class Distribution — {target}","quantity":"class counts and proportions","rationale":"Check class balance and rare categories before classification interpretation.","priority":92})
    if dates and nums:
        plan.append({"kind":"time_series","fields":[dates[0],target if target in nums else nums[0]],"title":f"{target if target in nums else nums[0]} over {dates[0]}","quantity":"time trend / temporal variability","rationale":"Inspect temporal structure and abrupt changes relevant to validation and availability.","priority":85})
    if cats and nums:
        preferred=sorted(cats,key=lambda c:(0 if c.lower() in {"city","region","country","category","segment"} else 1,df[c].nunique()))[0]
        m=target if target in nums else nums[0]
        plan.append({"kind":"category_bar","fields":[preferred,m],"title":f"{m} by {preferred}","quantity":"grouped sum/mean and ranking","rationale":"Compare an important measure across a useful categorical dimension.","priority":83})
    if cats:
        c=cats[0]; plan.append({"kind":"count","fields":[c],"title":f"Record Distribution — {c}","quantity":"category counts","rationale":"Check prevalence and imbalance in an important dimension.","priority":70})
    seen=set(); out=[]
    for p in sorted(plan,key=lambda x:-x.get("priority",0)):
        k=(p["kind"],tuple(p["fields"]))
        if k not in seen: seen.add(k); out.append(p)
    return out[:6]


def _llm_refine_plan(state, deterministic, quantities):
    """Perform the llm refine plan operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    cfg=provider_from_state(state); llm=get_llm(cfg)
    if llm is None: return deterministic, "LLM refinement unavailable; deterministic analytical planner retained."
    allowed=[{"kind":p["kind"],"fields":p["fields"],"quantity":p["quantity"],"title":p["title"]} for p in deterministic]
    prompt=("You are the Plot Agent of a professional data-analysis application. "
            "Use the supplied Data Scientist evidence and analytical quantities. Return JSON only with a key 'plots'. "
            "Select 3-6 plots from the allowed candidates; do not invent columns or metrics. "
            "Prefer plots that answer the analytical question, expose important distributions/relationships, "
            "and support model diagnostics. Preserve fields from the candidates.\n\n"
            f"Quantities:\n{json.dumps(json_safe(quantities),indent=2)[:30000]}\n\nAllowed candidates:\n{json.dumps(allowed,indent=2)}")
    try:
        response=llm.invoke([HumanMessage(content=prompt)])
        text=response.content if hasattr(response,"content") else str(response)
        start=text.find("{"); end=text.rfind("}")
        obj=json.loads(text[start:end+1]) if start>=0 and end>start else {}
        requested=obj.get("plots",[])
        lookup={(p["kind"],tuple(p["fields"])):p for p in deterministic}
        refined=[]
        for r in requested:
            key=(r.get("kind"),tuple(r.get("fields",[])))
            if key in lookup: refined.append({**lookup[key],"llm_priority":len(refined)})
        if refined: return refined[:6], text[:5000]
    except Exception as exc:
        return deterministic, f"LLM refinement failed safely: {exc}"
    return deterministic, "LLM returned no valid candidate selection; deterministic plan retained."


def _plan_node(state: PlotAgentState):
    """Perform the plan node operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df=state.get("dataframe")
    if df is None or df.empty: return {"status":"error","summary":"No active dataset is available for Agent Plot."}
    target=state.get("target") if state.get("target") in df.columns else None
    quantities=_analytical_quantities(df,target,state.get("ml_results"),state.get("dl_results"))
    deterministic=_deterministic_plan(df,target,state.get("ml_results"),state.get("dl_results"))
    refined,reason=_llm_refine_plan(state,deterministic,quantities)
    mem=AgentMemory("Plot",30); mem.remember("plot_plan",[{"kind":p["kind"],"fields":p["fields"],"quantity":p.get("quantity")} for p in refined])
    return {"plan":refined,"plot_quantities":quantities,"llm_reasoning":reason,"memory":mem.to_dict(),"status":"planned","messages":[AIMessage(content=f"Agent Plot selected {len(refined)} analytical plots using Data Scientist evidence and provider-aware planning.")]}


def _render_node(state: PlotAgentState):
    """Perform the render node operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df=state.get("dataframe"); plots=[]
    try:
        import plotly.express as px
    except Exception as exc:
        return {"status":"ok","plots":[{k:p.get(k) for k in ("kind","title","fields","quantity","rationale")} for p in state.get("plan",[])],"summary":f"Plot plan created; Plotly renderer unavailable: {exc}"}
    for p in state.get("plan",[]):
        try:
            kind,fields=p["kind"],p["fields"]; fig=None
            if kind=="correlation_heatmap":
                fig=px.imshow(df[fields].corr(numeric_only=True),text_auto=".2f",aspect="auto",title=p["title"],color_continuous_scale="RdBu_r")
            elif kind in {"scatter","target_scatter"}:
                work=df[[fields[0],fields[1]]].dropna(); fig=px.scatter(work,x=fields[0],y=fields[1],trendline="ols" if len(work)>=5 else None,title=p["title"])
            elif kind=="distribution": fig=px.histogram(df,x=fields[0],marginal="box",title=p["title"])
            elif kind=="time_series":
                work=df[[fields[0],fields[1]]].dropna().sort_values(fields[0]); fig=px.line(work,x=fields[0],y=fields[1],title=p["title"])
            elif kind=="category_bar":
                work=df[[fields[0],fields[1]]].dropna().groupby(fields[0],as_index=False)[fields[1]].sum().sort_values(fields[1],ascending=False).head(30); fig=px.bar(work,x=fields[0],y=fields[1],title=p["title"],text_auto=".3s")
            elif kind=="count":
                work=df[fields[0]].astype(str).value_counts().head(30).rename_axis(fields[0]).reset_index(name="Count"); fig=px.bar(work,x=fields[0],y="Count",title=p["title"],text_auto=True)
            if fig is None: continue
            fig.update_layout(template="plotly_white",margin=dict(l=60,r=30,t=70,b=70),font=dict(size=13))
            insights=[]
            if kind=="category_bar":
                top=df.groupby(fields[0],dropna=False)[fields[1]].sum().sort_values(ascending=False).head(3); insights.append("Top categories by aggregated measure: "+", ".join(f"{k} ({v:.3g})" for k,v in top.items()))
            if kind in {"scatter","target_scatter"}:
                c=df[[fields[0],fields[1]]].corr().iloc[0,1]
                if np.isfinite(c): insights.append(f"Pearson correlation = {c:.3f}; association only, not a causal conclusion.")
            plots.append({"kind":kind,"title":p["title"],"fields":fields,"quantity":p.get("quantity"),"rationale":p["rationale"],"insights":insights,"figure_json":fig.to_json()})
        except Exception as exc:
            plots.append({"kind":p["kind"],"title":p["title"],"fields":p["fields"],"quantity":p.get("quantity"),"rationale":p["rationale"],"error":str(exc)})
    model_df=_model_comparison(state.get("ml_results"),state.get("dl_results"))
    if not model_df.empty:
        metric="R2" if "R2" in set(model_df["Metric"]) else ("f1_weighted" if "f1_weighted" in set(model_df["Metric"]) else model_df["Metric"].iloc[0])
        md=model_df[model_df["Metric"]==metric]
        if not md.empty:
            fig=px.bar(md,x="Model",y="Value",color="Agent",text="Value",title=f"Model Comparison — {metric}",barmode="group"); fig.update_layout(template="plotly_white")
            plots.append({"kind":"model_comparison","title":f"Model Comparison — {metric}","fields":["Model",metric],"quantity":"recorded model metric","rationale":"Supplemental comparison of recorded model evaluation evidence.","insights":[],"figure_json":fig.to_json()})
    return {"status":"ok","plots":plots,"plot_quantities":state.get("plot_quantities",{}),"llm_reasoning":state.get("llm_reasoning",""),"summary":f"Agent Plot generated {len(plots)} analytical visualizations using Data Scientist evidence, deterministic validation and LLM-assisted selection."}

workflow=StateGraph(PlotAgentState); workflow.add_node("plan",_plan_node); workflow.add_node("render",_render_node); workflow.set_entry_point("plan"); workflow.add_edge("plan","render"); workflow.add_edge("render",END); agent_plot_app=workflow.compile()
