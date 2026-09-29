from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph

from .graph_ml import run_ml_step
from .graph_dl import run_dl_step
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    dataframe: pd.DataFrame
    current_step: str
    stage_summary: str
    approved_steps: list[str]
    rejected_steps: list[str]
    user_approved: bool
    needs_approval: bool
    route: str
    route_locked: bool
    target: str
    ml_results: dict[str, Any]
    dl_results: dict[str, Any]
    step_index: int
    api_key: str
    model_name: str
    max_steps: int
    abort_requested: bool
    leakage_gate: dict[str, Any]
    dataset_card: dict[str, Any]
    human_approval_evidence: list[dict[str, Any]]
    compute_mode: str
    target_analysis: dict[str, Any]
    feature_candidates: list[str]
    memory: list[dict[str, Any]]


def _master_route(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    if rows >= 10000 or cols >= 60:
        return "DL"
    if rows >= 3000 and cols >= 20:
        return "COMPETE"
    return "ML"


def _master_proposal(state: AgentState) -> dict:
    df = state.get("dataframe")
    if df is None:
        return {"current_step": "Error", "stage_summary": "No dataframe is loaded.", "needs_approval": False}
    route = state.get("route") or _master_route(df)
    analysis = state.get("target_analysis") or recommend_targets_and_features(df)
    target = state.get("target") or analysis.get("suggested_target") or df.columns[-1]
    if route == "COMPETE":
        # Desktop version executes ML first, then DL after the ML approval.
        next_route = "COMPETE"
    else:
        next_route = route
    summary = (f"Master Agent proposes the {next_route} route for {len(df):,} rows and {len(df.columns):,} columns. "
               f"Target candidate: '{target}'. Approve this routing decision before the specialist agent runs.")
    mem=list(state.get("memory", [])); mem.append({"agent":"Master","kind":"routing","target":target,"route":next_route})
    return {"route": next_route, "route_locked": True, "target": target, "memory": mem[-20:],
            "current_step": "MASTER: route decision", "stage_summary": summary, "target_analysis": analysis,
            "feature_candidates": analysis.get("suggested_features", []),
            "needs_approval": True, "messages": [AIMessage(content=summary)]}


def _execute_specialist(state: AgentState) -> dict:
    route = state.get("route", "ML")
    idx = state.get("step_index", 0)
    try:
        # COMPETE is deliberately explicit: Master → ML → human approval → DL → human approval.
        use_dl = route == "DL" or (route == "COMPETE" and bool(state.get("ml_results")) and not state.get("dl_results"))
        if use_dl:
            result = run_dl_step(state["dataframe"], state.get("target"), seed=42 + idx, compute_mode=state.get("compute_mode", "CPU"))
            return {"dl_results": result, "current_step": "DL Agent: model analysis",
                    "stage_summary": result.get("summary", result.get("message", "DL step completed.")),
                    "needs_approval": True, "step_index": idx + 1,
                    "messages": [AIMessage(content=result.get("summary", str(result)))]}
        result = run_ml_step(state["dataframe"], state.get("target"), seed=42 + idx)
        return {"ml_results": result, "current_step": "ML Agent: model analysis",
                "stage_summary": result.get("summary", result.get("message", "ML step completed.")),
                "needs_approval": True, "step_index": idx + 1,
                "messages": [AIMessage(content=result.get("summary", str(result)))]}
    except Exception as exc:
        return {"current_step":"ERROR", "stage_summary":f"Specialist agent failed safely: {exc}", "needs_approval":False, "agent_error":str(exc)}


def step_node(state: AgentState):
    """One graph invocation = one gated step; the GUI supplies approval between invocations."""
    if state.get("abort_requested"):
        return {"current_step":"ABORTED", "stage_summary":"Agent run aborted safely by the user.", "needs_approval":False, "user_approved":False}
    if state.get("step_index",0) >= state.get("max_steps",20):
        return {"current_step":"ABORTED", "stage_summary":"Safety limit reached; agent loop stopped. Review the evidence before starting a new run.", "needs_approval":False, "user_approved":False}
    approved = bool(state.get("user_approved", False))
    step = state.get("current_step", "")
    if not step or step == "Error":
        return _master_proposal(state)
    if not approved:
        # Rejected proposals are deliberately repeated/revised, never auto-approved.
        if step.startswith("MASTER"):
            out = _master_proposal(state)
            out["stage_summary"] = "The previous routing proposal was rejected. " + out["stage_summary"]
            return out
        out = _execute_specialist(state)
        out["stage_summary"] = "The previous specialist step was rejected. Re-running it with a new seed. " + out["stage_summary"]
        return out
    if step.startswith("MASTER"):
        return _execute_specialist(state)
    # COMPETE is the explicit two-specialist route from the reference architecture.
    # ML is approved first; after that approval, DL is proposed and separately gated.
    if state.get("route") == "COMPETE" and step.startswith("ML") and not state.get("dl_results"):
        return _execute_specialist(state)
    return {"current_step": "DONE", "stage_summary": "The approved specialist analysis is complete. Results are available to Agent Plot and Agent Report.",
            "needs_approval": False, "user_approved": False}


workflow = StateGraph(AgentState)
workflow.add_node("step", step_node)
workflow.set_entry_point("step")
workflow.add_edge("step", END)
agent_ds_app = workflow.compile()
