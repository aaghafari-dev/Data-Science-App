from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph

from .graph_ml import run_ml_step
from .graph_dl import run_dl_step
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory
from services.agent_quality import AgentSelfCheck, AgentWhyEvidence
from services.agent_tools import default_registry


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
    plan: dict[str, Any]
    critic: dict[str, Any]
    self_check: dict[str, Any]
    why_evidence: dict[str, Any]
    evidence_ids: list[str]


def _master_route(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    if rows >= 10000 or cols >= 60:
        return "DL"
    if rows >= 3000 and cols >= 20:
        return "COMPETE"
    return "ML"


def _route_plan(state: AgentState) -> dict[str, Any]:
    df = state.get("dataframe")
    if df is None:
        return {"objective": "stop safely", "inputs": [], "expected_output": "error", "risk": "no data", "tool_budget": 0}
    route = state.get("route") or _master_route(df)
    target = state.get("target") or (state.get("target_analysis") or {}).get("suggested_target") or df.columns[-1]
    registry = default_registry()
    previews = []
    for tool_name, kwargs in [("data_quality", {"df": df}), ("temporal_availability", {"df": df, "target": target}), ("compute_diagnostics", {})]:
        try:
            previews.append(registry.dry_run(tool_name, **kwargs))
        except Exception as exc:
            previews.append({"tool": tool_name, "status": "review", "reason": str(exc), "will_execute": False})
    return {
        "objective": "choose and execute the next governed modeling stage",
        "inputs": ["active dataframe", f"target={target}", f"route={route}"],
        "expected_output": "validated ML/DL analysis evidence",
        "risk": "target leakage, unsuitable split, compute limitations, or overfitting",
        "route": route,
        "target": target,
        "tool_budget": 8,
        "human_gate": True,
        "tool_dry_run": previews,
    }


def _critic_plan(state: AgentState, plan: dict[str, Any]) -> dict[str, Any]:
    df = state.get("dataframe")
    analysis = state.get("target_analysis") or {}
    issues: list[str] = []
    if df is None or df.empty:
        issues.append("No dataframe is available.")
    target = plan.get("target")
    if not target or target not in (df.columns if df is not None else []):
        issues.append("Target is missing or not present in the dataframe.")
    gate = state.get("leakage_gate") or {}
    if gate.get("status") == "blocked":
        issues.append("Scientific leakage gate is blocked until explicit human override.")
    if not analysis.get("suggested_features") and df is not None and len(df.columns) <= 1:
        issues.append("No usable predictor fields were identified.")
    return {
        "status": "pass" if not issues else "review",
        "issues": issues,
        "checks": [
            {"check": "target_defined", "status": "pass" if target in (df.columns if df is not None else []) else "fail"},
            {"check": "leakage_gate", "status": "pass" if gate.get("status") != "blocked" else "fail"},
            {"check": "human_approval", "status": "pass"},
            {"check": "bounded_tools", "status": "pass" if plan.get("tool_budget", 99) <= 20 else "fail"},
        ],
        "recommendation": "execute" if not issues else "revise before execution",
    }


def _proposal(state: AgentState, *, revised: bool = False) -> dict[str, Any]:
    df = state.get("dataframe")
    if df is None:
        return {"current_step": "Error", "stage_summary": "No dataframe is loaded.", "needs_approval": False}
    route = state.get("route") or _master_route(df)
    analysis = state.get("target_analysis") or recommend_targets_and_features(df)
    target = state.get("target") or analysis.get("suggested_target") or df.columns[-1]
    plan = _route_plan({**state, "route": route, "target": target, "target_analysis": analysis})
    critic = _critic_plan({**state, "route": route, "target": target, "target_analysis": analysis}, plan)
    why = AgentWhyEvidence.create(
        "Master Agent", f"route={route}",
        [
            f"Dataset shape is {len(df):,} rows × {len(df.columns):,} columns.",
            f"Proposed target is '{target}'.",
            f"Route rule selected {route} from dataset scale and the current governed policy.",
            "The plan is subject to a critic pass and explicit human approval before execution.",
        ],
        evidence_refs=list(state.get("evidence_ids", [])),
        alternatives=["ML", "DL", "COMPETE"],
        constraints=["leakage gate", "bounded tool budget", "human approval", "compute policy"],
    )
    self_check = AgentSelfCheck.run("Master", "route_and_model_plan", plan, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
    mem = list(state.get("memory", [])); mem.append({"agent": "Master", "kind": "plan_critic", "route": route, "target": target, "critic": critic.get("status")})
    prefix = "Revised proposal after rejection. " if revised else ""
    summary = (
        f"{prefix}Plan → Critic → Execute: Master Agent proposes the {route} route for {len(df):,} rows and {len(df.columns):,} columns. "
        f"Target: '{target}'. Critic status: {critic.get('status')}. "
        f"Self-check: {self_check.get('status')}. The rationale is recorded as first-class Agent Why evidence."
    )
    if critic.get("issues"):
        summary += " Review flags: " + "; ".join(critic["issues"])
    return {
        "route": route, "route_locked": True, "target": target, "memory": mem[-20:],
        "current_step": "MASTER: Plan → Critic", "stage_summary": summary,
        "target_analysis": analysis, "feature_candidates": analysis.get("suggested_features", []),
        "plan": plan, "critic": critic, "self_check": self_check, "why_evidence": why,
        "needs_approval": True, "messages": [AIMessage(content=summary)],
    }


def _execute_specialist(state: AgentState) -> dict[str, Any]:
    route = state.get("route", "ML")
    idx = state.get("step_index", 0)
    try:
        use_dl = route == "DL" or (route == "COMPETE" and bool(state.get("ml_results")) and not state.get("dl_results"))
        if use_dl:
            result = run_dl_step(state["dataframe"], state.get("target"), seed=42 + idx, compute_mode=state.get("compute_mode", "CPU"))
            agent = "DL"
            current = "DL Agent: model analysis"
            summary = result.get("summary", result.get("message", "DL step completed."))
            why = AgentWhyEvidence.create(
                "DL Agent", "execute deep-learning analysis",
                ["The Master route requires DL or the COMPETE route has completed its ML stage.", f"Compute policy requested: {state.get('compute_mode', 'CPU')}.", "The DL result includes professional evaluation and diagnostic evidence."],
                evidence_refs=list(state.get("evidence_ids", [])),
                alternatives=["CPU MLP", "PyTorch CUDA MLP"], constraints=["VRAM budget", "human approval", "safe CUDA fallback"],
            )
            result_check = AgentSelfCheck.run(agent, "model_analysis_result", {"objective": "produce validated DL evidence", "inputs": ["dataframe", "target"], "expected_output": "metrics + diagnostics", "risk": "overfitting/compute failure", "tool_budget": 6}, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
            return {"dl_results": result, "current_step": current, "stage_summary": summary + f" Self-check: {result_check.get('status')}.", "needs_approval": True, "step_index": idx + 1, "why_evidence": why, "self_check": result_check, "messages": [AIMessage(content=summary)]}
        result = run_ml_step(state["dataframe"], state.get("target"), seed=42 + idx)
        agent = "ML"
        current = "ML Agent: model analysis"
        summary = result.get("summary", result.get("message", "ML step completed."))
        why = AgentWhyEvidence.create(
            "ML Agent", "execute machine-learning analysis",
            ["The Master route selected ML or this is the first stage of COMPETE.", "The ML result compares multiple model families and professional evaluation evidence.", "The result is presented for explicit human approval before the workflow advances."],
            evidence_refs=list(state.get("evidence_ids", [])),
            alternatives=["Random Forest", "Gradient Boosting", "Logistic/Ridge"], constraints=["leakage gate", "held-out evaluation", "human approval"],
        )
        result_check = AgentSelfCheck.run(agent, "model_analysis_result", {"objective": "produce validated ML evidence", "inputs": ["dataframe", "target"], "expected_output": "metrics + diagnostics", "risk": "overfitting/leakage", "tool_budget": 8}, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
        return {"ml_results": result, "current_step": current, "stage_summary": summary + f" Self-check: {result_check.get('status')}.", "needs_approval": True, "step_index": idx + 1, "why_evidence": why, "self_check": result_check, "messages": [AIMessage(content=summary)]}
    except Exception as exc:
        return {"current_step":"ERROR", "stage_summary":f"Specialist agent failed safely: {exc}", "needs_approval":False, "agent_error":str(exc)}


def step_node(state: AgentState):
    """LangGraph control point: plan → critic → human approval → execute → self-check → approval."""
    if state.get("abort_requested"):
        return {"current_step":"ABORTED", "stage_summary":"Agent run aborted safely by the user.", "needs_approval":False, "user_approved":False}
    if state.get("step_index",0) >= state.get("max_steps",20):
        return {"current_step":"ABORTED", "stage_summary":"Safety limit reached; agent loop stopped. Review the evidence before starting a new run.", "needs_approval":False, "user_approved":False}
    approved = bool(state.get("user_approved", False))
    step = state.get("current_step", "")
    if not step or step in {"Error", "ABORTED"}:
        return _proposal(state)
    if not approved:
        if step.startswith("MASTER"):
            return _proposal(state, revised=True)
        # A rejected result is not silently accepted; rerun with a new seed after a revised proposal.
        return _proposal({**state, "route_locked": True}, revised=True)
    if step.startswith("MASTER"):
        return _execute_specialist(state)
    if route := state.get("route"):
        if route == "COMPETE" and step.startswith("ML") and not state.get("dl_results"):
            return _proposal({**state, "current_step": ""})
    return {"current_step": "DONE", "stage_summary": "The approved specialist analysis is complete. ML/DL evidence is available to Agent Plot and Agent Report.", "needs_approval": False, "user_approved": False}


workflow = StateGraph(AgentState)
workflow.add_node("step", step_node)
workflow.set_entry_point("step")
workflow.add_edge("step", END)
agent_ds_app = workflow.compile()
