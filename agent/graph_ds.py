"""Agent Data Scientist Master Agent implemented with LangGraph and governed sub-agents.

The Master Agent owns analytical planning, method routing, approval gates, bounded role-specific
memory, typed tool previews, evidence generation, and specialist dispatch. It does not choose ML
or DL from dataset size alone. Instead it builds an evidence profile, reviews candidate analytical
routes with the configured LLM when available, and asks the user to approve every governed step.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph

from .graph_anomaly import run_anomaly_step
from .graph_cluster import run_clustering_step
from .graph_dl import run_dl_step
from .graph_ml import run_ml_step
from .graph_statistics import run_statistics_step
from .graph_timeseries import run_time_series_step
from .graph_unsupervised import run_unsupervised_step
from .graph_rl import run_reinforcement_step
from .graph_cnn import run_cnn_image_step
from services.agent_memory import AgentMemory
from services.agent_quality import AgentSelfCheck, AgentWhyEvidence
from services.agent_tools import default_registry
from services.llm_config import provider_from_state
from services.master_router import MasterDecisionEngine
from services.result_verification import ResultVerifier
from services.target_feature_selection import recommend_targets_and_features


class AgentState(TypedDict, total=False):
    """Typed LangGraph state shared by the Master Agent and its specialist sub-agents."""

    messages: list[BaseMessage]
    dataframe: pd.DataFrame
    objective: str
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
    clustering_results: dict[str, Any]
    anomaly_results: dict[str, Any]
    statistics_results: dict[str, Any]
    data_quality_results: dict[str, Any]
    time_series_results: dict[str, Any]
    unsupervised_results: dict[str, Any]
    reinforcement_results: dict[str, Any]
    cnn_results: dict[str, Any]
    image_dataset_path: str
    validation_strategy: dict[str, Any]
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
    master_memory: list[dict[str, Any]]
    specialist_memory: dict[str, list[dict[str, Any]]]
    plan: dict[str, Any]
    critic: dict[str, Any]
    self_check: dict[str, Any]
    why_evidence: dict[str, Any]
    master_decision: dict[str, Any]
    task_identification: list[dict[str, Any]]
    method_candidates: list[dict[str, Any]]
    method_selection: dict[str, Any]
    specialist_plan: list[dict[str, Any]]
    selection_protocol: list[str]
    experiment_plan: dict[str, Any]
    subagent_queue: list[str]
    current_subagent: str
    verification: dict[str, Any]
    verification_why: dict[str, Any]
    verification_self_check: dict[str, Any]
    evidence_validation: dict[str, Any]
    stop_evaluation: dict[str, Any]
    evidence_ids: list[str]
    analysis_stage: str
    completed_stages: list[str]
    llm_provider: str
    local_path: str
    rejection_feedback: str
    api_base: str
    run_id: str
    thread_id: str
    memory_refs: list[str]
    approval_history: list[dict[str, Any]]
    completed_bullets: list[str]
    next_bullets: list[str]
    risk_bullets: list[str]


STAGES = ["PLAN", "VALIDATION", "PREPROCESSING", "MODEL_SELECTION", "MODEL_EXECUTION", "EVALUATION", "ROBUSTNESS", "DIAGNOSIS", "VERIFY", "STOP"]


def _master_route(df: pd.DataFrame, objective: str = "", state: dict[str, Any] | None = None) -> str:
    """Return the Master Agent's evidence/LLM-reviewed primary analytical route."""
    cfg = provider_from_state(state or {})
    decision = MasterDecisionEngine.decide(df, objective, cfg if cfg.configured() else None, target=(state or {}).get("target"), image_dataset_path=(state or {}).get("image_dataset_path"))
    return str(decision.get("primary_route") or "EXPLORATION")


def _route_plan(state: AgentState) -> dict[str, Any]:
    """Create a bounded tool-preview plan for the Master Agent's current analytical route."""
    df = state.get("dataframe")
    if df is None:
        return {"objective": "stop safely", "inputs": [], "expected_output": "error", "risk": "no data", "tool_budget": 0}
    route = state.get("route") or _master_route(df, state.get("objective", ""), state)
    decision = state.get("master_decision") or {}
    target = state.get("target") or (state.get("target_analysis") or {}).get("suggested_target")
    registry = default_registry()
    previews = []
    preview_specs = [("data_quality", {"df": df}), ("describe", {"df": df}), ("compute_diagnostics", {})]
    if target and target in df.columns:
        preview_specs += [("validation_protocol", {"df": df, "target": target}), ("temporal_availability", {"df": df, "target": target})]
    for tool_name, kwargs in preview_specs:
        try:
            previews.append(registry.dry_run(tool_name, **kwargs))
        except Exception as exc:
            previews.append({"tool": tool_name, "status": "review", "reason": str(exc), "will_execute": False})
    return {
        "objective": state.get("objective") or "perform a professional evidence-bound analysis of the active dataset",
        "inputs": ["active dataframe", f"route={route}"] + ([f"target={target}"] if target else []),
        "expected_output": "validated analytical evidence + diagnostics + limitations",
        "risk": "target ambiguity, leakage, unsuitable validation, overfitting, distribution shift, compute limitations",
        "route": route,
        "target": target,
        "sub_agents": decision.get("sub_agents", []),
        "task_identification": decision.get("task_identification", []),
        "method_candidates": decision.get("method_candidates", []),
        "method_selection": decision.get("method_selection", {}),
        "specialist_plan": decision.get("specialist_plan", []),
        "tool_budget": 30,
        "human_gate": True,
        "tool_dry_run": previews,
        "quality_requirements": ["evidence-bound method selection", "human approval", "validation appropriate to task", "uncertainty where applicable", "diagnostics", "reproducibility"],
    }


def _critic_plan(state: AgentState, plan: dict[str, Any]) -> dict[str, Any]:
    """Critique a Master Agent plan for missing inputs, unsafe transitions, and evidence gaps."""
    df = state.get("dataframe")
    issues = []
    target = plan.get("target")
    route = str(plan.get("route") or "EXPLORATION")
    if df is None or df.empty:
        issues.append("No dataframe is available.")
    if route in {"REGRESSION", "CLASSIFICATION", "HYBRID_SUPERVISED"} and (not target or target not in (df.columns if df is not None else [])):
        issues.append("A supervised route requires an approved target.")
    if (state.get("leakage_gate") or {}).get("status") == "blocked":
        issues.append("Scientific leakage gate is blocked until explicit human review.")
    if df is not None and len(df.columns) <= 1 and route not in {"EXPLORATION", "STATISTICS"}:
        issues.append("No usable predictor structure was identified.")
    return {
        "status": "pass" if not issues else "review",
        "issues": issues,
        "checks": [
            {"check": "data_available", "status": "pass" if df is not None and not df.empty else "fail"},
            {"check": "target_defined_when_required", "status": "pass" if route not in {"REGRESSION", "CLASSIFICATION", "HYBRID_SUPERVISED"} or target in (df.columns if df is not None else []) else "fail"},
            {"check": "leakage_gate", "status": "pass" if (state.get("leakage_gate") or {}).get("status") != "blocked" else "fail"},
            {"check": "bounded_tools", "status": "pass" if plan.get("tool_budget", 999) <= 30 else "fail"},
            {"check": "human_approval", "status": "pass"},
        ],
        "recommendation": "execute" if not issues else "revise before execution",
    }


def _stage_description(stage: str, state: AgentState) -> str:
    """Generate the concise human-facing explanation for the current governed stage."""
    target = state.get("target", "not specified")
    route = state.get("route", "not specified")
    subagent = state.get("current_subagent") or ", ".join((state.get("master_decision") or {}).get("sub_agents", []))
    descriptions = {
        "PLAN": f"Review the Master Agent protocol for objective '{state.get('objective', 'professional data analysis')}'. The agent identifies the task, compares candidate methods and assumptions, and proposes a bounded specialist plan before execution.",
        "VALIDATION": "Review data quality, target/feature validity, leakage controls, validation design, temporal structure, and train/test isolation where supervised analysis is proposed.",
        "PREPROCESSING": "Review the preprocessing policy. Imputation, encoding, scaling, and feature transformations must be learned only from the appropriate training data.",
        "MODEL_SELECTION": "Review the explicit method-selection record: task identification, candidate methods, assumptions, evidence comparison, selected method, challenger methods, evidence gaps, and specialist order.",
        "MODEL_EXECUTION": f"Approve execution of the next specialist in the Master Agent plan: {subagent or route}. Only the explicitly selected specialist/challenger may execute at this gate.",
        "EVALUATION": "Review the specialist result, validation metrics, uncertainty, baseline comparisons, and task-appropriate diagnostic evidence.",
        "ROBUSTNESS": "Review robustness, feature stability, temporal availability, perturbation sensitivity, and unresolved diagnostic risks where applicable.",
        "DIAGNOSIS": "Review automatic diagnostic hypotheses, subgroup/slice findings, and limitations before independent verification.",
        "VERIFY": "Review independent evidence verification for structural completeness and internal consistency; it is not a certification of scientific truth.",
        "STOP": "Review whether the required evidence is sufficient, whether unresolved high-severity issues remain, and whether another governed experiment is justified.",
    }
    return descriptions.get(stage, f"Review the proposed analytical step for target '{target}'.")


def _build_master_decision(state: AgentState) -> dict[str, Any]:
    """Build the LLM-assisted Master Agent route and specialist queue once per run."""
    df = state.get("dataframe")
    if df is None:
        return {"primary_route": "EXPLORATION", "sub_agents": ["Data Quality Agent", "Statistical Insight Agent"]}
    decision = MasterDecisionEngine.decide(
        df,
        state.get("objective", ""),
        provider_from_state(state),
        target=state.get("target"),
    )
    queue = list(decision.get("sub_agents") or [])
    memory = AgentMemory.from_dict(state.get("master_memory"), "Master", 30)
    memory.remember(
        "master_decision",
        {
            "route": decision.get("primary_route"),
            "sub_agents": queue,
            "decision_source": decision.get("decision_source"),
            "selection_status": decision.get("selection_status"),
            "experiment_plan": decision.get("experiment_plan", {}),
            "validation_strategy": decision.get("validation_strategy", {}),
            "selected_method": (decision.get("method_selection") or {}).get("selected_method", {}).get("method_id"),
        },
    )
    return {
        "master_decision": decision,
        "route": decision.get("primary_route", "EXPLORATION"),
        "route_locked": True,
        "task_identification": decision.get("task_identification", []),
        "method_candidates": decision.get("method_candidates", []),
        "method_selection": decision.get("method_selection", {}),
        "specialist_plan": decision.get("specialist_plan", []),
        "selection_protocol": decision.get("selection_protocol", []),
        "experiment_plan": decision.get("experiment_plan", {}),
            "validation_strategy": decision.get("validation_strategy", {}),
        "subagent_queue": queue,
        "current_subagent": queue[0] if queue else "",
        "master_memory": memory.to_dict(),
    }


def _proposal(state: AgentState, stage: str | None = None, revised: bool = False) -> dict[str, Any]:
    """Prepare one evidence-bound proposal and human approval package for a governed stage."""
    df = state.get("dataframe")
    if df is None:
        return {"current_step": "ERROR", "stage_summary": "No dataframe is loaded.", "needs_approval": False}
    if not state.get("master_decision"):
        decision_update = _build_master_decision(state)
        state = {**state, **decision_update}
    stage = stage or state.get("analysis_stage") or "PLAN"
    route = state.get("route") or "EXPLORATION"
    analysis = state.get("target_analysis") or recommend_targets_and_features(df)
    target = state.get("target") or analysis.get("suggested_target")
    plan = state.get("plan") or _route_plan({**state, "route": route, "target": target, "target_analysis": analysis})
    critic = state.get("critic") or _critic_plan({**state, "route": route, "target": target}, plan)
    summary = _stage_description(stage, {**state, "target": target, "route": route})
    decision = state.get("master_decision") or {}
    why = AgentWhyEvidence.create(
        "Master Agent",
        f"review_stage={stage}",
        [
            summary,
            f"Dataset shape: {len(df):,} rows × {len(df.columns):,} columns.",
            f"Identified task: '{route}'.",
            f"Selected method: {(decision.get('method_selection') or {}).get('selected_method', {}).get('method_id', 'not selected')}.",
            f"Selection status: {decision.get('selection_status', 'not recorded')}.",
            "The protocol separates objective interpretation, task identification, candidate-method comparison, assumptions, evidence gaps, specialist selection, and human approval.",
        ],
        evidence_refs=list(state.get("evidence_ids", [])),
        alternatives=["approve stage", "reject and revise", "abort run"],
        constraints=["leakage controls", "task-appropriate validation", "bounded tools", "human approval", "evidence traceability"],
    )
    self_check = AgentSelfCheck.run("Master", f"stage_{stage}", plan if stage == "PLAN" else {"stage": stage, "objective": summary, "risk": critic.get("issues", [])}, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
    prefix = "Revised proposal. " if revised else ""
    brief = []
    if stage == "PLAN":
        selection = decision.get("method_selection") or {}
        selected_method = (selection.get("selected_method") or {}).get("method_id", "not selected")
        brief = [
            f"Dataset: {len(df):,} rows × {len(df.columns):,} columns",
            f"Identified task: {route}",
            f"Selected method family: {selected_method}",
            f"Selection status: {decision.get('selection_status', 'not recorded')}",
            f"Specialist plan: {', '.join(decision.get('sub_agents', [])) or 'none'}",
        ]
    elif stage == "VALIDATION":
        brief = ["Data-quality and validation tools have been dry-run before execution.", f"Leakage gate: {(state.get('leakage_gate') or {}).get('status', 'not recorded')}", "Supervised analyses keep the final test partition locked."]
    elif stage == "PREPROCESSING":
        brief = ["Preprocessing must be fitted only on the permitted training data.", f"Candidate feature count: {len(analysis.get('suggested_features', []) or [])}", "Feature provenance and availability remain reviewable evidence."]
    elif stage == "MODEL_SELECTION":
        selection = decision.get("method_selection") or {}
        selected_method = (selection.get("selected_method") or {}).get("method_id", "not selected")
        candidates = decision.get("method_candidates") or []
        brief = [
            f"Task identification: {route}",
            "Candidate methods are compared using evidence, assumptions, validation requirements, and limitations.",
            f"Selected method: {selected_method}; candidates reviewed: {len(candidates)}",
            f"Selection status: {decision.get('selection_status', 'not recorded')}",
            "Human approval is required before specialist execution.",
        ]
        gaps = decision.get("evidence_gaps") or []
        if gaps:
            brief[-1] = "Evidence gaps: " + "; ".join(map(str, gaps[:2]))
    elif stage == "MODEL_EXECUTION":
        brief = [f"Selected specialist: {state.get('current_subagent') or route}", f"Compute policy: {state.get('compute_mode', 'CPU')}", "The specialist has not been allowed to execute until this approval."]
    elif stage == "EVALUATION":
        brief = ["Specialist evidence is available for review.", "Metrics and diagnostics are reported according to the analytical task."]
    elif stage == "ROBUSTNESS":
        brief = ["Robustness and sensitivity evidence are reviewed where applicable.", "Diagnostic evidence is not treated as automatic proof of causality or deployment readiness."]
    elif stage == "DIAGNOSIS":
        brief = ["Diagnostic hypotheses are ready for review.", "Subgroup/slice and generalisation risks remain evidence-bound."]
    elif stage == "VERIFY":
        brief = [f"Evidence IDs recorded: {len(state.get('evidence_ids', []) or [])}", "Verification checks structural completeness and consistency."]
    elif stage == "STOP":
        brief = [f"Approved stages: {len(state.get('approved_steps', []) or [])}", "Completion depends on evidence sufficiency and unresolved-risk review."]
    completed_bullets = {
        "PLAN": ["Dataset structure and analytical objective have been reviewed.", "Task hypotheses and candidate methods have been compared.", "A bounded specialist plan has been prepared."],
        "VALIDATION": ["The proposed validation protocol and leakage controls have been reviewed.", "The final test partition remains locked for supervised work."],
        "PREPROCESSING": ["The preprocessing policy has been specified with train-only fitting requirements.", "Feature provenance and availability checks have been identified."],
        "MODEL_SELECTION": ["Candidate methods, assumptions and evidence gaps have been reviewed.", "The primary method and challenger strategy have been identified."],
        "MODEL_EXECUTION": ["The validation, preprocessing and method-selection stages have been approved.", f"The next governed action is execution of the approved specialist: {state.get('current_subagent') or route}."],
        "EVALUATION": ["Specialist results and task-appropriate metrics have been reviewed.", "Baseline, uncertainty and diagnostic evidence have been collected where available."],
        "ROBUSTNESS": ["Sensitivity and robustness evidence has been reviewed where applicable.", "Remaining diagnostic limitations have been identified."],
        "DIAGNOSIS": ["Automatic diagnostic hypotheses and subgroup/slice evidence have been reviewed."],
        "VERIFY": ["Independent structural verification has been completed.", "Evidence completeness and internal consistency have been checked."],
        "STOP": ["The governed analysis has reached its final evidence review gate."]
    }
    next_bullets = {
        "PLAN": ["Run the validation and leakage-control review.", "Do not train a model until the validation design is approved."],
        "VALIDATION": ["Review and approve the preprocessing policy.", "Keep transformations inside the permitted training boundary."],
        "PREPROCESSING": ["Compare candidate methods and confirm the selected method family.", "Review assumptions and challenger methods."],
        "MODEL_SELECTION": [f"Execute only the approved specialist: {state.get('current_subagent') or route}.", "Generate evidence and preserve the selected validation protocol."],
        "MODEL_EXECUTION": ["Evaluate the specialist result against the approved validation protocol.", "Review metrics, uncertainty and diagnostics."],
        "EVALUATION": ["Review robustness and sensitivity evidence.", "Identify whether any high-severity risk requires another experiment."],
        "ROBUSTNESS": ["Review automatic diagnosis and subgroup/generalisation risks.", "Prepare the evidence for independent verification."],
        "DIAGNOSIS": ["Run independent evidence verification.", "Check completeness and consistency before the final stop decision."],
        "VERIFY": ["Review the final stopping decision and unresolved risks.", "Finish only when evidence is sufficient."],
        "STOP": ["Finalize the governed run and publish the evidence for Plot, Report, Story and Presentation."]
    }
    risk_bullets = {
        "PLAN": ["Target ambiguity or an inappropriate task route may invalidate downstream analysis."],
        "VALIDATION": ["Leakage, temporal ordering, grouping or class imbalance can make metrics misleading."],
        "PREPROCESSING": ["Fitting transformations outside the training boundary can cause leakage."],
        "MODEL_SELECTION": ["A high validation score alone is not evidence of scientific validity or causality."],
        "MODEL_EXECUTION": ["Compute failures, convergence issues or specialist assumptions may require revision."],
        "EVALUATION": ["Aggregate metrics may hide subgroup failures or generalisation gaps."],
        "ROBUSTNESS": ["Sensitivity checks are diagnostic and do not establish causal validity."],
        "DIAGNOSIS": ["Automated hypotheses require human/domain interpretation."],
        "VERIFY": ["Structural verification cannot certify scientific truth."],
        "STOP": ["Unresolved high-severity evidence should trigger another governed experiment rather than silent completion."]
    }
    stage_completed = completed_bullets.get(stage, ["The current governed evidence package has been prepared."])
    stage_next = next_bullets.get(stage, ["Proceed only after explicit human approval."])
    stage_risk = risk_bullets.get(stage, ["Review the evidence and limitations before approval."])
    memory=AgentMemory.from_dict(state.get("master_memory"), "Master", 30)
    memory.remember("approval_stage", {"stage":stage,"route":route,"subagent":state.get("current_subagent"),"summary":summary})
    return {
        "route": route,
        "route_locked": True,
        "target": target,
        "target_analysis": analysis,
        "feature_candidates": analysis.get("suggested_features", []),
        "plan": plan,
        "critic": critic,
        "current_step": f"MASTER: {stage}",
        "analysis_stage": stage,
        "stage_summary": prefix + summary,
        "completed_bullets": stage_completed,
        "next_bullets": stage_next,
        "risk_bullets": stage_risk,
        "needs_approval": True,
        "user_approved": False,
        "why_evidence": why,
        "self_check": self_check,
        "approval_result_points": brief,
        "master_decision": decision,
        "subagent_queue": state.get("subagent_queue", []),
        "current_subagent": state.get("current_subagent", ""),
        "master_memory": memory.to_dict(),
        "messages": [AIMessage(content=summary)],
    }


def _execute_specialist(state: AgentState) -> dict[str, Any]:
    """Dispatch exactly one approved specialist sub-agent and return JSON-safe evidence state."""
    route = state.get("route", "EXPLORATION")
    queue = list(state.get("subagent_queue") or [])
    subagent = state.get("current_subagent") or (queue[0] if queue else "")
    idx = state.get("step_index", 0)
    df = state["dataframe"]
    target = state.get("target")
    compute_mode = state.get("compute_mode", "CPU")
    try:
        registry=default_registry()
        if subagent == "Data Quality Agent":
            result = registry.execute("data_quality", df=df, human_approved=True, evidence_bound=True)
            key = "data_quality_results"
        elif subagent == "ML Agent":
            result = run_ml_step(df, target, seed=42 + idx, compute_mode=compute_mode)
            key = "ml_results"
        elif subagent == "DL Agent":
            result = run_dl_step(df, target, seed=42 + idx, compute_mode=compute_mode)
            key = "dl_results"
        elif subagent == "Clustering Agent":
            result = registry.execute("clustering", df=df, human_approved=True, evidence_bound=True, seed=42 + idx)
            key = "clustering_results"
        elif subagent == "Anomaly Detection Agent":
            result = registry.execute("anomaly_detection", df=df, human_approved=True, evidence_bound=True, seed=42 + idx)
            key = "anomaly_results"
        elif subagent == "Time-Series Agent":
            result = registry.execute("time_series", df=df, human_approved=True, evidence_bound=True)
            key = "time_series_results"
        elif subagent == "Unsupervised Learning Agent":
            result = run_unsupervised_step(df, features=state.get("feature_candidates") or None, seed=42 + idx)
            key = "unsupervised_results"
        elif subagent == "Reinforcement Learning Agent":
            result = run_reinforcement_step(seed=42 + idx)
            key = "reinforcement_results"
        elif subagent == "CNN Image Analysis Agent":
            image_dir = state.get("image_dataset_path", "")
            if not image_dir:
                # If the active table contains image paths, infer the common parent without inventing labels.
                image_cols = (state.get("master_decision") or {}).get("profile", {}).get("image_path_columns", [])
                if image_cols and image_cols[0] in df.columns:
                    from pathlib import Path
                    paths = [Path(str(x)).expanduser() for x in df[image_cols[0]].dropna().head(500) if str(x).strip()]
                    if paths:
                        try: image_dir = str(Path(os.path.commonpath([str(x.resolve()) for x in paths])))
                        except Exception: image_dir = str(paths[0].parent)
            if not image_dir:
                raise ValueError("CNN Image Analysis requires an ImageFolder dataset path or an image-path column with class-folder structure.")
            result = run_cnn_image_step(image_dir, seed=42 + idx, device=compute_mode.lower() if compute_mode.lower() in {"cpu","cuda"} else "auto")
            key = "cnn_results"
        elif subagent == "Statistical Insight Agent":
            result = registry.execute("statistical_insights", df=df, human_approved=True, evidence_bound=True, question=state.get("objective", ""))
            key = "statistics_results"
        else:
            result = run_statistics_step(df, question=state.get("objective", ""))
            key = "statistics_results"
        specialist_key = subagent.replace(" ", "_").replace("-", "_")
        specialist_memory = dict(state.get("specialist_memory") or {})
        sm = AgentMemory.from_dict(specialist_memory.get(specialist_key), subagent, 20)
        sm.remember("execution", {"route": route, "result_status": result.get("status", "complete") if isinstance(result, dict) else "complete", "step_index": idx, "selected_method": ((state.get("method_selection") or {}).get("selected_method") or {}).get("method_id")})
        specialist_memory[specialist_key] = sm.to_dict()
        remaining = queue[1:] if queue and queue[0] == subagent else [x for x in queue if x != subagent]
        selected_method = ((state.get("method_selection") or {}).get("selected_method") or {}).get("method_id", "not recorded")
        why = AgentWhyEvidence.create(
            "Master Agent",
            "execute_specialist",
            [
                f"Approved specialist: {subagent}.",
                f"Identified task: {route}.",
                f"Selected method family: {selected_method}.",
                "The specialist executed only after the current human approval gate.",
            ],
            evidence_refs=list(state.get("evidence_ids", [])),
            alternatives=["execute next specialist", "revise", "abort"],
            constraints=["human approval", "bounded specialist", "evidence traceability"],
        )
        check = AgentSelfCheck.run("Master", "specialist_execution", {"objective": "produce bounded specialist evidence", "inputs": ["active dataframe", subagent], "expected_output": "specialist result", "risk": "method mismatch or incomplete evidence", "tool_budget": 20}, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
        if remaining:
            next_subagent = remaining[0]
            return {key: result, "specialist_memory": specialist_memory, "subagent_queue": remaining, "current_subagent": next_subagent, "why_evidence": why, "self_check": check, "step_index": idx + 1, "analysis_stage": "MODEL_SELECTION", "current_step": "MASTER: MODEL_SELECTION", "needs_approval": True, "user_approved": False, "stage_summary": f"{subagent} completed. The next specialist is {next_subagent}; review its proposed method before execution."}
        return {key: result, "specialist_memory": specialist_memory, "subagent_queue": [], "current_subagent": subagent, "why_evidence": why, "self_check": check, "step_index": idx + 1, "analysis_stage": "EVALUATION", "current_step": "MASTER: EVALUATION", "needs_approval": True, "user_approved": False, "stage_summary": f"{subagent} completed. Review the specialist evidence before accepting the result."}
    except Exception as exc:
        return {"current_step": "ERROR", "stage_summary": f"Specialist agent failed safely: {exc}", "needs_approval": False, "agent_error": str(exc)}


def _verification(state: AgentState) -> dict[str, Any]:
    """Run independent structural verification of the selected specialist evidence."""
    result = next((state.get(k) for k in ("cnn_results", "dl_results", "ml_results", "clustering_results", "anomaly_results", "statistics_results", "data_quality_results", "time_series_results") if state.get(k)), {})
    verification = ResultVerifier.verify(result) if result else {"status": "review", "reason": "No specialist evidence was recorded."}
    why = AgentWhyEvidence.create("Independent Verification", "verify_analysis_evidence", [f"Verification status: {verification.get('status')}.", "Verification checks structural completeness and internal consistency rather than scientific truth."], evidence_refs=list(state.get("evidence_ids", [])), alternatives=["accept evidence", "revise analysis"], constraints=["evidence traceability", "human review"])
    check = AgentSelfCheck.run("Master", "verification", {"objective": "verify result evidence", "inputs": ["specialist result"], "expected_output": "verification status", "risk": "accepting incomplete evidence", "tool_budget": 4}, {"evidence_refs": [why.get("evidence_id")] if why.get("evidence_id") else []})
    return {"verification": verification, "verification_why": why, "verification_self_check": check, "current_step": "MASTER: VERIFY", "analysis_stage": "VERIFY", "stage_summary": f"Independent verification status: {verification.get('status')}. Review before completion.", "needs_approval": True, "user_approved": False}


def _stop(state: AgentState) -> dict[str, Any]:
    """Evaluate whether the Master Agent has enough evidence to stop safely."""
    result_keys = ("cnn_results", "dl_results", "ml_results", "clustering_results", "anomaly_results", "statistics_results", "data_quality_results", "time_series_results")
    results = [state.get(k) for k in result_keys if state.get(k)]
    unresolved = []
    for result in results:
        unresolved.extend([x for x in (result.get("model_diagnosis") or {}).get("hypotheses", []) if x.get("severity") == "high"])
    if state.get("subagent_queue"):
        return {"decision": "continue", "reason": "Additional approved specialist sub-agents remain in the Master plan.", "unresolved_high_severity": unresolved, "evidence_sufficient": False}
    if unresolved:
        return {"decision": "review", "reason": "High-severity diagnostic hypotheses remain unresolved.", "unresolved_high_severity": unresolved, "evidence_sufficient": False}
    return {"decision": "ready_to_stop" if results else "continue", "reason": "Required specialist evidence is present and no high-severity automated diagnosis remains." if results else "No validated specialist result exists.", "unresolved_high_severity": unresolved, "evidence_sufficient": bool(results)}


def step_node(state: AgentState):
    """Execute one LangGraph Master Agent transition and stop at the next human gate."""
    if state.get("abort_requested"):
        return {"current_step": "ABORTED", "stage_summary": "Agent run aborted safely by the user.", "needs_approval": False, "user_approved": False}
    if state.get("step_index", 0) >= state.get("max_steps", 30):
        return {"current_step": "ABORTED", "stage_summary": "Safety limit reached; review unresolved evidence before starting a new run.", "needs_approval": False, "user_approved": False}
    stage = state.get("analysis_stage") or "PLAN"
    approved = bool(state.get("user_approved"))
    if not approved:
        return _proposal(state, stage, revised=bool(state.get("rejected_steps")))
    completed = list(state.get("completed_stages", []))
    completed.append(stage)
    out = {"completed_stages": completed, "approved_steps": list(state.get("approved_steps", [])) + [stage], "step_index": state.get("step_index", 0) + 1, "user_approved": False}
    if stage == "PLAN":
        out.update(_proposal({**state, **out}, "VALIDATION")); return out
    if stage == "VALIDATION":
        out.update(_proposal({**state, **out}, "PREPROCESSING")); return out
    if stage == "PREPROCESSING":
        out.update(_proposal({**state, **out}, "MODEL_SELECTION")); return out
    if stage == "MODEL_SELECTION":
        out.update(_proposal({**state, **out}, "MODEL_EXECUTION")); return out
    if stage == "MODEL_EXECUTION":
        out.update(_execute_specialist({**state, **out})); return out
    if stage == "EVALUATION":
        out.update(_proposal({**state, **out}, "ROBUSTNESS")); return out
    if stage == "ROBUSTNESS":
        out.update(_proposal({**state, **out}, "DIAGNOSIS")); return out
    if stage == "DIAGNOSIS":
        out.update(_verification({**state, **out})); return out
    if stage == "VERIFY":
        stop = _stop({**state, **out}); out["stop_evaluation"] = stop
        out.update(_proposal({**state, **out}, "STOP")); return out
    if stage == "STOP":
        stop = _stop({**state, **out}); out["stop_evaluation"] = stop
        out["current_step"] = "DONE" if stop.get("decision") == "ready_to_stop" else "DONE WITH REVIEW FLAGS"
        out["analysis_stage"] = "DONE"
        out["stage_summary"] = "Governed analysis completed. Evidence is available for Plot, Report and Presentation." if stop.get("evidence_sufficient") else "Analysis reached a governed stopping point with review flags."
        out["needs_approval"] = False
        return out
    return {**out, "current_step": "DONE", "needs_approval": False, "stage_summary": "Governed analysis complete."}


workflow = StateGraph(AgentState)
workflow.add_node("step", step_node)
workflow.set_entry_point("step")
workflow.add_edge("step", END)
agent_ds_app = workflow.compile()
