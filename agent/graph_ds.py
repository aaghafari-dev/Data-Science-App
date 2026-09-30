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
from services.result_verification import ResultVerifier

class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]; dataframe: pd.DataFrame; current_step: str; stage_summary: str
    approved_steps: list[str]; rejected_steps: list[str]; user_approved: bool; needs_approval: bool
    route: str; route_locked: bool; target: str; ml_results: dict[str,Any]; dl_results: dict[str,Any]
    step_index: int; api_key: str; model_name: str; max_steps: int; abort_requested: bool
    leakage_gate: dict[str,Any]; dataset_card: dict[str,Any]; human_approval_evidence: list[dict[str,Any]]
    compute_mode: str; target_analysis: dict[str,Any]; feature_candidates: list[str]; memory: list[dict[str,Any]]
    plan: dict[str,Any]; critic: dict[str,Any]; self_check: dict[str,Any]; why_evidence: dict[str,Any]
    verification: dict[str,Any]; verification_why: dict[str,Any]; verification_self_check: dict[str,Any]
    evidence_validation: dict[str,Any]; stop_evaluation: dict[str,Any]; evidence_ids: list[str]
    analysis_stage: str; completed_stages: list[str]

STAGES = ["PLAN","VALIDATION","PREPROCESSING","MODEL_SELECTION","MODEL_EXECUTION","EVALUATION","ROBUSTNESS","DIAGNOSIS","VERIFY","STOP"]


def _master_route(df):
    rows, cols=df.shape
    if rows>=10000 or cols>=60:return "DL"
    if rows>=3000 and cols>=20:return "COMPETE"
    return "ML"


def _route_plan(state):
    df=state.get("dataframe")
    if df is None:return {"objective":"stop safely","inputs":[],"expected_output":"error","risk":"no data","tool_budget":0}
    route=state.get("route") or _master_route(df); target=state.get("target") or (state.get("target_analysis") or {}).get("suggested_target") or df.columns[-1]
    registry=default_registry(); previews=[]
    for tool_name,kwargs in [("data_quality",{"df":df}),("validation_protocol",{"df":df,"target":target}),("temporal_availability",{"df":df,"target":target}),("compute_diagnostics",{})]:
        try: previews.append(registry.dry_run(tool_name,**kwargs))
        except Exception as exc: previews.append({"tool":tool_name,"status":"review","reason":str(exc),"will_execute":False})
    return {"objective":"produce a reliable predictive analysis with auditable evidence","inputs":["active dataframe",f"target={target}",f"route={route}"],"expected_output":"validated ML/DL analysis evidence + diagnostics + limitations","risk":"target leakage, unsuitable validation, overfitting, distribution shift, compute limitations","route":route,"target":target,"tool_budget":20,"human_gate":True,"tool_dry_run":previews,"quality_requirements":["training-only model selection","locked final test","uncertainty","robustness","diagnosis","evidence traceability"]}


def _critic_plan(state,plan):
    df=state.get("dataframe"); issues=[]; target=plan.get("target")
    if df is None or df.empty: issues.append("No dataframe is available.")
    if not target or target not in (df.columns if df is not None else []): issues.append("Target is missing or not present in the dataframe.")
    if (state.get("leakage_gate") or {}).get("status")=="blocked": issues.append("Scientific leakage gate is blocked until explicit human override.")
    if df is not None and len(df.columns)<=1: issues.append("No usable predictor fields were identified.")
    return {"status":"pass" if not issues else "review","issues":issues,"checks":[{"check":"target_defined","status":"pass" if target in (df.columns if df is not None else []) else "fail"},{"check":"leakage_gate","status":"pass" if (state.get("leakage_gate") or {}).get("status")!="blocked" else "fail"},{"check":"bounded_tools","status":"pass" if plan.get("tool_budget",99)<=20 else "fail"},{"check":"human_approval","status":"pass"}],"recommendation":"execute" if not issues else "revise before execution"}


def _stage_description(stage,state):
    target=state.get("target","not specified"); route=state.get("route","not specified"); result=state.get("ml_results") or state.get("dl_results")
    descriptions={
        "PLAN":f"Review the complete Master Agent plan for target '{target}', route '{route}', risk controls, tool budget and dry-run information.",
        "VALIDATION":"Review the proposed validation protocol, leakage gate, train/test isolation and temporal/group considerations before any model execution.",
        "PREPROCESSING":"Review the preprocessing policy. Imputation, categorical encoding and scaling must be learned only from training folds; leakage variables must be excluded or explicitly acknowledged.",
        "MODEL_SELECTION":"Review the candidate model families and selection metric. Hyperparameter tuning must use training-only cross-validation; the final test partition remains locked.",
        "MODEL_EXECUTION":"Approve execution of the ML/DL specialist under the selected compute policy. This is the only stage that starts model fitting.",
        "EVALUATION":"Review the completed model evaluation: baselines, cross-validation, locked-test metrics, calibration and prediction uncertainty. No additional fitting occurs at this gate.",
        "ROBUSTNESS":"Review perturbation/robustness evidence, feature stability, temporal availability and counterfactual leakage diagnostics before accepting the model evidence.",
        "DIAGNOSIS":"Review automatic model-diagnosis hypotheses, subgroup/slice findings and unresolved risks. Decide whether the evidence is sufficient for verification.",
        "VERIFY":"Review independent structural/evidence verification. This does not certify scientific truth; it checks evidence completeness and internal consistency.",
        "STOP":"Review whether required evidence is sufficient and whether any high-severity diagnostic issue remains unresolved. Approve completion or continue to the next governed experiment.",
    }
    return descriptions.get(stage, "Review the proposed analytical step.")


def _proposal(state,stage=None,revised=False):
    df=state.get("dataframe")
    if df is None:return {"current_step":"ERROR","stage_summary":"No dataframe is loaded.","needs_approval":False}
    stage=stage or state.get("analysis_stage") or "PLAN"; route=state.get("route") or _master_route(df); analysis=state.get("target_analysis") or recommend_targets_and_features(df); target=state.get("target") or analysis.get("suggested_target") or df.columns[-1]
    plan=state.get("plan") or _route_plan({**state,"route":route,"target":target,"target_analysis":analysis}); critic=state.get("critic") or _critic_plan({**state,"route":route,"target":target},plan)
    summary=_stage_description(stage,{**state,"target":target,"route":route})
    why=AgentWhyEvidence.create("Master Agent",f"review_stage={stage}",[summary,f"Dataset shape: {len(df):,} rows × {len(df.columns):,} columns.",f"Target: '{target}'.",f"Route: '{route}'.", "Every governed analytical stage requires explicit human approval before it can advance."],evidence_refs=list(state.get("evidence_ids",[])),alternatives=["approve stage","reject and revise","abort run"],constraints=["leakage gate","locked final test","compute policy","tool budget","human approval"])
    self_check=AgentSelfCheck.run("Master",f"stage_{stage}",plan if stage=="PLAN" else {"stage":stage,"objective":summary,"risk":critic.get("issues",[])}, {"evidence_refs":[why.get("evidence_id")] if why.get("evidence_id") else []})
    prefix="Revised proposal. " if revised else ""
    return {"route":route,"route_locked":True,"target":target,"target_analysis":analysis,"feature_candidates":analysis.get("suggested_features",[]),"plan":plan,"critic":critic,"current_step":f"MASTER: {stage}","analysis_stage":stage,"stage_summary":prefix+summary,f"needs_approval":True,"user_approved":False,"why_evidence":why,"self_check":self_check,"messages":[AIMessage(content=summary)]}


def _execute_specialist(state):
    route=state.get("route","ML"); idx=state.get("step_index",0)
    try:
        use_dl=route=="DL" or (route=="COMPETE" and bool(state.get("ml_results")) and not state.get("dl_results"))
        if use_dl:
            result=run_dl_step(state["dataframe"],state.get("target"),seed=42+idx,compute_mode=state.get("compute_mode","CPU")); agent="DL"; key="dl_results"
        else:
            result=run_ml_step(state["dataframe"],state.get("target"),seed=42+idx,compute_mode=state.get("compute_mode","CPU")); agent="ML"; key="ml_results"
        why=AgentWhyEvidence.create(f"{agent} Agent","execute_model_fitting",[f"Master route requires {agent} at this stage.",f"Compute policy requested: {state.get('compute_mode','CPU')}.","The specialist returns training-only selection and locked-test evaluation evidence."],evidence_refs=list(state.get("evidence_ids",[])),alternatives=["CPU execution","GPU execution when validated"],constraints=["VRAM/runtime safety","locked test","training-only model selection"])
        check=AgentSelfCheck.run(agent,"model_execution",{"objective":"produce model and evaluation evidence","inputs":["dataframe","target"],"expected_output":"model results","risk":"overfitting/leakage/compute failure","tool_budget":12},{"evidence_refs":[why.get("evidence_id")] if why.get("evidence_id") else []})
        return {key:result,"why_evidence":why,"self_check":check,"step_index":idx+1,"analysis_stage":"EVALUATION","current_step":f"MASTER: EVALUATION","needs_approval":True,"user_approved":False,"stage_summary":f"{agent} model execution completed. Review professional evaluation before accepting the result."}
    except Exception as exc:
        return {"current_step":"ERROR","stage_summary":f"Specialist agent failed safely: {exc}","needs_approval":False,"agent_error":str(exc)}


def _verification(state):
    result=state.get("dl_results") or state.get("ml_results") or {}; verification=ResultVerifier.verify(result)
    why=AgentWhyEvidence.create("Independent Verification","verify_analysis_evidence",[f"Verification status: {verification.get('status')}.","Verification checks structural completeness and internal consistency rather than scientific truth."],evidence_refs=list(state.get("evidence_ids",[])),alternatives=["accept evidence","revise analysis"],constraints=["locked test","professional evaluation","evidence traceability"])
    check=AgentSelfCheck.run("Master","verification",{"objective":"verify result evidence","inputs":["specialist result"],"expected_output":"verification status","risk":"accepting incomplete evidence","tool_budget":4},{"evidence_refs":[why.get("evidence_id")] if why.get("evidence_id") else []})
    return {"verification":verification,"verification_why":why,"verification_self_check":check,"current_step":"MASTER: VERIFY","analysis_stage":"VERIFY","stage_summary":f"Independent verification status: {verification.get('status')}. Review before completion.","needs_approval":True,"user_approved":False}


def _stop(state):
    route=state.get("route"); ml=state.get("ml_results"); dl=state.get("dl_results"); result=dl if route=="DL" else ml if route=="ML" else dl or ml
    unresolved=[]
    if result: unresolved=[x for x in (result.get("model_diagnosis") or {}).get("hypotheses",[]) if x.get("severity")=="high"]
    if route=="COMPETE" and ml and not dl: return {"decision":"continue","reason":"COMPETE route requires the second specialist model family.","unresolved_high_severity":unresolved,"evidence_sufficient":False}
    if unresolved:return {"decision":"review","reason":"High-severity diagnostic hypotheses remain unresolved.","unresolved_high_severity":unresolved,"evidence_sufficient":False}
    return {"decision":"ready_to_stop" if result else "continue","reason":"Required evidence is present and no high-severity automated diagnosis remains." if result else "No validated result exists.","unresolved_high_severity":unresolved,"evidence_sufficient":bool(result)}


def step_node(state: AgentState):
    if state.get("abort_requested"): return {"current_step":"ABORTED","stage_summary":"Agent run aborted safely by the user.","needs_approval":False,"user_approved":False}
    if state.get("step_index",0)>=state.get("max_steps",20): return {"current_step":"ABORTED","stage_summary":"Safety limit reached; review unresolved evidence before a new run.","needs_approval":False,"user_approved":False}
    stage=state.get("analysis_stage") or "PLAN"; approved=bool(state.get("user_approved"))
    if not approved:
        return _proposal(state,stage,revised=bool(state.get("rejected_steps")))
    completed=list(state.get("completed_stages",[])); completed.append(stage)
    out={"completed_stages":completed,"approved_steps":list(state.get("approved_steps",[]))+[stage],"step_index":state.get("step_index",0)+1,"user_approved":False}
    if stage=="PLAN": out.update(_proposal({**state,**out},"VALIDATION")); return out
    if stage=="VALIDATION": out.update(_proposal({**state,**out},"PREPROCESSING")); return out
    if stage=="PREPROCESSING": out.update(_proposal({**state,**out},"MODEL_SELECTION")); return out
    if stage=="MODEL_SELECTION": out.update(_proposal({**state,**out},"MODEL_EXECUTION")); return out
    if stage=="MODEL_EXECUTION": out.update(_execute_specialist({**state,**out})); return out
    if stage=="EVALUATION": out.update(_proposal({**state,**out},"ROBUSTNESS")); return out
    if stage=="ROBUSTNESS": out.update(_proposal({**state,**out},"DIAGNOSIS")); return out
    if stage=="DIAGNOSIS": out.update(_verification({**state,**out})); return out
    if stage=="VERIFY":
        stop=_stop({**state,**out}); out["stop_evaluation"]=stop
        if stop["decision"]=="continue" and state.get("route")=="COMPETE" and state.get("ml_results") and not state.get("dl_results"):
            out.update(_proposal({**state,**out,"analysis_stage":"MODEL_EXECUTION"},"MODEL_EXECUTION")); return out
        out.update(_proposal({**state,**out},"STOP")); return out
    if stage=="STOP":
        stop=_stop({**state,**out}); out["stop_evaluation"]=stop; out["current_step"]="DONE" if stop.get("decision")=="ready_to_stop" else "DONE WITH REVIEW FLAGS"; out["analysis_stage"]="DONE"; out["stage_summary"]="Governed analysis completed. Required evidence is available for Plot, Report and Presentation." if stop.get("evidence_sufficient") else "Analysis reached a governed stopping point with review flags."; out["needs_approval"]=False; return out
    return {**out,"current_step":"DONE","needs_approval":False,"stage_summary":"Governed analysis complete."}

workflow=StateGraph(AgentState); workflow.add_node("step",step_node); workflow.set_entry_point("step"); workflow.add_edge("step",END); agent_ds_app=workflow.compile()
