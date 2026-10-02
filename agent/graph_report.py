"""Module duty: Graph report.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from fpdf import FPDF

from .prompts import REPORT_AGENT_SYSTEM_PROMPT
from services.agent_memory import AgentMemory
from services.llm_config import provider_from_state, get_llm
from services.serialization import json_safe
from services.narrative_engine import EvidenceNarrativeEngine, REPORT_SECTIONS
from services.output_quality import OutputQualityGate


class ReportState(TypedDict, total=False):
    """LangGraph state for evidence audit, LLM report planning, and PDF rendering."""
    analysis_evidence: dict[str, Any]
    api_key: str
    model_name: str
    llm_provider: str
    local_path: str
    pdf_buffer: io.BytesIO
    report_text: str
    report_plan: dict[str, Any]
    evidence_audit: dict[str, Any]
    messages: list
    memory: list[dict[str,Any]]


def _evidence_text(evidence):
    """Serialize report evidence into a bounded JSON-safe prompt payload."""
    return json.dumps(json_safe(evidence), default=str, indent=2)[:90000]


def _audit_evidence(evidence):
    """Audit that the report has the required Data Scientist and Plot evidence inputs."""
    ds_keys=["Master Decision","ML Agent","DL Agent","Clustering Agent","Anomaly Detection Agent","Statistical Insight Agent","Data Quality Agent","Time-Series Agent","CNN Image Analysis Agent"]
    present=[k for k in ds_keys if evidence.get(k)]
    plot=evidence.get("Plot Agent") or {}
    return {"status":"ready" if present and isinstance(plot,dict) else "review", "data_scientist_evidence":present, "plot_evidence_count":len(plot.get("plots",[])) if isinstance(plot,dict) else 0, "required_inputs":["Data Scientist Evidence","Plot Evidence","LLM"]}


def _fallback_report(evidence):
    """Perform the fallback report operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    target=evidence.get("target","not recorded"); route=evidence.get("master_route","not recorded")
    ml=evidence.get("ML Agent") or {}; dl=evidence.get("DL Agent") or {}; cnn=evidence.get("CNN Image Analysis Agent") or {}; plot=evidence.get("Plot Agent") or {}
    lines=[
        "1. Executive Summary",
        f"The governed analysis evaluated target '{target}' using the Master Agent route '{route}'. This report is restricted to recorded analytical evidence and explicitly documented human approvals.",
        "",
        "2. Introduction and Analytical Objective",
        "The objective was to evaluate the predictive structure of the supplied dataset using leakage-aware preprocessing, training-only model selection, locked-test evaluation and diagnostic analysis. The report distinguishes observed measurements from interpretation.",
        "",
        "3. Data and Governance",
        f"Dataset card: {json.dumps(evidence.get('dataset_card',{}),default=str)[:7000]}",
        f"Leakage gate: {json.dumps(evidence.get('leakage_gate',{}),default=str)[:5000]}",
        f"Target/feature analysis: {json.dumps(evidence.get('target_feature_analysis',{}),default=str)[:7000]}",
        "",
        "4. Methods and Validation",
        f"ML validation protocol: {json.dumps(ml.get('validation_protocol',{}),default=str)[:7000]}",
        f"DL validation protocol: {json.dumps(dl.get('validation_protocol',{}),default=str)[:7000]}",
        "All reported model-selection procedures are intended to use training-only validation. The final test partition is reserved for generalisation assessment.",
        "",
        "5. Results",
        f"ML Agent: {ml.get('summary','No ML result recorded.')}",
        f"ML best model: {ml.get('best_model','not recorded')}; metrics: {json.dumps(ml.get('best_metrics',{}),default=str)}",
        f"DL Agent: {dl.get('summary','No DL result recorded.')}",
        f"CNN Image Analysis Agent: {cnn.get('status','No CNN result recorded.')}; metrics: {json.dumps(cnn.get('metrics',{}),default=str)}",
        f"DL best model: {dl.get('best_model','not recorded')}; metrics: {json.dumps(dl.get('best_metrics',{}),default=str)}",
        "",
        "6. Evaluation, Uncertainty and Robustness",
        f"ML evaluation: {json.dumps(ml.get('evaluation',{}),default=str)[:12000]}",
        f"DL evaluation: {json.dumps(dl.get('evaluation',{}),default=str)[:12000]}",
        f"ML feature stability: {json.dumps(ml.get('feature_stability',{}),default=str)[:7000]}",
        f"ML temporal availability: {json.dumps(ml.get('temporal_availability',{}),default=str)[:7000]}",
        f"ML counterfactual leakage: {json.dumps(ml.get('counterfactual_leakage',{}),default=str)[:7000]}",
        f"Model diagnosis: {json.dumps(ml.get('model_diagnosis',{}),default=str)[:7000]}",
        "",
        "7. Visual Results",
        f"Agent Plot generated {len(plot.get('plots',[]) if isinstance(plot,dict) else [])} evidence-linked visualizations.",
    ]
    if isinstance(plot,dict):
        for p in plot.get("plots",[]):
            lines.append(f"• {p.get('title')} — quantity: {p.get('quantity')}; fields: {p.get('fields')}; rationale: {p.get('rationale')}; findings: {p.get('insights',[])}")
    lines += [
        "",
        "8. Discussion",
        "The recorded results should be interpreted in the context of the observed dataset, target definition, feature availability and validation strategy. Associations in exploratory plots do not establish causality. Prediction uncertainty and robustness diagnostics are evidence about model behaviour, not guarantees of future performance.",
        "",
        "9. Conclusion",
        "The analysis reached the conclusions supported by the recorded model metrics, validation evidence, diagnostics and visualizations. Any deployment or scientific interpretation beyond this evidence requires additional domain and operational review.",
        "",
        "10. Recommendations and Next Checks",
        "• Review unresolved model-diagnosis flags before deployment.\n• Reassess feature availability under the intended prediction timestamp.\n• Monitor production drift and subgroup performance.\n• Preserve the analysis recipe, model card, dataset card and evidence DAG with the delivered model.",
        "",
        "11. Human Oversight and Reproducibility",
        f"Approval gates recorded: {len(evidence.get('human_approval_evidence',[]))}",
        f"Agent Why evidence: {json.dumps(evidence.get('agent_why',[]),default=str)[:9000]}",
        f"Agent Self-Checks: {json.dumps(evidence.get('agent_self_checks',[]),default=str)[:9000]}",
        f"Compute policy: {json.dumps(evidence.get('compute_info',{}),default=str)[:4000]}",
        "",
        "12. Evidence Appendix",
        f"Evidence DAG: {json.dumps(evidence.get('evidence_dag',{}),default=str)[:12000]}",
        "The report must be read together with the underlying dataset and recorded analysis artifacts; it is not a substitute for the primary data.",
    ]
    return "\n".join(lines)


def _render_plot_images(evidence):
    """Perform the render plot images operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    images=[]; plot=evidence.get("Plot Agent") or {}
    for i,item in enumerate(plot.get("plots",[]) if isinstance(plot,dict) else []):
        fj=item.get("figure_json")
        if not fj: continue
        path=Path(tempfile.gettempdir())/f"dssp_report_plot_{os.getpid()}_{i}.png"
        try:
            import plotly.io as pio
            pio.from_json(fj).write_image(str(path),format="png",width=1200,height=700,scale=1)
            if path.exists(): images.append((str(path),item.get("title",f"Visualization {i+1}"),item.get("rationale","")))
        except Exception: pass
    return images


def _pdf_bytes(text,evidence):
    """Perform the pdf bytes operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    pdf=FPDF(); pdf.set_margins(17,17,17); pdf.set_auto_page_break(auto=True,margin=15); pdf.set_title("Data Science Studio Pro — Professional Analysis Report")
    # Title page follows the supplied professional report pattern: title, subtitle,
    # presenter/date and analysis metadata.
    pdf.add_page(); pdf.set_font("Helvetica","B",22); pdf.set_text_color(13,71,161); pdf.multi_cell(0,12,"Data Science Studio Pro",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.set_font("Helvetica","B",17); pdf.multi_cell(0,10,"Professional Data Analysis Report",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.set_text_color(40,40,40); pdf.ln(15); pdf.set_font("Helvetica",size=12); pdf.multi_cell(0,7,f"Target: {evidence.get('target','Not specified')}\nMaster route: {evidence.get('master_route','Not specified')}\nGenerated: {datetime.now():%Y-%m-%d %H:%M}",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(18); pdf.set_font("Helvetica",size=10); pdf.set_text_color(100,100,100); pdf.multi_cell(0,6,"Evidence-bound report. Interpret all conclusions within the recorded dataset, validation protocol and human approvals.",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.set_text_color(0,0,0)
    pdf.add_page(); pdf.set_font("Helvetica","B",16); pdf.multi_cell(0,8,"Table of Contents",new_x="LMARGIN",new_y="NEXT"); pdf.set_font("Helvetica",size=10)
    toc=["1. Executive Summary","2. Introduction and Analytical Objective","3. Data and Governance","4. Methods and Validation","5. Results","6. Evaluation, Uncertainty and Robustness","7. Visual Results","8. Discussion","9. Conclusion","10. Recommendations and Next Checks","11. Human Oversight and Reproducibility","12. Evidence Appendix"]
    for x in toc: pdf.multi_cell(0,6,x,new_x="LMARGIN",new_y="NEXT")
    images=_render_plot_images(evidence)
    for raw in str(text).splitlines():
        line=raw.strip()
        if not line: pdf.ln(3); continue
        if len(line)>=3 and line[0].isdigit() and ". " in line[:4]:
            pdf.set_font("Helvetica","B",14); pdf.set_text_color(13,71,161); pdf.ln(2)
        else:
            pdf.set_font("Helvetica",size=10); pdf.set_text_color(0,0,0)
        safe=line.encode("latin-1","replace").decode("latin-1")
        for wrapped in textwrap.wrap(safe,width=105,break_long_words=True,break_on_hyphens=False) or [" "]:
            pdf.multi_cell(0,5.5,wrapped,new_x="LMARGIN",new_y="NEXT")
        if line=="7. Visual Results":
            for path,title,rationale in images:
                try:
                    pdf.set_font("Helvetica","B",11); pdf.multi_cell(0,6,title,new_x="LMARGIN",new_y="NEXT"); pdf.image(path,w=175); pdf.set_font("Helvetica",size=9); pdf.multi_cell(0,5,textwrap.fill(rationale,100),new_x="LMARGIN",new_y="NEXT"); pdf.ln(3)
                except Exception: pass
    for path,_,_ in images:
        try: Path(path).unlink(missing_ok=True)
        except Exception: pass
    return bytes(pdf.output())


def generate_report_plan_node(state):
    """Audit Data Scientist/Plot evidence and ask the configured LLM for a constrained report plan."""
    evidence=json_safe(state.get("analysis_evidence") or {})
    audit=_audit_evidence(evidence)
    llm=get_llm(provider_from_state(state))
    plan={
        "title":"Professional Data Analysis Report",
        "sections":["Executive Summary","Analytical Objective","Data and Governance","Methodology","Validation Strategy","Results","Robustness and Uncertainty","Visual Findings","Human Oversight","Limitations and Open Questions","Reproducibility","Evidence Appendix"],
        "evidence_policy":"Use only Data Scientist Evidence and Plot Evidence supplied in state; do not invent metrics, sample sizes, methods, or causal claims.",
    }
    if llm is not None:
        prompt=(REPORT_AGENT_SYSTEM_PROMPT+"\n\nYou are planning a professional PDF report. Return JSON with keys title, sections, emphasis, unresolved_questions. Use ONLY the supplied Data Scientist Evidence and Plot Evidence. Do not invent facts.\n\nData Scientist Evidence:\n"+json.dumps({k:v for k,v in evidence.items() if k!="Plot Agent"},default=str)[:65000]+"\n\nPlot Evidence:\n"+json.dumps(evidence.get("Plot Agent") or {},default=str)[:25000])
        try:
            response=llm.invoke([HumanMessage(content=prompt)])
            text=response.content if hasattr(response,"content") else str(response)
            match=__import__("re").search(r"\{.*\}",text,__import__("re").S)
            if match:
                candidate=json.loads(match.group(0))
                if isinstance(candidate,dict):
                    plan.update({k:candidate[k] for k in ("title","sections","emphasis","unresolved_questions") if k in candidate})
        except Exception:
            pass
    return {"evidence_audit":audit,"report_plan":plan,"messages":[AIMessage(content="Report plan prepared from Data Scientist Evidence + Plot Evidence + configured LLM.")]}


def generate_report_narrative_node(state):
    """Generate section-level, coherent evidence-bound report prose."""
    evidence=json_safe(state.get("analysis_evidence") or {})
    plan=state.get("report_plan") or {}
    llm=get_llm(provider_from_state(state))
    sections=EvidenceNarrativeEngine.deterministic(evidence)
    if llm is not None:
        prompt=EvidenceNarrativeEngine.build_prompt(evidence, format_name="professional analytical report") + "\n\nApproved report plan:\n" + json.dumps(plan,default=str)
        try:
            response=llm.invoke([HumanMessage(content=prompt)])
            raw=response.content if hasattr(response,"content") else str(response)
            match=__import__("re").search(r"\{.*\}",raw,__import__("re").S)
            candidate=json.loads(match.group(0)) if match else {}
            if isinstance(candidate,dict):
                for key in REPORT_SECTIONS:
                    if isinstance(candidate.get(key),str) and candidate[key].strip(): sections[key]=candidate[key].strip()
        except Exception:
            pass
    text="\n\n".join(f"{i}. {key.replace('_',' ').title()}\n{sections[key]}" for i,key in enumerate(REPORT_SECTIONS,1))
    quality=OutputQualityGate.report(text,evidence)
    return {"report_text":text,"report_sections":sections,"output_quality":quality,"messages":[AIMessage(content="Coherent section-level report narrative generated from structured evidence.")]}

def render_report_pdf_node(state):
    """Render the approved report narrative and linked plot evidence into a PDF buffer."""
    evidence=json_safe(state.get("analysis_evidence") or {})
    text=state.get("report_text") or _fallback_report(evidence)
    memory=AgentMemory("Report",20)
    memory.remember("report_context",{"target":evidence.get("target"),"plots":len((evidence.get("Plot Agent") or {}).get("plots",[])) if isinstance(evidence.get("Plot Agent"),dict) else 0})
    return {"pdf_buffer":io.BytesIO(_pdf_bytes(text,evidence)),"memory":memory.to_dict(),"messages":[AIMessage(content="Professional PDF report rendered from Data Scientist Evidence + Plot Evidence + LLM narrative.")]}


workflow=StateGraph(ReportState)
workflow.add_node("evidence_audit_and_plan",generate_report_plan_node)
workflow.add_node("llm_narrative",generate_report_narrative_node)
workflow.add_node("python_pdf",render_report_pdf_node)
workflow.set_entry_point("evidence_audit_and_plan")
workflow.add_edge("evidence_audit_and_plan","llm_narrative")
workflow.add_edge("llm_narrative","python_pdf")
workflow.add_edge("python_pdf",END)
agent_report_app=workflow.compile()
