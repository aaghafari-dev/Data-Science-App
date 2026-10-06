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
    ram_gb: float | None
    gpu_vram_gb: float | None
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
    """Create a coherent executive-quality report without dumping raw JSON into the main narrative."""
    target=evidence.get("target") or "not recorded"; route=evidence.get("master_route") or "exploratory analysis"
    ml=evidence.get("ML Agent") or {}; dl=evidence.get("DL Agent") or {}; cnn=evidence.get("CNN Image Analysis Agent") or {}
    plot=evidence.get("Plot Agent") or {}; leakage=evidence.get("leakage_gate") or {}; validation=evidence.get("validation_strategy") or {}
    card=evidence.get("dataset_card") or {}
    plots=plot.get("plots",[]) if isinstance(plot,dict) else []
    def metric_sentence(bundle, label):
        best=bundle.get("best_model") or bundle.get("best_model_name")
        metrics=bundle.get("best_metrics") or bundle.get("metrics") or {}
        if not isinstance(metrics,dict): metrics={}
        selected=[]
        preferred=("R2","RMSE","MAE","accuracy","balanced_accuracy","f1_macro","f1_weighted","precision_macro","recall_macro")
        for k in preferred:
            if k in metrics: selected.append(f"{k}={metrics[k]}")
        if not selected:
            for k,v in list(metrics.items())[:4]: selected.append(f"{k}={v}")
        if best: return f"{label} selected {best} as the recorded best candidate, with " + ", ".join(selected) + "."
        return f"{label} produced recorded evidence, but no best-model designation was available in the supplied state."
    def dataset_sentence():
        if not isinstance(card,dict): return "The dataset card was not available in structured form."
        rows=card.get("rows") or card.get("n_rows") or "not recorded"; cols=card.get("columns") or card.get("n_columns") or "not recorded"
        return f"The active dataset contains approximately {rows} rows and {cols} columns according to the recorded dataset card. Interpretation is conditional on the observed sample, feature availability and provenance."
    leak_status=leakage.get("status","not recorded") if isinstance(leakage,dict) else "not recorded"
    val_name=validation.get("protocol") or validation.get("strategy") or validation.get("name") or "the recorded validation protocol"
    lines=[
        "1. Executive Summary",
        f"The governed analysis addressed the analytical objective for target '{target}' using the Master Agent route '{route}'. The workflow separated task identification, validation design, specialist execution and evidence review, with human approval gates retained in the project record. The principal findings below are limited to the evidence actually produced by the application.",
        "",
        "2. Analytical Objective and Scope",
        f"The objective was to determine the most defensible analytical strategy for the active dataset and to communicate the resulting evidence without exceeding what the data and validation design support. The selected route was {route}. This report is an analytical record, not a causal or deployment guarantee.",
        "",
        "3. Data, Governance and Leakage Control",
        dataset_sentence(),
        f"The Scientific Data Leakage Gate recorded status '{leak_status}'. Predictors are expected to be evaluated for target derivation, future-information availability, duplicate/overlap structure and identifier-like behaviour before model selection. The final interpretation should therefore retain the recorded leakage assessment and any unresolved review flags.",
        "",
        "4. Methodology and Validation",
        f"The Master Agent evaluated candidate methods before specialist execution. Validation was treated as an analytical design decision; the recorded protocol was {val_name}. Model selection and preprocessing are intended to remain separated from the final evaluation data, with robustness and diagnostic evidence considered separately from primary performance.",
        "",
        "5. Results",
        metric_sentence(ml,"The ML analysis") if ml else "No ML result was recorded.",
        metric_sentence(dl,"The DL analysis") if dl else "No conventional DL result was recorded.",
        (f"The CNN Image Analysis Agent recorded status '{cnn.get('status','not recorded')}' with metrics {', '.join(f'{k}={v}' for k,v in list((cnn.get('metrics') or {}).items())[:6])}." if cnn else "No CNN image-analysis result was recorded."),
        "",
        "6. Robustness, Uncertainty and Diagnostics",
        "Primary performance was interpreted separately from robustness. Feature stability, temporal-availability checks, counterfactual leakage tests, model diagnosis and perturbation evidence should be read as information about where the observed result may change. They do not constitute a guarantee of future performance.",
        "",
        "7. Visual Findings",
        f"The Plot Agent produced {len(plots)} evidence-linked visualization(s). Each visualization is treated as a communication view of recorded analytical evidence. Associations are descriptive unless an appropriate inferential design supports a stronger claim.",
    ]
    for pitem in plots:
        if not isinstance(pitem,dict): continue
        title=pitem.get("title") or "Untitled visualization"; fields=pitem.get("fields") or []
        insight=" ".join(str(x) for x in (pitem.get("insights") or []) if x)
        lines.append(f"• {title}: fields={', '.join(map(str,fields))}. {insight or pitem.get('rationale','Evidence-linked visualization supporting the recorded analysis.')}")
    lines += [
        "",
        "8. Discussion",
        "Taken together, the recorded evidence supports interpretation of the selected analytical task under the observed dataset and validation design. The application deliberately distinguishes observed relationships and predictive behaviour from causal explanations. Domain knowledge remains necessary to determine whether the observed patterns are scientifically or operationally meaningful.",
        "",
        "9. Limitations and Open Questions",
        "The principal limitations are the observed sample, feature availability, validation assumptions, model specification, unresolved evidence gaps and the distinction between predictive association and causal explanation. Any decision beyond these boundaries requires additional domain, experimental or operational evidence.",
        "",
        "10. Recommendations and Next Checks",
        "1. Review every unresolved leakage, validation and model-diagnostic flag before external use.\n2. Confirm that all predictors would be available at the intended decision time.\n3. Compare the selected model against a transparent baseline and an appropriate challenger where justified.\n4. Preserve dataset, analysis recipe, model/evidence versions and approval records.\n5. For scientific or high-impact decisions, perform independent domain validation before deployment or causal interpretation.",
        "",
        "11. Human Oversight and Reproducibility",
        f"The evidence record contains {len(evidence.get('human_approval_evidence',[]))} human approval event(s). Agent Why and Agent Self-Check evidence are retained separately so that analytical decisions can be reviewed. The reproducibility package should be retained with the dataset version, random seeds, provider configuration and generated artifacts.",
        "",
        "12. Evidence Appendix",
        "The Evidence Appendix contains the machine-readable governance and provenance record. Raw evidence is intentionally separated from the executive narrative so that the report remains readable while retaining an auditable trail for technical review.",
        json.dumps({"leakage_gate":leakage,"validation_strategy":validation,"evidence_dag":evidence.get("evidence_dag",{})},default=str,indent=2)[:18000],
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
            if path.exists(): images.append((str(path),item.get("title",f"Visualization {i+1}"),item.get("caption") or item.get("rationale",""),item.get("explanation", "")))
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
            for path,title,caption,explanation in images:
                try:
                    pdf.set_font("Helvetica","B",11); pdf.multi_cell(0,6,title,new_x="LMARGIN",new_y="NEXT"); pdf.image(path,w=175); pdf.set_font("Helvetica","I",9); pdf.multi_cell(0,5,textwrap.fill("Caption: "+caption,100),new_x="LMARGIN",new_y="NEXT"); pdf.set_font("Helvetica",size=9); pdf.multi_cell(0,5,textwrap.fill("Explanation: "+explanation,100),new_x="LMARGIN",new_y="NEXT"); pdf.ln(3)
                except Exception: pass
    for path,_,_,_ in images:
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
