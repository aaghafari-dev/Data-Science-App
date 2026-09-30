from __future__ import annotations
from typing import Any, TypedDict
import io, json, textwrap, tempfile, os
from pathlib import Path
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from fpdf import FPDF
from .prompts import REPORT_AGENT_SYSTEM_PROMPT
from .local_llm import LocalLLMLoader
from services.agent_memory import AgentMemory
import config

class ReportState(TypedDict, total=False):
    analysis_evidence: dict[str, Any]
    api_key: str
    model_name: str
    pdf_buffer: io.BytesIO
    report_text: str
    messages: list
    memory: list[dict[str,Any]]


def _safe(v):
    if isinstance(v, dict): return {str(k): _safe(x) for k,x in v.items() if k not in {"model_object","figure"}}
    if isinstance(v, list): return [_safe(x) for x in v]
    try:
        import pandas as pd, numpy as np
        if isinstance(v, pd.DataFrame): return {"type":"DataFrame","shape":list(v.shape),"columns":[str(c) for c in v.columns],"preview":v.head(15).to_dict(orient="records")}
        if isinstance(v, pd.Series): return v.head(30).tolist()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer,np.floating)): return v.item()
    except Exception: pass
    try: json.dumps(v); return v
    except Exception: return str(v)


def get_llm(state):
    model_name=state.get("model_name","")
    if model_name in config.LOCAL_LLM_MODELS: return LocalLLMLoader().get_llm(model_name)
    if not state.get("api_key"):
        for local_name, rel in config.LOCAL_LLM_MODELS.items():
            cache_path=os.path.join(config.HF_CACHE_DIR, rel)
            if os.path.exists(cache_path): return LocalLLMLoader().get_llm(local_name)
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name,api_key=state["api_key"],temperature=0)
    except Exception: return None


def _evidence_text(state):
    return json.dumps(_safe(state.get("analysis_evidence") or {}),default=str,indent=2)[:80000]


def _fallback_report(evidence):
    master=evidence.get("master_route","not recorded"); target=evidence.get("target","not recorded")
    ml=evidence.get("ML Agent") or {}; dl=evidence.get("DL Agent") or {}; plot=evidence.get("Plot Agent") or {}
    lines=[
        "Data Science Studio Pro — Professional AI Agent Report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "1. Executive Summary",
        f"The governed Master Agent analysed the dataset for target '{target}' using route '{master}'. The conclusions in this report are restricted to recorded evidence. Human approval gates and validation evidence are preserved in the audit trail.",
        "",
        "2. Analytical Objective and Dataset",
        f"Target: {target}",
        f"Dataset governance: {json.dumps(evidence.get('dataset_card',{}),default=str)[:6000]}",
        f"Leakage gate: {json.dumps(evidence.get('leakage_gate',{}),default=str)[:5000]}",
        "",
        "3. Methodology and Validation",
        f"ML validation: {json.dumps((ml.get('validation_protocol') or (ml.get('evaluation') or {}).get('best_model',{}).get('validation_protocol',{})),default=str)[:6000]}",
        f"DL validation: {json.dumps((dl.get('validation_protocol') or (dl.get('evaluation') or {}).get('best_model',{}).get('validation_protocol',{})),default=str)[:6000]}",
        "Preprocessing and feature transformations must be interpreted as fold-local whenever stated by the recorded pipeline. The final test partition is reserved for final generalisation assessment.",
        "",
        "4. Model Results and Professional Evaluation",
        f"ML: {ml.get('summary','No ML result recorded.')}",
        f"DL: {dl.get('summary','No DL result recorded.')}",
        f"ML evaluation: {json.dumps(ml.get('evaluation',{}),default=str)[:10000]}",
        f"DL evaluation: {json.dumps(dl.get('evaluation',{}),default=str)[:10000]}",
        "",
        "5. Diagnostics and Robustness",
        f"ML feature stability: {json.dumps(ml.get('feature_stability',{}),default=str)[:5000]}",
        f"ML temporal availability: {json.dumps(ml.get('temporal_availability',{}),default=str)[:5000]}",
        f"ML counterfactual leakage: {json.dumps(ml.get('counterfactual_leakage',{}),default=str)[:5000]}",
        f"ML diagnosis: {json.dumps(ml.get('model_diagnosis',{}),default=str)[:6000]}",
        f"DL diagnosis: {json.dumps(dl.get('model_diagnosis',{}),default=str)[:6000]}",
        "",
        "6. Visual Findings",
        f"Agent Plot generated {len((plot.get('plots') or [])) if isinstance(plot,dict) else 0} analytical visualizations.",
    ]
    if isinstance(plot,dict):
        for item in plot.get("plots",[]):
            lines.append(f"• {item.get('title')} — fields: {item.get('fields')}; rationale: {item.get('rationale')}; findings: {item.get('insights',[])}")
    lines += [
        "",
        "7. Human Oversight and Agent Decision Evidence",
        f"Approval gates recorded: {len(evidence.get('human_approval_evidence',[]))}",
        json.dumps(evidence.get('agent_why',[]),default=str,indent=2)[:10000],
        json.dumps(evidence.get('agent_self_checks',[]),default=str,indent=2)[:10000],
        "",
        "8. Limitations and Deployment Considerations",
        "Predictive performance is conditional on the evaluated population, validation strategy and preprocessing protocol. Prediction uncertainty is not automatically a deployment guarantee. Distribution shift, subgroup performance, feature availability and operational costs should be reviewed before deployment.",
        "",
        "9. Reproducibility and Evidence Appendix",
        f"Compute information: {json.dumps(evidence.get('compute_info',{}),default=str)[:4000]}",
        f"Evidence DAG: {json.dumps(evidence.get('evidence_dag',{}),default=str)[:12000]}",
    ]
    return "\n".join(lines)


def _render_plot_images(evidence):
    images=[]; plot=evidence.get("Plot Agent") or {}
    for i,item in enumerate(plot.get("plots",[]) if isinstance(plot,dict) else []):
        fig_json=item.get("figure_json")
        if not fig_json: continue
        path=Path(tempfile.gettempdir())/f"dssp_report_plot_{os.getpid()}_{i}.png"
        try:
            import plotly.io as pio
            fig=pio.from_json(fig_json)
            fig.write_image(str(path),format="png",width=1100,height=650,scale=1)
            if path.exists(): images.append((str(path),item.get("title",f"Visualization {i+1}"),item.get("rationale","")))
        except Exception:
            try: path.unlink(missing_ok=True)
            except Exception: pass
    return images


def _pdf_bytes(text, evidence):
    pdf=FPDF(); pdf.set_margins(17,17,17); pdf.set_auto_page_break(auto=True,margin=15); pdf.set_title("Data Science Studio Pro — Professional AI Agent Report")
    pdf.add_page(); pdf.set_font("Helvetica","B",18); pdf.cell(0,12,"Data Science Studio Pro",new_x="LMARGIN",new_y="NEXT")
    pdf.set_font("Helvetica","B",14); pdf.cell(0,10,"Professional AI Agent Report",new_x="LMARGIN",new_y="NEXT")
    pdf.set_font("Helvetica",size=9); pdf.set_text_color(90,90,90); pdf.cell(0,7,f"Generated {datetime.now():%Y-%m-%d %H:%M}",new_x="LMARGIN",new_y="NEXT"); pdf.set_text_color(0,0,0); pdf.ln(4)
    images=_render_plot_images(evidence)
    image_iter=iter(images)
    for raw in str(text).replace("\t","    ").splitlines():
        line=raw.strip()
        if not line: pdf.ln(3); continue
        if line[:2].isdigit() and ". " in line[:5]:
            pdf.set_font("Helvetica","B",13); pdf.ln(2)
        else: pdf.set_font("Helvetica",size=10)
        safe=line.encode("latin-1","replace").decode("latin-1")
        for wrapped in textwrap.wrap(safe,width=105,break_long_words=True,break_on_hyphens=False) or [" "]:
            pdf.multi_cell(0,5.5,wrapped,new_x="LMARGIN",new_y="NEXT")
        if line == "6. Visual Findings":
            for path,title,rationale in images:
                try:
                    pdf.set_font("Helvetica","B",11); pdf.multi_cell(0,6,title,new_x="LMARGIN",new_y="NEXT")
                    pdf.image(path,w=175); pdf.set_font("Helvetica",size=9); pdf.multi_cell(0,5,textwrap.fill(rationale,100),new_x="LMARGIN",new_y="NEXT"); pdf.ln(3)
                except Exception: pass
    for path,_,_ in images:
        try: Path(path).unlink(missing_ok=True)
        except Exception: pass
    return bytes(pdf.output())


def generate_report_node(state):
    evidence=_safe(state.get("analysis_evidence") or {})
    llm=get_llm(state)
    if llm is not None:
        try:
            instruction=(REPORT_AGENT_SYSTEM_PROMPT+"\n\nProduce a professional analytical report with numbered sections, explicit methodology, model evaluation, uncertainty, robustness, visual findings, limitations, deployment considerations and evidence traceability. Use the supplied Agent Data Scientist and Agent Plot evidence. Do not invent metrics.\n\nEvidence:\n"+json.dumps(evidence,default=str,indent=2)[:80000])
            response=llm.invoke([HumanMessage(content=instruction)])
            report_text=response.content if hasattr(response,"content") else str(response)
        except Exception as exc:
            report_text=_fallback_report(evidence)+f"\n\nLLM generation note: {exc}"
    else:
        report_text=_fallback_report(evidence)
    mem=AgentMemory("Report",20); mem.remember("evidence_summary", {"keys": list(evidence.keys()),"visualizations":len((evidence.get("Plot Agent") or {}).get("plots",[])) if isinstance(evidence.get("Plot Agent"),dict) else 0})
    return {"pdf_buffer":io.BytesIO(_pdf_bytes(report_text,evidence)),"report_text":report_text,"memory":mem.to_dict(),"messages":[AIMessage(content="Professional evidence-bound report generated with narrative interpretation and embedded analytical visualizations where the rendering backend is available.")]}

report_workflow=StateGraph(ReportState); report_workflow.add_node("generate_report",generate_report_node); report_workflow.set_entry_point("generate_report"); report_workflow.add_edge("generate_report",END); agent_report_app=report_workflow.compile()
