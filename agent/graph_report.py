from __future__ import annotations
from typing import Any, TypedDict
import io, json, textwrap
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

def get_llm(state):
    model_name=state.get("model_name","")
    if model_name in config.LOCAL_LLM_MODELS: return LocalLLMLoader().get_llm(model_name)
    if not state.get("api_key"):
        # Prefer a downloaded local model automatically when no API key is configured.
        for local_name, rel in config.LOCAL_LLM_MODELS.items():
            cache_path = __import__("os").path.join(config.HF_CACHE_DIR, rel)
            if __import__("os").path.exists(cache_path):
                return LocalLLMLoader().get_llm(local_name)
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name,api_key=state["api_key"],temperature=0)
    except Exception: return None

def _evidence_text(state):
    evidence=state.get("analysis_evidence") or {}
    return json.dumps(evidence,default=str,indent=2)[:60000]

def _fallback_report(evidence):
    master=evidence.get("master_route","not recorded"); target=evidence.get("target","not recorded")
    ml=evidence.get("ML Agent") or {}; dl=evidence.get("DL Agent") or {}; plot=evidence.get("Plot Agent") or {}
    lines=[
        "Data Science Studio Pro — AI Agent Report", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "", "Executive Summary",
        f"The Master Agent selected the recorded route '{master}' for target '{target}'. The report is evidence-bound: statements below are based only on recorded analysis artifacts.",
        "", "Dataset and Governance",
        f"Dataset Card: {json.dumps(evidence.get('dataset_card',{}),default=str)[:5000]}",
        f"Scientific/Data Leakage Gate: {json.dumps(evidence.get('leakage_gate',{}),default=str)[:5000]}",
        "", "Model Results",
        f"ML Agent: {ml.get('summary','No ML result recorded.')}",
        f"DL Agent: {dl.get('summary','No DL result recorded.')}",
        f"Plot Agent: {plot.get('summary','No Plot result recorded.')}",
        "", "Interpretation and Advice",
        "The recorded metrics should be interpreted together with the uncertainty, data-quality and leakage checks. Before deployment, review the model card, confirm the evaluation population and validate that no future information entered the predictors.",
        "", "Human Oversight and Reproducibility",
        f"Human approval evidence: {len(evidence.get('human_approval_evidence',[]))} recorded gate(s).",
        "Limitations: this report does not claim causal relationships, generalisation beyond the evaluated data, or production readiness unless those claims are explicitly supported by evidence.",
        "", "Evidence appendix", json.dumps(evidence,default=str,indent=2)[:20000]
    ]
    return "\n".join(lines)

def _pdf_bytes(text):
    pdf=FPDF(); pdf.set_margins(18,18,18); pdf.add_page(); pdf.set_auto_page_break(auto=True,margin=16)
    pdf.set_title("Data Science Studio Pro — AI Agent Report")
    pdf.set_font("Helvetica",size=11)
    # fpdf2 raises 'Not enough horizontal space...' for long unbroken tokens.
    for raw in str(text).replace("\t","    ").splitlines():
        if not raw.strip(): pdf.ln(4); continue
        safe=raw.encode("latin-1","replace").decode("latin-1")
        for line in textwrap.wrap(safe,width=40,break_long_words=True,break_on_hyphens=False,replace_whitespace=False) or [" "]:
            pdf.multi_cell(0,6.5,line,new_x="LMARGIN",new_y="NEXT")
    return bytes(pdf.output())

def generate_report_node(state):
    evidence=_evidence_text(state); llm=get_llm(state)
    if llm is not None:
        try:
            prompt=REPORT_AGENT_SYSTEM_PROMPT+"\n\nEvidence:\n"+evidence
            response=llm.invoke([HumanMessage(content=prompt)])
            report_text=response.content if hasattr(response,"content") else str(response)
        except Exception as exc:
            report_text=_fallback_report(state.get("analysis_evidence") or {})+f"\n\nLLM note: {exc}"
    else:
        report_text=_fallback_report(state.get("analysis_evidence") or {})
    mem=AgentMemory("Report",20); mem.remember("evidence_summary", {"keys": list((state.get("analysis_evidence") or {}).keys())})
    return {"pdf_buffer":io.BytesIO(_pdf_bytes(report_text)),"report_text":report_text, "memory":mem.to_dict(),
            "messages":[AIMessage(content="AI Agent Report generated with narrative interpretation, highlights, advice, limitations and evidence provenance.")]}

report_workflow=StateGraph(ReportState); report_workflow.add_node("generate_report",generate_report_node); report_workflow.set_entry_point("generate_report"); report_workflow.add_edge("generate_report",END); agent_report_app=report_workflow.compile()
