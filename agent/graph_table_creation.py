from __future__ import annotations

from typing import Any, TypedDict
from pathlib import Path
import re
from io import StringIO
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None


class TableCreationState(TypedDict, total=False):
    messages: list[BaseMessage]
    mode: str
    url: str
    api_key: str
    api_header: str
    prompt: str
    memory: list[dict[str, Any]]
    raw_content: str
    dataframe: pd.DataFrame
    status: str
    summary: str
    error: str


def _plan(state):
    mode = state.get("mode", "Web Scraping")
    msg = f"AI Agent Table Creation plans a {mode} workflow for {state.get('url','the supplied source')}."
    return {"status":"planned", "summary":msg, "messages":[AIMessage(content=msg)]}


def _fetch(state):
    url = state.get("url", "").strip()
    if not url: return {"status":"error", "error":"A URL is required."}
    headers = {"User-Agent":"DataScienceStudioPro/18.3 (+data-analysis-agent)"}
    mode = state.get("mode", "Web Scraping")
    if mode == "API Key":
        header = state.get("api_header") or "Authorization"
        key = state.get("api_key", "")
        if key: headers[header] = key if header.lower() == "authorization" and key.lower().startswith("bearer ") else (f"Bearer {key}" if header.lower()=="authorization" else key)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return {"raw_content":response.text, "status":"fetched", "summary":f"Retrieved {len(response.content):,} bytes from the source."}
    except Exception as exc:
        return {"status":"error", "error":str(exc)}


def _extract(state):
    raw = state.get("raw_content", "")
    mode = state.get("mode", "Web Scraping")
    try:
        if mode == "API Key":
            import json
            obj = json.loads(raw)
            if isinstance(obj, list): df = pd.json_normalize(obj)
            elif isinstance(obj, dict):
                candidate = next((v for v in obj.values() if isinstance(v,list)), obj)
                df = pd.json_normalize(candidate)
            else: df = pd.DataFrame({"value":[obj]})
        else:
            soup = BeautifulSoup(raw, "html.parser")
            tables = pd.read_html(StringIO(str(soup)))
            if not tables: raise ValueError("No HTML table was found on the page.")
            # Choose the largest table; the user can refine it after creation.
            df = max(tables, key=lambda x: x.shape[0]*max(1,x.shape[1])).copy()
        return {"dataframe":df, "status":"validated", "summary":f"Created a table with {len(df):,} rows and {len(df.columns):,} columns."}
    except Exception as exc:
        return {"status":"error", "error":str(exc)}


def _finalize(state):
    memory = list(state.get("memory", []))
    memory.append({"url":state.get("url"), "mode":state.get("mode"), "summary":state.get("summary")})
    return {"memory":memory[-10:], "status":"complete", "summary":state.get("summary","Table creation completed."),
            "messages":[AIMessage(content=state.get("summary","Table creation completed."))]}


def _route(state):
    if state.get("status") == "error": return "end"
    step = state.get("status")
    return "fetch" if step == "planned" else "extract" if step == "fetched" else "finalize" if step == "validated" else "end"

workflow = StateGraph(TableCreationState)
workflow.add_node("plan", _plan); workflow.add_node("fetch", _fetch); workflow.add_node("extract", _extract); workflow.add_node("finalize", _finalize)
workflow.set_entry_point("plan")
workflow.add_conditional_edges("plan", _route, {"fetch":"fetch","end":END})
workflow.add_conditional_edges("fetch", _route, {"extract":"extract","end":END})
workflow.add_conditional_edges("extract", _route, {"finalize":"finalize","end":END})
workflow.add_edge("finalize", END)
agent_table_creation_app = workflow.compile()  # Sensitive API credentials are intentionally not checkpoint-persisted; bounded memory is explicit state.
