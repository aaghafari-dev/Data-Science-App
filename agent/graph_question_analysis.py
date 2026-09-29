from __future__ import annotations
from typing import Any, TypedDict
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from services.target_feature_selection import recommend_targets_and_features

class QuestionState(TypedDict, total=False):
    question: str; dataframe: pd.DataFrame; target_candidates: list; feature_candidates: list
    suggested_target: str; suggested_features: list; plan: list; memory: list[dict[str,Any]]; status: str; summary: str

def _understand(s):
    df=s.get("dataframe")
    rec=recommend_targets_and_features(df)
    q=s.get("question","")
    plan=["Inspect core dtypes and data quality", "Identify target candidates and possible leakage", "Identify candidate predictors", "Select an analysis/model route only after human review"]
    summary=f"Question Agent interpreted: '{q}'. Suggested target candidate: {rec.get('suggested_target') or 'none'}; candidate features: {len(rec.get('suggested_features',[]))}."
    mem=list(s.get("memory",[])); mem.append({"question":q,"suggested_target":rec.get("suggested_target"),"warnings":rec.get("warnings",[])})
    return {"target_candidates":rec["target_candidates"],"feature_candidates":rec["feature_candidates"],"suggested_target":rec.get("suggested_target"),"suggested_features":rec.get("suggested_features",[]),"plan":plan,"memory":mem[-10:],"status":"awaiting_review","summary":summary,"messages":[AIMessage(content=summary)]}
workflow=StateGraph(QuestionState); workflow.add_node("understand",_understand); workflow.set_entry_point("understand"); workflow.add_edge("understand",END); question_analysis_app=workflow.compile()
