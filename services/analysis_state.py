from __future__ import annotations
from enum import Enum

class AnalysisState(str, Enum):
    DATA_LOADED="DATA_LOADED"; QUALITY_CHECKED="QUALITY_CHECKED"; CONTRACT_VALIDATED="CONTRACT_VALIDATED"
    ANALYSIS_READY="ANALYSIS_READY"; MODEL_READY="MODEL_READY"; MODEL_VALIDATED="MODEL_VALIDATED"
    REPORT_READY="REPORT_READY"; PRESENTATION_READY="PRESENTATION_READY"; SHAREABLE="SHAREABLE"; MONITORED="MONITORED"

_ALLOWED={
    "DATA_LOADED":{"QUALITY_CHECKED"}, "QUALITY_CHECKED":{"CONTRACT_VALIDATED","ANALYSIS_READY"},
    "CONTRACT_VALIDATED":{"ANALYSIS_READY"}, "ANALYSIS_READY":{"MODEL_READY","REPORT_READY"},
    "MODEL_READY":{"MODEL_VALIDATED","REPORT_READY"}, "MODEL_VALIDATED":{"REPORT_READY","MONITORED"},
    "REPORT_READY":{"PRESENTATION_READY","SHAREABLE"}, "PRESENTATION_READY":{"SHAREABLE"},
    "SHAREABLE":{"MONITORED"}, "MONITORED":set()
}
class AnalysisStateMachine:
    def __init__(self,state="DATA_LOADED"): self.state=state
    def can_transition(self,next_state): return next_state in _ALLOWED.get(self.state,set())
    def transition(self,next_state):
        if not self.can_transition(next_state): raise ValueError(f"Invalid analysis transition: {self.state} → {next_state}")
        self.state=next_state; return self.state
    def to_dict(self): return {"state":self.state,"allowed_next":sorted(_ALLOWED.get(self.state,set()))}
