from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

class AgentRunConsole:
    def __init__(self): self.events: list[dict[str,Any]]=[]
    def log(self, agent: str, node: str, status: str, message: str, *, evidence_id=None):
        self.events.append({"time":datetime.now(timezone.utc).isoformat(),"agent":agent,"node":node,"status":status,"message":message,"evidence_id":evidence_id})
        self.events=self.events[-500:]
    def clear(self): self.events.clear()
    def to_text(self): return "\n".join(f"[{e['time']}] {e['agent']} | {e['node']} | {e['status']} | {e['message']}" for e in self.events)
