"""Module duty: Agent console.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

class AgentRunConsole:
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.events: list[dict[str,Any]]=[]
    def log(self, agent: str, node: str, status: str, message: str, *, evidence_id=None):
        """Perform the log operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.events.append({"time":datetime.now(timezone.utc).isoformat(),"agent":agent,"node":node,"status":status,"message":message,"evidence_id":evidence_id})
        self.events=self.events[-500:]
    def clear(self):
        """Perform the clear operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.events.clear()
    def to_text(self):
        """Perform the to text operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return "\n".join(f"[{e['time']}] {e['agent']} | {e['node']} | {e['status']} | {e['message']}" for e in self.events)
