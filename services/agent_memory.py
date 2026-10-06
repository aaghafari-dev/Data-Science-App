"""Module duty: Agent memory.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import os

@dataclass
class AgentMemory:
    """Small, bounded, role-specific working memory. Secrets are never stored."""
    role: str
    max_items: int = 20
    items: list[dict[str, Any]] = field(default_factory=list)

    def remember(self, kind: str, content: Any, *, evidence_id: str | None = None):
        """Perform the remember operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        safe_content = self._sanitize(content)
        safe = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind,
                "content": safe_content, "evidence_id": evidence_id}
        self.items.append(safe)
        run_id = os.environ.get("DSP_AGENT_RUN_ID", "").strip()
        if run_id:
            try:
                from services.agent_memory_store import AgentMemoryStore
                AgentMemoryStore().record(run_id, self.role, kind, safe_content, evidence_id=evidence_id)
            except Exception:
                pass
        self.items = self.items[-self.max_items:]


    @staticmethod
    def _sanitize(value: Any) -> Any:
        """Remove credential-like fields before analytical memory is persisted."""
        secret_names={"api_key","apikey","password","passwd","secret","token","authorization","access_token"}
        if isinstance(value, dict):
            return {str(k): "***redacted***" if str(k).lower() in secret_names else AgentMemory._sanitize(v) for k,v in value.items()}
        if isinstance(value, list):
            return [AgentMemory._sanitize(v) for v in value]
        if isinstance(value, tuple):
            return [AgentMemory._sanitize(v) for v in value]
        return value

    def recall(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Perform the recall operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if kind is None:
            return list(self.items)
        return [x for x in self.items if x.get("kind") == kind]

    def to_dict(self):
        """Perform the to dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {"role": self.role, "items": self.recall()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, role: str, max_items: int = 20):
        """Perform the from dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        obj = cls(role=role, max_items=max_items)
        if isinstance(data, dict):
            obj.items = list(data.get("items", []))[-max_items:]
        return obj
