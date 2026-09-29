from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import json, hashlib

class AnalysisRecipe:
    """Immutable-ish record of how an analysis was produced."""
    def __init__(self): self.steps: list[dict[str, Any]] = []
    def add(self, action: str, parameters: dict[str, Any] | None = None, *, artifact_ids=None, evidence_ids=None):
        self.steps.append({"time":datetime.now(timezone.utc).isoformat(),"action":action,
                           "parameters":parameters or {},"artifact_ids":artifact_ids or [],"evidence_ids":evidence_ids or []})
    def to_dict(self): return {"version":"1.0","steps":self.steps,"recipe_id":self.fingerprint()}
    def fingerprint(self): return hashlib.sha256(json.dumps(self.steps,sort_keys=True,default=str).encode()).hexdigest()[:16]
