from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
import hashlib, json


class AnalysisRecipe:
    """Versioned, replayable record of how an analysis was produced.

    Replay is intentionally explicit: the recipe describes requested actions but
    never bypasses current validation, permissions or human approval gates.
    """
    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.version = "2.0"

    def add(self, action: str, parameters: dict[str, Any] | None = None, *, artifact_ids=None, evidence_ids=None, status: str = "complete"):
        self.steps.append({"time":datetime.now(timezone.utc).isoformat(),"action":action,"parameters":parameters or {},"artifact_ids":artifact_ids or [],"evidence_ids":evidence_ids or [],"status":status})

    def to_dict(self):
        return {"version":self.version,"steps":self.steps,"recipe_id":self.fingerprint()}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.steps,sort_keys=True,default=str).encode()).hexdigest()[:16]

    def validate(self) -> dict[str, Any]:
        checks=[]
        checks.append({"check":"steps_present","status":"pass" if self.steps else "review"})
        invalid=[i for i,s in enumerate(self.steps) if not s.get("action")]
        checks.append({"check":"action_names","status":"fail" if invalid else "pass","invalid_indices":invalid})
        missing_evidence=[i for i,s in enumerate(self.steps) if s.get("status")=="complete" and s.get("action") in {"model_result","agent_start","evidence_validation"} and not (s.get("evidence_ids") or s.get("parameters"))]
        checks.append({"check":"governance_metadata","status":"review" if missing_evidence else "pass","indices":missing_evidence})
        status="fail" if any(x["status"]=="fail" for x in checks) else ("review" if any(x["status"]=="review" for x in checks) else "pass")
        return {"status":status,"checks":checks,"recipe_id":self.fingerprint()}

    def replay(self, executor: Callable[[str, dict[str, Any]], Any], *, approved: bool = False) -> list[Any]:
        if not approved:
            raise PermissionError("Recipe replay requires explicit human approval.")
        validation=self.validate()
        if validation["status"] == "fail":
            raise ValueError(f"Recipe validation failed: {validation}")
        outputs=[]
        for step in self.steps:
            outputs.append(executor(step["action"], dict(step.get("parameters") or {})))
        return outputs
