"""Module duty: Artifacts.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib, json

@dataclass
class Artifact:
    artifact_id: str
    kind: str
    name: str
    version: int
    parents: list[str] = field(default_factory=list)
    metadata: dict[str,Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ArtifactStore:
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.artifacts: dict[str,Artifact]={}
    def register(self, kind,name,metadata=None,parents=None):
        """Perform the register operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        payload=json.dumps(metadata or {},sort_keys=True,default=str)
        aid=hashlib.sha256(f"{kind}|{name}|{payload}|{parents or []}".encode()).hexdigest()[:20]
        version=1+max([a.version for a in self.artifacts.values() if a.name==name],default=0)
        a=Artifact(aid,kind,name,version,parents or [],metadata or {}); self.artifacts[aid]=a; return a
    def lineage(self, artifact_id):
        """Perform the lineage operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        seen=set(); out=[]
        def walk(aid):
            """Perform the walk operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            if aid in seen or aid not in self.artifacts:return
            seen.add(aid); a=self.artifacts[aid]; out.append(a)
            for p in a.parents: walk(p)
        walk(artifact_id); return out
    def to_dict(self):
        """Perform the to dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {k:a.__dict__ for k,a in self.artifacts.items()}
