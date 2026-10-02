"""Module duty: Evidence validation.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any


class EvidenceValidator:
    """Validate that evidence objects and report-facing claims are traceable."""
    @staticmethod
    def validate(evidence: list[dict[str, Any]], required_kinds: list[str] | None = None) -> dict[str, Any]:
        """Perform the validate operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        ids = {x.get("evidence_id") for x in evidence if x.get("evidence_id")}
        checks = []
        missing_parent = []
        for x in evidence:
            for p in x.get("parent_ids", []) or []:
                if p not in ids:
                    missing_parent.append((x.get("evidence_id"), p))
        checks.append({"check": "unique_ids", "status": "pass" if len(ids) == len([x for x in evidence if x.get("evidence_id")]) else "fail"})
        checks.append({"check": "parent_references", "status": "pass" if not missing_parent else "review", "missing": missing_parent[:20]})
        if required_kinds:
            kinds = {x.get("kind") for x in evidence}
            missing = [k for k in required_kinds if k not in kinds]
            checks.append({"check": "required_evidence", "status": "pass" if not missing else "review", "missing": missing})
        status = "fail" if any(c["status"] == "fail" for c in checks) else ("review" if any(c["status"] == "review" for c in checks) else "pass")
        return {"status": status, "checks": checks, "evidence_count": len(evidence)}
