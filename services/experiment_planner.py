"""Evidence-driven next-experiment planning for the Master Agent."""

from __future__ import annotations

from typing import Any


class ExperimentPlanner:
    """Translate unresolved evidence into a small, reviewable next-step experiment plan."""

    @staticmethod
    def plan(decision: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a bounded experiment proposal without executing it."""
        evidence = evidence or {}
        gaps = list(decision.get("evidence_gaps") or [])
        selection = decision.get("method_selection") or {}
        selected = (selection.get("selected_method") or {}).get("method_id")
        task = decision.get("primary_route")
        actions: list[dict[str, Any]] = []
        if not evidence.get("data_quality_reviewed"):
            actions.append({"action": "data_quality_review", "purpose": "resolve missingness, duplicates, constants and structural risks", "approval_required": True})
        if task in {"REGRESSION", "CLASSIFICATION", "HYBRID_SUPERVISED"} and not evidence.get("validation_reviewed"):
            actions.append({"action": "validation_protocol_review", "purpose": "confirm split strategy, leakage controls and task-appropriate metrics", "approval_required": True})
        if selected and selection.get("challenger_method_ids"):
            actions.append({"action": "challenger_comparison", "purpose": "compare the approved challenger under the same validation protocol", "approval_required": True})
        if gaps:
            actions.append({"action": "resolve_evidence_gap", "purpose": str(gaps[0]), "approval_required": True})
        if not actions:
            actions.append({"action": "independent_verification", "purpose": "check evidence completeness and internal consistency before stopping", "approval_required": True})
        return {"task": task, "selected_method": selected, "actions": actions[:5], "stop_rule": "Do not execute an additional experiment unless the user approves the specific proposed action."}
