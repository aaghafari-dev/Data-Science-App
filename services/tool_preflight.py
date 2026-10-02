"""Module duty: Tool preflight.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import time
import pandas as pd


@dataclass
class ToolPreflight:
    status: str
    tool: str
    checks: list[dict[str, Any]]
    estimated_rows: int | None = None
    estimated_memory_mb: float | None = None
    estimated_seconds: float | None = None

    def to_dict(self):
        """Perform the to dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {"status": self.status, "tool": self.tool, "checks": self.checks, "estimated_rows": self.estimated_rows, "estimated_memory_mb": self.estimated_memory_mb, "estimated_seconds": self.estimated_seconds}


def estimate_dataframe(df: pd.DataFrame | None) -> tuple[int | None, float | None]:
    """Perform the estimate dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if df is None:
        return None, None
    try:
        return int(len(df)), float(df.memory_usage(deep=True).sum() / (1024**2))
    except Exception:
        return len(df), None


def preflight(tool_name: str, *, df: pd.DataFrame | None = None, required_inputs: list[str] | None = None, budget: dict[str, Any] | None = None, requires_gpu: bool = False, gpu_available: bool | None = None, requires_human_approval: bool = False, requires_evidence: bool = False, human_approved: bool = False, evidence_bound: bool = False) -> ToolPreflight:
    """Perform the preflight operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    rows, mem = estimate_dataframe(df)
    budget = budget or {}
    checks = []
    checks.append({"check": "inputs", "status": "pass" if required_inputs is not None else "review", "required": required_inputs or []})
    max_rows = budget.get("max_rows")
    checks.append({"check": "row_budget", "status": "pass" if max_rows is None or rows is None or rows <= max_rows else "fail", "rows": rows, "max_rows": max_rows})
    checks.append({"check": "gpu_requirement", "status": "pass" if not requires_gpu or gpu_available else "fail"})
    checks.append({"check": "human_approval", "status": "pass" if (not requires_human_approval or human_approved) else "fail"})
    checks.append({"check": "evidence_binding", "status": "pass" if (not requires_evidence or evidence_bound) else "fail"})
    status = "blocked" if any(c["status"] == "fail" for c in checks) else ("review" if any(c["status"] == "review" for c in checks) else "ready")
    sec = float(budget.get("max_seconds")) if budget.get("max_seconds") is not None else None
    return ToolPreflight(status, tool_name, checks, rows, mem, sec)
