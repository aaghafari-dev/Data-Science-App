from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import pandas as pd


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    capabilities: tuple[str, ...]
    handler: Callable[..., Any]
    budget: dict[str, float] = field(default_factory=dict)
    requires_evidence: bool = False
    requires_human_approval: bool = False


class TypedAgentToolRegistry:
    """Typed, permissioned tool registry with visible execution budgets."""
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown agent tool: {name}")
        return self._tools[name]

    def describe(self):
        return [{
            "name": s.name, "version": s.version, "description": s.description,
            "capabilities": list(s.capabilities), "input_schema": s.input_schema,
            "budget": dict(s.budget), "requires_evidence": s.requires_evidence,
            "requires_human_approval": s.requires_human_approval,
        } for s in self._tools.values()]

    def dry_run(self, name: str, **kwargs) -> dict[str, Any]:
        """Return a deterministic execution preview without invoking the handler."""
        spec = self.get(name)
        missing = [k for k, v in spec.input_schema.get("required", {}).items() if k not in kwargs or kwargs[k] is None]
        row_count = None
        for value in kwargs.values():
            if isinstance(value, pd.DataFrame): row_count = len(value); break
        budget = dict(spec.budget)
        budget_status = "within_budget"
        if row_count is not None and budget.get("max_rows") is not None and row_count > budget["max_rows"]: budget_status = "row_budget_exceeded"
        return {"status":"ready" if not missing and budget_status=="within_budget" else "review",
                "tool":name,"version":spec.version,"description":spec.description,"capabilities":list(spec.capabilities),
                "required_inputs":list(spec.input_schema.get("required",{}).keys()),"missing_inputs":missing,
                "budget":budget,"observed_rows":row_count,"budget_status":budget_status,
                "requires_evidence":spec.requires_evidence,"requires_human_approval":spec.requires_human_approval,
                "will_execute":False,"note":"Dry-run only; no handler was executed and no data were modified."}

    def execute(self, name: str, *, human_approved: bool = False, evidence_bound: bool = False, **kwargs):
        spec = self.get(name)
        missing = [k for k, v in spec.input_schema.get("required", {}).items() if k not in kwargs or kwargs[k] is None]
        if missing:
            raise ValueError(f"Tool '{name}' requires: {', '.join(missing)}")
        if spec.requires_human_approval and not human_approved:
            raise PermissionError(f"Tool '{name}' requires explicit human approval before execution.")
        if spec.requires_evidence and not evidence_bound:
            raise PermissionError(f"Tool '{name}' requires evidence binding before execution.")
        max_rows = spec.budget.get("max_rows")
        if max_rows is not None:
            for value in kwargs.values():
                if isinstance(value, pd.DataFrame) and len(value) > max_rows:
                    raise ValueError(f"Tool '{name}' budget exceeded: {len(value):,} rows > {int(max_rows):,} allowed.")
        t0 = time.monotonic()
        result = spec.handler(**kwargs)
        max_seconds = spec.budget.get("max_seconds")
        if max_seconds is not None and (time.monotonic() - t0) > max_seconds:
            raise TimeoutError(f"Tool '{name}' exceeded its {max_seconds:.1f}s execution budget.")
        return result


def default_registry() -> TypedAgentToolRegistry:
    from services.data_quality import DataQualityEngine
    from services.data_contracts import DataContractEngine
    from services.error_analysis import ErrorAnalysisEngine
    from services.statistical_analysis import StatisticalAnalysisEngine
    from services.agent_quality import TemporalAvailabilityAnalyzer, ModelDiagnosisEngine, RobustnessPerturbationAnalyzer, prediction_uncertainty_for_model
    from services.compute_backend import ComputeBackend

    r = TypedAgentToolRegistry()
    read_budget = {"max_rows": 1_000_000, "max_seconds": 30}
    r.register(ToolSpec("data_quality", "1.0", "Profile quality and missingness.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataQualityEngine.evaluate(df), budget=read_budget))
    r.register(ToolSpec("data_contract", "1.0", "Build or validate a schema contract.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataContractEngine.build(df), budget=read_budget))
    r.register(ToolSpec("describe", "1.0", "Compute descriptive statistics.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: StatisticalAnalysisEngine.describe(df), budget=read_budget))
    r.register(ToolSpec("classification_error", "1.0", "Compute classification error diagnostics.", {"required":{"y_true":"array","y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.classification, budget={"max_seconds":20}))
    r.register(ToolSpec("regression_error", "1.0", "Compute regression error diagnostics.", {"required":{"y_true":"array","y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.regression, budget={"max_seconds":20}))
    r.register(ToolSpec("temporal_availability", "1.0", "Build a feature availability matrix over time.", {"required":{"df":"DataFrame"}}, ("read_data", "diagnostics"), lambda df, target=None: TemporalAvailabilityAnalyzer.run(df, target=target), budget=read_budget))
    r.register(ToolSpec("model_diagnosis", "1.0", "Create evidence-based model diagnostic hypotheses.", {"required":{"task":"str", "evaluation":"dict"}}, ("model_results", "diagnostics"), ModelDiagnosisEngine.diagnose, budget={"max_seconds":20}))
    r.register(ToolSpec("robustness_perturbation", "1.0", "Test bounded numeric jitter, missingness and clipping sensitivity.", {"required":{"pipeline":"model", "X_test":"DataFrame", "y_test":"array", "task":"str"}}, ("model_results", "robustness"), RobustnessPerturbationAnalyzer.run, budget={"max_seconds":60}, requires_evidence=True))
    r.register(ToolSpec("prediction_uncertainty", "1.0", "Estimate prediction-level uncertainty diagnostics.", {"required":{"pipeline":"model", "X_train":"DataFrame", "y_train":"array", "X_test":"DataFrame", "y_pred":"array", "task":"str"}}, ("model_results", "uncertainty"), prediction_uncertainty_for_model, budget={"max_seconds":30}, requires_evidence=True))
    r.register(ToolSpec("compute_diagnostics", "1.0", "Inspect NVIDIA hardware, PyTorch CUDA and runtime readiness.", {"required":{}}, ("hardware",), lambda: ComputeBackend.detect(runtime_validate=False).to_dict(), budget={"max_seconds":10}))
    return r
