from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import pandas as pd

from services.tool_preflight import preflight
from services.analysis_cache import AnalysisCache


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
    requires_gpu: bool = False
    deterministic: bool = True
    cacheable: bool = False


class TypedAgentToolRegistry:
    """Typed, permissioned, preflighted and cache-aware analytical tool registry."""
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self.cache = AnalysisCache(max_items=32)

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
            "requires_gpu": s.requires_gpu, "deterministic": s.deterministic,
            "cacheable": s.cacheable,
        } for s in self._tools.values()]

    def dry_run(self, name: str, **kwargs) -> dict[str, Any]:
        spec = self.get(name)
        missing = [k for k, v in spec.input_schema.get("required", {}).items() if k not in kwargs or kwargs[k] is None]
        df = next((v for v in kwargs.values() if isinstance(v, pd.DataFrame)), None)
        pf = preflight(name, df=df, required_inputs=list(spec.input_schema.get("required", {}).keys()), budget=spec.budget,
                       requires_gpu=spec.requires_gpu, gpu_available=kwargs.get("gpu_available"),
                       requires_human_approval=spec.requires_human_approval, requires_evidence=spec.requires_evidence,
                       human_approved=kwargs.get("human_approved", False), evidence_bound=kwargs.get("evidence_bound", False))
        status = "ready" if not missing and pf.status in {"ready", "review"} else "review"
        return {"status":status,"tool":name,"version":spec.version,"description":spec.description,"capabilities":list(spec.capabilities),
                "required_inputs":list(spec.input_schema.get("required",{}).keys()),"missing_inputs":missing,
                "budget":dict(spec.budget),"observed_rows":len(df) if df is not None else None,
                "estimated_memory_mb":pf.estimated_memory_mb,"estimated_seconds":pf.estimated_seconds,
                "preflight":pf.to_dict(),"requires_evidence":spec.requires_evidence,"requires_human_approval":spec.requires_human_approval,
                "will_execute":False,"note":"Dry-run only; no handler was executed and no data were modified."}

    def execute(self, name: str, *, human_approved: bool = False, evidence_bound: bool = False, gpu_available: bool | None = None, **kwargs):
        spec = self.get(name)
        missing = [k for k, v in spec.input_schema.get("required", {}).items() if k not in kwargs or kwargs[k] is None]
        if missing:
            raise ValueError(f"Tool '{name}' requires: {', '.join(missing)}")
        if spec.requires_human_approval and not human_approved:
            raise PermissionError(f"Tool '{name}' requires explicit human approval before execution.")
        if spec.requires_evidence and not evidence_bound:
            raise PermissionError(f"Tool '{name}' requires evidence binding before execution.")
        df = next((v for v in kwargs.values() if isinstance(v, pd.DataFrame)), None)
        pf = preflight(name, df=df, required_inputs=list(spec.input_schema.get("required", {}).keys()), budget=spec.budget,
                       requires_gpu=spec.requires_gpu, gpu_available=gpu_available,
                       requires_human_approval=spec.requires_human_approval, requires_evidence=spec.requires_evidence,
                       human_approved=human_approved, evidence_bound=evidence_bound)
        if pf.status == "blocked":
            raise PermissionError(f"Tool '{name}' preflight blocked execution: {pf.to_dict()}")
        if spec.cacheable and spec.deterministic:
            params = {k:v for k,v in kwargs.items() if not isinstance(v,pd.DataFrame) and k not in {"api_key"}}
            cached, hit = self.cache.get_or_compute(name, lambda: self._timed_execute(spec, kwargs), df=df, params=params)
            return cached
        return self._timed_execute(spec, kwargs)

    @staticmethod
    def _timed_execute(spec: ToolSpec, kwargs: dict[str, Any]):
        t0 = time.monotonic()
        result = spec.handler(**kwargs)
        max_seconds = spec.budget.get("max_seconds")
        if max_seconds is not None and (time.monotonic() - t0) > max_seconds:
            raise TimeoutError(f"Tool '{spec.name}' exceeded its {max_seconds:.1f}s execution budget.")
        return result


def default_registry() -> TypedAgentToolRegistry:
    from services.data_quality import DataQualityEngine
    from services.data_contracts import DataContractEngine
    from services.error_analysis import ErrorAnalysisEngine
    from services.statistical_analysis import StatisticalAnalysisEngine
    from services.agent_quality import TemporalAvailabilityAnalyzer, ModelDiagnosisEngine, RobustnessPerturbationAnalyzer, prediction_uncertainty_for_model
    from services.compute_backend import ComputeBackend
    from services.data_slice_analysis import DataSliceAnalyzer
    from services.evaluation_protocol import ValidationProtocolAdvisor
    from services.feature_provenance import FeatureProvenanceEngine
    from services.method_selection import AnalyticalMethodAdvisor

    r = TypedAgentToolRegistry()
    read_budget = {"max_rows": 1_000_000, "max_seconds": 30}
    r.register(ToolSpec("data_quality", "2.0", "Profile quality and missingness.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataQualityEngine.evaluate(df), budget=read_budget, cacheable=True))
    r.register(ToolSpec("data_contract", "2.0", "Build or validate a schema contract.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataContractEngine.build(df), budget=read_budget, cacheable=True))
    r.register(ToolSpec("describe", "2.0", "Compute descriptive statistics.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: StatisticalAnalysisEngine.describe(df), budget=read_budget, cacheable=True))
    r.register(ToolSpec("classification_error", "1.1", "Compute classification error diagnostics.", {"required":{"y_true":"array", "y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.classification, budget={"max_seconds":20}))
    r.register(ToolSpec("regression_error", "1.1", "Compute regression error diagnostics.", {"required":{"y_true":"array", "y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.regression, budget={"max_seconds":20}))
    r.register(ToolSpec("validation_protocol", "2.0", "Advise on random/stratified/group/time validation and audit split integrity.", {"required":{"df":"DataFrame"}}, ("read_data", "validation"), lambda df, target=None: ValidationProtocolAdvisor.infer(df,target).to_dict(), budget=read_budget, cacheable=True))
    r.register(ToolSpec("temporal_availability", "2.0", "Build a feature availability matrix over time.", {"required":{"df":"DataFrame"}}, ("read_data", "diagnostics"), lambda df, target=None: TemporalAvailabilityAnalyzer.run(df,target=target), budget=read_budget, cacheable=True))
    r.register(ToolSpec("feature_provenance", "1.0", "Build feature role, missingness and name-flag provenance evidence.", {"required":{"df":"DataFrame"}}, ("read_data", "provenance"), lambda df, target=None: FeatureProvenanceEngine.build(df,target=target), budget=read_budget, cacheable=True))
    r.register(ToolSpec("method_selection", "1.0", "Suggest analytical/statistical methods from the question and explicit assumptions.", {"required":{"question":"str"}}, ("statistics", "method_selection"), lambda question, df=None: AnalyticalMethodAdvisor.advise(question,df), budget={"max_seconds":10}, cacheable=True))
    r.register(ToolSpec("model_diagnosis", "2.0", "Create evidence-based model diagnostic hypotheses.", {"required":{"task":"str", "evaluation":"dict"}}, ("model_results", "diagnostics"), ModelDiagnosisEngine.diagnose, budget={"max_seconds":20}, cacheable=True))
    r.register(ToolSpec("data_slice_analysis", "1.0", "Find materially different performance slices with minimum-sample safeguards.", {"required":{"X":"DataFrame", "y":"array", "predictions":"array", "task":"str"}}, ("model_results", "diagnostics"), DataSliceAnalyzer.run, budget={"max_seconds":30}))
    r.register(ToolSpec("robustness_perturbation", "2.0", "Test bounded jitter, missingness and clipping sensitivity.", {"required":{"pipeline":"model", "X_test":"DataFrame", "y_test":"array", "task":"str"}}, ("model_results", "robustness"), RobustnessPerturbationAnalyzer.run, budget={"max_seconds":60}, requires_evidence=True))
    r.register(ToolSpec("prediction_uncertainty", "2.0", "Estimate prediction-level uncertainty diagnostics.", {"required":{"pipeline":"model", "X_train":"DataFrame", "y_train":"array", "X_test":"DataFrame", "y_pred":"array", "task":"str"}}, ("model_results", "uncertainty"), prediction_uncertainty_for_model, budget={"max_seconds":60}, requires_evidence=True))
    r.register(ToolSpec("compute_diagnostics", "2.0", "Inspect NVIDIA hardware, PyTorch CUDA and runtime readiness.", {"required":{}}, ("hardware",), lambda: ComputeBackend.detect(runtime_validate=False).to_dict(), budget={"max_seconds":10}, cacheable=False))
    return r
