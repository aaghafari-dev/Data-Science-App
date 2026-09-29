from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd

@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    capabilities: tuple[str, ...]
    handler: Callable[..., Any]

class TypedAgentToolRegistry:
    """Typed, explicit tool registry. Agents can call registered tools only."""
    def __init__(self): self._tools: dict[str, ToolSpec] = {}
    def register(self, spec: ToolSpec): self._tools[spec.name] = spec
    def get(self, name: str) -> ToolSpec:
        if name not in self._tools: raise KeyError(f"Unknown agent tool: {name}")
        return self._tools[name]
    def describe(self):
        return [{"name": s.name, "version": s.version, "description": s.description,
                 "capabilities": list(s.capabilities), "input_schema": s.input_schema} for s in self._tools.values()]
    def execute(self, name: str, **kwargs):
        spec = self.get(name)
        missing = [k for k,v in spec.input_schema.get("required", {}).items() if k not in kwargs or kwargs[k] is None]
        if missing: raise ValueError(f"Tool '{name}' requires: {', '.join(missing)}")
        return spec.handler(**kwargs)

def default_registry() -> TypedAgentToolRegistry:
    from services.data_quality import DataQualityEngine
    from services.data_contracts import DataContractEngine
    from services.error_analysis import ErrorAnalysisEngine
    from services.statistical_analysis import StatisticalAnalysisEngine
    r = TypedAgentToolRegistry()
    r.register(ToolSpec("data_quality", "1.0", "Profile quality and missingness.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataQualityEngine.evaluate(df)))
    r.register(ToolSpec("data_contract", "1.0", "Build or validate a schema contract.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: DataContractEngine.build(df)))
    r.register(ToolSpec("describe", "1.0", "Compute descriptive statistics.", {"required":{"df":"DataFrame"}}, ("read_data",), lambda df: StatisticalAnalysisEngine.describe(df)))
    r.register(ToolSpec("classification_error", "1.0", "Compute classification error diagnostics.", {"required":{"y_true":"array","y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.classification))
    r.register(ToolSpec("regression_error", "1.0", "Compute regression error diagnostics.", {"required":{"y_true":"array","y_pred":"array"}}, ("model_results",), ErrorAnalysisEngine.regression))
    return r
