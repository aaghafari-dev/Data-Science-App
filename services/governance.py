from __future__ import annotations

"""Professional governance, reproducibility and evidence services.

The services are deliberately dependency-light so the desktop application can run
without MLflow/DVC/SHAP/etc. They produce JSON-serialisable evidence records that
can be attached to experiments, agents, reports and the visual evidence DAG.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable
import json
import re

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def frame_fingerprint(df: pd.DataFrame) -> str:
    if df is None:
        return "none"
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    schema = "|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode()
    return sha256(schema + payload).hexdigest()


@dataclass
class EvidenceObject:
    evidence_id: str
    kind: str
    title: str
    created_at: str = field(default_factory=_now)
    source: str = "Data Science Studio Pro"
    status: str = "complete"
    data: dict[str, Any] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class HumanApprovalEvidence:
    """First-class audit object for every human gate."""

    @staticmethod
    def create(step: str, decision: str, summary: str, actor: str = "user", evidence_ids: list[str] | None = None) -> dict[str, Any]:
        return EvidenceObject(
            evidence_id=f"approval-{sha256(f'{step}|{decision}|{_now()}'.encode()).hexdigest()[:12]}",
            kind="human_approval",
            title=f"Human approval: {step}",
            status=decision,
            data={"step": step, "decision": decision, "actor": actor, "summary": summary},
            parent_ids=evidence_ids or [],
        ).to_dict()


class ScientificDataLeakageGate:
    """Conservative pre-model checks for common scientific/data leakage modes."""

    SUSPICIOUS_PATTERNS = (
        r"target", r"label", r"outcome", r"ground.?truth", r"future", r"post", r"result", r"prediction", r"fold", r"split"
    )

    def evaluate(self, df: pd.DataFrame, target: str | None = None, test_df: pd.DataFrame | None = None) -> dict[str, Any]:
        if df is None or df.empty:
            return {"status": "blocked", "severity": "high", "issues": ["No data available."], "checks": []}
        target = target or df.columns[-1]
        issues: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []

        duplicate_rows = int(df.duplicated().sum())
        checks.append({"check": "duplicate_rows", "count": duplicate_rows, "status": "warning" if duplicate_rows else "pass"})
        if duplicate_rows:
            issues.append({"severity": "medium", "issue": f"{duplicate_rows:,} duplicate rows may cross train/test boundaries."})

        if target not in df.columns:
            issues.append({"severity": "high", "issue": f"Target '{target}' is not present."})
        else:
            suspicious = []
            for col in df.columns:
                if col == target:
                    continue
                if any(re.search(p, str(col), flags=re.I) for p in self.SUSPICIOUS_PATTERNS):
                    suspicious.append(col)
            checks.append({"check": "suspicious_feature_names", "columns": suspicious, "status": "review" if suspicious else "pass"})
            if suspicious:
                issues.append({"severity": "high", "issue": "Potential target/future-information features require scientific review.", "columns": suspicious})

            if df[target].nunique(dropna=True) <= 1:
                issues.append({"severity": "high", "issue": "Target has one unique value; model evaluation is not meaningful."})

            exact_corr = []
            for col in df.select_dtypes(include=np.number).columns:
                if col != target:
                    try:
                        corr = float(df[[col, target]].corr().iloc[0, 1])
                        if np.isfinite(corr) and abs(corr) >= 0.999999:
                            exact_corr.append(col)
                    except Exception:
                        pass
            checks.append({"check": "near_perfect_numeric_correlation", "columns": exact_corr, "status": "review" if exact_corr else "pass"})
            if exact_corr:
                issues.append({"severity": "high", "issue": "Feature(s) are nearly identical to the target and may represent leakage.", "columns": exact_corr})

        if test_df is not None and not test_df.empty:
            common = set(df.columns) & set(test_df.columns)
            train_hash = pd.util.hash_pandas_object(df[list(common)].astype(str), index=False)
            test_hash = pd.util.hash_pandas_object(test_df[list(common)].astype(str), index=False)
            overlap = int(len(set(train_hash.tolist()) & set(test_hash.tolist())))
            checks.append({"check": "train_test_row_overlap", "overlap": overlap, "status": "blocked" if overlap else "pass"})
            if overlap:
                issues.append({"severity": "high", "issue": f"{overlap:,} exact feature rows overlap between train and test."})

        severity = "high" if any(x["severity"] == "high" for x in issues) else ("medium" if issues else "pass")
        return {"status": "blocked" if severity == "high" else "review" if severity == "medium" else "pass",
                "severity": severity, "target": target, "issues": issues, "checks": checks,
                "dataset_fingerprint": frame_fingerprint(df)}


class DatasetCardBuilder:
    @staticmethod
    def build(df: pd.DataFrame, source: str = "unknown", purpose: str = "analysis") -> dict[str, Any]:
        if df is None:
            return {}
        return {
            "card_type": "Dataset Card", "created_at": _now(), "source": source, "purpose": purpose,
            "fingerprint": frame_fingerprint(df), "rows": int(len(df)), "columns": int(len(df.columns)),
            "columns_detail": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                                "unique": int(df[c].nunique(dropna=True))} for c in df.columns],
            "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
        }


class ModelCardBuilder:
    @staticmethod
    def build(result: dict[str, Any], dataset_card: dict[str, Any] | None = None, leakage_gate: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "card_type": "Model Card", "created_at": _now(), "agent": result.get("agent"),
            "task": result.get("task"), "target": result.get("target"), "best_model": result.get("best_model"),
            "metrics": result.get("best_metrics", {}), "models_compared": list((result.get("models") or {}).keys()),
            "dataset_fingerprint": (dataset_card or {}).get("fingerprint"),
            "leakage_gate_status": (leakage_gate or {}).get("status"),
            "validation_protocol": result.get("validation_protocol"),
            "selection_policy": result.get("selection_policy"),
            "professional_evaluation": result.get("evaluation"),
            "diagnosis": result.get("model_diagnosis"),
            "feature_set_challenge": result.get("feature_set_challenge"),
            "limitations": ["Performance depends on the supplied dataset, deployment population and validation strategy.",
                            "The final test set is intended to remain locked during model selection.",
                            "Uncertainty and robustness diagnostics are conditional on their stated assumptions.",
                            "Automated diagnostic hypotheses require domain review and are not causal proof."],
        }


class StatisticalUncertainty:
    @staticmethod
    def bootstrap_mean(values: Any, n_boot: int = 1000, confidence: float = 0.95, seed: int = 42) -> dict[str, float]:
        x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
        if len(x) < 2:
            return {"estimate": float(x.mean()) if len(x) else float("nan"), "lower": float("nan"), "upper": float("nan"), "confidence": confidence}
        rng = np.random.default_rng(seed)
        samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        alpha = (1 - confidence) / 2
        return {"estimate": float(x.mean()), "lower": float(np.quantile(samples, alpha)), "upper": float(np.quantile(samples, 1 - alpha)), "confidence": confidence}

    @staticmethod
    def bootstrap_metric(y_true: Any, y_pred: Any, metric: Callable[[Any, Any], float], n_boot: int = 500, confidence: float = .95, seed: int = 42) -> dict[str, float]:
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if len(yt) < 2 or len(yt) != len(yp):
            return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan"), "confidence": confidence}
        rng = np.random.default_rng(seed); idx = rng.integers(0, len(yt), size=(n_boot, len(yt)))
        vals = np.asarray([metric(yt[i], yp[i]) for i in idx])
        alpha = (1 - confidence) / 2
        return {"estimate": float(metric(yt, yp)), "lower": float(np.quantile(vals, alpha)), "upper": float(np.quantile(vals, 1 - alpha)), "confidence": confidence}


class ExperimentComparator:
    @staticmethod
    def compare(records: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for r in records:
            metrics = r.get("metrics") or {}
            row = {"experiment": r.get("experiment") or r.get("id") or "unknown",
                   "agent": r.get("agent", ""), "model": r.get("model", ""),
                   "task": r.get("task", ""), "dataset_fingerprint": r.get("dataset_fingerprint", ""),
                   "route": r.get("route", ""), "step_index": r.get("step_index", "")}
            for key, value in metrics.items():
                if isinstance(value, dict) and {"estimate", "lower", "upper"}.issubset(value):
                    row[f"{key}.estimate"] = value.get("estimate"); row[f"{key}.lower"] = value.get("lower"); row[f"{key}.upper"] = value.get("upper")
                else:
                    row[key] = value
            rows.append(row)
        return pd.DataFrame(rows)


class ModelPromotionRegistry:
    STAGES = ("development", "candidate", "staging", "production", "archived")

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}

    def register(self, model_id: str, metrics: dict[str, Any], evidence_ids: list[str] | None = None, stage: str = "development") -> dict[str, Any]:
        if stage not in self.STAGES:
            raise ValueError(f"Invalid model stage: {stage}")
        record = {"model_id": model_id, "metrics": _jsonable(metrics), "stage": stage, "updated_at": _now(), "evidence_ids": evidence_ids or []}
        self.records[model_id] = record; return record

    def promote(self, model_id: str, stage: str, approval: dict[str, Any]) -> dict[str, Any]:
        if model_id not in self.records: raise KeyError(model_id)
        if approval.get("kind") != "human_approval" or approval.get("data", {}).get("decision") != "approved":
            raise PermissionError("Model promotion requires a first-class human approval evidence object.")
        if stage not in self.STAGES: raise ValueError(stage)
        self.records[model_id]["stage"] = stage; self.records[model_id]["updated_at"] = _now(); self.records[model_id]["evidence_ids"].append(approval["evidence_id"])
        return self.records[model_id]


class DataDiff:
    @staticmethod
    def compare(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
        before_cols, after_cols = set(before.columns), set(after.columns)
        added, removed = sorted(after_cols - before_cols), sorted(before_cols - after_cols)
        common = sorted(before_cols & after_cols)
        changed = []
        for c in common:
            if str(before[c].dtype) != str(after[c].dtype):
                changed.append({"column": c, "before_dtype": str(before[c].dtype), "after_dtype": str(after[c].dtype)})
        return {"before_fingerprint": frame_fingerprint(before), "after_fingerprint": frame_fingerprint(after),
                "rows_before": len(before), "rows_after": len(after), "row_delta": len(after)-len(before),
                "columns_added": added, "columns_removed": removed, "dtype_changes": changed}


class EvidenceDAG:
    def __init__(self): self.nodes: dict[str, dict[str, Any]] = {}; self.edges: list[tuple[str, str, str]] = []
    def add(self, evidence: dict[str, Any]) -> str:
        eid = evidence.get("evidence_id") or f"e-{len(self.nodes)+1}"
        self.nodes[eid] = evidence
        for parent in evidence.get("parent_ids", []): self.edges.append((parent, eid, "supports"))
        return eid
    def to_dict(self): return {"nodes": list(self.nodes.values()), "edges": [list(e) for e in self.edges]}
    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        def node_id(value: str) -> str:
            return re.sub(r"\W", "_", str(value))
        for eid, n in self.nodes.items():
            safe_id = node_id(eid)
            title = str(n.get("title", eid)).replace('"', "'")
            lines.append(f'  {safe_id}["{title}"]')
        for a, b, label in self.edges:
            lines.append(f'  {node_id(a)} -->|{label}| {node_id(b)}')
        return "\n".join(lines)



class AgentEvaluation:
    @staticmethod
    def evaluate(run_state: dict[str, Any]) -> dict[str, Any]:
        approvals = run_state.get("approved_steps", [])
        rejects = run_state.get("rejected_steps", [])
        ml = run_state.get("ml_results") or {}; dl = run_state.get("dl_results") or {}
        return {"evaluation_type": "agent_run", "created_at": _now(),
                "approval_count": len(approvals), "rejection_count": len(rejects),
                "completed": bool(ml or dl), "has_plot_evidence": bool(run_state.get("plot_results")),
                "route": run_state.get("route"), "criteria": {"human_gated": bool(approvals or rejects), "evidence_bound": bool(ml or dl)}}


@dataclass
class AnalysisExtension:
    name: str
    version: str
    capabilities: list[str]
    handler: Callable[..., Any]
    input_schema: dict[str, Any] = field(default_factory=dict)


class ControlledAnalysisExtensionAPI:
    """Typed, allow-listed extension registry; no unrestricted plugin execution."""
    def __init__(self): self._extensions: dict[str, AnalysisExtension] = {}
    def register(self, extension: AnalysisExtension):
        if not extension.name or not extension.version: raise ValueError("Extension name/version required")
        self._extensions[extension.name] = extension
    def list(self) -> list[dict[str, Any]]:
        return [{"name": e.name, "version": e.version, "capabilities": e.capabilities, "input_schema": e.input_schema} for e in self._extensions.values()]
    def execute(self, name: str, capability: str, **kwargs):
        e = self._extensions[name]
        if capability not in e.capabilities: raise PermissionError(f"Capability '{capability}' is not allowed for extension '{name}'.")
        return e.handler(**kwargs)
