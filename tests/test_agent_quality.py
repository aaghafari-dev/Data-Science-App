import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from services.model_preprocessing import build_preprocessor
from services.agent_quality import (
    ProfessionalEvaluationEngine, FeatureStabilityAnalyzer, TemporalAvailabilityAnalyzer,
    CounterfactualLeakageAnalyzer, ModelDiagnosisEngine, AgentWhyEvidence, AgentSelfCheck, RobustnessPerturbationAnalyzer, PredictionUncertaintyEngine,
)


def test_professional_evaluation_and_diagnostics():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120), "group": np.where(np.arange(120) % 2, "A", "B")})
    y = (X["a"] + rng.normal(scale=.4, size=120) > 0).astype(int)
    pipe = Pipeline([("pre", build_preprocessor(X)), ("model", LogisticRegression(max_iter=1000))])
    tr, te = np.arange(90), np.arange(90, 120)
    pipe.fit(X.iloc[tr], y[tr])
    bundle = ProfessionalEvaluationEngine.evaluate_bundle(pipe, X.iloc[tr], X.iloc[te], y[tr], y[te], "classification")
    assert bundle["evaluation_level"] == "professional"
    assert bundle["baseline"]["model"] == "Dummy baseline"
    assert "cross_validation" in bundle


def test_temporal_availability_flags_late_feature():
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "stable": np.arange(n), "late": [np.nan] * 30 + list(range(30)), "target": np.arange(n) * .1})
    out = TemporalAvailabilityAnalyzer.run(df, target="target")
    assert out["status"] == "ok"
    assert any(x["feature"] == "late" for x in out["late_features"])


def test_agent_why_and_self_check_are_first_class_objects():
    why = AgentWhyEvidence.create("Master Agent", "route=ML", ["small dataset", "target is defined"], ["gate-1"])
    assert why["kind"] == "agent_why"
    assert why["evidence_id"].startswith("why-")
    check = AgentSelfCheck.run("Master", "route", {"objective": "route", "inputs": ["df"], "expected_output": "route", "risk": "leakage", "tool_budget": 4}, {"evidence_refs": [why["evidence_id"]]})
    assert check["status"] == "pass"
    assert check["approval_required"] is True


def test_feature_stability_returns_json_safe_structure():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"x1": rng.normal(size=80), "x2": rng.normal(size=80)})
    y = X["x1"] * 2 + rng.normal(size=80)
    def factory():
        return Pipeline([("pre", build_preprocessor(X)), ("model", Ridge())])
    out = FeatureStabilityAnalyzer.run(factory, X, y, "regression", repeats=2, max_features=2)
    assert out["status"] == "ok"
    assert set(out["stable_features"]).issubset(set(X.columns))


def test_counterfactual_leakage_diagnostic_runs():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"leak": rng.normal(size=100), "noise": rng.normal(size=100)})
    y = X["leak"] + rng.normal(scale=.05, size=100)
    def factory(cols):
        return Pipeline([("pre", build_preprocessor(X[cols])), ("model", Ridge())])
    out = CounterfactualLeakageAnalyzer.run(factory, X, y, "regression", ["leak"])
    assert out["status"] == "ok"
    assert out["tests"][0]["feature"] == "leak"
    diag = ModelDiagnosisEngine.diagnose("regression", {"cross_validation": {"status": "ok", "metrics": {}}, "calibration": {"status": "not_applicable"}}, leakage=out)
    assert "hypotheses" in diag


def test_typed_tool_registry_exposes_budgets_and_permissions():
    from services.agent_tools import default_registry
    registry = default_registry()
    names = {x["name"] for x in registry.describe()}
    assert {"temporal_availability", "model_diagnosis", "compute_diagnostics"}.issubset(names)
    for item in registry.describe():
        assert "budget" in item


def test_robustness_and_prediction_uncertainty():
    rng=np.random.default_rng(12)
    X=pd.DataFrame({"x":rng.normal(size=120),"z":rng.normal(size=120)})
    y=X["x"]*2+rng.normal(scale=.2,size=120)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    pipe=Pipeline([("pre",build_preprocessor(X)),("model",Ridge())]); pipe.fit(X.iloc[:90],y.iloc[:90])
    out=RobustnessPerturbationAnalyzer.run(pipe,X.iloc[90:],y.iloc[90:],"regression")
    assert out["status"]=="ok" and len(out["tests"])==3
    u=PredictionUncertaintyEngine.regression(pipe,X.iloc[:90],y.iloc[:90],X.iloc[90:],pipe.predict(X.iloc[90:]))
    assert u["status"]=="ok" and "residual_quantiles" in u

def test_tool_dry_run_does_not_execute():
    from services.agent_tools import default_registry
    registry=default_registry()
    df=pd.DataFrame({"x":[1,2,3]})
    out=registry.dry_run("data_quality",df=df)
    assert out["status"]=="ready" and out["will_execute"] is False
