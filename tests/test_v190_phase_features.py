"""Module duty: Test v190 phase features.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from services.model_preprocessing import build_preprocessor
from services.evaluation_protocol import ValidationProtocolAdvisor
from services.feature_set_analysis import FeatureSetChallenge
from services.data_slice_analysis import DataSliceAnalyzer
from services.feature_provenance import FeatureProvenanceEngine
from services.tool_preflight import preflight
from services.experiment_comparator import ExperimentComparator
from services.evidence_validation import EvidenceValidator
from services.analysis_recipe import AnalysisRecipe
from services.method_selection import AnalyticalMethodAdvisor
from services.resource_policy import ResourcePolicy
from services.agent_tools import default_registry


def test_validation_protocol_detects_time_and_locks_integrity():
    """Perform the test validation protocol detects time and locks integrity operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    n=80
    df=pd.DataFrame({"date":pd.date_range("2025-01-01",periods=n),"x":np.arange(n),"target":np.arange(n)*.2})
    plan=ValidationProtocolAdvisor.infer(df,"target")
    assert plan.strategy=="time"
    tr,te=ValidationProtocolAdvisor.split(df,"target",plan)
    audit=ValidationProtocolAdvisor.audit(tr,te,"target",plan.group_column)
    assert audit["status"]=="pass"
    assert tr["date"].max()<te["date"].min()


def test_feature_set_challenge_and_slices_are_structured():
    """Perform the test feature set challenge and slices are structured operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    rng=np.random.default_rng(3)
    X=pd.DataFrame({"x":rng.normal(size=100),"z":rng.normal(size=100),"group":np.where(np.arange(100)%2,"A","B")})
    y=X.x*2+rng.normal(scale=.2,size=100)
    def factory(cols):
        """Perform the factory operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return Pipeline([("pre",build_preprocessor(X[cols])),("model",Ridge())])
    out=FeatureSetChallenge.run(factory,X.iloc[:70],X.iloc[70:],y.iloc[:70],y.iloc[70:],"regression",stable_features=["x","z"])
    assert out["status"]=="ok"
    assert any(x["feature_set"]=="stable_features_only" for x in out["sets"])
    pred=factory(["x","z","group"]).fit(X.iloc[:70],y.iloc[:70]).predict(X.iloc[70:])
    slices=DataSliceAnalyzer.run(X.iloc[70:],y.iloc[70:],pred,"regression",min_rows=5)
    assert slices["status"]=="ok"


def test_provenance_preflight_cache_and_comparison():
    """Perform the test provenance preflight cache and comparison operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df=pd.DataFrame({"x":[1,2,3],"future_result":[3,4,5],"target":[2,4,6]})
    prov=FeatureProvenanceEngine.build(df,"target")
    assert prov["status"]=="ok"
    assert "future" in prov["features"][1]["name_flags"]
    pf=preflight("x",df=df,required_inputs=["df"],budget={"max_rows":10})
    assert pf.status=="ready"
    comp=ExperimentComparator.compare([{"experiment":"a","task":"regression","dataset_fingerprint":"x","validation_strategy":"time","metrics":{"r2":.5}},{"experiment":"b","task":"regression","dataset_fingerprint":"x","validation_strategy":"time","metrics":{"r2":.6}}],"r2")
    assert comp["comparable"] is True
    assert ResourcePolicy.plan(1000,10)["parallel_jobs"]>=1


def test_recipe_evidence_and_method_advisor():
    """Perform the test recipe evidence and method advisor operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    recipe=AnalysisRecipe(); recipe.add("agent_start",{"target":"y"})
    assert recipe.validate()["status"] in {"pass","review"}
    try:
        recipe.replay(lambda a,p: a, approved=False)
        assert False
    except PermissionError:
        pass
    ev=[{"evidence_id":"a","kind":"model_card","parent_ids":[]},{"evidence_id":"b","kind":"professional_evaluation","parent_ids":["a"]}]
    assert EvidenceValidator.validate(ev,["model_card","professional_evaluation"])["status"]=="pass"
    advice=AnalyticalMethodAdvisor.advise("compare two groups")
    assert advice["status"]=="ok" and advice["human_review_required"] is True


def test_tool_registry_exposes_professional_phase_tools():
    """Perform the test tool registry exposes professional phase tools operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    names={x["name"] for x in default_registry().describe()}
    assert {"validation_protocol","feature_provenance","method_selection","data_slice_analysis"}.issubset(names)
