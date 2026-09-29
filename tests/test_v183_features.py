import pandas as pd
import numpy as np
from services.target_feature_selection import recommend_targets_and_features
from services.model_preprocessing import build_preprocessor
from services.agent_tools import default_registry
from services.analysis_recipe import AnalysisRecipe
from services.analysis_state import AnalysisStateMachine, AnalysisState
from services.artifacts import ArtifactStore
from services.compute_backend import ComputeBackend


def test_target_feature_recommendation_is_explicit():
    df=pd.DataFrame({"id":[1,2,3,4],"feature":[1.,2.,3.,4.],"target":[0,1,0,1]})
    r=recommend_targets_and_features(df)
    assert r["suggested_target"] == "target"
    assert "feature" in r["suggested_features"]
    assert "id" not in r["suggested_features"]


def test_all_missing_columns_are_excluded_from_preprocessor():
    df=pd.DataFrame({"num":[1.,2.,3.],"empty":[np.nan,np.nan,np.nan],"cat":["a","b","a"],"when":pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03"])})
    pre=build_preprocessor(df)
    out=pre.fit_transform(df)
    assert out.shape[0]==3


def test_professional_agent_foundations():
    reg=default_registry(); assert any(x["name"]=="data_quality" for x in reg.describe())
    recipe=AnalysisRecipe(); recipe.add("load",{"source":"x.csv"}); assert recipe.to_dict()["recipe_id"]
    sm=AnalysisStateMachine(AnalysisState.DATA_LOADED.value); sm.transition(AnalysisState.QUALITY_CHECKED.value); assert sm.state==AnalysisState.QUALITY_CHECKED.value
    store=ArtifactStore(); a=store.register("dataset","demo",{"rows":3}); b=store.register("model","rf",{},[a.artifact_id]); assert store.lineage(b.artifact_id)[-1].artifact_id==a.artifact_id
    assert ComputeBackend.resolve("CPU").mode=="CPU"
