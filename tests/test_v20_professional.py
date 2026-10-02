"""Module duty: Test v20 professional.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import json
import pandas as pd
import numpy as np
import pytest


def test_presentation_json_safe_dataframe():
    """Perform the test presentation json safe dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    pytest.importorskip("langchain_core")
    from agent.graph_presentation import _json_safe
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    safe = _json_safe({"frame": df})
    json.dumps(safe)
    assert safe["frame"]["__type__"] == "DataFrame"
    assert safe["frame"]["shape"] == [2, 2]


def test_supervised_candidate_families_are_complete():
    """Perform the test supervised candidate families are complete operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from agent.graph_ml import _candidates
    cls = _candidates("classification", 42, 1)
    reg = _candidates("regression", 42, 1)
    for name in ("Logistic Regression", "SVM", "Random Forest", "Decision Tree", "KNN", "Naive Bayes", "Neural Network"):
        assert name in cls
    for name in ("OLS", "Ridge", "Lasso", "Elastic Net", "Polynomial Regression", "SVR", "Decision Tree", "Random Forest", "Gradient Boosting", "Neural Network"):
        assert name in reg


def test_supervised_metric_helpers_are_professional():
    """Perform the test supervised metric helpers are professional operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from services.supervised_analysis import _classification_metrics, _regression_metrics
    cls = _classification_metrics(np.array(["a","a","b","b"]), np.array(["a","b","b","b"]), np.array([[.8,.2],[.4,.6],[.2,.8],[.1,.9]]))
    assert "accuracy" in cls and "f1_weighted" in cls and "roc_auc" in cls and "pr_auc" in cls
    reg = _regression_metrics(np.array([1.,2.,3.]), np.array([1.2,1.9,2.7]))
    assert all(k in reg for k in ("mae","rmse","r2","median_absolute_error","mean_bias_error"))


def test_master_agent_requires_each_governed_stage_approval():
    """Perform the test master agent requires each governed stage approval operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from agent.graph_ds import agent_ds_app
    df = pd.DataFrame({"x": range(30), "y": [i % 3 for i in range(30)], "target": [i % 2 for i in range(30)]})
    state = {"dataframe": df, "target": "target", "compute_mode": "CPU", "max_steps": 20,
             "approved_steps": [], "rejected_steps": [], "user_approved": False,
             "step_index": 0, "evidence_ids": [], "memory": [], "target_analysis": {"suggested_target":"target","suggested_features":["x","y"]},
             "leakage_gate": {"status":"pass"}, "human_approval_evidence": []}
    first = agent_ds_app.invoke(state)
    assert first["needs_approval"] is True
    assert first["analysis_stage"] == "PLAN"
    second = agent_ds_app.invoke({**state, **first, "user_approved": True})
    assert second["needs_approval"] is True
    assert second["analysis_stage"] == "VALIDATION"
    assert second["approved_steps"][-1] == "PLAN"
