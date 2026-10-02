"""Module duty: Test v201 features.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import json
import pandas as pd


def test_json_safe_never_emits_dataframe():
    """Perform the test json safe never emits dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from services.serialization import json_safe
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    safe = json_safe({"df": df})
    json.dumps(safe)
    assert safe["df"]["__type__"] == "DataFrame"
    assert safe["df"]["shape"] == [2, 2]


def test_llm_provider_gate_blocks_unconfigured():
    """Perform the test llm provider gate blocks unconfigured operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from services.llm_config import LLMConfig, preflight
    assert preflight(LLMConfig()).get("status") == "blocked"


def test_sheet_manager_opens_with_sheet_one_only():
    """Perform the test sheet manager opens with sheet one only operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from core.sheet_manager import SheetManager
    sm = SheetManager()
    assert list(sm.sheets) == ["Sheet 1"]
    assert sm.active_sheet == "Sheet 1"


def test_plot_plan_contains_quantities():
    """Perform the test plot plan contains quantities operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    import pytest
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from agent.graph_plot import _deterministic_plan
    df = pd.DataFrame({"City":["A","B","A","B"],"Sales":[1,3,2,4],"Experience":[1,2,3,4]})
    plans = _deterministic_plan(df, "Sales", None, None)
    assert plans
    assert all("quantity" in p for p in plans)


def test_api_model_catalog_is_broad_and_editable():
    """Perform the test api model catalog is broad and editable operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    import config
    assert any(x.startswith("gpt-") for x in config.CLOUD_LLM_MODELS)
    assert any(x.startswith("claude") for x in config.CLOUD_LLM_MODELS)
    assert any(x.startswith("gemini") for x in config.CLOUD_LLM_MODELS)
    assert any(x.startswith("deepseek") for x in config.CLOUD_LLM_MODELS)
    assert any(x.startswith("grok") for x in config.CLOUD_LLM_MODELS)
    assert "mistral-small-4" in config.CLOUD_LLM_MODELS


def test_api_provider_inference():
    """Perform the test api provider inference operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from services.llm_config import infer_api_provider
    assert infer_api_provider("claude-sonnet-4-6") == "anthropic"
    assert infer_api_provider("gemini-2.5-pro") == "google"
    assert infer_api_provider("deepseek-v4-pro") == "deepseek"
    assert infer_api_provider("grok-4.7") == "xai"
    assert infer_api_provider("gpt-5.6") == "openai"
