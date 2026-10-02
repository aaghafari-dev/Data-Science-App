"""Module duty: Test agent imports.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import importlib.util
import pytest


def test_agent_modules_import_when_langgraph_stack_exists():
    """Perform the test agent modules import when langgraph stack exists operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if not importlib.util.find_spec("langgraph") or not importlib.util.find_spec("langchain_core"):
        pytest.skip("LangGraph/LangChain are optional in the execution environment")
    import agent.graph_ds  # noqa: F401
    import agent.graph_report  # noqa: F401
    import agent.graph_plot  # noqa: F401
