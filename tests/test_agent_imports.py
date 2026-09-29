import importlib.util
import pytest


def test_agent_modules_import_when_langgraph_stack_exists():
    if not importlib.util.find_spec("langgraph") or not importlib.util.find_spec("langchain_core"):
        pytest.skip("LangGraph/LangChain are optional in the execution environment")
    import agent.graph_ds  # noqa: F401
    import agent.graph_report  # noqa: F401
    import agent.graph_plot  # noqa: F401
