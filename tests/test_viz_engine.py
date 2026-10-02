"""Module duty: Test viz engine.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
from core.viz_engine import VizEngine


def test_multiple_shelves_and_orientation():
    """Perform the test multiple shelves and orientation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df = pd.DataFrame({"Region": ["West", "West", "East"], "Category": ["A", "B", "A"], "Sales": [10, 20, 30]})
    plan = VizEngine.plan_shelves(df, ["Region", "Category"], ["Sales"])
    assert plan.orientation == "horizontal"
    assert plan.dimensions == ["Region", "Category"]
    out = VizEngine.aggregate_for_shelves(df, ["Region", "Category"], ["Sales"])
    assert list(out.columns) == ["Region", "Category", "Sales"]
    assert len(out) == 3


def test_reverse_measure_orientation():
    """Perform the test reverse measure orientation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df = pd.DataFrame({"Region": ["West", "East"], "Sales": [10, 30]})
    plan = VizEngine.plan_shelves(df, ["Sales"], ["Region"])
    assert plan.orientation == "vertical"


def test_dimensions_only_create_record_count():
    """Perform the test dimensions only create record count operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df = pd.DataFrame({"Region": ["West", "West", "East"]})
    out = VizEngine.aggregate_for_shelves(df, ["Region"], [])
    assert "Number of Records" in out.columns
    assert out["Number of Records"].sum() == 3
