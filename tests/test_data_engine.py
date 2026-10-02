"""Module duty: Test data engine.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
from core.data_engine import DataEngine


def test_persistent_filter_and_remove():
    """Perform the test persistent filter and remove operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    e = DataEngine(); df = pd.DataFrame({"Region": ["West", "East", "West"], "Sales": [10, 20, 30]})
    e.set_active_dataframe(df)
    e.set_filter("Region", {"kind": "categorical", "values": ["West"]})
    assert len(e.df) == 2
    e.set_filter("Sales", {"kind": "numeric", "min": 15, "max": 30})
    assert len(e.df) == 1 and e.df.iloc[0]["Sales"] == 30
    e.remove_filter("Region")
    assert len(e.df) == 2
