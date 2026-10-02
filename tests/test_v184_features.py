"""Module duty: Test v184 features.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
import numpy as np
import py_compile
from pathlib import Path
from services.target_feature_selection import recommend_targets_and_features


def test_target_validation_flags_math_redundancy():
    """Perform the test target validation flags math redundancy operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    x=np.arange(60,dtype=float)
    df=pd.DataFrame({"Current":x,"CurrentSquared":x**2,"Target":x**2,"City":["A","B","C"]*20})
    rec=recommend_targets_and_features(df)
    assert rec["redundancy_pairs"]
    assert any("direction" in w.lower() for w in rec["warnings"])


def test_core_python_sources_compile():
    """Perform the test core python sources compile operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    root=Path(__file__).parents[1]
    for rel in ["main.py","agent/graph_plot.py","agent/graph_report.py","agent/graph_presentation.py"]:
        py_compile.compile(str(root/rel),doraise=True)
