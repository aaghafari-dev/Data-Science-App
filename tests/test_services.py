"""Module duty: Test services.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
from services.lineage import DatasetLineage, dataset_fingerprint
from services.privacy import detect_pii, anonymize_pii


def test_lineage_and_hash(tmp_path):
    """Perform the test lineage and hash operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df = pd.DataFrame({"a": [1, 2]})
    h = dataset_fingerprint(df); assert len(h) == 64
    lin = DatasetLineage(str(tmp_path)); rec = lin.snapshot(df, operation="load")
    assert rec["dataset_hash"] == h


def test_privacy_baseline():
    """Perform the test privacy baseline operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    df = pd.DataFrame({"email": ["a@example.com", "b@example.com"], "x": [1, 2]})
    findings = detect_pii(df); assert findings["email"] == "email"
    out = anonymize_pii(df, ["email"]); assert not out["email"].equals(df["email"])
