"""Module duty: Privacy.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import re
import pandas as pd


def detect_pii(df: pd.DataFrame) -> dict:
    """Perform the detect pii operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    patterns = {
        "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
        "phone": re.compile(r"^\+?[0-9 ()-]{7,}$"),
    }
    findings = {}
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(500)
        scores = {name: float(sample.map(lambda x: bool(p.search(x))).mean()) for name, p in patterns.items()} if len(sample) else {}
        if scores and max(scores.values()) >= .5:
            findings[col] = max(scores, key=scores.get)
    try:
        from presidio_analyzer import AnalyzerEngine
        findings["_engine"] = "Presidio available"
    except Exception:
        findings["_engine"] = "regex baseline; Presidio optional"
    return findings


def anonymize_pii(df: pd.DataFrame, columns: list[str] | None = None, seed: int = 42) -> pd.DataFrame:
    """Perform the anonymize pii operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    out = df.copy(); columns = columns or list(out.columns)
    try:
        from faker import Faker
        fake = Faker(); Faker.seed(seed)
        for col in columns:
            if col in out.columns:
                out[col] = [fake.uuid4() for _ in range(len(out))]
    except Exception:
        for col in columns:
            if col in out.columns:
                out[col] = [f"anon_{i:06d}" for i in range(len(out))]
    return out
