"""Module duty: Duckdb workspace.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

class DuckDBWorkspace:
    """Optional analytical SQL layer. Falls back with a clear dependency message."""
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            import duckdb
            self.duckdb = duckdb
        except Exception:
            self.duckdb = None

    @property
    def available(self):
        """Perform the available operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return self.duckdb is not None

    def query(self, df: pd.DataFrame, sql: str) -> pd.DataFrame:
        """Perform the query operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.duckdb is None: raise RuntimeError("DuckDB is not installed. Install duckdb to use the Analytical Workspace.")
        con = self.duckdb.connect(database=":memory:")
        try:
            con.register("data", df)
            return con.execute(sql).df()
        finally:
            con.close()

    def profile_query(self, df: pd.DataFrame) -> str:
        """Perform the profile query operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return "SELECT * FROM data LIMIT 100;"
