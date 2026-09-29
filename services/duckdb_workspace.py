from __future__ import annotations

from pathlib import Path
import pandas as pd

class DuckDBWorkspace:
    """Optional analytical SQL layer. Falls back with a clear dependency message."""
    def __init__(self):
        try:
            import duckdb
            self.duckdb = duckdb
        except Exception:
            self.duckdb = None

    @property
    def available(self): return self.duckdb is not None

    def query(self, df: pd.DataFrame, sql: str) -> pd.DataFrame:
        if self.duckdb is None: raise RuntimeError("DuckDB is not installed. Install duckdb to use the Analytical Workspace.")
        con = self.duckdb.connect(database=":memory:")
        try:
            con.register("data", df)
            return con.execute(sql).df()
        finally:
            con.close()

    def profile_query(self, df: pd.DataFrame) -> str:
        return "SELECT * FROM data LIMIT 100;"
