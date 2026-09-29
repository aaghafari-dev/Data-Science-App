from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass
class BackendPlan:
    backend: str
    reason: str
    estimated_rows: int
    estimated_columns: int

class LargeDataEngine:
    """Progressive backend selector: Pandas -> Polars -> DuckDB -> Dask.
    It never silently changes scientific results; the selected backend is recorded.
    """
    def __init__(self, pandas_row_limit: int = 1_000_000): self.pandas_row_limit = pandas_row_limit
    def plan(self, df: pd.DataFrame) -> BackendPlan:
        rows, cols = (0,0) if df is None else df.shape
        if rows <= self.pandas_row_limit:
            return BackendPlan("pandas", "Dataset is within the configured in-memory threshold.", rows, cols)
        try:
            import polars  # noqa: F401
            return BackendPlan("polars", "Large tabular data; Polars is available.", rows, cols)
        except Exception: pass
        try:
            import duckdb  # noqa: F401
            return BackendPlan("duckdb", "Large analytical workload; DuckDB is available.", rows, cols)
        except Exception: pass
        try:
            import dask.dataframe  # noqa: F401
            return BackendPlan("dask", "Large workload; Dask is available.", rows, cols)
        except Exception: pass
        return BackendPlan("pandas-chunked", "No accelerated backend installed; use chunked Pandas operations.", rows, cols)
