"""Module duty: Api service.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any


def create_app(data_engine=None):
    """Perform the create app operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="Data Science Studio Pro API", version="1.0")

    @app.get("/health")
    def health():
        """Perform the health operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {"status": "ok"}

    @app.get("/data/summary")
    def summary():
        """Perform the summary operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if data_engine is None or data_engine.df is None: raise HTTPException(404, "No dataset loaded")
        df = data_engine.df
        return {"rows": len(df), "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict()}

    @app.get("/data/profile")
    def profile():
        """Perform the profile operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if data_engine is None or data_engine.df is None: raise HTTPException(404, "No dataset loaded")
        return {"missing": data_engine.get_missing_values(), "duplicates": int(data_engine.df.duplicated().sum())}

    return app
