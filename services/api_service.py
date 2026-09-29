from __future__ import annotations

from typing import Any


def create_app(data_engine=None):
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="Data Science Studio Pro API", version="1.0")

    @app.get("/health")
    def health(): return {"status": "ok"}

    @app.get("/data/summary")
    def summary():
        if data_engine is None or data_engine.df is None: raise HTTPException(404, "No dataset loaded")
        df = data_engine.df
        return {"rows": len(df), "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict()}

    @app.get("/data/profile")
    def profile():
        if data_engine is None or data_engine.df is None: raise HTTPException(404, "No dataset loaded")
        return {"missing": data_engine.get_missing_values(), "duplicates": int(data_engine.df.duplicated().sum())}

    return app
