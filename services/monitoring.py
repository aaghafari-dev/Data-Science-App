from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

class ModelMonitoring:
    @staticmethod
    def numeric_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        common=[c for c in reference.columns if c in current.columns]
        for c in common:
            if not pd.api.types.is_numeric_dtype(reference[c]) or not pd.api.types.is_numeric_dtype(current[c]): continue
            a=pd.to_numeric(reference[c],errors="coerce").dropna(); b=pd.to_numeric(current[c],errors="coerce").dropna()
            if a.empty or b.empty: continue
            ref_mean=float(a.mean()); cur_mean=float(b.mean()); ref_std=float(a.std(ddof=0)); cur_std=float(b.std(ddof=0))
            shift=abs(cur_mean-ref_mean)/(ref_std+1e-12)
            rows.append({"feature":c,"reference_mean":ref_mean,"current_mean":cur_mean,"standardized_shift":float(shift),"status":"review" if shift>=0.5 else "stable"})
        return pd.DataFrame(rows)

    @staticmethod
    def categorical_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for c in [x for x in reference.columns if x in current.columns]:
            if pd.api.types.is_numeric_dtype(reference[c]): continue
            a=reference[c].astype(str).value_counts(normalize=True); b=current[c].astype(str).value_counts(normalize=True)
            cats=set(a.index)|set(b.index); tv=0.5*sum(abs(float(a.get(k,0))-float(b.get(k,0))) for k in cats)
            rows.append({"feature":c,"total_variation":float(tv),"status":"review" if tv>=0.2 else "stable"})
        return pd.DataFrame(rows)

    @staticmethod
    def summary(reference, current):
        numeric=ModelMonitoring.numeric_drift(reference,current); categorical=ModelMonitoring.categorical_drift(reference,current)
        return {"numeric":numeric.to_dict("records"),"categorical":categorical.to_dict("records"),"review_count":int((numeric.get("status",pd.Series(dtype=str))=="review").sum()) + int((categorical.get("status",pd.Series(dtype=str))=="review").sum())}
