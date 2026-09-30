from __future__ import annotations

from typing import Any
import numpy as np


class ResultVerifier:
    """Independent consistency checks over agent-produced evidence."""
    @staticmethod
    def verify(result: dict[str, Any] | None) -> dict[str, Any]:
        if not result:
            return {"status":"fail","checks":[{"check":"result_present","status":"fail"}]}
        checks=[]
        checks.append({"check":"status_ok","status":"pass" if result.get("status")=="ok" else "fail","value":result.get("status")})
        checks.append({"check":"target_defined","status":"pass" if result.get("target") else "fail"})
        checks.append({"check":"model_defined","status":"pass" if result.get("best_model") else "fail"})
        evaluation=result.get("evaluation") or {}
        checks.append({"check":"professional_evaluation","status":"pass" if evaluation.get("level")=="professional" else "review"})
        checks.append({"check":"test_locked","status":"pass" if evaluation.get("test_set_locked") is True else "review"})
        selection=result.get("selection_policy") or {}
        checks.append({"check":"selection_policy","status":"pass" if selection.get("basis") else "review"})
        diagnosis=result.get("model_diagnosis") or {}
        checks.append({"check":"diagnosis_present","status":"pass" if diagnosis.get("status")=="ok" else "review"})
        status="fail" if any(c["status"]=="fail" for c in checks) else ("review" if any(c["status"]=="review" for c in checks) else "pass")
        return {"status":status,"checks":checks,"interpretation":"Verification checks internal consistency and governance evidence; it does not certify scientific truth."}
