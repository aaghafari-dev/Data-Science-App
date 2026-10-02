"""Deterministic quality gates for generated reports and presentations."""
from __future__ import annotations
import re
from typing import Any

class OutputQualityGate:
    """Check narrative outputs against structured evidence without using an LLM."""
    @staticmethod
    def report(text: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Check report completeness, fragmentation risk and unsupported numeric claims."""
        required=["Executive Summary","Objective","Results","Limitations","Conclusion"]
        missing=[x for x in required if x.lower() not in (text or "").lower()]
        fragments=[x for x in (text or "").splitlines() if x.strip() and len(x.strip().split()) < 4 and not x.strip().startswith(tuple(str(i)+"." for i in range(1,10)))]
        numeric_claims=re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|\b)", text or "")
        evidence_text=str(evidence)
        unsupported=[]
        for value in numeric_claims[:100]:
            if value not in evidence_text: unsupported.append(value)
        return {"status":"pass" if not missing and len(fragments)<5 and not unsupported else "review", "missing_sections":missing, "fragment_count":len(fragments), "unsupported_numeric_tokens":unsupported[:20], "checks":["section completeness","fragmentation risk","numeric evidence traceability"]}

    @staticmethod
    def presentation(slides: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
        """Check that slides have one purpose, concise body text and evidence context."""
        issues=[]
        for i,slide in enumerate(slides,1):
            if not slide.get("title"): issues.append(f"slide {i}: missing title")
            if not slide.get("purpose"): issues.append(f"slide {i}: missing analytical purpose")
            if len(str(slide.get("body","")).split()) > 110: issues.append(f"slide {i}: body too dense; move detail to speaker notes")
            if slide.get("figure_json") and not slide.get("notes"): issues.append(f"slide {i}: visual lacks speaker-note context")
        return {"status":"pass" if not issues else "review", "issues":issues, "slide_count":len(slides), "checks":["title","one analytical purpose","text density","visual context"]}
