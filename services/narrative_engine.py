"""Professional evidence-to-narrative generation for reports and presentations."""
from __future__ import annotations
import json
from typing import Any
from services.serialization import json_safe

REPORT_SECTIONS=("executive_summary","objective","data_governance","methodology","validation","results","robustness","visual_findings","discussion","limitations","conclusion","recommendations","reproducibility")

class EvidenceNarrativeEngine:
    """Convert structured evidence into coherent section-level narratives rather than sentence fragments."""
    @staticmethod
    def compact(evidence: dict[str, Any], limit: int=65000) -> str:
        """Create a bounded evidence packet for an LLM."""
        return json.dumps(json_safe(evidence), default=str, indent=2)[:limit]

    @classmethod
    def build_prompt(cls, evidence, audience="senior data scientists", format_name="report"):
        """Build a narrative-generation prompt that enforces continuity and evidence discipline."""
        return f"""You are the senior analytical editor for Data Science Studio Pro. Produce a coherent {format_name} for {audience}.\n\nWrite connected paragraphs, not sentence fragments and not a dump of metrics. Each section must have a clear purpose and logical transition. Use only supplied evidence. Never invent values, methods, sample sizes, causal explanations, or deployment claims. Distinguish observation, interpretation and limitation. Prefer precise prose over generic filler.\n\nReturn JSON with exactly these keys: {', '.join(REPORT_SECTIONS)}. Each value must be a concise but complete paragraph or two, except recommendations which may be a short numbered list.\n\nEvidence:\n{cls.compact(evidence)}"""

    @staticmethod
    def slide_text(text: str, max_words: int = 70) -> str:
        """Condense coherent narrative into a complete slide message without sentence-fragment truncation."""
        import re
        source = " ".join(str(text or "").split())
        if not source:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", source)
        selected = []
        count = 0
        for sentence in sentences:
            words = sentence.split()
            if selected and count + len(words) > max_words:
                break
            if not selected and len(words) > max_words:
                return " ".join(words[:max_words]).rsplit(" ", 1)[0] + "…"
            selected.append(sentence); count += len(words)
            if len(selected) >= 2:
                break
        return " ".join(selected)

    @classmethod
    def deterministic(cls, evidence: dict[str, Any]) -> dict[str, str]:
        """Create a coherent fallback narrative when no LLM is available."""
        target=evidence.get("target") or "no supervised target"
        route=evidence.get("master_route") or "exploratory analysis"
        card=evidence.get("dataset_card") or {}; ml=evidence.get("ML Agent") or {}; dl=evidence.get("DL Agent") or {}; plot=evidence.get("Plot Agent") or {}
        plots=plot.get("plots",[]) if isinstance(plot,dict) else []
        return {
            "executive_summary": f"The governed analysis examined the active dataset using the Master Agent route {route}. The workflow combined data-quality controls, task-specific validation, specialist analysis and evidence-linked visualisation. The principal target was {target}. Conclusions below are limited to the recorded evidence and the approved analytical protocol.",
            "objective": f"The analytical objective was to determine which evidence-supported analytical strategy is appropriate for the supplied data and to quantify the resulting findings without introducing unsupported assumptions. The Master Agent separated task identification, method comparison, validation design and specialist execution so that each transition remained reviewable.",
            "data_governance": f"The analysis used the recorded dataset card and leakage controls as the governing description of the data. Dataset metadata include {json.dumps(card,default=str)[:1200]}. Any interpretation should therefore be understood as conditional on the observed sample, feature availability and provenance recorded in the project.",
            "methodology": f"The Master Agent evaluated candidate analytical routes before specialist execution. The selected route was {route}; specialist evidence was generated only after the corresponding human approval gates. Preprocessing and model-selection decisions are intended to remain separated from the final evaluation data.",
            "validation": f"Validation was treated as an analytical design decision rather than an afterthought. The project records the selected validation protocol and its assumptions; final performance should therefore be interpreted together with the validation population, resampling strategy and any grouping or temporal constraints recorded in the evidence.",
            "results": f"The recorded specialist results provide the quantitative basis for the analysis. ML evidence is summarised as {ml.get('summary','not recorded')}; DL evidence is summarised as {dl.get('summary','not recorded')}. Where numerical metrics are present, they should be read in the context of their validation protocol rather than as standalone measures of generalisation.",
            "robustness": "Robustness evidence is considered separately from primary performance. Stability, diagnostic and perturbation results identify conditions under which the observed result may change; they do not constitute a guarantee of future behaviour.",
            "visual_findings": f"The Plot Agent produced {len(plots)} evidence-linked visualisation(s). Each visual should be interpreted as a view of the underlying recorded data and analysis rather than as independent evidence. Visual associations are descriptive unless an appropriate inferential design supports a stronger claim.",
            "discussion": "Taken together, the results provide evidence about the selected analytical task under the recorded dataset and validation design. Interpretation should distinguish predictive performance or structural patterns from causal explanations, and should retain the limitations identified by the diagnostic and governance layers.",
            "limitations": "Important limitations arise from the observed sample, feature availability, validation assumptions, model specification and unresolved evidence gaps. The application therefore preserves limitations explicitly rather than converting them into confident conclusions.",
            "conclusion": "The analysis supports only the conclusions directly justified by the recorded evidence. The Master Agent's role is to make the analytical reasoning traceable and reproducible; domain experts remain responsible for deciding whether the evidence is sufficient for scientific, operational or deployment decisions.",
            "recommendations": "1. Review unresolved evidence gaps and diagnostic flags.\n2. Confirm that all predictors would be available at the intended decision time.\n3. Preserve the dataset, analysis recipe, model/evidence versions and approval record.\n4. Perform domain-specific validation before deployment or causal interpretation.",
            "reproducibility": f"The project records the analysis route, human approvals, specialist evidence and evidence identifiers. The reproducibility record should be retained with the dataset version, model configuration, random seeds, provider configuration and generated artifacts."
        }
