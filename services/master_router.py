"""Evidence-first method selection for the Agent Data Scientist Master Agent.

The Master Agent must separate *evidence generation* from *analytical selection*.  This
module therefore implements an explicit protocol:

1. interpret the analytical objective;
2. identify plausible analytical tasks;
3. generate executable candidate methods from the observed data structure;
4. state assumptions, evidence requirements, validation strategy, and risks;
5. compare candidates without hiding the comparison inside a row/column heuristic;
6. ask the configured LLM to review the task and method selection when available; and
7. produce a human-reviewable specialist execution plan.

A deterministic fallback is retained only as a transparent *provisional evidence proposal*.
It is never labelled as equivalent to LLM-assisted analytical reasoning and remains subject to
the existing human approval gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.llm_config import LLMConfig, get_llm
from services.method_registry import AnalyticalMethodRegistry
from services.experiment_planner import ExperimentPlanner
from services.validation_strategy import ValidationStrategyAgent


@dataclass
class AnalysisProfile:
    """JSON-safe description of the data, approved target, and analytical context."""

    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    image_path_columns: list[str]
    image_dataset_path: str | None
    high_cardinality_columns: list[str]
    low_cardinality_numeric_columns: list[str]
    missing_heavy_columns: list[str]
    approved_target: str | None
    target_type: str | None
    objective: str
    explicit_target: bool


@dataclass
class TaskHypothesis:
    """A candidate analytical task with explicit supporting evidence and blockers."""

    task_id: str
    evidence_strength: float
    evidence: list[str]
    blockers: list[str]
    target_required: bool
    validation_focus: list[str]


@dataclass
class MethodCandidate:
    """A candidate method/family with assumptions, evidence requirements, and risks."""

    method_id: str
    task_id: str
    specialist: str
    role: str
    evidence_for: list[str]
    assumptions: list[str]
    evidence_requirements: list[str]
    validation_protocol: list[str]
    limitations: list[str]
    challenger_eligible: bool


class MasterDecisionEngine:
    """Run the Agent Data Scientist's explicit task-to-method selection protocol."""

    ROUTES = (
        "EXPLORATION",
        "STATISTICS",
        "REGRESSION",
        "CLASSIFICATION",
        "CLUSTERING",
        "ANOMALY",
        "TIME_SERIES",
        "HYBRID_SUPERVISED",
        "UNSUPERVISED",
        "REINFORCEMENT_LEARNING",
        "CNN_IMAGE_CLASSIFICATION",
    )

    SPECIALISTS = (
        "Data Quality Agent",
        "Statistical Insight Agent",
        "ML Agent",
        "DL Agent",
        "Clustering Agent",
        "Anomaly Detection Agent",
        "Time-Series Agent",
        "Unsupervised Learning Agent",
        "Reinforcement Learning Agent",
        "CNN Image Analysis Agent",
        "Feature Engineering Agent",
        "Experiment Planner Agent",
    )

    TASK_ALIASES = {
        "regression": "REGRESSION",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
        "anomaly": "ANOMALY",
        "statistics": "STATISTICS",
        "exploration": "EXPLORATION",
        "time_series": "TIME_SERIES",
        "hybrid_supervised": "HYBRID_SUPERVISED",
        "unsupervised": "UNSUPERVISED",
        "reinforcement": "REINFORCEMENT_LEARNING",
        "reinforcement_learning": "REINFORCEMENT_LEARNING",
        "cnn": "CNN_IMAGE_CLASSIFICATION",
        "cnn_image_analysis": "CNN_IMAGE_CLASSIFICATION",
        "image_classification": "CNN_IMAGE_CLASSIFICATION",
    }

    @staticmethod
    def profile(
        df: pd.DataFrame,
        objective: str = "",
        target: str | None = None,
        image_dataset_path: str | None = None,
    ) -> AnalysisProfile:
        """Build an evidence profile without silently inventing a supervised target."""
        numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [
            str(c)
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        dates = [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        image_path_columns = [str(c) for c in df.columns if any(token in str(c).lower() for token in ("image", "img", "filepath", "file_path", "image_path")) and df[c].dropna().astype(str).str.contains(r"\.(?:jpg|jpeg|png|bmp|tif|tiff|webp)$", case=False, regex=True).mean() >= 0.50]
        high_card = [
            str(c)
            for c in df.columns
            if df[c].nunique(dropna=True) > max(50, int(len(df) * 0.25))
        ]
        low_num = [
            c
            for c in numeric
            if df[c].nunique(dropna=True) <= min(20, max(3, int(len(df) * 0.05)))
        ]
        missing_heavy = [str(c) for c in df.columns if float(df[c].isna().mean()) >= 0.30]
        approved_target = str(target) if target is not None and str(target) in df.columns else None
        target_type = MasterDecisionEngine._target_type(df[approved_target]) if approved_target else None
        return AnalysisProfile(
            rows=int(len(df)),
            columns=int(len(df.columns)),
            numeric_columns=numeric[:50],
            categorical_columns=categorical[:50],
            datetime_columns=dates[:20],
            high_cardinality_columns=high_card[:30],
            low_cardinality_numeric_columns=low_num[:30],
            missing_heavy_columns=missing_heavy[:30],
            approved_target=approved_target,
            target_type=target_type,
            objective=objective.strip(),
            image_path_columns=image_path_columns[:10],
            image_dataset_path=str(image_dataset_path) if image_dataset_path else None,
            explicit_target=approved_target is not None,
        )

    @staticmethod
    def _target_type(series: pd.Series) -> str:
        """Classify an approved target as categorical or continuous numeric."""
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            return "classification"
        if pd.api.types.is_numeric_dtype(series):
            unique = series.nunique(dropna=True)
            threshold = min(20, max(2, int(len(series) * 0.05)))
            return "classification" if unique <= threshold else "regression"
        return "classification"

    @staticmethod
    def _contains(objective: str, terms: tuple[str, ...]) -> bool:
        """Return whether the analytical objective contains a supplied concept."""
        text = objective.lower()
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)

    @classmethod
    def identify_tasks(cls, profile: AnalysisProfile) -> list[dict[str, Any]]:
        """Generate transparent task hypotheses from objective and data evidence."""
        obj = profile.objective
        tasks: dict[str, TaskHypothesis] = {}

        def add(task_id: str, strength: float, evidence: list[str], blockers: list[str], target_required: bool, validation: list[str]) -> None:
            """Register one task hypothesis in the local evidence set."""
            tasks[task_id] = TaskHypothesis(task_id, strength, evidence, blockers, target_required, validation)

        add("EXPLORATION", 0.55, ["Exploratory analysis is a valid baseline when the question is broad or under-specified."], [], False, ["schema and missingness review", "distribution and relationship inspection"])

        if cls._contains(obj, ("correlation", "association", "compare groups", "hypothesis", "statistical", "significance", "inference")):
            add("STATISTICS", 0.92, ["The objective explicitly requests statistical association, comparison, or inference."], [], False, ["distribution checks", "effect-size context", "multiple-comparison awareness where applicable"])
        elif not obj:
            add("STATISTICS", 0.50, ["Descriptive statistical evidence is useful for an under-specified objective."], [], False, ["descriptive statistics", "association diagnostics"])

        if profile.explicit_target and profile.target_type == "regression":
            evidence = ["An approved target is continuous numeric."]
            if cls._contains(obj, ("predict", "forecast", "estimate", "regress", "model")):
                evidence.append("The objective contains predictive/modeling language.")
            add("REGRESSION", 0.90, evidence, [], True, ["train-only preprocessing", "task-appropriate cross-validation", "locked test evaluation", "residual/error diagnostics"])
        if profile.explicit_target and profile.target_type == "classification":
            evidence = ["An approved target has categorical/class structure."]
            if cls._contains(obj, ("predict", "classify", "classification", "label", "model")):
                evidence.append("The objective contains predictive/classification language.")
            add("CLASSIFICATION", 0.90, evidence, [], True, ["stratification where valid", "train-only preprocessing", "locked test evaluation", "class-imbalance diagnostics"])

        if cls._contains(obj, ("cluster", "clusters", "segment", "unsupervised", "group similar", "segmentation")):
            blockers = [] if len(profile.numeric_columns) >= 2 else ["At least two numerical variables are required by the current clustering specialist."]
            add("CLUSTERING", 0.96, ["The objective explicitly requests unsupervised grouping."], blockers, False, ["scaling/imputation", "cluster validity metrics", "domain validation of cluster meaning"])
        elif len(profile.numeric_columns) >= 2 and not profile.explicit_target:
            add("CLUSTERING", 0.45, ["Multiple numerical variables permit exploratory unsupervised structure discovery."], [], False, ["scaling/imputation", "cluster validity metrics"])

        if cls._contains(obj, ("unsupervised", "latent structure", "dimensionality reduction", "reduce dimensions", "pca", "umap", "t-sne", "association rule")):
            blockers = [] if len(profile.numeric_columns) >= 2 else ["At least two numerical variables are required for the current unsupervised structure-discovery workflow."]
            add("UNSUPERVISED", 0.97, ["The objective explicitly requests unsupervised structure discovery or dimensionality reduction."], blockers, False, ["feature representation", "imputation/scaling", "stability", "domain validation"])

        if cls._contains(obj, ("reinforcement learning", "q-learning", "sarsa", "policy", "reward", "agent environment")):
            add("REINFORCEMENT_LEARNING", 0.99, ["The objective explicitly describes a sequential decision problem or reinforcement-learning terminology."], [], False, ["state/action definition", "reward design", "environment specification", "policy evaluation", "seed stability"])

        if cls._contains(obj, ("cnn", "convolutional neural network", "image classification", "image analysis", "computer vision", "transfer learning", "fine-tune images")) or profile.image_path_columns:
            blockers = [] if (profile.image_path_columns or profile.image_dataset_path) else ["An image-path column or an ImageFolder dataset is required for CNN image analysis."]
            add("CNN_IMAGE_CLASSIFICATION", 0.99, ["The objective or dataset indicates an image-classification problem."] if profile.image_path_columns else ["The objective explicitly requests CNN/image analysis."], blockers, False, ["class-balanced split", "augmentation review", "transfer-learning policy", "validation/test isolation", "class-wise metrics", "error and explainability review"])

        if cls._contains(obj, ("anomaly", "outlier", "fraud", "novelty", "abnormal")):
            blockers = [] if len(profile.numeric_columns) >= 2 else ["At least two numerical variables are required by the current anomaly specialist."]
            add("ANOMALY", 0.94, ["The objective explicitly requests anomaly/novelty analysis."], blockers, False, ["scaling/imputation", "contamination sensitivity", "domain review of flagged observations"])

        if (profile.datetime_columns and profile.numeric_columns) or cls._contains(obj, ("forecast", "time series", "temporal", "trend over time")):
            blockers = [] if profile.datetime_columns and profile.numeric_columns else ["A date-like column and numeric value are required for the current time-series specialist."]
            evidence = ["Date/time and numeric measurement structure is available."] if profile.datetime_columns and profile.numeric_columns else ["The objective explicitly requests temporal analysis but required data structure is not yet confirmed."]
            add("TIME_SERIES", 0.88 if not blockers else 0.30, evidence, blockers, False, ["chronological ordering", "time-aware validation", "baseline forecast comparison"])

        if profile.explicit_target and cls._contains(obj, ("compare models", "benchmark", "compare machine learning", "machine learning and deep learning", "ml and dl")):
            add("HYBRID_SUPERVISED", 0.86, ["The objective explicitly asks for a supervised method comparison."], [], True, ["same validation protocol", "locked test set", "paired metric comparison", "compute-aware challenger design"])

        ranked = sorted(tasks.values(), key=lambda x: (-x.evidence_strength, x.task_id))
        return [asdict(x) for x in ranked]

    @classmethod
    def candidate_methods(cls, profile: AnalysisProfile, task_id: str) -> list[dict[str, Any]]:
        """Create executable candidate method families with explicit assumptions and evidence needs."""
        p = profile
        methods: list[MethodCandidate] = []
        if task_id in {"REGRESSION", "CLASSIFICATION"}:
            methods.append(MethodCandidate(
                "classical_ml", task_id, "ML Agent", "primary",
                ["Classical ML is appropriate for structured/tabular data and provides a broad model-family comparison."],
                ["Predictor/target relationship is learnable from available features.", "Training folds are representative enough for the selected validation protocol."],
                ["approved target", "usable predictors", "leakage review", "validation strategy"],
                ["fold-local preprocessing", "cross-validation", "locked test evaluation", "task-specific metrics"],
                ["Model family performance can be sensitive to preprocessing and distribution shift."], True,
            ))
            if len(p.numeric_columns) >= 5 and p.rows >= 500:
                methods.append(MethodCandidate(
                    "deep_learning", task_id, "DL Agent", "challenger",
                    ["A neural-network challenger becomes technically plausible with a moderately sized structured dataset and multiple numerical predictors."],
                    ["The sample contains enough independent information to support a neural model.", "Compute and convergence budget is sufficient."],
                    ["approved target", "multiple usable predictors", "sufficient sample size", "compute policy"],
                    ["strict train/validation isolation", "early stopping", "locked test evaluation", "comparison against classical baseline"],
                    ["Neural models may add complexity without improving generalisation on small tabular datasets."], True,
                ))
        elif task_id == "CLUSTERING":
            methods.append(MethodCandidate(
                "clustering_analysis", task_id, "Clustering Agent", "primary",
                ["The clustering specialist compares K-Means, Agglomerative, Gaussian Mixture and DBSCAN candidates under a common preprocessing policy."],
                ["Numerical representation is meaningful after scaling.", "Cluster structure is reasonably compact/separable."],
                ["at least two numerical variables", "missingness policy", "scaling policy"],
                ["median imputation", "standard scaling", "multiple k candidates", "silhouette/Calinski-Harabasz/Davies-Bouldin"],
                ["Validity metrics do not establish semantic or causal meaning of clusters."], False,
            ))
        elif task_id == "UNSUPERVISED":
            methods.append(MethodCandidate(
                "unsupervised_structure_discovery", task_id, "Unsupervised Learning Agent", "primary",
                ["PCA, clustering-family comparison and anomaly screening can reveal latent structure without an approved target."],
                ["Feature representation is meaningful after preprocessing.", "Unsupervised patterns require domain validation."],
                ["at least two numeric variables", "missingness policy", "scaling policy", "stability assessment"],
                ["PCA explained variance", "clustering validity", "repeatability", "anomaly sensitivity"],
                ["Unsupervised structure is descriptive and does not establish causal meaning."], True,
            ))
        elif task_id == "REINFORCEMENT_LEARNING":
            methods.append(MethodCandidate(
                "tabular_rl", task_id, "Reinforcement Learning Agent", "primary",
                ["The objective contains explicit sequential decision/reward semantics."],
                ["A finite state/action representation and meaningful reward function can be defined.", "Policy evaluation is possible without leakage from future outcomes."],
                ["state definition", "action definition", "reward definition", "environment dynamics"],
                ["multiple random seeds", "learning curve", "policy stability", "off-policy evaluation where applicable"],
                ["The bounded tabular implementation is not a substitute for domain-specific environment design."], False,
            ))
        elif task_id == "CNN_IMAGE_CLASSIFICATION":
            methods.append(MethodCandidate(
                "cnn_transfer_learning", task_id, "CNN Image Analysis Agent", "primary",
                ["A dedicated CNN specialist supports pretrained convolutional architectures, augmentation, transfer learning and controlled fine-tuning."],
                ["Images are representative of the intended classification task.", "Train/validation/test separation is independent of augmentation and model fitting."],
                ["image dataset", "class labels", "class balance", "augmentation policy", "validation/test split", "compute policy"],
                ["stratified class review", "locked test set", "class-wise precision/recall/F1", "confusion matrix", "multiple-seed confirmation where practical"],
                ["CNN performance can be sensitive to dataset shift, label quality, augmentation and pretrained-domain mismatch."], False,
            ))
        elif task_id == "ANOMALY":
            methods.append(MethodCandidate(
                "isolation_forest", task_id, "Anomaly Detection Agent", "primary",
                ["Isolation Forest is implemented by the current bounded anomaly specialist."],
                ["Rare observations can be represented in the numerical feature space.", "The contamination assumption is an acceptable starting point."],
                ["at least two numerical variables", "scaling/imputation policy", "contamination sensitivity"],
                ["median imputation", "standard scaling", "contamination sensitivity review", "domain review"],
                ["Anomaly flags are model-dependent and are not automatically true anomalies."], False,
            ))
        elif task_id == "TIME_SERIES":
            methods.append(MethodCandidate(
                "temporal_baseline", task_id, "Time-Series Agent", "primary",
                ["The time-series specialist provides temporal ordering, trend, and bounded forecasting evidence."],
                ["A meaningful date/value relationship exists.", "Future observations are not allowed to leak into training."],
                ["date-like column", "numeric value", "chronological ordering"],
                ["time-aware split", "baseline comparison", "forecast diagnostics"],
                ["Forecast quality depends strongly on sampling regularity and structural breaks."], False,
            ))
        elif task_id == "STATISTICS":
            methods.append(MethodCandidate(
                "descriptive_association", task_id, "Statistical Insight Agent", "primary",
                ["Descriptive and association evidence is available without making unsupported causal claims."],
                ["Variables and sampling context are sufficient for the requested descriptive question."],
                ["schema", "variable types", "missingness context"],
                ["descriptive summaries", "association diagnostics", "uncertainty context where available"],
                ["Association does not establish causality or temporal direction."], False,
            ))
        elif task_id == "EXPLORATION":
            methods.append(MethodCandidate(
                "evidence_profile", task_id, "Statistical Insight Agent", "primary",
                ["A structured evidence profile is the safest starting point for an under-specified objective."],
                ["The user is willing to refine the objective after initial evidence review."],
                ["schema", "missingness", "distributions", "associations"],
                ["data quality", "descriptive statistics", "visual inspection"],
                ["Exploration does not answer a predictive question without a defined task."], False,
            ))
        elif task_id == "HYBRID_SUPERVISED":
            base_task = "REGRESSION" if profile.target_type == "regression" else "CLASSIFICATION"
            methods.extend(cls.candidate_methods(profile, base_task))
            for method in methods:
                method["task_id"] = "HYBRID_SUPERVISED"
                method["role"] = "primary" if method["method_id"] == "classical_ml" else "challenger"
        return [AnalyticalMethodRegistry.annotate(asdict(x)) for x in methods]

    @classmethod
    def compare_methods(cls, methods: list[dict[str, Any]], profile: AnalysisProfile) -> list[dict[str, Any]]:
        """Create an explicit evidence comparison instead of hiding method choice in a threshold."""
        comparison = []
        for method in methods:
            evidence_score = 0.0
            reasons = list(method.get("evidence_for", []))
            if method["specialist"] == "ML Agent":
                evidence_score += 0.55
                if profile.rows >= 100:
                    evidence_score += 0.15
                if profile.columns <= 500:
                    evidence_score += 0.10
            elif method["specialist"] == "CNN Image Analysis Agent":
                evidence_score += 0.90
                if profile.image_path_columns:
                    evidence_score += 0.10
            elif method["specialist"] == "DL Agent":
                evidence_score += 0.25
                if profile.rows >= 500:
                    evidence_score += 0.20
                if len(profile.numeric_columns) >= 5:
                    evidence_score += 0.20
            else:
                evidence_score += 0.60
            comparison.append({
                "method_id": method["method_id"],
                "specialist": method["specialist"],
                "role": method["role"],
                "evidence_score": round(min(evidence_score, 1.0), 3),
                "evidence": reasons,
                "assumptions": method["assumptions"],
                "validation_protocol": method["validation_protocol"],
                "limitations": method["limitations"],
                "capabilities": method.get("capabilities", {}),
            })
        return comparison

    @classmethod
    def _task_by_id(cls, tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        """Return one task hypothesis by identifier."""
        return next((x for x in tasks if x.get("task_id") == task_id), None)

    @classmethod
    def _method_by_id(cls, methods: list[dict[str, Any]], method_id: str) -> dict[str, Any] | None:
        """Return one method candidate by identifier."""
        return next((x for x in methods if x.get("method_id") == method_id), None)

    @classmethod
    def _fallback_selection(cls, profile: AnalysisProfile, tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Produce a transparent provisional selection when LLM reasoning is unavailable."""
        viable = [x for x in tasks if not x.get("blockers")]
        task = viable[0] if viable else tasks[0]
        methods = cls.candidate_methods(profile, task["task_id"])
        comparison = cls.compare_methods(methods, profile)
        selected = max(comparison, key=lambda x: x.get("evidence_score", 0.0)) if comparison else None
        return task, ([selected] if selected else [])

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from an LLM response."""
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    @classmethod
    def _llm_task_review(cls, llm: Any, profile: AnalysisProfile, tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Ask the configured LLM to identify the task while remaining bounded by supplied evidence."""
        payload = json.dumps({"profile": asdict(profile), "task_hypotheses": tasks}, default=str)
        prompt = (
            "You are the task-identification stage of a professional Agent Data Scientist. "
            "Follow this order: analytical objective -> task identification. Use only the supplied evidence. "
            "Do not invent a target, data type, temporal structure, or domain fact. Select exactly one task_id "
            "from the supplied hypotheses, explain why, list rejected task_ids and unresolved ambiguities, and "
            "state whether the task is blocked. Return JSON only with keys: task_id, rationale, rejected_task_ids, "
            "ambiguities, blocked.\n\nEvidence:\n" + payload
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            data = cls._extract_json(response.content if hasattr(response, "content") else str(response))
            if not data:
                return None
            task_id = str(data.get("task_id", "")).upper()
            allowed = {x["task_id"] for x in tasks}
            if task_id not in allowed:
                return None
            return {
                "task_id": task_id,
                "rationale": data.get("rationale", []),
                "rejected_task_ids": [x for x in data.get("rejected_task_ids", []) if x in allowed],
                "ambiguities": data.get("ambiguities", []),
                "blocked": bool(data.get("blocked", False)),
            }
        except Exception:
            return None

    @classmethod
    def _llm_method_review(
        cls,
        llm: Any,
        profile: AnalysisProfile,
        task: dict[str, Any],
        methods: list[dict[str, Any]],
        comparison: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask the configured LLM to compare candidate methods and select a bounded specialist plan."""
        payload = json.dumps({"profile": asdict(profile), "task": task, "methods": methods, "comparison": comparison}, default=str)
        prompt = (
            "You are the method-selection stage of a professional Agent Data Scientist. "
            "Follow this order: candidate methods -> assumptions -> evidence comparison -> specialist selection. "
            "Use only supplied evidence. Prefer the least complex method that adequately answers the task unless "
            "the evidence supports a more complex challenger. Do not use dataset size as a standalone decision rule. "
            "Return JSON only with keys: selected_method_id, challenger_method_ids, rationale, assumptions_to_verify, "
            "evidence_gaps, specialist_order. selected_method_id must be one supplied method_id; challenger IDs must "
            "also be supplied. specialist_order may contain only known specialist names.\n\nEvidence:\n" + payload
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            data = cls._extract_json(response.content if hasattr(response, "content") else str(response))
            if not data:
                return None
            allowed_methods = {x["method_id"] for x in methods}
            selected = str(data.get("selected_method_id", ""))
            if selected not in allowed_methods:
                return None
            challengers = [x for x in data.get("challenger_method_ids", []) if x in allowed_methods and x != selected]
            specialist_order = [x for x in data.get("specialist_order", []) if x in cls.SPECIALISTS]
            return {
                "selected_method_id": selected,
                "challenger_method_ids": challengers,
                "rationale": data.get("rationale", []),
                "assumptions_to_verify": data.get("assumptions_to_verify", []),
                "evidence_gaps": data.get("evidence_gaps", []),
                "specialist_order": specialist_order,
            }
        except Exception:
            return None

    @classmethod
    def _specialist_plan(
        cls,
        task_id: str,
        methods: list[dict[str, Any]],
        selection: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Translate the selected method into a bounded execution plan for specialist agents."""
        selected = cls._method_by_id(methods, selection.get("selected_method_id", ""))
        if selected is None:
            return []
        plan = [
            {"specialist": "Data Quality Agent", "role": "prerequisite", "method_id": selected["method_id"], "purpose": "verify data quality and structural risks before analytical execution"},
            {"specialist": "Statistical Insight Agent", "role": "context", "method_id": selected["method_id"], "purpose": "provide descriptive/association evidence relevant to method assumptions"},
        ]
        primary = {"specialist": selected["specialist"], "role": selected["role"], "method_id": selected["method_id"], "purpose": f"execute the selected {selected['method_id']} analytical method family"}
        plan.append(primary)
        for challenger_id in selection.get("challenger_method_ids", []):
            candidate = cls._method_by_id(methods, challenger_id)
            if candidate and candidate.get("challenger_eligible"):
                plan.append({"specialist": candidate["specialist"], "role": "challenger", "method_id": challenger_id, "purpose": f"challenge the primary method under the same validation protocol"})
        return plan

    @classmethod
    def decide(
        cls,
        df: pd.DataFrame,
        objective: str,
        llm_config: LLMConfig | None = None,
        target: str | None = None,
        image_dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """Run the full task-to-method protocol and return a human-reviewable decision package."""
        profile = cls.profile(df, objective, target=target, image_dataset_path=image_dataset_path)
        tasks = cls.identify_tasks(profile)
        if not tasks:
            tasks = [asdict(TaskHypothesis("EXPLORATION", 0.10, ["No specialized task evidence was identified."], [], False, ["schema review"]))]

        fallback_task, fallback_selected = cls._fallback_selection(profile, tasks)
        task_review: dict[str, Any] = {
            "status": "not_run",
            "selected_task_id": fallback_task["task_id"],
            "source": "provisional_evidence_only",
        }
        method_review: dict[str, Any] = {"status": "not_run", "source": "provisional_evidence_only"}
        selected_task = fallback_task
        task_methods = cls.candidate_methods(profile, selected_task["task_id"])
        comparison = cls.compare_methods(task_methods, profile)
        fallback_method = fallback_selected[0] if fallback_selected else None
        llm_used = False

        if llm_config is not None and llm_config.configured():
            llm = get_llm(llm_config)
            if llm is not None:
                task_result = cls._llm_task_review(llm, profile, tasks)
                if task_result:
                    reviewed_task = cls._task_by_id(tasks, task_result["task_id"])
                    if reviewed_task and not reviewed_task.get("blockers") and not task_result.get("blocked"):
                        selected_task = reviewed_task
                        task_review = {"status": "llm_selected", "source": "configured_llm", **task_result}
                        task_methods = cls.candidate_methods(profile, selected_task["task_id"])
                        comparison = cls.compare_methods(task_methods, profile)
                        method_result = cls._llm_method_review(llm, profile, selected_task, task_methods, comparison)
                        if method_result:
                            method_review = {"status": "llm_selected", "source": "configured_llm", **method_result}
                            llm_used = True

        if method_review.get("status") != "llm_selected":
            selected_method_id = fallback_method.get("method_id") if fallback_method else (comparison[0]["method_id"] if comparison else "")
            method_review = {
                "status": "provisional_heuristic",
                "source": "transparent_evidence_fallback",
                "selected_method_id": selected_method_id,
                "challenger_method_ids": [],
                "rationale": ["LLM-assisted method selection was not available; this is a provisional evidence proposal and requires human review."],
                "assumptions_to_verify": [],
                "evidence_gaps": ["Independent analytical reasoning from the configured LLM was not available."],
                "specialist_order": [],
            }

        selected_method = cls._method_by_id(task_methods, method_review.get("selected_method_id", "")) or (task_methods[0] if task_methods else None)
        if selected_method is None:
            selected_method = {
                "method_id": "evidence_profile",
                "task_id": "EXPLORATION",
                "specialist": "Statistical Insight Agent",
                "role": "primary",
                "evidence_for": ["No executable specialized method was identified."],
                "assumptions": [],
                "evidence_requirements": ["schema"],
                "validation_protocol": ["data review"],
                "limitations": ["Method selection remains unresolved."],
                "challenger_eligible": False,
            }

        selection = {
            **method_review,
            "selected_method": selected_method,
            "comparison": comparison,
            "task_id": selected_task["task_id"],
            "task_hypothesis": selected_task,
        }
        specialist_plan = cls._specialist_plan(selected_task["task_id"], task_methods, selection)
        # Governance order is intentionally deterministic: data quality and statistical context
        # precede the selected analytical specialist. The LLM may recommend specialists, but it cannot
        # bypass these prerequisites by returning an arbitrary execution order.

        sub_agents = [x["specialist"] for x in specialist_plan]
        decision_source = "evidence_protocol_plus_llm_selection" if llm_used else "transparent_evidence_fallback"
        validation_strategy = ValidationStrategyAgent.propose(df, selected_task.get("task_id", "EXPLORATION"), target=target)
        experiment_plan = ExperimentPlanner.plan({
            "primary_route": selected_task["task_id"],
            "method_selection": selection,
            "evidence_gaps": method_review.get("evidence_gaps", []),
        })
        return {
            "primary_route": selected_task["task_id"],
            "sub_agents": sub_agents,
            "candidate_routes": tasks[:8],
            "profile": asdict(profile),
            "task_identification": tasks,
            "method_candidates": task_methods,
            "evidence_comparison": comparison,
            "method_selection": selection,
            "specialist_plan": specialist_plan,
            "decision_source": decision_source,
            "selection_status": "llm_selected" if llm_used else "provisional_heuristic_requires_human_review",
            "rationale": list(task_review.get("rationale", [])) + list(method_review.get("rationale", [])),
            "assumptions_to_verify": method_review.get("assumptions_to_verify", []),
            "evidence_gaps": method_review.get("evidence_gaps", []),
            "experiment_plan": experiment_plan,
            "validation_strategy": validation_strategy,
            "task_review": task_review,
            "method_review": method_review,
            "requires_human_approval": True,
            "selection_protocol": [
                "analytical objective",
                "task identification",
                "candidate methods",
                "assumptions",
                "evidence comparison",
                "specialist selection",
                "human approval",
            ],
        }

    @classmethod
    def _subagents_for(cls, route: str, profile: AnalysisProfile) -> list[str]:
        """Return the bounded specialist names associated with a provisional task route."""
        methods = cls.candidate_methods(profile, route)
        if not methods:
            return ["Data Quality Agent", "Statistical Insight Agent"]
        plan = cls._specialist_plan(route, methods, {"selected_method_id": methods[0]["method_id"], "challenger_method_ids": []})
        return [x["specialist"] for x in plan]
