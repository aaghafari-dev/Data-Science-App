"""Tests for V20.2 Master Agent routing, clustering, and paired shelf semantics."""

import pandas as pd

from agent.graph_cluster import run_clustering_step
from services.master_router import MasterDecisionEngine
from core.viz_engine import VizEngine


def test_numeric_rows_columns_remain_paired_observations():
    """Ensure one numeric field on each opposite shelf is not collapsed to one point."""
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [2.0, 4.0, 1.0, 8.0]})
    plan = VizEngine.plan_shelves(df, ["Y"], ["X"])
    plot_df = VizEngine.aggregate_for_shelves(df, ["Y"], ["X"])
    assert plan.quantitative_pair is True
    assert plan.x_field == "X"
    assert plan.y_field == "Y"
    assert len(plot_df) == len(df)


def test_master_router_detects_clustering_intent():
    """Ensure clustering intent produces a clustering route and specialist."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 1, 3, 4, 6, 5], "group": [1, 1, 2, 2, 3, 3]})
    decision = MasterDecisionEngine.decide(df, "Find clusters and segment similar observations")
    assert decision["primary_route"] == "CLUSTERING"
    assert "Clustering Agent" in decision["sub_agents"]


def test_clustering_specialist_returns_validation_metrics():
    """Ensure the clustering specialist returns multiple validity metrics."""
    df = pd.DataFrame({"x": [0, 0.1, 0.2, 5, 5.1, 5.2], "y": [0, 0.2, -0.1, 5, 4.9, 5.2]})
    result = run_clustering_step(df, seed=42, max_k=4)
    assert result["status"] == "ok"
    assert result["best_metrics"]["silhouette"] >= -1
    assert "calinski_harabasz" in result["best_metrics"]
    assert "davies_bouldin" in result["best_metrics"]


def test_master_selection_protocol_is_explicit_and_not_size_only():
    """Ensure the Master decision exposes task, method, assumptions, evidence, and specialist stages."""
    df = pd.DataFrame({
        "x1": range(120),
        "x2": [i * 0.5 for i in range(120)],
        "x3": [i % 7 for i in range(120)],
        "target": [float(i) + (i % 3) * 0.1 for i in range(120)],
    })
    decision = MasterDecisionEngine.decide(df, "Predict the target using an appropriate method", target="target")
    assert decision["selection_protocol"] == [
        "analytical objective",
        "task identification",
        "candidate methods",
        "assumptions",
        "evidence comparison",
        "specialist selection",
        "human approval",
    ]
    assert decision["task_identification"]
    assert decision["method_candidates"]
    assert decision["evidence_comparison"]
    assert decision["method_selection"]["selected_method"]["method_id"]
    assert decision["specialist_plan"][0]["specialist"] == "Data Quality Agent"
    assert decision["selection_status"] == "provisional_heuristic_requires_human_review"


def test_master_does_not_invent_supervised_target_for_unsupervised_objective():
    """Ensure an unsupervised objective is not converted into hidden supervised routing."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "label": [0, 1, 0, 1]})
    decision = MasterDecisionEngine.decide(df, "Find clusters of similar observations")
    assert decision["primary_route"] == "CLUSTERING"
    assert decision["profile"]["approved_target"] is None
    assert decision["profile"]["explicit_target"] is False


def test_data_quality_typed_tool_compatibility_entry_point():
    """Ensure the typed agent registry can call the compatibility evaluate entry point."""
    from services.data_quality import DataQualityEngine
    df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
    result = DataQualityEngine.evaluate(df, target="y")
    assert result["status"] in {"pass", "review"}
    assert result["rows"] == 3


def test_professional_clustering_workspace_service_compares_methods():
    """Ensure the senior-analyst clustering service compares multiple method families."""
    from services.clustering_analysis import ClusteringAnalysisEngine
    df = pd.DataFrame({"x": [0, .1, .2, 5, 5.1, 5.2], "y": [0, .2, -.1, 5, 4.9, 5.2]})
    result = ClusteringAnalysisEngine.run(df, k_min=2, k_max=3)
    assert result["status"] == "ok"
    assert len(result["candidates"]) >= 4
    assert result["selected_method"]["algorithm"] in {"K-Means", "Agglomerative", "Gaussian Mixture", "DBSCAN"}
    assert "cluster_profile" in result


def test_qwen_memory_estimator_uses_local_weight_files_when_available(tmp_path):
    """Ensure local LLM resource estimation is based on measured checkpoint size when possible."""
    import pytest
    pytest.importorskip("transformers")
    from agent.local_llm import LocalLLMLoader
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"torch_dtype":"bfloat16"}', encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(b"0" * (1024 * 1024))
    estimate, basis, observed = LocalLLMLoader._model_memory_estimate(str(model_dir), "Qwen2.5-3B-Instruct")
    assert observed > 0
    assert "measured local weight files" in basis
    assert estimate > observed
