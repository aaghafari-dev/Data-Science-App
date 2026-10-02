"""Professional clustering-analysis services for Data Science Studio Pro.

The service layer is intentionally GUI-independent so the same analytical protocol can be
used by the desktop workspace, the Master Agent, tests, and exported analysis artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class ClusteringAnalysisEngine:
    """Run reproducible, evidence-rich unsupervised clustering analyses."""

    @staticmethod
    def _numeric_frame(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Return finite numeric features selected from the active dataframe."""
        if df is None or df.empty:
            raise ValueError("No data is available for clustering analysis.")
        selected = features or [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        selected = [c for c in selected if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(selected) < 2:
            raise ValueError("Clustering requires at least two numeric features.")
        x = df[selected].replace([np.inf, -np.inf], np.nan)
        x = x.loc[:, x.notna().any(axis=0)]
        selected = [str(c) for c in x.columns]
        if len(selected) < 2:
            raise ValueError("At least two numeric features containing finite observations are required.")
        return x, selected

    @staticmethod
    def _preprocess(
        df: pd.DataFrame,
        features: list[str] | None = None,
        scaling: str = "standard",
        imputation: str = "median",
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Apply deterministic imputation and scaling without mutating the source dataframe."""
        x, selected = ClusteringAnalysisEngine._numeric_frame(df, features)
        imputer = SimpleImputer(strategy=imputation)
        transformed = imputer.fit_transform(x)
        if scaling == "minmax":
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(transformed)
        elif scaling == "robust":
            scaler = RobustScaler()
            transformed = scaler.fit_transform(transformed)
        elif scaling == "none":
            scaler = None
        else:
            scaling = "standard"
            scaler = StandardScaler()
            transformed = scaler.fit_transform(transformed)
        meta = {
            "features": selected,
            "imputation": imputation,
            "scaling": scaling,
            "rows": int(len(x)),
            "missing_values_before_imputation": int(x.isna().sum().sum()),
        }
        return transformed, selected, meta

    @staticmethod
    def _metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Calculate bounded internal cluster-validity metrics."""
        unique, counts = np.unique(labels, return_counts=True)
        mask = labels >= 0
        x_eval = x[mask]
        labels_eval = labels[mask]
        unique_eval = np.unique(labels_eval)
        result: dict[str, Any] = {
            "n_clusters": int(len(unique_eval)),
            "noise_fraction": float(np.mean(~mask)),
            "cluster_sizes": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        }
        if len(unique_eval) >= 2 and len(x_eval) > len(unique_eval):
            result["silhouette"] = float(silhouette_score(x_eval, labels_eval))
            result["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, labels_eval))
            result["davies_bouldin"] = float(davies_bouldin_score(x_eval, labels_eval))
        else:
            result.update({"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None})
        return result

    @staticmethod
    def run(
        df: pd.DataFrame,
        features: list[str] | None = None,
        scaling: str = "standard",
        imputation: str = "median",
        algorithms: list[str] | None = None,
        k_min: int = 2,
        k_max: int = 8,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run multiple bounded clustering families and compare their validity evidence."""
        x, selected, prep = ClusteringAnalysisEngine._preprocess(df, features, scaling, imputation)
        k_min = max(2, int(k_min))
        k_max = min(max(k_min, int(k_max)), max(k_min, min(12, len(x) - 1)))
        algorithms = algorithms or ["K-Means", "Agglomerative", "Gaussian Mixture", "DBSCAN"]
        results: list[dict[str, Any]] = []

        if "K-Means" in algorithms:
            for k in range(k_min, k_max + 1):
                model = KMeans(n_clusters=k, n_init=10, random_state=seed)
                labels = model.fit_predict(x)
                metrics = ClusteringAnalysisEngine._metrics(x, labels)
                results.append({"algorithm": "K-Means", "parameters": {"k": k}, "metrics": metrics, "inertia": float(model.inertia_)})

        if "Agglomerative" in algorithms:
            for k in range(k_min, k_max + 1):
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = model.fit_predict(x)
                results.append({"algorithm": "Agglomerative", "parameters": {"k": k, "linkage": "ward"}, "metrics": ClusteringAnalysisEngine._metrics(x, labels)})

        if "Gaussian Mixture" in algorithms:
            for k in range(k_min, k_max + 1):
                model = GaussianMixture(n_components=k, random_state=seed, covariance_type="full")
                labels = model.fit_predict(x)
                item = {"algorithm": "Gaussian Mixture", "parameters": {"k": k, "covariance_type": "full"}, "metrics": ClusteringAnalysisEngine._metrics(x, labels), "aic": float(model.aic(x)), "bic": float(model.bic(x))}
                results.append(item)

        if "DBSCAN" in algorithms:
            model = DBSCAN(eps=float(dbscan_eps), min_samples=max(2, int(dbscan_min_samples)))
            labels = model.fit_predict(x)
            results.append({"algorithm": "DBSCAN", "parameters": {"eps": float(dbscan_eps), "min_samples": int(dbscan_min_samples)}, "metrics": ClusteringAnalysisEngine._metrics(x, labels)})

        candidates = [r for r in results if r.get("metrics", {}).get("silhouette") is not None]
        selected_result = max(candidates, key=lambda r: float(r["metrics"]["silhouette"])) if candidates else (results[0] if results else None)
        labels = None
        if selected_result:
            algorithm = selected_result["algorithm"]
            params = selected_result["parameters"]
            if algorithm == "K-Means":
                labels = KMeans(n_clusters=params["k"], n_init=10, random_state=seed).fit_predict(x)
            elif algorithm == "Agglomerative":
                labels = AgglomerativeClustering(n_clusters=params["k"], linkage=params.get("linkage", "ward")).fit_predict(x)
            elif algorithm == "Gaussian Mixture":
                labels = GaussianMixture(n_components=params["k"], random_state=seed, covariance_type=params.get("covariance_type", "full")).fit_predict(x)
            elif algorithm == "DBSCAN":
                labels = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(x)

        profiles = ClusteringAnalysisEngine.profile(df, selected, labels) if labels is not None else {}
        return {
            "status": "ok" if results else "error",
            "agent": "Clustering Analysis",
            "features": selected,
            "preprocessing": prep,
            "candidates": results,
            "selected_method": selected_result,
            "labels": labels.tolist() if labels is not None else [],
            "cluster_profile": profiles,
            "interpretation_note": "Internal validity metrics support method comparison but do not establish that clusters are scientifically or operationally meaningful. Domain validation is required.",
            "reproducibility": {"seed": int(seed), "algorithms": algorithms, "k_range": [k_min, k_max]},
        }

    @staticmethod
    def profile(df: pd.DataFrame, features: list[str], labels: np.ndarray | list[int]) -> dict[str, Any]:
        """Summarize feature distributions and sizes for the selected clustering solution."""
        frame = df[features].copy().apply(pd.to_numeric, errors="coerce")
        labels_arr = np.asarray(labels)
        work = frame.copy()
        work["__cluster__"] = labels_arr
        numeric = work.groupby("__cluster__")[features].agg(["mean", "median", "std"])
        sizes = work["__cluster__"].value_counts().sort_index()
        return {
            "sizes": {str(k): int(v) for k, v in sizes.items()},
            "feature_summary": json_safe_frame(numeric),
        }

    @staticmethod
    def stability_analysis(
        df: pd.DataFrame,
        features: list[str],
        algorithm: str,
        parameters: dict[str, Any],
        repeats: int = 5,
    ) -> dict[str, Any]:
        """Estimate solution stability across repeated seeds using adjusted Rand index."""
        x, _, _ = ClusteringAnalysisEngine._preprocess(df, features, "standard", "median")
        labels_list: list[np.ndarray] = []
        for seed in range(max(2, int(repeats))):
            if algorithm == "K-Means":
                labels = KMeans(n_clusters=int(parameters["k"]), n_init=10, random_state=seed).fit_predict(x)
            elif algorithm == "Agglomerative":
                labels = AgglomerativeClustering(n_clusters=int(parameters["k"]), linkage=parameters.get("linkage", "ward")).fit_predict(x)
            elif algorithm == "Gaussian Mixture":
                labels = GaussianMixture(n_components=int(parameters["k"]), random_state=seed, covariance_type=parameters.get("covariance_type", "full")).fit_predict(x)
            elif algorithm == "DBSCAN":
                labels = DBSCAN(eps=float(parameters["eps"]), min_samples=int(parameters["min_samples"])).fit_predict(x)
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            labels_list.append(labels)
        pairwise = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                pairwise.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
        return {
            "algorithm": algorithm,
            "repeats": len(labels_list),
            "pairwise_ari": pairwise,
            "mean_ari": float(np.mean(pairwise)) if pairwise else None,
            "min_ari": float(np.min(pairwise)) if pairwise else None,
            "interpretation": "Higher adjusted-Rand agreement indicates greater repeatability under the tested random seeds; it does not prove population-level cluster stability.",
        }

    @staticmethod
    def pca_projection(df: pd.DataFrame, features: list[str], labels: list[int] | None = None) -> dict[str, Any]:
        """Create a two-component PCA projection for cluster visualization."""
        x, selected, prep = ClusteringAnalysisEngine._preprocess(df, features, "standard", "median")
        if x.shape[1] < 2:
            raise ValueError("PCA visualization requires at least two numeric features.")
        pca = PCA(n_components=2, random_state=42)
        z = pca.fit_transform(x)
        return {
            "features": selected,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "projection": [[float(a), float(b)] for a, b in z],
            "labels": list(labels) if labels is not None else [],
            "preprocessing": prep,
        }


def json_safe_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Convert a pandas table into a compact JSON-safe nested mapping."""
    return {
        str(index): {str(col): (None if pd.isna(value) else float(value)) for col, value in row.items()}
        for index, row in frame.iterrows()
    }
