"""Professional unsupervised-learning analysis engine."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class UnsupervisedAnalysisEngine:
    """Provide reproducible dimensionality reduction, clustering and anomaly analysis."""
    @staticmethod
    def prepare(df, features, imputation="median", scaling="standard"):
        """Prepare numeric features using a deterministic preprocessing policy."""
        x = df[features].apply(pd.to_numeric, errors="coerce")
        x = SimpleImputer(strategy=imputation).fit_transform(x)
        scaler = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(), "none": None}.get(scaling)
        if scaler is not None: x = scaler.fit_transform(x)
        return np.asarray(x, dtype=float)

    @classmethod
    def run(cls, df, features=None, imputation="median", scaling="standard", seed=42):
        """Run PCA, clustering-family comparison and anomaly detectors."""
        features = features or list(df.select_dtypes(include=[np.number]).columns)
        if len(features) < 2: raise ValueError("Unsupervised analysis requires at least two numeric features.")
        x = cls.prepare(df, features, imputation, scaling)
        pca_n = min(5, x.shape[1], x.shape[0])
        pca = PCA(n_components=pca_n, random_state=seed).fit(x)
        labels = {}
        for name, model in {
            "K-Means": KMeans(n_clusters=min(3, max(2, len(x)//50)), random_state=seed, n_init=10),
            "Agglomerative": AgglomerativeClustering(n_clusters=min(3, max(2, len(x)//50))),
            "Gaussian Mixture": GaussianMixture(n_components=min(3, max(2, len(x)//50)), random_state=seed),
            "DBSCAN": DBSCAN(eps=0.8, min_samples=max(5, min(20, len(x)//100 or 5)))
        }.items():
            labels[name] = (model.fit_predict(x) if name != "Gaussian Mixture" else model.fit(x).predict(x)).tolist()
        anomalies = {
            "Isolation Forest": IsolationForest(contamination="auto", random_state=seed).fit_predict(x).tolist(),
            "Local Outlier Factor": LocalOutlierFactor(novelty=False).fit_predict(x).tolist()
        }
        return {"features": features, "preprocessing": {"imputation": imputation, "scaling": scaling}, "pca": {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "components": pca.components_.tolist(), "projection": pca.transform(x)[:, :2].tolist()}, "cluster_labels": labels, "anomaly_labels": anomalies, "interpretation_note": "Unsupervised validity metrics and anomaly flags require domain validation; they do not establish semantic meaning or causality."}
