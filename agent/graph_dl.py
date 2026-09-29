from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from services.governance import StatisticalUncertainty
from services.error_analysis import ErrorAnalysisEngine
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory
from services.compute_backend import ComputeBackend
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error


def _torch_mlp(Xtr, Xte, ytr, yte, task, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr), dtype=torch.float32)
    Xv = torch.tensor(Xte.toarray() if hasattr(Xte, "toarray") else np.asarray(Xte), dtype=torch.float32)
    if task == "classification":
        yt = torch.tensor(np.asarray(ytr), dtype=torch.long)
        yv = torch.tensor(np.asarray(yte), dtype=torch.long)
        out_dim = int(np.max(np.asarray(ytr))) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        yt = torch.tensor(np.asarray(ytr, dtype=np.float32).reshape(-1,1), dtype=torch.float32)
        yv = torch.tensor(np.asarray(yte, dtype=np.float32).reshape(-1,1), dtype=torch.float32)
        out_dim = 1
        loss_fn = nn.MSELoss()
    model = nn.Sequential(nn.Linear(Xt.shape[1],128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,out_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = Xt.to(device), yt.to(device)
    for _ in range(120):
        opt.zero_grad(); pred=model(Xt); loss=loss_fn(pred,yt); loss.backward(); opt.step()
    with torch.no_grad(): pred=model(Xv.to(device)).detach().cpu().numpy()
    if task == "classification":
        pred_labels=pred.argmax(axis=1)
        metrics={"accuracy":float(accuracy_score(np.asarray(yte),pred_labels)),"f1_weighted":float(f1_score(np.asarray(yte),pred_labels,average="weighted"))}
        evaluation=ErrorAnalysisEngine.classification(yte,pred_labels)
    else:
        pred_values=pred.reshape(-1)
        metrics={"r2":float(r2_score(np.asarray(yte),pred_values)),"mae":float(mean_absolute_error(np.asarray(yte),pred_values)),"rmse":float(mean_squared_error(np.asarray(yte),pred_values)**.5)}
        evaluation=ErrorAnalysisEngine.regression(yte,pred_values)
    metrics["error_analysis"]=evaluation
    return model, metrics


def run_dl_step(df: pd.DataFrame, target: str | None = None, seed: int = 42, compute_mode: str = "CPU") -> dict[str, Any]:
    """Master-agent DL internal agent with optional CUDA execution and safe CPU fallback."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available."}
    rec = recommend_targets_and_features(df)
    target = target or rec.get("suggested_target") or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    compute = ComputeBackend.resolve(compute_mode)
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target]); y = work[target]
    task = "classification" if (y.dtype == object or y.nunique() <= min(20, max(2, int(len(y) * .05)))) else "regression"
    if task == "classification": y = pd.Series(pd.factorize(y.astype(str))[0], index=y.index)
    from services.model_preprocessing import build_preprocessor
    pre = build_preprocessor(X)
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed, stratify=strat)
    t0=time.time(); backend="scikit-learn MLP"; model_obj=None
    if compute.mode in {"GPU","CPU+GPU"} and compute.gpu_available:
        try:
            Xtr_t=pre.fit_transform(Xtr); Xte_t=pre.transform(Xte)
            if Xtr_t.shape[0] <= 100000 and Xtr_t.shape[1] <= 5000:
                import torch
                model_obj, metrics = _torch_mlp(Xtr_t,Xte_t,ytr,yte,task,seed,"cuda")
                backend="PyTorch CUDA MLP"
            else:
                compute.message += " Dataset too large for the bounded CUDA tabular path; using CPU MLP instead."
        except Exception as exc:
            compute.message += f" CUDA execution failed safely ({exc}); using CPU MLP instead."
    if model_obj is None:
        estimator = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, early_stopping=True, random_state=seed) if task=="classification" else MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500, early_stopping=True, random_state=seed)
        pipe=Pipeline([("pre",pre),("model",estimator)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); model_obj=pipe
        if task=="classification": metrics={"accuracy":float(accuracy_score(yte,pred)),"f1_weighted":float(f1_score(yte,pred,average="weighted"))}; evaluation=ErrorAnalysisEngine.classification(yte,pred)
        else: metrics={"r2":float(r2_score(yte,pred)),"mae":float(mean_absolute_error(yte,pred)),"rmse":float(mean_squared_error(yte,pred)**.5)}; evaluation=ErrorAnalysisEngine.regression(yte,pred)
        metrics["error_analysis"]=evaluation
    primary="f1_weighted" if task=="classification" else "r2"
    return {"status":"ok","agent":"DL","target":target,"task":task,"backend":backend,
            "models": {"MLP": {"metrics":metrics,"seconds":round(time.time()-t0,3)}},
            "best_model":"MLP","best_metrics":metrics,"model_object":model_obj,"compute":compute.__dict__,
            "target_analysis":rec,"memory":AgentMemory("DL",20).to_dict(),
            "summary":f"DL trained an MLP using the {backend} backend. {compute.message}"}
