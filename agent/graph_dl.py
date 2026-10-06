"""Module duty: Graph dl.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from services.mlp_training import make_mlp, fit_with_convergence_audit
from sklearn.model_selection import train_test_split
from services.evaluation_protocol import ValidationProtocolAdvisor
from services.resource_policy import ResourcePolicy
from services.feature_provenance import FeatureProvenanceEngine
from services.data_slice_analysis import DataSliceAnalyzer
from sklearn.pipeline import Pipeline
from services.error_analysis import ErrorAnalysisEngine
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory
from services.compute_backend import ComputeBackend
from services.agent_quality import ProfessionalEvaluationEngine, TemporalAvailabilityAnalyzer, ModelDiagnosisEngine, RobustnessPerturbationAnalyzer, DLValidationEngine
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error


def _torch_mlp(Xtr, Xte, ytr, yte, task, seed, device, epochs=120):
    """Perform the torch mlp operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr), dtype=torch.float32)
    Xv = torch.tensor(Xte.toarray() if hasattr(Xte, "toarray") else np.asarray(Xte), dtype=torch.float32)
    if task == "classification":
        yt = torch.tensor(np.asarray(ytr), dtype=torch.long)
        out_dim = int(np.max(np.asarray(ytr))) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        yt = torch.tensor(np.asarray(ytr, dtype=np.float32).reshape(-1,1), dtype=torch.float32)
        out_dim = 1
        loss_fn = nn.MSELoss()
    model = nn.Sequential(nn.Linear(Xt.shape[1],128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,out_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = Xt.to(device), yt.to(device)
    for _ in range(int(epochs)):
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
    """Master-agent DL internal agent with hardware-aware execution and professional evaluation."""
    if df is None or df.empty:
        return {"status": "error", "message": "No data available."}
    rec = recommend_targets_and_features(df)
    target = target or rec.get("suggested_target") or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    compute = ComputeBackend.resolve(compute_mode, runtime_validate=True)
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target]); y = work[target]
    task = "classification" if (y.dtype == object or y.nunique() <= min(20, max(2, int(len(y) * .05)))) else "regression"
    if task == "classification": y = pd.Series(pd.factorize(y.astype(str))[0], index=y.index)
    validation_plan = ValidationProtocolAdvisor.infer(work, target)
    if validation_plan.strategy == "time" and validation_plan.time_column in work.columns:
        ordered = work.assign(__dsp_time=pd.to_datetime(work[validation_plan.time_column], errors="coerce")).sort_values("__dsp_time").drop(columns=["__dsp_time"])
        cut=max(1,min(len(ordered)-1,int(round(len(ordered)*.8))))
        train_df,test_df=ordered.iloc[:cut].copy(),ordered.iloc[cut:].copy()
    elif validation_plan.strategy == "group" and validation_plan.group_column in work.columns:
        train_df,test_df=ValidationProtocolAdvisor.split(work,target,validation_plan,test_size=.2,seed=seed)
    else:
        strat=y if task=="classification" and y.value_counts().min()>=2 else None
        train_df,test_df=train_test_split(work,test_size=.2,random_state=seed,stratify=strat)
    Xtr,ytr=train_df.drop(columns=[target]),train_df[target]
    Xte,yte=test_df.drop(columns=[target]),test_df[target]
    if task=="classification":
        classes = {v:i for i,v in enumerate(pd.Series(y.astype(str)).dropna().unique())}
        ytr = ytr.astype(str).map(classes).astype(int)
        yte = yte.astype(str).map(classes).astype(int)
    from services.model_preprocessing import build_preprocessor
    pre = build_preprocessor(Xtr)
    t0=time.time(); backend="scikit-learn MLP"; model_obj=None
    if compute.mode in {"GPU","CPU+GPU"} and compute.gpu_available:
        try:
            Xtr_t=pre.fit_transform(Xtr); Xte_t=pre.transform(Xte)
            # MX250-class devices have only 2 GB VRAM. Bound the dense tensor path.
            estimated_gb = (Xtr_t.shape[0] * Xtr_t.shape[1] * 4) / (1024**3)
            vram_budget = max(0.25, (compute.vram_free_gb or 1.0) * 0.55)
            if Xtr_t.shape[0] <= 50000 and Xtr_t.shape[1] <= 2500 and estimated_gb <= vram_budget:
                import torch
                model_obj, metrics = _torch_mlp(Xtr_t,Xte_t,ytr,yte,task,seed,"cuda")
                backend="PyTorch CUDA MLP"
            else:
                compute.diagnostics.append(f"CUDA tabular path skipped: estimated input tensor {estimated_gb:.2f} GB exceeds bounded budget {vram_budget:.2f} GB or dimensions are too large.")
                compute.message += " Large/low-VRAM workload routed to CPU MLP."
        except Exception as exc:
            compute.diagnostics.append(f"CUDA execution failed safely: {exc}")
            compute.message += " CUDA execution failed safely; using CPU MLP."
    if model_obj is None:
        estimator = make_mlp(task, seed)
        pipe=Pipeline([("pre",build_preprocessor(X)),("model",estimator)])
        model_obj, convergence = fit_with_convergence_audit(
            pipe,
            lambda: pipe.fit(Xtr, ytr),
            retry_factory=(lambda: Pipeline([("pre",build_preprocessor(X)),("model",make_mlp(task, seed, max_iter=3000))]))
        )
        pred=model_obj.predict(Xte)
        metrics_warning=convergence.get("warnings", [])
        if task=="classification": metrics={"accuracy":float(accuracy_score(yte,pred)),"f1_weighted":float(f1_score(yte,pred,average="weighted"))}; evaluation=ErrorAnalysisEngine.classification(yte,pred)
        else: metrics={"r2":float(r2_score(yte,pred)),"mae":float(mean_absolute_error(yte,pred)),"rmse":float(mean_squared_error(yte,pred)**.5)}; evaluation=ErrorAnalysisEngine.regression(yte,pred)
        metrics["error_analysis"]=evaluation
        metrics["convergence"]=convergence
    primary="f1_weighted" if task=="classification" else "r2"
    pred_final = model_obj.predict(Xte) if hasattr(model_obj, "predict") else (
        np.argmax(model_obj(torch.tensor(Xte.toarray() if hasattr(Xte,"toarray") else np.asarray(Xte),dtype=torch.float32).to("cuda" if compute.mode in {"GPU","CPU+GPU"} and compute.gpu_available else "cpu")).detach().cpu().numpy(),axis=1)
        if task=="classification" else model_obj(torch.tensor(Xte.toarray() if hasattr(Xte,"toarray") else np.asarray(Xte),dtype=torch.float32).to("cuda" if compute.mode in {"GPU","CPU+GPU"} and compute.gpu_available else "cpu")).detach().cpu().numpy().reshape(-1)
    )
    professional = ProfessionalEvaluationEngine.evaluate_holdout(yte, pred_final, task)
    baseline = ProfessionalEvaluationEngine.baseline(ytr, yte, task)
    if hasattr(model_obj, "predict"):
        try:
            professional["cross_validation"] = ProfessionalEvaluationEngine.cross_validation(model_obj, Xtr, ytr, task, seed=seed, n_jobs=1)
            if task == "classification":
                professional["calibration"] = ProfessionalEvaluationEngine.classification_calibration(model_obj, Xte, yte)
        except Exception as exc:
            professional["cross_validation"] = {"status":"error","reason":str(exc)}
    else:
        professional["cross_validation"] = {"status":"not_available_for_torch_backend","reason":"The bounded GPU MLP path uses a locked holdout plus seed stability and robustness diagnostics."}
        professional["calibration"] = {"status":"not_available_for_torch_backend"}

    # For the torch path, record probability-confidence uncertainty where possible.
    if task=="classification" and not hasattr(model_obj,"predict_proba"):
        try:
            import torch
            with torch.no_grad():
                logits=model_obj(torch.tensor(Xte.toarray() if hasattr(Xte,"toarray") else np.asarray(Xte),dtype=torch.float32).to("cuda" if compute.mode in {"GPU","CPU+GPU"} and compute.gpu_available else "cpu"))
                probs=torch.softmax(logits,dim=1).detach().cpu().numpy()
            confidence=np.max(probs,axis=1); entropy=-(probs*np.log(np.clip(probs,1e-12,1))).sum(axis=1)
            professional["prediction_uncertainty"]={"status":"ok","method":"softmax_confidence_entropy","mean_confidence":float(confidence.mean()),"p10_confidence":float(np.quantile(confidence,.10)),"p50_confidence":float(np.quantile(confidence,.50)),"mean_entropy":float(entropy.mean()),"interpretation":"Softmax confidence/entropy is a diagnostic and should not be treated as calibrated probability without calibration validation."}
        except Exception as exc:
            professional["prediction_uncertainty"]={"status":"error","reason":str(exc)}
    # Run the same bounded perturbation protocol for the neural backend.
    try:
        if hasattr(model_obj, "predict"):
            robustness = RobustnessPerturbationAnalyzer.run(model_obj, Xte, yte, task, seed=seed)
        else:
            import torch
            device = "cuda" if compute.mode in {"GPU", "CPU+GPU"} and compute.gpu_available else "cpu"
            def torch_predict(raw_df):
                """Perform the torch predict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
                arr = pre.transform(raw_df)
                tensor = torch.tensor(arr.toarray() if hasattr(arr,"toarray") else np.asarray(arr), dtype=torch.float32).to(device)
                with torch.no_grad(): out=model_obj(tensor).detach().cpu().numpy()
                return np.argmax(out,axis=1) if task=="classification" else out.reshape(-1)
            robustness = RobustnessPerturbationAnalyzer.run(model_obj, Xte, yte, task, seed=seed, predict_fn=torch_predict)
    except Exception as exc:
        robustness={"status":"error","reason":str(exc)}
    professional["robustness_perturbation"] = robustness
    seed_stability={"status":"not_run","reason":"Dataset/workload exceeds bounded DL seed-stability policy."}
    if len(Xtr)<=3000 and Xtr.shape[1]<=500:
        try:
            if backend=="PyTorch CUDA MLP" or (compute.mode=="CPU" and model_obj is not None and not hasattr(model_obj,"predict")):
                device = "cuda" if backend=="PyTorch CUDA MLP" else "cpu"
                Xtr_seed=pre.transform(Xtr); Xte_seed=pre.transform(Xte)
                def _seed_run(rs):
                    """Perform the seed run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
                    _m,_metrics=_torch_mlp(Xtr_seed,Xte_seed,ytr,yte,task,rs,device,epochs=60)
                    return {"metrics":_metrics}
                seed_stability=DLValidationEngine.seed_stability(_seed_run,[seed,seed+1],task)
            elif backend=="scikit-learn MLP":
                from sklearn.neural_network import MLPClassifier as _MC, MLPRegressor as _MR
                from sklearn.pipeline import Pipeline as _Pipe
                def _seed_run(rs):
                    """Perform the seed run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
                    est=make_mlp(task, rs, max_iter=1500)
                    pp=_Pipe([("pre",build_preprocessor(Xtr)),("model",est)])
                    pp.fit(Xtr,ytr); pr=pp.predict(Xte)
                    mm={"accuracy":float(accuracy_score(yte,pr)),"f1_weighted":float(f1_score(yte,pr,average="weighted"))} if task=="classification" else {"r2":float(r2_score(yte,pr)),"mae":float(mean_absolute_error(yte,pr)),"rmse":float(mean_squared_error(yte,pr)**.5)}
                    return {"metrics":mm}
                seed_stability=DLValidationEngine.seed_stability(_seed_run,[seed,seed+1],task)
        except Exception as exc:
            seed_stability={"status":"error","reason":str(exc)}
    professional["seed_stability"]=seed_stability
    try:
        professional["data_slices"] = DataSliceAnalyzer.run(Xte, yte, pred_final, task)
    except Exception:
        professional["data_slices"] = {"status":"error"}
    temporal = TemporalAvailabilityAnalyzer.run(df, target=target)
    provenance = FeatureProvenanceEngine.build(df, target=target, approved_features=list(Xtr.columns))
    baseline_eval = {"cross_validation":{"status":"not_available_for_neural_backend","reason":"The bounded DL path uses held-out evaluation plus seed/robustness diagnostics; a full neural CV can be enabled for larger compute budgets."}, "calibration":{"status":"not_available"}, "baseline":baseline, "prediction_uncertainty":professional.get("prediction_uncertainty",{}), "robustness_perturbation":robustness, "data_slices":professional.get("data_slices")}
    diagnosis = ModelDiagnosisEngine.diagnose(task, baseline_eval, None, temporal, None)
    validation_audit=ValidationProtocolAdvisor.audit(train_df,test_df,target=target,group_column=validation_plan.group_column)
    resource=ResourcePolicy.plan(len(Xtr),Xtr.shape[1],compute_mode=compute.mode,vram_gb=compute.vram_total_gb)
    return {"status":"ok","agent":"DL","target":target,"task":task,"backend":backend,
            "models": {"MLP": {"metrics":metrics,"seconds":round(time.time()-t0,3),"professional_evaluation":professional}},
            "best_model":"MLP","best_metrics":metrics,"model_object":model_obj,"compute":compute.to_dict(),
            "evaluation":{"level":"professional","best_model":professional,"baseline":baseline,"cross_validation":{"status":"not_available_for_neural_backend"},"prediction_uncertainty":professional.get("prediction_uncertainty"),"robustness_perturbation":professional.get("robustness_perturbation"),"data_slices":professional.get("data_slices"),"seed_stability":seed_stability,"test_set_locked":True},
            "validation_protocol":validation_plan.to_dict(),"validation_audit":validation_audit,"feature_provenance":provenance,"resource_policy":resource,
            "temporal_availability":temporal,"model_diagnosis":diagnosis,
            "target_analysis":rec,"memory":AgentMemory("DL",20).to_dict(),
            "summary":f"DL trained an MLP using the {backend} backend and evaluated held-out performance against a baseline with diagnostic evidence. {compute.message}"}
