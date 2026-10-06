"""Module duty: Graph ml.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations
from typing import Any
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from services.mlp_training import make_mlp, fit_search_with_convergence_audit
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from services.evaluation_protocol import ValidationProtocolAdvisor
from services.resource_policy import ResourcePolicy
from services.feature_provenance import FeatureProvenanceEngine
from services.feature_set_analysis import FeatureSetChallenge
from services.error_analysis import ErrorAnalysisEngine
from services.target_feature_selection import recommend_targets_and_features
from services.agent_memory import AgentMemory
from services.agent_quality import ProfessionalEvaluationEngine, FeatureStabilityAnalyzer, TemporalAvailabilityAnalyzer, CounterfactualLeakageAnalyzer, ModelDiagnosisEngine
from services.model_preprocessing import build_preprocessor


def _split(work,target,task,plan,seed):
    """Perform the split operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if plan.strategy=="time" and plan.time_column in work.columns:
        ordered=work.assign(__dsp_time=pd.to_datetime(work[plan.time_column],errors="coerce")).sort_values("__dsp_time").drop(columns=["__dsp_time"])
        cut=max(1,min(len(ordered)-1,int(round(len(ordered)*.8)))); return ordered.iloc[:cut].copy(),ordered.iloc[cut:].copy()
    if plan.strategy=="group" and plan.group_column in work.columns:
        return ValidationProtocolAdvisor.split(work,target,plan,test_size=.2,seed=seed)
    y=work[target]; strat=y if task=="classification" and y.value_counts().min()>=2 else None
    return train_test_split(work,test_size=.2,random_state=seed,stratify=strat)


def _task(y):
    """Perform the task operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return "classification" if (y.dtype==object or str(y.dtype).startswith("category") or y.nunique()<=min(20,max(2,int(len(y)*.05)))) else "regression"


def _candidates(task,seed,parallel_jobs):
    """Perform the candidates operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if task=="classification":
        models={
            "Logistic Regression":(LogisticRegression(max_iter=3000),{"model__C":[.1,1,10]}),
            "SVM":(SVC(probability=True),{"model__C":[.5,1,10],"model__kernel":["rbf","linear"]}),
            "Random Forest":(RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=parallel_jobs),{"model__max_depth":[None,8,16],"model__min_samples_leaf":[1,3]}),
            "Decision Tree":(DecisionTreeClassifier(random_state=seed),{"model__max_depth":[None,4,8,16]}),
            "KNN":(KNeighborsClassifier(),{"model__n_neighbors":[3,5,9],"model__weights":["uniform","distance"]}),
            "Naive Bayes":(GaussianNB(),{"model__var_smoothing":[1e-9,1e-8,1e-7]}),
            "Neural Network":(make_mlp("classification", seed),{"model__alpha":[1e-5,1e-4,1e-3]}),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"]=(XGBClassifier(n_estimators=200,eval_metric="logloss",random_state=seed,n_jobs=parallel_jobs),{"model__max_depth":[3,6],"model__learning_rate":[.03,.1]})
        except Exception: pass
        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"]=(LGBMClassifier(n_estimators=200,random_state=seed,verbosity=-1,n_jobs=parallel_jobs),{"model__num_leaves":[15,31],"model__learning_rate":[.03,.1]})
        except Exception: pass
        return models
    models={
        "OLS":(LinearRegression(),{}),
        "Ridge":(Ridge(),{"model__alpha":[.01,.1,1,10]}),
        "Lasso":(Lasso(max_iter=20000,tol=1e-4),{"model__alpha":[.0001,.001,.01,.1]}),
        "Elastic Net":(ElasticNet(max_iter=20000,tol=1e-4),{"model__alpha":[.001,.01,.1],"model__l1_ratio":[.2,.5,.8]}),
        "Polynomial Regression":(Pipeline([("poly",PolynomialFeatures(degree=2,include_bias=False)),("linear",Ridge())]),{"model__linear__alpha":[.1,1,10]}),
        "SVR":(SVR(),{"model__C":[.5,1,10],"model__epsilon":[.05,.1,.2]}),
        "Decision Tree":(DecisionTreeRegressor(random_state=seed),{"model__max_depth":[None,5,10,20]}),
        "Random Forest":(RandomForestRegressor(n_estimators=250,random_state=seed,n_jobs=parallel_jobs),{"model__max_depth":[None,10,20]}),
        "Gradient Boosting":(GradientBoostingRegressor(random_state=seed),{"model__n_estimators":[100,200],"model__learning_rate":[.03,.1],"model__max_depth":[2,3]}),
        "Neural Network":(make_mlp("regression", seed),{"model__alpha":[1e-5,1e-4,1e-3]}),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"]=(XGBRegressor(n_estimators=200,random_state=seed,n_jobs=parallel_jobs,objective="reg:squarederror"),{"model__max_depth":[3,6],"model__learning_rate":[.03,.1]})
    except Exception: pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"]=(LGBMRegressor(n_estimators=200,random_state=seed,verbosity=-1,n_jobs=parallel_jobs),{"model__num_leaves":[15,31],"model__learning_rate":[.03,.1]})
    except Exception: pass
    return models


def _safe_metrics(y,pred,task,proba=None):
    """Perform the safe metrics operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if task=="classification":
        out={"accuracy":float(accuracy_score(y,pred)),"f1_weighted":float(f1_score(y,pred,average="weighted",zero_division=0))}
        if proba is not None:
            try:
                from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
                if proba.shape[1]==2:
                    out["roc_auc"]=float(roc_auc_score(y,proba[:,1])); out["pr_auc"]=float(average_precision_score(y,proba[:,1]))
                out["log_loss"]=float(log_loss(y,proba))
            except Exception: pass
        return out
    return {"r2":float(r2_score(y,pred)),"mae":float(mean_absolute_error(y,pred)),"rmse":float(np.sqrt(mean_squared_error(y,pred)))}


def _suspicious_features(rec):
    """Perform the suspicious features operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    out=[]
    for item in rec.get("feature_candidates",[]):
        if isinstance(item,dict):
            name=item.get("column"); reasons=" ".join(item.get("reasons",[])).lower()
            if name and any(k in reasons for k in ("future","near-perfect","leakage","power-")): out.append(str(name))
    return out[:10]


def run_ml_step(df: pd.DataFrame, target: str|None=None, seed:int=42, compute_mode:str="CPU") -> dict[str,Any]:
    """Perform the run ml step operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if df is None or df.empty: return {"status":"error","message":"No data available."}
    rec=recommend_targets_and_features(df); target=target or rec.get("suggested_target") or df.columns[-1]
    if target not in df.columns: raise ValueError(f"Target column not found: {target}")
    work=df.dropna(subset=[target]).copy(); task=_task(work[target]);
    if task=="classification": work[target]=work[target].astype(str)
    plan=ValidationProtocolAdvisor.infer(work,target); train_df,test_df=_split(work,target,task,plan,seed)
    Xtr,ytr=train_df.drop(columns=[target]),train_df[target]; Xte,yte=test_df.drop(columns=[target]),test_df[target]
    resource=ResourcePolicy.plan(len(Xtr),Xtr.shape[1],compute_mode=compute_mode)
    models=_candidates(task,seed,resource["parallel_jobs"])
    # Naive Bayes requires dense transformed data. Wrap only this estimator.
    results={}; fitted={}
    for name,(estimator,params) in models.items():
        model_est=estimator
        if name=="Naive Bayes":
            model_est=Pipeline([("dense",FunctionTransformer(lambda x: x.toarray() if hasattr(x,"toarray") else x,accept_sparse=True)),("nb",GaussianNB())])
            params={"model__nb__var_smoothing":[1e-9,1e-8,1e-7]}
        pipe=Pipeline([("pre",build_preprocessor(Xtr)),("model",model_est)])
        try:
            t0=time.time(); cv_folds=3 if task=="regression" else min(3,int(ytr.value_counts().min()))
            if params:
                search=RandomizedSearchCV(pipe,params,n_iter=min(3,max(1,len(next(iter(params.values()))))),scoring="f1_weighted" if task=="classification" else "neg_root_mean_squared_error",cv=cv_folds,n_jobs=1,refit=True,random_state=seed,return_train_score=True)
            else:
                search=GridSearchCV(pipe,params,scoring="f1_weighted" if task=="classification" else "neg_root_mean_squared_error",cv=cv_folds,n_jobs=1,refit=True,return_train_score=True)
            convergence=fit_search_with_convergence_audit(search, Xtr, ytr)
            pred=search.predict(Xte); proba=search.predict_proba(Xte) if hasattr(search,"predict_proba") else None
            metrics=_safe_metrics(yte,pred,task,proba); prof=ProfessionalEvaluationEngine.evaluate_bundle(search.best_estimator_,Xtr,Xte,ytr,yte,task,seed,n_jobs=1)
            results[name]={"status":"ok","metrics":metrics,"cv_selection_score":float(search.best_score_),"best_params":search.best_params_,"seconds":round(time.time()-t0,3),"professional_evaluation":prof,"convergence":convergence,"convergence_warnings":convergence.get("warnings",[]),"converged":bool(convergence.get("converged",False))}
            fitted[name]=search.best_estimator_
        except Exception as exc:
            results[name]={"status":"unavailable","error":str(exc),"metrics":{}}
    valid=[(n,v.get("cv_selection_score")) for n,v in results.items() if v.get("status")=="ok" and v.get("cv_selection_score") is not None]
    if not valid: raise RuntimeError("No candidate model completed successfully. Review preprocessing, class distribution and compute resources.")
    best=max(valid,key=lambda x:x[1])[0] if task=="classification" else max(valid,key=lambda x:x[1])[0]
    best_pipe=fitted[best]
    # Advanced diagnostics are calculated from the selected training/test protocol.
    def factory(cols):
        """Perform the factory operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return Pipeline([("pre",build_preprocessor(Xtr[list(cols)])),("model",clone(best_pipe.named_steps["model"]))])
    stability=FeatureStabilityAnalyzer.run(lambda: clone(best_pipe), Xtr,ytr,task,repeats=3,max_features=25,seed=seed)
    temporal=TemporalAvailabilityAnalyzer.run(df,target=target); suspicious=_suspicious_features(rec)
    leakage=CounterfactualLeakageAnalyzer.run(lambda cols: Pipeline([("pre",build_preprocessor(Xtr[list(cols)])),("model",clone(best_pipe.named_steps["model"]))]),Xtr,ytr,task,suspicious,seed) if suspicious else {"status":"not_available","reason":"No high-priority suspicious features were identified."}
    feature_sets=FeatureSetChallenge.run(lambda cols: Pipeline([("pre",build_preprocessor(Xtr[list(cols)])),("model",clone(best_pipe.named_steps["model"]))]),Xtr,Xte,ytr,yte,task,stable_features=[],suspicious_features=suspicious)
    provenance=FeatureProvenanceEngine.build(df,target=target,approved_features=list(Xtr.columns))
    diagnosis=ModelDiagnosisEngine.diagnose(task,results[best]["professional_evaluation"],stability,temporal,leakage)
    primary="f1_weighted" if task=="classification" else "r2"
    return {"status":"ok","agent":"ML","target":target,"task":task,"models":results,"best_model":best,"best_metrics":results[best]["metrics"],"model_object":best_pipe,
            "evaluation":{"level":"professional","best_model":results[best]["professional_evaluation"],"baseline":results[best]["professional_evaluation"].get("baseline"),"cross_validation":results[best]["professional_evaluation"].get("cross_validation"),"prediction_uncertainty":results[best]["professional_evaluation"].get("prediction_uncertainty"),"robustness_perturbation":results[best]["professional_evaluation"].get("robustness_perturbation"),"calibration":results[best]["professional_evaluation"].get("calibration"),"data_slices":results[best]["professional_evaluation"].get("data_slices"),"test_set_locked":True},
            "feature_stability":stability,"feature_set_challenge":feature_sets,"temporal_availability":temporal,"counterfactual_leakage":leakage,"model_diagnosis":diagnosis,"feature_provenance":provenance,
            "validation_protocol":plan.to_dict(),"validation_audit":ValidationProtocolAdvisor.audit(train_df,test_df,target=target,group_column=plan.group_column),"resource_policy":resource,
            "selection_policy":{"basis":"training-only cross-validation with hyperparameter search","primary_metric":primary,"test_set_locked":True,"methods_included":list(models.keys())},
            "summary":f"ML evaluated {len([x for x in results.values() if x.get('status')=='ok'])} candidate families ({', '.join(models.keys())}) with fold-local preprocessing, hyperparameter search, locked-test evaluation, calibration/uncertainty and diagnostics. Selected {best} using training-only cross-validation.","target_analysis":rec,"memory":AgentMemory("ML",20).to_dict()}
