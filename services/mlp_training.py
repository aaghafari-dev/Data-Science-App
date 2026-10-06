"""Centralized, auditable scikit-learn MLP training policy.

This module is the single source of truth for tabular MLP configuration used by
Supervised Analysis, the ML Agent and the DL Agent.  Convergence warnings are
captured as structured evidence; they are never globally suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor


MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER = 1500
MLP_RETRY_MAX_ITER = 3000
MLP_N_ITER_NO_CHANGE = 30
MLP_TOL = 1e-4
MLP_VALIDATION_FRACTION = 0.15


@dataclass(frozen=True)
class MLPPolicy:
    """Explicit, reproducible training policy."""
    hidden_layer_sizes: tuple[int, ...] = MLP_HIDDEN_LAYERS
    max_iter: int = MLP_MAX_ITER
    retry_max_iter: int = MLP_RETRY_MAX_ITER
    n_iter_no_change: int = MLP_N_ITER_NO_CHANGE
    tol: float = MLP_TOL
    validation_fraction: float = MLP_VALIDATION_FRACTION
    early_stopping: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_mlp(task: str, seed: int, *, max_iter: int | None = None,
             hidden_layer_sizes: tuple[int, ...] | None = None,
             alpha: float | None = None):
    """Create the canonical MLP estimator for classification or regression."""
    policy = MLPPolicy()
    kwargs = {
        "hidden_layer_sizes": hidden_layer_sizes or policy.hidden_layer_sizes,
        "max_iter": max_iter or policy.max_iter,
        "early_stopping": policy.early_stopping,
        "n_iter_no_change": policy.n_iter_no_change,
        "tol": policy.tol,
        "validation_fraction": policy.validation_fraction,
        "random_state": seed,
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    if task == "classification":
        return MLPClassifier(**kwargs)
    if task == "regression":
        return MLPRegressor(**kwargs)
    raise ValueError(f"Unsupported MLP task: {task}")


def _audit(estimator: Any, warning_messages: list[str], *, retried: bool = False) -> dict[str, Any]:
    """Build structured convergence evidence from a fitted estimator or Pipeline."""
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model", estimator)
    n_iter = getattr(model, "n_iter_", None)
    max_iter = getattr(model, "max_iter", None)
    if n_iter is None or max_iter is None:
        return {
            "status": "not_applicable",
            "converged": None,
            "n_iter": None,
            "max_iter": None,
            "warning_count": len(warning_messages),
            "warnings": warning_messages[:5],
            "retried": bool(retried),
            "policy": MLPPolicy().to_dict(),
        }
    converged = not bool(warning_messages)
    if n_iter < max_iter:
        converged = True
    status = "converged" if converged else "max_iter_reached"
    return {
        "status": status,
        "converged": bool(converged),
        "n_iter": int(n_iter) if n_iter is not None else None,
        "max_iter": int(max_iter) if max_iter is not None else None,
        "tol": float(getattr(model, "tol", MLP_TOL)),
        "n_iter_no_change": int(getattr(model, "n_iter_no_change", MLP_N_ITER_NO_CHANGE)),
        "validation_fraction": float(getattr(model, "validation_fraction", MLP_VALIDATION_FRACTION)),
        "final_loss": float(getattr(model, "loss_", float("nan"))) if getattr(model, "loss_", None) is not None else None,
        "warning_count": len(warning_messages),
        "warnings": warning_messages[:5],
        "retried": bool(retried),
        "policy": MLPPolicy().to_dict(),
    }


def fit_with_convergence_audit(estimator: Any, fit_callable: Callable[[], Any],
                               *, retry_factory: Callable[[], Any] | None = None):
    """Fit once, capture convergence warnings, optionally retry once.

    The retry is only performed when a caller explicitly supplies a retry
    factory.  This avoids silently doubling expensive GridSearchCV workloads.
    """
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = fit_callable()
    warning_messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    retried = False

    if warning_messages and retry_factory is not None:
        retry_estimator = retry_factory()
        with warnings.catch_warnings(record=True) as retry_caught:
            warnings.simplefilter("always")
            fitted = retry_estimator.fit()
        warning_messages = [str(w.message) for w in retry_caught if issubclass(w.category, ConvergenceWarning)]
        retried = True
        estimator = retry_estimator
    else:
        estimator = fitted

    return fitted, _audit(estimator, warning_messages, retried=retried)


def fit_search_with_convergence_audit(searcher: Any, X, y, *, model_step: str = "model") -> dict[str, Any]:
    """Run Grid/RandomizedSearchCV without console warning spam and audit its best MLP.

    Convergence warnings from individual CV fits are counted and preserved in
    the returned evidence.  The search itself is not retried because doing so
    would multiply the complete hyperparameter-search cost.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        searcher.fit(X, y)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    best = searcher.best_estimator_.named_steps.get(model_step, searcher.best_estimator_)
    audit = _audit(best, convergence)
    audit["cv_warning_count"] = len(convergence)
    audit["cv_warning_samples"] = convergence[:5]
    return audit
