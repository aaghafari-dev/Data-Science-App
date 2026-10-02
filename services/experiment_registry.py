"""Module duty: Experiment registry.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    """MLflow-compatible local experiment ledger; optionally mirrors to MLflow."""
    def __init__(self, root: str = "runs"):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        self.run_dir = self.root / self.run_id; self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.run_dir / "manifest.jsonl"

    def log(self, event: str, **payload: Any):
        """Perform the log operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with self.events.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def start(self, dataset_hash: str | None = None, task: str | None = None):
        """Perform the start operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.log("run_started", dataset_hash=dataset_hash, task=task,
                 python=platform.python_version(), platform=platform.platform())

    def log_metrics(self, model: str, metrics: dict[str, Any], params: dict[str, Any] | None = None):
        """Perform the log metrics operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.log("model_result", model=model, metrics=metrics, params=params or {})
        try:
            import mlflow
            mlflow.log_params(params or {})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception:
            pass

    def finish(self, **payload):
        """Perform the finish operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.log("run_finished", **payload)
