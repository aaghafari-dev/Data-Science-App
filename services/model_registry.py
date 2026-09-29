from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import joblib


class ModelRegistry:
    """Small local model registry with versioned artifacts and model metadata."""
    def __init__(self, root: str = "model_registry"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        self.records = json.loads(self.index.read_text(encoding="utf-8")) if self.index.exists() else []

    def register(self, name: str, model: Any, metrics: dict[str, Any], tags: dict[str, Any] | None = None) -> dict:
        version = 1 + max([int(r.get("version", 0)) for r in self.records if r.get("name") == name] or [0])
        path = self.root / f"{name.replace(' ', '_')}_v{version}.joblib"
        joblib.dump(model, path)
        rec = {"name": name, "version": version, "artifact": str(path), "metrics": metrics,
               "tags": tags or {}, "created_at": datetime.now(timezone.utc).isoformat(), "stage": "candidate"}
        self.records.append(rec); self.index.write_text(json.dumps(self.records, indent=2, default=str), encoding="utf-8")
        return rec
