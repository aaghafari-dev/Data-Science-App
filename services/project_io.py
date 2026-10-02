"""Module duty: Project io.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

"""Safe, portable JSON project manifest utilities.

This module deliberately does not deserialize arbitrary Python pickle objects.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

APP_VERSION = "20.1"


def _jsonable(value: Any) -> Any:
    """Perform the jsonable operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_manifest(*, dataset_fingerprint: str | None, source: str | None,
                   rows: int, columns: int, view_state: dict[str, Any],
                   filters: list[Any], evidence: list[Any]) -> dict[str, Any]:
    """Perform the build manifest operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return {
        "format": "dssp-project-manifest",
        "format_version": 1,
        "app_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"fingerprint": dataset_fingerprint, "source": source, "rows": rows, "columns": columns},
        "view_state": _jsonable(view_state),
        "filters": _jsonable(filters),
        "evidence": _jsonable(evidence),
    }


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Perform the save manifest operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    Path(path).write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Perform the load manifest operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "dssp-project-manifest":
        raise ValueError("This file is not a valid Data Science Studio Pro manifest.")
    return data
