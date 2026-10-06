"""Persistent, secret-free Agent memory and LangGraph run trace store.

The store is deliberately small and append-only. It records analytical context, approvals,
run identifiers and evidence references, never API keys or raw credentials.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import uuid

class AgentMemoryStore:
    def __init__(self, path: str | None = None, max_records: int = 5000):
        default = Path(tempfile.gettempdir()) / "DataScienceStudioPro" / "agent_memory.jsonl"
        self.path = Path(path or os.environ.get("DSP_AGENT_MEMORY_PATH", str(default)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records

    def record(self, run_id: str, role: str, kind: str, content: Any, *, evidence_id: str | None = None) -> str:
        ref = uuid.uuid4().hex
        payload = {"memory_id": ref, "time": datetime.now(timezone.utc).isoformat(), "run_id": run_id or "unbound", "role": role, "kind": kind, "content": content, "evidence_id": evidence_id}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._trim()
        return ref

    def recall(self, *, run_id: str | None = None, role: str | None = None, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows=[]
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(line))
            except Exception: continue
        if run_id: rows=[x for x in rows if x.get("run_id")==run_id]
        if role: rows=[x for x in rows if x.get("role")==role]
        if kind: rows=[x for x in rows if x.get("kind")==kind]
        return rows[-max(1, int(limit)):]

    def _trim(self):
        if not self.path.exists(): return
        lines=self.path.read_text(encoding="utf-8").splitlines()
        if len(lines)>self.max_records:
            self.path.write_text("\n".join(lines[-self.max_records:])+"\n",encoding="utf-8")
