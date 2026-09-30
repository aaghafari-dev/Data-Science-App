from __future__ import annotations

from typing import Any
import os


class ResourcePolicy:
    """CPU/GPU/parallelism policy that prevents uncontrolled oversubscription."""
    @staticmethod
    def plan(rows: int, features: int, compute_mode: str = "CPU", vram_gb: float | None = None, requested_jobs: int | None = None) -> dict[str, Any]:
        cpu = os.cpu_count() or 2
        jobs = requested_jobs or min(max(cpu - 1, 1), 8)
        if rows * max(features, 1) > 5_000_000:
            jobs = min(jobs, 4)
        if compute_mode == "CPU+GPU":
            jobs = min(jobs, 4)
        if compute_mode == "GPU":
            jobs = 1
        low_vram = vram_gb is not None and vram_gb <= 2.25
        if low_vram:
            jobs = min(jobs, 2)
        return {"cpu_count": cpu, "parallel_jobs": max(1, jobs), "compute_mode": compute_mode, "low_vram": low_vram, "reason": "Bounded parallelism prevents CPU oversubscription and protects low-VRAM systems."}
