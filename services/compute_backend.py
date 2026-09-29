from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ComputeInfo:
    mode: str
    gpu_available: bool
    gpu_name: str | None
    cuda_version: str | None
    message: str

class ComputeBackend:
    MODES=("CPU","CPU+GPU","GPU")
    @staticmethod
    def detect() -> ComputeInfo:
        try:
            import torch
            ok=bool(torch.cuda.is_available())
            name=torch.cuda.get_device_name(0) if ok else None
            return ComputeInfo("CPU",ok,name,getattr(torch.version,"cuda",None),"CUDA GPU detected." if ok else "GPU not found; CPU is available.")
        except Exception as exc:
            return ComputeInfo("CPU",False,None,None,f"GPU detection unavailable; CPU is available ({exc}).")
    @classmethod
    def resolve(cls, requested: str) -> ComputeInfo:
        info=cls.detect(); requested=requested if requested in cls.MODES else "CPU"
        if requested in {"GPU","CPU+GPU"} and not info.gpu_available:
            return ComputeInfo("CPU",False,None,info.cuda_version,f"GPU mode requested, but no CUDA GPU was found. Falling back to CPU.")
        return ComputeInfo(requested,info.gpu_available,info.gpu_name,info.cuda_version,f"Compute mode: {requested}. GPU: {info.gpu_name or 'not available'}.")
