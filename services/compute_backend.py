"""Module duty: Compute backend.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import re
import subprocess
import sys
from typing import Any


@dataclass
class ComputeInfo:
    mode: str
    requested_mode: str
    gpu_available: bool
    hardware_gpu_detected: bool
    gpu_name: str | None
    vram_total_gb: float | None
    vram_free_gb: float | None
    driver_version: str | None
    cuda_version: str | None
    torch_version: str | None
    compute_capability: str | None
    torch_cuda_built: bool
    runtime_validated: bool
    supported_architecture: bool | None
    message: str
    diagnostics: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Perform the to dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return asdict(self)


class ComputeBackend:
    """Professional hardware/runtime detection and compute-policy resolver.

    Detection deliberately separates three facts:
      1. NVIDIA hardware exists;
      2. the installed PyTorch build contains CUDA support;
      3. the CUDA runtime can actually execute a tiny operation on that device.

    This prevents a CPU-only/unsupported PyTorch build from being reported as
    "GPU not found" when Windows can see the NVIDIA adapter.
    """

    MODES = ("CPU", "CPU+GPU", "GPU")
    _ARCH_HINTS = {"MX250": "6.1", "MX230": "6.1", "MX150": "6.1"}

    @staticmethod
    def _nvidia_smi() -> dict[str, Any]:
        """Perform the nvidia smi operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            candidates=["nvidia-smi"]
            if os.name=="nt": candidates.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
            proc=None; last_error="nvidia-smi not found"
            for executable in candidates:
                try:
                    proc=subprocess.run([executable,"--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5,creationflags=creationflags)
                    if proc.returncode==0 and proc.stdout.strip(): break
                    last_error=(proc.stderr or "nvidia-smi returned no GPU information").strip()
                except Exception as exc: last_error=str(exc)
            if proc is None or proc.returncode != 0 or not proc.stdout.strip():
                return {"available": False, "error": last_error}
            first = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) < 4:
                return {"available": False, "error": "Unexpected nvidia-smi output."}
            return {"available": True, "name": parts[0], "vram_total_gb": float(parts[1]) / 1024, "vram_free_gb": float(parts[2]) / 1024, "driver_version": parts[3]}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    @classmethod
    def detect(cls, runtime_validate: bool = False) -> ComputeInfo:
        """Perform the detect operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        smi = cls._nvidia_smi()
        hardware = bool(smi.get("available"))
        name = smi.get("name")
        diagnostics: list[str] = []
        if smi.get("error"):
            diagnostics.append(f"nvidia-smi: {smi['error']}")
        torch_version = cuda_version = None
        torch_cuda_built = False
        cuda_available = False
        supported_arch = None
        cc = None
        runtime_validated = False
        try:
            import torch
            torch_version = getattr(torch, "__version__", None)
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            torch_cuda_built = bool(cuda_version)
            cuda_available = bool(torch.cuda.is_available()) if torch_cuda_built else False
            if cuda_available:
                idx = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(idx)
                name = name or device_name
                capability = torch.cuda.get_device_capability(idx)
                cc = f"{capability[0]}.{capability[1]}"
                arch_list = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
                supported_arch = (f"sm_{capability[0]}{capability[1]}" in arch_list) if arch_list else None
                if supported_arch is False:
                    diagnostics.append(f"PyTorch CUDA build does not list sm_{capability[0]}{capability[1]} in torch.cuda.get_arch_list().")
                if runtime_validate:
                    runtime_validated = cls._cuda_smoke_test()
                    if not runtime_validated:
                        diagnostics.append("CUDA runtime smoke test failed in an isolated subprocess.")
        except Exception as exc:
            diagnostics.append(f"PyTorch CUDA inspection failed: {exc}")
        if name and not cc:
            for hint, hint_cc in cls._ARCH_HINTS.items():
                if hint.lower() in str(name).lower():
                    cc = hint_cc
                    diagnostics.append(f"Known device hint: {name} → compute capability {hint_cc} (hardware hint; PyTorch runtime capability should be verified separately).")
                    break
        usable = hardware and torch_cuda_built and cuda_available and (supported_arch is not False) and (runtime_validated if runtime_validate else True)
        if hardware and not torch_cuda_built:
            diagnostics.append("NVIDIA hardware is present, but the installed PyTorch package is CPU-only (torch.version.cuda is empty).")
        elif hardware and torch_cuda_built and not cuda_available:
            diagnostics.append("NVIDIA hardware is present, but PyTorch cannot initialize CUDA in this Python environment.")
        if usable:
            message = f"CUDA GPU ready: {name or 'NVIDIA GPU'}"
        elif hardware:
            message = f"NVIDIA GPU detected ({name or 'unknown'}), but CUDA execution is not currently usable by PyTorch."
        else:
            message = "No NVIDIA GPU was detected through nvidia-smi; CPU mode is available."
        return ComputeInfo(
            mode="CPU", requested_mode="CPU", gpu_available=usable, hardware_gpu_detected=hardware,
            gpu_name=name, vram_total_gb=smi.get("vram_total_gb"), vram_free_gb=smi.get("vram_free_gb"),
            driver_version=smi.get("driver_version"), cuda_version=cuda_version, torch_version=torch_version,
            compute_capability=cc, torch_cuda_built=torch_cuda_built, runtime_validated=runtime_validated,
            supported_architecture=supported_arch, message=message, diagnostics=diagnostics,
        )

    @staticmethod
    def _cuda_smoke_test() -> bool:
        """Validate CUDA in a separate Python process so a bad CUDA context cannot poison the GUI."""
        code = "import torch; x=torch.ones(1,device='cuda'); y=x+1; torch.cuda.synchronize(); assert float(y.item())==2.0"
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=12, creationflags=creationflags)
            return proc.returncode == 0
        except Exception:
            return False

    @classmethod
    def resolve(cls, requested: str, runtime_validate: bool = True) -> ComputeInfo:
        """Perform the resolve operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        requested = requested if requested in cls.MODES else "CPU"
        info = cls.detect(runtime_validate=runtime_validate and requested in {"GPU", "CPU+GPU"})
        info.requested_mode = requested
        if requested == "CPU":
            info.mode = "CPU"
            info.message = "CPU mode selected explicitly; GPU execution is disabled for this run."
            return info
        if not info.gpu_available:
            # CPU+GPU is an opportunistic policy; GPU is a strict request but the
            # application remains safe by refusing CUDA rather than crashing.
            info.mode = "CPU"
            detail = " Hardware is visible but PyTorch CUDA is not usable." if info.hardware_gpu_detected else " No CUDA-capable GPU is usable."
            info.message = f"{requested} requested; using CPU safely.{detail}"
            return info
        # 2 GB class devices should be treated as a bounded accelerator, not a
        # general-purpose large-model backend.
        if info.vram_total_gb is not None and info.vram_total_gb <= 2.25:
            info.diagnostics.append("Low-VRAM accelerator policy: keep CUDA workloads small and use CPU for large models/data transforms.")
            if info.gpu_name and "MX250" in info.gpu_name.upper():
                info.diagnostics.append("MX250 profile detected: 2 GB-class VRAM is not treated as a safe local-LLM GPU backend.")
        info.mode = requested
        info.message = f"Compute mode: {requested}; GPU runtime validated on {info.gpu_name or 'NVIDIA GPU'}."
        return info
