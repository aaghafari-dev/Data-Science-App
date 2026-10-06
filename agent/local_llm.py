"""Module duty: Local llm.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, GenerationConfig
from langchain_huggingface import HuggingFacePipeline
import config


class LocalLLMLoader:
    """Resource-aware local LLM loader.

    A 2 GB-class GPU such as an MX250 is treated as an accelerator for small
    tensor workloads, not as a safe device for multi-billion-parameter LLMs.
    Large models are therefore routed to CPU or rejected before a large memory
    allocation can crash the application.
    """
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.loaded_model = None
        self.loaded_model_name = None

    @staticmethod
    def _system_memory_gb() -> float | None:
        """Perform the system memory gb operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            import psutil
            return float(psutil.virtual_memory().available) / (1024 ** 3)
        except Exception:
            return None

    @staticmethod
    def _model_size_hint(model_name: str) -> float:
        """Return a coarse parameter-count hint in billions when no model files are available."""
        n = model_name.lower()
        if "7b" in n: return 7.0
        if "3b" in n: return 3.0
        if "1.5b" in n: return 1.5
        if "0.5b" in n: return 0.5
        if "medium" in n: return 0.35
        if "small" in n: return 0.12
        if "base" in n: return 0.25
        if "gpt2" in n: return 0.12
        return 1.0

    @staticmethod
    def _model_memory_estimate(model_path: str, model_name: str) -> tuple[float, str, float]:
        """Estimate resident weight memory from actual local files/configuration when possible.

        The previous implementation multiplied a name-based parameter hint by 4 bytes and a
        35%% overhead factor. That is deliberately conservative but can be misleading for
        models distributed in BF16/FP16 or quantized weights. This estimator first measures
        local weight files and then adds a bounded runtime allowance.
        """
        path = Path(model_path)
        weight_bytes = 0
        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            for item in path.rglob(pattern):
                try:
                    weight_bytes += item.stat().st_size
                except OSError:
                    pass
        dtype = "float32"
        bytes_per_weight = 4.0
        config_files = list(path.rglob("config.json")) if path.exists() else []
        if config_files:
            try:
                cfg = json.loads(config_files[0].read_text(encoding="utf-8"))
                raw = str(cfg.get("torch_dtype") or cfg.get("dtype") or "").lower()
                if "bfloat16" in raw or raw in {"bf16", "bfloat16"}:
                    dtype, bytes_per_weight = "bfloat16", 2.0
                elif "float16" in raw or raw in {"fp16", "float16", "half"}:
                    dtype, bytes_per_weight = "float16", 2.0
                elif "int8" in raw:
                    dtype, bytes_per_weight = "int8", 1.0
                elif "int4" in raw or "4bit" in raw:
                    dtype, bytes_per_weight = "int4", 0.5
            except Exception:
                pass
        if weight_bytes > 0:
            observed_gb = weight_bytes / (1024 ** 3)
            # File size is a better lower bound than parameter-name heuristics. Add 20%% for
            # tokenizer/model metadata and normal inference buffers, not an arbitrary 35%%.
            estimate = observed_gb * 1.20
            return estimate, f"measured local weight files ({observed_gb:.2f} GB) + runtime allowance", observed_gb
        param_count_b = LocalLLMLoader._model_size_hint(model_name)
        observed_lower_bound = param_count_b * (1024 ** 3) * bytes_per_weight
        params_gb = observed_lower_bound / (1024 ** 3)
        estimate = params_gb * 1.20
        return estimate, f"parameter-name estimate using {dtype} ({param_count_b:g}B parameters × {bytes_per_weight:g} bytes/weight)", params_gb

    def _resolve_path(self, model_name: str, custom_path: str = "") -> str:
        """Perform the resolve path operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if custom_path.strip():
            return custom_path.strip()
        rel = config.LOCAL_LLM_MODELS.get(model_name)
        if not rel:
            return ""
        base = Path(config.HF_CACHE_DIR) / rel
        snapshots = base / "snapshots"
        if snapshots.exists():
            folders = [x for x in snapshots.iterdir() if x.is_dir()]
            if folders:
                return str(folders[0])
        return str(base)

    @staticmethod
    def _gpu_memory() -> dict[str, float | None]:
        """Return total and currently free CUDA memory in GiB without allocating tensors."""
        try:
            if not torch.cuda.is_available():
                return {"total_gb": None, "free_gb": None}
            free_b, total_b = torch.cuda.mem_get_info(0)
            return {"total_gb": float(total_b) / (1024 ** 3), "free_gb": float(free_b) / (1024 ** 3)}
        except Exception:
            return {"total_gb": None, "free_gb": None}

    @classmethod
    def resource_preflight(cls, model_name: str, custom_path: str = "", ram_budget_gb: float | None = None, gpu_budget_gb: float | None = None) -> dict:
        """Assess local-model feasibility without penalising an already-resident shared model.

        Agent Data Scientist, Plot, Report and Presentation are governed separately, but
        they may intentionally use the same local checkpoint. Once that checkpoint is
        resident in the shared LLM runtime, another Agent must not pretend that the full
        model will be allocated again. A small execution-time headroom check is still
        returned so the caller can see that the model is reused rather than reloaded.
        """
        # The application-level LLM runtime may already have the exact checkpoint loaded.
        # This is the key distinction between *model feasibility* and *additional memory
        # required to load another copy*.
        try:
            import services.llm_config as llm_config_module
            shared = getattr(llm_config_module, "_LOCAL_LOADER", None)
            if shared is not None and shared.loaded_model is not None and shared.loaded_model_name == model_name:
                return {
                    "status": "ready", "model": model_name, "resident": True,
                    "estimate_gb": 0.0, "cpu_required_gb": 0.0, "gpu_required_gb": 0.0,
                    "available_ram_gb": shared._system_memory_gb(),
                    "configured_ram_gb": ram_budget_gb,
                    "gpu_total_gb": shared._gpu_memory().get("total_gb"),
                    "gpu_free_gb": shared._gpu_memory().get("free_gb"),
                    "configured_gpu_vram_gb": gpu_budget_gb,
                    "issues": [],
                    "reason": "The selected local checkpoint is already resident in the shared LLM runtime; the Agent will reuse it instead of loading a second copy.",
                }
        except Exception:
            pass

        loader = cls()
        model_path = loader._resolve_path(model_name, custom_path)
        if not model_path or not Path(model_path).exists():
            return {"status":"blocked", "reason":f"Local model path was not found: {model_path}"}
        estimate, basis, observed = loader._model_memory_estimate(model_path, model_name)
        # Runtime headroom is deliberately separate from raw weight size. A model that fits only
        # exactly into VRAM is not considered safe because KV/cache and framework buffers remain.
        gpu_required = estimate * 1.15
        cpu_required = estimate + max(0.75, min(2.0, estimate * 0.15))
        available_ram = loader._system_memory_gb()
        gpu = loader._gpu_memory()
        budget_ram = float(ram_budget_gb) if ram_budget_gb is not None else available_ram
        budget_gpu = float(gpu_budget_gb) if gpu_budget_gb is not None else gpu.get("free_gb")
        issues=[]
        if budget_ram is not None and budget_ram < cpu_required:
            issues.append(f"configured RAM budget {budget_ram:.1f} GiB < estimated CPU working set {cpu_required:.1f} GiB")
        if available_ram is not None and available_ram < cpu_required:
            issues.append(f"currently free system RAM {available_ram:.1f} GiB < estimated CPU working set {cpu_required:.1f} GiB")
        if budget_gpu is not None and budget_gpu > 0 and budget_gpu < gpu_required:
            issues.append(f"configured GPU VRAM budget {budget_gpu:.1f} GiB < estimated GPU working set {gpu_required:.1f} GiB")
        if gpu.get("free_gb") is not None and gpu.get("free_gb") < gpu_required and (gpu_budget_gb is not None or gpu.get("total_gb",0) >= 4):
            issues.append(f"currently free GPU VRAM {gpu['free_gb']:.1f} GiB < estimated GPU working set {gpu_required:.1f} GiB")
        return {
            "status":"blocked" if issues else "ready", "model":model_name, "estimate_gb":estimate,
            "cpu_required_gb":cpu_required, "gpu_required_gb":gpu_required, "estimate_basis":basis,
            "observed_weight_gb":observed, "available_ram_gb":available_ram,
            "configured_ram_gb":ram_budget_gb, "gpu_total_gb":gpu.get("total_gb"),
            "gpu_free_gb":gpu.get("free_gb"), "configured_gpu_vram_gb":gpu_budget_gb,
            "issues":issues,
            "reason":("; ".join(issues) if issues else "Selected resource profile is sufficient for the estimated inference working set."),
        }

    def load_model(self, model_name, custom_path: str = "", ram_budget_gb: float | None = None, gpu_budget_gb: float | None = None):
        """Perform the load model operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if model_name not in config.LOCAL_LLM_MODELS and not custom_path.strip():
            return None
        model_path = self._resolve_path(model_name, custom_path)
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Local LLM model path was not found: {model_path}")

        resource = self.resource_preflight(model_name, custom_path, ram_budget_gb=ram_budget_gb, gpu_budget_gb=gpu_budget_gb)
        if resource.get("status") == "blocked":
            raise MemoryError(
                f"Local model '{model_name}' cannot be safely loaded with the selected resource profile. "
                f"Estimated CPU working set: {resource.get('cpu_required_gb', 0):.1f} GiB; "
                f"estimated GPU working set: {resource.get('gpu_required_gb', 0):.1f} GiB. "
                + resource.get("reason", "Insufficient RAM/VRAM.")
                + " Choose a larger resource profile, a quantized checkpoint, CPU execution, or an API model."
            )

        try:
            import warnings
            kwargs = {}
            use_cuda = False
            if torch.cuda.is_available():
                try:
                    gpu = self._gpu_memory()
                    requested_gpu = float(gpu_budget_gb) if gpu_budget_gb is not None else float(gpu.get("free_gb") or 0)
                    required_gpu = float(resource.get("gpu_required_gb") or 0)
                    use_cuda = requested_gpu >= required_gpu and float(gpu.get("free_gb") or 0) >= required_gpu
                except Exception:
                    use_cuda = False
            if use_cuda:
                kwargs = {"device_map": "auto", "dtype": torch.float16, "low_cpu_mem_usage": True}
            else:
                # Prefer BF16 for checkpoints that are distributed in BF16 when the
                # installed CPU runtime advertises support; otherwise use FP32.
                cpu_bf16 = False
                try:
                    cpu_bf16 = bool(getattr(torch.cpu, "is_bf16_supported", lambda: False)())
                except Exception:
                    cpu_bf16 = False
                requested_dtype = torch.bfloat16 if cpu_bf16 else torch.float32
                kwargs = {"device_map": "cpu", "dtype": requested_dtype, "low_cpu_mem_usage": True}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                except Exception:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)

            tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Normalise generation configuration for all governed Agents. Qwen BPE
            # tokenizers must not use WordPiece clean-up, and deterministic generation
            # must not carry stale checkpoint values such as max_length=20 or sampling
            # flags that are invalid when do_sample=False.
            try:
                generation_config = GenerationConfig(
                    do_sample=False,
                    max_new_tokens=256,
                    max_length=None,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                model.generation_config = generation_config
            except Exception:
                generation_config = getattr(model, "generation_config", None)
                if generation_config is not None:
                    for key, value in (("max_length", None), ("temperature", None), ("top_p", None), ("top_k", None)):
                        try: setattr(generation_config, key, value)
                        except Exception: pass
                    try: generation_config.do_sample = False
                    except Exception: pass
                    try: generation_config.max_new_tokens = 256
                    except Exception: pass
                    try: generation_config.pad_token_id = tokenizer.pad_token_id
                    except Exception: pass
            pipe_kind = "text2text-generation" if "flan" in model_name.lower() else "text-generation"
            pipe_kwargs = {"model": model, "tokenizer": tokenizer}
            if generation_config is not None:
                pipe_kwargs["generation_config"] = generation_config
            pipe = pipeline(pipe_kind, **pipe_kwargs)
            self.loaded_model = HuggingFacePipeline(pipeline=pipe)
            self.loaded_model_name = model_name
            return self.loaded_model
        except Exception as exc:
            self.loaded_model = None
            self.loaded_model_name = None
            raise RuntimeError(f"Could not safely load local LLM '{model_name}': {exc}") from exc

    def get_llm(self, model_name, custom_path: str = "", ram_budget_gb: float | None = None, gpu_budget_gb: float | None = None):
        """Perform the get llm operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        # If the exact checkpoint is already resident, reuse it before running a
        # second RAM preflight. Otherwise the already-loaded model would be counted
        # against available RAM and could incorrectly block Plot/Report/Presentation
        # after Agent Data Scientist has loaded the same small local model.
        if self.loaded_model is not None and self.loaded_model_name == model_name:
            return self.loaded_model
        return self.load_model(model_name, custom_path=custom_path, ram_budget_gb=ram_budget_gb, gpu_budget_gb=gpu_budget_gb)
