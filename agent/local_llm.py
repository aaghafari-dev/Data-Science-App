"""Module duty: Local llm.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
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
        params_gb = self_hint = LocalLLMLoader._model_size_hint(model_name) * bytes_per_weight / (1024 ** 0)
        estimate = params_gb * 1.20
        return estimate, f"parameter-name estimate using {dtype}", params_gb

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

    def load_model(self, model_name, custom_path: str = ""):
        """Perform the load model operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if model_name not in config.LOCAL_LLM_MODELS and not custom_path.strip():
            return None
        model_path = self._resolve_path(model_name, custom_path)
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Local LLM model path was not found: {model_path}")

        available_ram = self._system_memory_gb()
        estimated_cpu_gb, estimate_basis, observed_weight_gb = self._model_memory_estimate(model_path, model_name)
        if available_ram is not None and available_ram < estimated_cpu_gb + 1.5:
            raise MemoryError(
                f"Local model '{model_name}' has an estimated working-set requirement of about {estimated_cpu_gb:.1f} GB "
                f"({estimate_basis}), while only {available_ram:.1f} GB RAM is currently available. "
                "This is a resource-safety estimate, not a claim that the model intrinsically requires that exact amount of RAM. "
                "Close other applications, use a smaller/quantized checkpoint, or use an API model if necessary."
            )

        try:
            import warnings
            kwargs = {}
            use_cuda = False
            if torch.cuda.is_available():
                try:
                    props = torch.cuda.get_device_properties(0)
                    # Do not place local LLM weights on 2 GB-class devices.
                    use_cuda = float(props.total_memory) >= 4 * 1024**3
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

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            pipe_kind = "text2text-generation" if "flan" in model_name.lower() else "text-generation"
            # Avoid generation_config + explicit-generation-argument conflicts.
            pipe = pipeline(
                pipe_kind,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            self.loaded_model = HuggingFacePipeline(pipeline=pipe)
            self.loaded_model_name = model_name
            return self.loaded_model
        except Exception as exc:
            self.loaded_model = None
            self.loaded_model_name = None
            raise RuntimeError(f"Could not safely load local LLM '{model_name}': {exc}") from exc

    def get_llm(self, model_name, custom_path: str = ""):
        """Perform the get llm operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return self.loaded_model if self.loaded_model_name == model_name else self.load_model(model_name, custom_path=custom_path)
